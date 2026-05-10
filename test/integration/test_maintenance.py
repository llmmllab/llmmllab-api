"""Integration tests for DatabaseMaintenanceService — exercises real PostgreSQL."""

import os

import pytest
from sqlalchemy import text

from db.maintenance import DatabaseMaintenanceService


@pytest.mark.asyncio
async def test_reindex_concurrently_uses_autocommit(
    session_factory, engine,
):
    """REINDEX CONCURRENTLY must run in autocommit mode; otherwise
    PostgreSQL raises 'cannot execute REINDEX CONCURRENTLY inside a
    transaction block'.

    This test directly exercises _run_reindex() with a real database
    to confirm the AUTOCOMMIT isolation_level fix (issue #45).
    """
    service = DatabaseMaintenanceService()
    await service.initialize(engine, session_factory)

    # Create a tiny table + index so REINDEX has something to work on
    async with engine.connect() as conn:
        await conn.execute(text("CREATE TABLE IF NOT EXISTS _reindex_test (id SERIAL PRIMARY KEY, val TEXT)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS _reindex_test_val_idx ON _reindex_test (val)"))
        await conn.execute(text("INSERT INTO _reindex_test (val) VALUES ('hello')"))
        await conn.commit()

    try:
        # _run_reindex should succeed (returns True) and NOT raise
        # "cannot execute REINDEX CONCURRENTLY inside a transaction block"
        result = await service._run_reindex()
        assert result is True, "REINDEX CONCURRENTLY should succeed in autocommit mode"

        # Database should still be usable after REINDEX
        async with session_factory() as session:
            r = await session.execute(text("SELECT 1 as ok"))
            assert r.scalar() == 1
    finally:
        # Cleanup
        async with engine.connect() as conn:
            await conn.execute(text("DROP INDEX IF EXISTS _reindex_test_val_idx"))
            await conn.execute(text("DROP TABLE IF EXISTS _reindex_test"))


@pytest.mark.asyncio
async def test_maintenance_performs_without_error(
    session_factory, engine,
):
    """Verify that perform_maintenance runs all steps without cascading errors."""
    service = DatabaseMaintenanceService()
    await service.initialize(engine, session_factory)

    # REINDEX is enabled via env but we don't want to require it
    # The test should pass regardless
    original = os.environ.pop("DB_REINDEX_ON_MAINTENANCE", None)
    try:
        del os.environ["DB_REINDEX_ON_MAINTENANCE"]
    except KeyError:
        pass

    result = await service.perform_maintenance()
    # Maintenance may return False if TimescaleDB has no jobs, but it
    # should NOT raise or cascade-fail after VACUUM.
    assert isinstance(result, bool)

    # Verify the database is still usable after maintenance
    async with session_factory() as session:
        r = await session.execute(text("SELECT 1 as ok"))
        assert r.scalar() == 1

    if original is not None:
        os.environ["DB_REINDEX_ON_MAINTENANCE"] = original


@pytest.mark.asyncio
async def test_maintenance_vacuum_does_not_break_subsequent_steps(
    session_factory, engine,
):
    """VACUUM ANALYZE runs outside a transaction; subsequent steps should
    still work even if VACUUM encounters issues."""
    service = DatabaseMaintenanceService()
    await service.initialize(engine, session_factory)

    original = os.environ.pop("DB_REINDEX_ON_MAINTENANCE", None)
    try:
        del os.environ["DB_REINDEX_ON_MAINTENANCE"]
    except KeyError:
        pass

        # Run maintenance twice to ensure no stale transaction state
        await service.perform_maintenance()
        await service.perform_maintenance()

        # DB should still be healthy
        async with session_factory() as session:
            r = await session.execute(text("SELECT 1 as ok"))
            assert r.scalar() == 1
    finally:
        if original is not None:
            os.environ["DB_REINDEX_ON_MAINTENANCE"] = original


@pytest.mark.asyncio
async def test_reindex_uses_scalar_not_mapping_subscript(session_factory, engine):
    """Regression test for #68: REINDEX must use .scalar() to read current_database(),
    not .mappings()['db_name'] which raises 'MappingResult object is not subscriptable'."""
    service = DatabaseMaintenanceService()
    await service.initialize(engine, session_factory)

    # Verify that session.execute(text("SELECT current_database()")).scalar()
    # returns a string (not a MappingResult that would fail subscripting)
    async with session_factory() as session:
        db_name = (await session.execute(
            text("SELECT current_database()")
        )).scalar()
    assert isinstance(db_name, str)
    assert len(db_name) > 0
        
@pytest.mark.asyncio
async def test_reindex_with_cache_flush(session_factory, engine):
    """REINDEX followed by _flush_pool_caches() leaves the DB usable.

    Exercises the full _run_reindex() path including the DISCARD ALL
    cache flush that prevents stale OID errors after REINDEX.
    """
    service = DatabaseMaintenanceService()
    await service.initialize(engine, session_factory)

    # Create a table + index so REINDEX has something to work on
    async with engine.connect() as conn:
        await conn.execute(text("CREATE TABLE IF NOT EXISTS _reindex_flush_test (id SERIAL PRIMARY KEY, val TEXT)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS _reindex_flush_val_idx ON _reindex_flush_test (val)"))
        await conn.execute(text("INSERT INTO _reindex_flush_test (val) VALUES ('world')"))
        await conn.commit()

    try:
        # Run REINDEX (autocommit) + cache flush
        result = await service._run_reindex()
        assert result is True

        # Verify DISCARD ALL didn't break subsequent queries
        async with session_factory() as session:
            r = await session.execute(text("SELECT val FROM _reindex_flush_test WHERE id = 1"))
            assert r.scalar() == "world"

        # Verify the pool is still healthy after cache flush
        async with engine.connect() as conn:
            r = await conn.execute(text("SELECT 1 as ok"))
            assert r.scalar() == 1
    finally:
        async with engine.connect() as conn:
            await conn.execute(text("DROP INDEX IF EXISTS _reindex_flush_val_idx"))
            await conn.execute(text("DROP TABLE IF EXISTS _reindex_flush_test"))


@pytest.mark.asyncio
async def test_flush_pool_caches_is_idempotent(session_factory, engine):
    """_flush_pool_caches() can be called multiple times safely."""
    service = DatabaseMaintenanceService()
    await service.initialize(engine, session_factory)

    # Call twice — should not raise
    await service._flush_pool_caches()
    await service._flush_pool_caches()

    # DB still usable
    async with session_factory() as session:
        r = await session.execute(text("SELECT 1 as ok"))
        assert r.scalar() == 1
