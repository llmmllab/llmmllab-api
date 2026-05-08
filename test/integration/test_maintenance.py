"""Integration tests for DatabaseMaintenanceService — exercises real PostgreSQL."""

import os

import pytest
from sqlalchemy import text

from db.maintenance import DatabaseMaintenanceService

# ---------------------------------------------------------------------------
# Regression test for issue #23: VACUUM ANALYZE must run in autocommit mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vacuum_analyze_runs_in_autocommit_mode(
    session_factory, engine,
):
    """Verify that _run_vacuum_analyze opens the connection with
    isolation_level='AUTOCOMMIT'.  Without this, PostgreSQL rejects
    VACUUM with 'cannot run VACUUM in a transaction block'.

    We confirm by actually executing VACUUM ANALYZE against a real
    TimescaleDB container and asserting it completes without error.
    """
    service = DatabaseMaintenanceService()
    await service.initialize(engine, session_factory)

    # Disable REINDEX so the test is fast and deterministic
    original = os.environ.pop("DB_REINDEX_ON_MAINTENANCE", None)
    try:
        del os.environ["DB_REINDEX_ON_MAINTENANCE"]
    except KeyError:
        pass

    # This would raise "cannot run VACUUM in a transaction block" if
    # the connection were NOT in autocommit mode.
    try:
        await service._run_vacuum_analyze()

        # DB should still be healthy
        async with session_factory() as session:
            r = await session.execute(text("SELECT 1 as ok"))
            assert r.scalar() == 1
    finally:
        if original is not None:
            os.environ["DB_REINDEX_ON_MAINTENANCE"] = original


@pytest.mark.asyncio
async def test_vacuum_analyze_requires_autocommit_mode(
    session_factory, engine,
):
    """Confirm that VACUUM ANALYZE succeeds with AUTOCOMMIT isolation
    and fails without it — proving the fix is necessary.

    This is a regression test for issue #23: the original code opened
    a regular connection (implicit transaction) and tried to run
    VACUUM ANALYZE, which PostgreSQL rejects with:
    'cannot run VACUUM in a transaction block'.
    """
    # Positive case: AUTOCOMMIT mode works
    async with engine.connect().execution_options(
        isolation_level="AUTOCOMMIT"
    ) as conn:
        await conn.execute(text("VACUUM ANALYZE"))

    # Negative case: an explicit transaction block rejects VACUUM.
    # We must BEGIN a transaction ourselves because NullPool + asyncpg
    # may not start a transaction implicitly on engine.connect().
    with pytest.raises(Exception, match="transaction"):
        conn = await engine.connect()
        try:
            await conn.execute(text("BEGIN"))
            await conn.execute(text("VACUUM ANALYZE"))
        except Exception:
            await conn.close()
            raise


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
