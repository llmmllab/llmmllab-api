"""
Unit tests for DatabaseMaintenanceService — VACUUM ANALYZE fix.

Verifies that VACUUM ANALYZE uses AUTOCOMMIT isolation level so it
executes outside a transaction block (PostgreSQL requirement).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from db.maintenance import DatabaseMaintenanceService


@pytest.mark.asyncio
class TestVacuumAnalyze:
    """VACUUM ANALYZE must use AUTOCOMMIT isolation level."""

    def _make_service(self):
        svc = DatabaseMaintenanceService()
        svc.engine = MagicMock()
        return svc

    async def test_vacuum_uses_autocommit_isolation(self):
        """_run_vacuum_analyze passes isolation_level=AUTOCOMMIT to connect()."""
        svc = self._make_service()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        svc.engine.connect = MagicMock(return_value=mock_ctx)

        await svc._run_vacuum_analyze()

        # Verify connect() was called with AUTOCOMMIT isolation
        svc.engine.connect.assert_called_once_with(
            execution_options={"isolation_level": "AUTOCOMMIT"}
        )

    async def test_vacuum_executes_vacuum_analyze(self):
        """_run_vacuum_analyze runs the VACUUM ANALYZE statement."""
        svc = self._make_service()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        svc.engine.connect = MagicMock(return_value=mock_ctx)

        await svc._run_vacuum_analyze()

        # Verify VACUUM ANALYZE was executed
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0][0]
        assert "VACUUM ANALYZE" in str(call_args)

    async def test_vacuum_handles_exception(self):
        """_run_vacuum_analyze catches and logs exceptions without re-raising."""
        svc = self._make_service()

        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=RuntimeError("DB error"))
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        svc.engine.connect = MagicMock(return_value=mock_ctx)

        # Should not raise
        await svc._run_vacuum_analyze()
