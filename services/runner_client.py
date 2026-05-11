"""
RunnerClient — HTTP client for the llmmllab-runner service pool.

Routes requests among multiple runner instances based on health and
hardware capability (VRAM). Manages server lifecycle (acquire, release,
shutdown) and model discovery across all runners.

Uses a persistent ``httpx.AsyncClient`` with connection pooling to avoid
the overhead of opening a new TCP connection for every request.

Server Handle Lifecycle
-----------------------
Every handle returned by ``acquire_server()`` is automatically registered
in an internal registry. On application shutdown, ``aclose()`` calls
``shutdown_all_handles()`` which sends DELETE requests to the runner for
each registered handle, ensuring no orphaned llama.cpp servers remain.

The ``num_ctx`` parameter on ``acquire_server()`` is forwarded to the
runner so it can refuse to start servers with insufficient context
(relative to ``CONTEXT_MINIMUM_RATIO``).
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import httpx

from config import MODEL_CACHE_REFRESH_SEC, RUNNER_ENDPOINTS
from models import Model, ModelTask
from utils.logging import llmmllogger

# Suppress verbose httpx/httpcore trace and debug logs at module level.
# This ensures they're silenced before any client is created, regardless of
# how the root logger is configured by uvicorn or structlog.
for _lib_name in (
    "httpx",
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.proxy",
    "hpack",
):
    logging.getLogger(_lib_name).setLevel(logging.WARNING)

logger = llmmllogger.bind(component="runner_client")

# Timeouts for different request categories (configurable via env)
_HEALTH_TIMEOUT = httpx.Timeout(float(os.environ.get("RUNNER_HEALTH_TIMEOUT_SEC", "5.0")))
_FAST_TIMEOUT = httpx.Timeout(float(os.environ.get("RUNNER_FAST_TIMEOUT_SEC", "10.0")))
_ACQUIRE_TIMEOUT = httpx.Timeout(float(os.environ.get("RUNNER_ACQUIRE_TIMEOUT_SEC", "150.0")))

# Circuit breaker thresholds (configurable via env)
_MAX_ACQUIRE_FAILURES = int(os.environ.get("RUNNER_MAX_ACQUIRE_FAILURES", "3"))
_UNHEALTHY_WINDOW = float(os.environ.get("RUNNER_UNHEALTHY_WINDOW_SEC", "60.0"))
# Per-endpoint connection retries during acquire
_ACQUIRE_RETRIES = int(os.environ.get("RUNNER_ACQUIRE_RETRIES", "2"))


@dataclass(frozen=True)
class ServerHandle:
    """Reference to an allocated llama.cpp server on a runner."""

    base_url: str
    server_id: str
    runner_host: str


class RunnerClient:
    """HTTP client that routes requests among multiple runner instances.

    Maintains a persistent ``httpx.AsyncClient`` for connection reuse.
    Call ``aclose()`` during application shutdown to clean up.
    """

    def __init__(self, endpoints: Optional[list[str]] = None):
        self._endpoints = endpoints if endpoints is not None else list(RUNNER_ENDPOINTS)
        self._healthy: list[str] = []
        self._client: Optional[httpx.AsyncClient] = None
        self._model_map: Dict[str, List[str]] = {}
        self._refresh_task: Optional[asyncio.Task] = None
        self._unhealthy_since: Dict[str, float] = {}
        self._acquire_failures: Dict[str, int] = {}
        # Circuit breaker: skip a runner if it has >= this many consecutive
        # acquire failures within the last UNHEALTHY_WINDOW seconds.
        self._MAX_ACQUIRE_FAILURES = _MAX_ACQUIRE_FAILURES
        self._UNHEALTHY_WINDOW = _UNHEALTHY_WINDOW
        # Track active server IDs per runner endpoint for cleanup on failure
        self._active_servers_by_endpoint: Dict[str, set[str]] = {}
        # Registry of active server handles for cleanup on shutdown.
        self._active_handles: set[ServerHandle] = set()

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily create a shared ``httpx.AsyncClient``."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=_FAST_TIMEOUT,
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                    keepalive_expiry=120,
                ),
            )
        return self._client

    # ------------------------------------------------------------------
    # Retry-After aware request proxying
    # ------------------------------------------------------------------

    async def proxy_request(
        self,
        handle: ServerHandle,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        timeout: float = 120.0,  # total backoff budget in seconds
    ) -> httpx.Response:
        """Send a request to a server handle, respecting Retry-After on 503.

        When the runner proxy returns 503 (all slots busy), reads the
        ``Retry-After`` header and sleeps before retrying.  Uses
        exponential backoff (2 s, 4 s, 8 s, …) capped by the ``Retry-After``
        value, and stops when the cumulative backoff would exceed *timeout*.

        Parameters
        ----------
        handle:
            The ``ServerHandle`` returned by ``acquire_server()``.
        method:
            HTTP method (``GET``, ``POST``, etc.).
        path:
            Path to append to the server's base URL.
        json:
            Optional JSON body.
        stream:
            If ``True`` the caller is responsible for draining and closing
            the response.
        timeout:
            Total backoff budget in seconds.  Once the cumulative sleep
            would exceed this, the last 503 response is returned.

        Returns
        -------
        httpx.Response
            The final response (may still be 503 if timeout exhausted).
        """
        url = f"{handle.base_url}/{path.lstrip('/')}"
        client = self._get_client()

        # First attempt (no backoff yet)
        response = await client.request(
            method=method,
            url=url,
            json=json,
            timeout=_ACQUIRE_TIMEOUT,
            stream=stream,
        )

        if response.status_code != 503:
            return response

        last_response = response
        elapsed = 0.0
        attempt = 0

        while elapsed < timeout:
            attempt += 1

            # Read Retry-After header (seconds as integer)
            retry_after = 30  # default fallback
            retry_header = response.headers.get("retry-after")
            if retry_header:
                try:
                    retry_after = int(retry_header)
                except (ValueError, TypeError):
                    pass

            # Exponential backoff: 2^attempt seconds, capped by Retry-After
            backoff = min(2**attempt, retry_after)

            if elapsed + backoff > timeout:
                break

            logger.warning(
                "Runner returned 503, backing off before retry",
                extra={
                    "server_id": handle.server_id,
                    "retry_after": retry_after,
                    "backoff": backoff,
                    "attempt": attempt,
                    "elapsed": round(elapsed, 1),
                },
            )
            await asyncio.sleep(backoff)
            elapsed += backoff

            response = await client.request(
                method=method,
                url=url,
                json=json,
                timeout=_ACQUIRE_TIMEOUT,
                stream=stream,
            )

            if response.status_code != 503:
                return response

            last_response = response

        logger.error(
            "Runner returned 503 after backoff timeout",
            extra={
                "server_id": handle.server_id,
                "elapsed": round(elapsed, 1),
                "timeout": timeout,
            },
        )

        return last_response

    async def shutdown_all_handles(self) -> None:
        """Shut down all registered server handles on the runner.

        Called during application shutdown to ensure no orphaned llama.cpp
        servers remain running on the runner nodes.
        """
        if not self._active_handles:
            return

        handles_to_shutdown = list(self._active_handles)
        logger.info(
            f"Shutting down {len(handles_to_shutdown)} active server handle(s)"
        )

        for handle in handles_to_shutdown:
            try:
                await self.shutdown_server(handle)
            except Exception as e:
                logger.warning(
                    f"Failed to shutdown handle {handle.server_id} during cleanup: {e}"
                )

        self._active_handles.clear()

    async def aclose(self) -> None:
        """Close the shared HTTP client and release active servers.  Call during app shutdown."""
        # Shut down all active server handles before closing the client
        await self.shutdown_all_handles()

        if self._refresh_task is not None:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            self._refresh_task = None
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def _is_circuit_open(self, endpoint: str) -> bool:
        """Check if the circuit breaker is open for this runner."""
        now = time.monotonic()
        # Reset if outside the unhealthy window
        if endpoint in self._unhealthy_since:
            if now - self._unhealthy_since[endpoint] > self._UNHEALTHY_WINDOW:
                self._unhealthy_since.pop(endpoint, None)
                self._acquire_failures.pop(endpoint, None)
                return False
        failures = self._acquire_failures.get(endpoint, 0)
        return failures >= self._MAX_ACQUIRE_FAILURES

    def _record_acquire_failure(self, endpoint: str) -> None:
        """Record an acquire failure and potentially open the circuit."""
        self._acquire_failures[endpoint] = self._acquire_failures.get(endpoint, 0) + 1
        self._unhealthy_since[endpoint] = time.monotonic()
        if endpoint in self._healthy:
            self._healthy.remove(endpoint)

    def _record_acquire_success(self, endpoint: str) -> None:
        """Reset failure count on success."""
        self._acquire_failures.pop(endpoint, None)
        self._unhealthy_since.pop(endpoint, None)

    def _trip_circuit_and_cleanup(self, endpoint: str) -> None:
        """Immediately open the circuit breaker and clean up orphaned servers.

        Called when a runner is determined to be unhealthy (connection error,
        HTTP error, or any acquire failure).  Forces the circuit open so we
        don't waste time retrying, and attempts to kill any known servers on
        this runner to free VRAM.
        """
        self._acquire_failures[endpoint] = self._MAX_ACQUIRE_FAILURES
        self._unhealthy_since[endpoint] = time.monotonic()
        if endpoint in self._healthy:
            self._healthy.remove(endpoint)
        asyncio.create_task(self._cleanup_endpoint(endpoint))

    async def _cleanup_endpoint(self, endpoint: str) -> None:
        """Best-effort attempt to kill all known servers on an endpoint.

        Called when a runner becomes unreachable or unhealthy, to free VRAM
        that would otherwise be wasted on orphaned servers.
        """
        server_ids = self._active_servers_by_endpoint.pop(endpoint, set())
        if not server_ids:
            return
        logger.info(
            f"Cleaning up {len(server_ids)} server(s) on unreachable endpoint {endpoint}"
        )
        client = self._get_client()
        for sid in server_ids:
            try:
                await client.delete(f"{endpoint}/v1/server/{sid}", timeout=_FAST_TIMEOUT)
                logger.info(f"Cleaned up server {sid} on {endpoint}")
            except Exception as e:
                logger.warning(
                    f"Failed to clean up server {sid} on {endpoint}: {e}"
                )

    def _is_connection_error(self, exc: Exception) -> bool:
        """Check if the exception is a connection-level error (disconnect, timeout, etc.)."""
        return isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout,
                                httpx.ReadTimeout, httpx.RemoteProtocolError,
                                ConnectionError))

    async def _health(self, endpoint: str) -> Optional[dict]:
        """Check health of a single runner. Returns health dict or None."""
        # Skip runners with open circuit breaker
        if self._is_circuit_open(endpoint):
            logger.debug(f"Skipping health check for {endpoint}: circuit breaker open")
            return None
        try:
            client = self._get_client()
            resp = await client.get(f"{endpoint}/health", timeout=_HEALTH_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                if endpoint not in self._healthy:
                    self._healthy.append(endpoint)
                return data
            else:
                if endpoint in self._healthy:
                    self._healthy.remove(endpoint)
                self._invalidate_model_map_for_endpoint(endpoint)
                return None
        except Exception as e:
            logger.warning(f"Runner {endpoint} health check failed: {e}")
            if endpoint in self._healthy:
                self._healthy.remove(endpoint)
           # Connection-level errors indicate an unreachable runner —
            # trip the circuit breaker and attempt to clean up orphaned
            # servers so VRAM isn't wasted.
            if self._is_connection_error(e):
                self._acquire_failures[endpoint] = self._MAX_ACQUIRE_FAILURES
                self._unhealthy_since[endpoint] = time.monotonic()
                asyncio.create_task(self._cleanup_endpoint(endpoint))
            self._invalidate_model_map_for_endpoint(endpoint)
            return None

    def _invalidate_model_map_for_endpoint(self, endpoint: str) -> None:
        """Remove an endpoint from the model map immediately when it becomes unhealthy.

        This avoids waiting for the next scheduled refresh (default 60 s) and
        prevents ``acquire_server()`` from routing to a dead runner.
        """
        for model_id, endpoints in list(self._model_map.items()):
            if endpoint in endpoints:
                endpoints.remove(endpoint)
                if not endpoints:
                    del self._model_map[model_id]
        logger.info(
            f"Invalidated model map for unhealthy endpoint {endpoint}",
        )

    async def validate_server_handle(self, handle: ServerHandle) -> bool:
        """Check if a server handle is still valid by hitting the server's /health.

        Returns ``True`` if the llama.cpp server behind the handle responds
        with HTTP 200, ``False`` otherwise.
        """
        try:
            client = self._get_client()
            resp = await client.get(
                f"{handle.base_url}/health",
                timeout=httpx.Timeout(3.0),
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Runner selection
    # ------------------------------------------------------------------

    async def _select_runner(self, model_id: str) -> Optional[str]:
        """Iterate endpoints, pick highest VRAM runner with matching model."""
        best_url = None
        best_vram = -1
        for endpoint in self._endpoints:
            health = await self._health(endpoint)
            if not health:
                continue
            models = health.get("models", [])
            if model_id not in models:
                continue
            vram = health.get("gpu", {}).get("available_vram_bytes", 0)
            if vram > best_vram:
                best_vram = vram
                best_url = endpoint
        return best_url

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def register_handle(self, handle: ServerHandle) -> None:
        """Register a server handle for lifecycle cleanup on shutdown."""
        self._active_handles.add(handle)

    def unregister_handle(self, handle: ServerHandle) -> None:
        """Unregister a server handle after it has been released/shutdown."""
        self._active_handles.discard(handle)

    async def acquire_server(self, model_id: str, num_ctx: Optional[int] = None, **kwargs) -> ServerHandle:
        """Acquire a new llama.cpp server from a runner.

        Uses the cached model map for fast routing. Falls back to
        health-check scan if the model isn't in the map.

        Parameters
        ----------
        model_id:
            Identifier of the model to load.
        num_ctx:
            Requested context window size.  Passed to the runner so it can
            refuse to start a server when the available context is smaller
            than ``CONTEXT_MINIMUM_RATIO * num_ctx``.
        **kwargs:
            Forwarded for compatibility (task, config_override).

        Returns
        -------
        ServerHandle
            Connection details for the allocated server (auto-registered
            for shutdown cleanup).

        Raises
        ------
        RuntimeError
            If no runner can satisfy the request.
        """
        payload: dict[str, Any] = {"model_id": model_id}
        if num_ctx is not None:
            payload["num_ctx"] = num_ctx
        config_override = kwargs.get("config_override")
        if config_override:
            payload["config_override"] = config_override

        # Fast path: use cached model map
        mapped_endpoints = self._model_map.get(model_id)
        if mapped_endpoints:
            ordered = list(mapped_endpoints)
        else:
            # Fallback: health-check scan
            best = await self._select_runner(model_id)
            if best:
                ordered = [best]
                for ep in self._endpoints:
                    if ep != best:
                        ordered.append(ep)
            else:
                ordered = list(self._endpoints)

        last_error = None
        for endpoint in ordered:
            # Skip runners with open circuit breaker
            if self._is_circuit_open(endpoint):
                logger.warning(
                    f"Skipping {endpoint}: circuit breaker open "
                    f"({self._acquire_failures.get(endpoint, 0)} failures)"
                )

                continue

            # Retry connection-level errors per endpoint (configurable via RUNNER_ACQUIRE_RETRIES)
            max_retries = _ACQUIRE_RETRIES
            for attempt in range(max_retries + 1):
                try:
                    client = self._get_client()
                    resp = await client.post(
                        f"{endpoint}/v1/server/create",
                        json=payload,
                        timeout=_ACQUIRE_TIMEOUT,
                    )

                    if resp.status_code == 507:
                        logger.warning(
                            f"Runner {endpoint} returned 507, trying next runner"
                        )
                        last_error = "Insufficient capacity"
                        break  # no point retrying 507

                    resp.raise_for_status()
                    data = resp.json()
                    handle = ServerHandle(
                        base_url=f"{endpoint}/v1/server/{data['server_id']}",
                        server_id=data["server_id"],
                        runner_host=endpoint,
                    )
                    logger.info(f"Acquired server {handle.server_id} from {endpoint}")
                    self._record_acquire_success(endpoint)
                    self._active_servers_by_endpoint.setdefault(endpoint, set()).add(data["server_id"])
                    self._active_handles.add(handle)
                    self._schedule_refresh()
                    return handle

                except Exception as e:
                    last_error = str(e)
                    is_conn_err = self._is_connection_error(e)

                    # Any acquire failure (connection error, HTTP error, etc.)
                    # means the runner is unhealthy.  Trip the circuit
                    # immediately and clean up orphaned servers so VRAM isn't
                    # wasted.
                    if is_conn_err:
                        logger.warning(
                            f"Connection error from {endpoint}, tripping circuit breaker: {e}"
                        )
                    else:
                        logger.warning(
                            f"Acquire error from {endpoint}, tripping circuit breaker: {e}"
                        )
                    self._trip_circuit_and_cleanup(endpoint)
                    break  # move to next endpoint

        raise RuntimeError(
            f"No healthy runner available for model {model_id}. "
            f"Last error: {last_error}"
        )

    async def release_server(self, handle: ServerHandle) -> None:
        """Release an acquired server back to the runner."""
        try:
            client = self._get_client()
            resp = await client.post(
                f"{handle.runner_host}/v1/server/{handle.server_id}/release"
            )
            resp.raise_for_status()
            logger.info(f"Released server {handle.server_id}")
            self._active_handles.discard(handle)
        except Exception as e:
            logger.error(f"Failed to release server {handle.server_id}: {e}")
            raise
        finally:
            # Remove from active tracking so cleanup doesn't try to kill it
            servers = self._active_servers_by_endpoint.get(handle.runner_host)
            if servers:
                servers.discard(handle.server_id)
            self.unregister_handle(handle)

    async def shutdown_server(self, handle: ServerHandle) -> None:
        """Permanently shut down a server on the runner."""
        try:
            client = self._get_client()
            resp = await client.delete(
                f"{handle.runner_host}/v1/server/{handle.server_id}"
            )
            resp.raise_for_status()
            logger.info(f"Shutdown server {handle.server_id}")
        except Exception as e:
            logger.error(f"Failed to shutdown server {handle.server_id}: {e}")
            raise
        finally:
            # Remove from active tracking
            servers = self._active_servers_by_endpoint.get(handle.runner_host)
            if servers:
                servers.discard(handle.server_id)
            self.unregister_handle(handle)

    # ------------------------------------------------------------------
    # Model map
    # ------------------------------------------------------------------

    async def refresh_model_map(self) -> None:
        """Query all runners and build a model_id -> [endpoints] map."""
        new_map: Dict[str, List[str]] = {}
        client = self._get_client()
        tasks = []
        for endpoint in self._endpoints:
            async def fetch_models(ep=endpoint):
                try:
                    resp = await client.get(f"{ep}/v1/models")
                    if resp.status_code == 200:
                        return [(m["id"], ep) for m in resp.json() if "id" in m]
                except Exception as e:
                    logger.warning(f"Failed to list models from {ep}: {e}")
                return []
            tasks.append(fetch_models())
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                for model_id, endpoint in result:
                    if model_id not in new_map:
                        new_map[model_id] = []
                    new_map[model_id].append(endpoint)
        self._model_map = new_map
        logger.info(f"Model map refreshed: {len(new_map)} models across {len(self._endpoints)} endpoints")

    def _schedule_refresh(self) -> None:
        """Schedule a model-map refresh after a delay, cancelling any pending one."""
        if self._refresh_task is not None:
            self._refresh_task.cancel()
        async def _do_refresh():
            try:
                await asyncio.sleep(MODEL_CACHE_REFRESH_SEC)
                await self.refresh_model_map()
            except asyncio.CancelledError:
                pass
            finally:
                self._refresh_task = None
        self._refresh_task = asyncio.create_task(_do_refresh())

    # ------------------------------------------------------------------
    # Slot availability
    # ------------------------------------------------------------------

    async def check_slot_availability(self, model_id: str) -> bool:
        """Check if a request for *model_id* can proceed without hitting 503.

        Returns ``True`` if an existing server has free slots, or if
        sufficient VRAM exists on a runner to start a new server.
        Returns ``True`` on any check failure (fail-open).
        """
        _slots_timeout = httpx.Timeout(3.0)
        client = self._get_client()

        # Case 1: Check active handles for free slots
        for handle in self._active_handles:
            try:
                resp = await client.get(
                    f"{handle.base_url}/slots",
                    timeout=_slots_timeout,
                )
                if resp.status_code == 200:
                    slots = resp.json()
                    if slots and any(
                        not s.get("is_processing", False) for s in slots
                    ):
                        return True
            except Exception:
                continue

        # Case 2: No active server with free slots — check if a new one
        # can be started (VRAM vs model size)
        endpoints = self._model_map.get(model_id, list(self._endpoints))
        for endpoint in endpoints:
            try:
                health_resp = await client.get(
                    f"{endpoint}/health",
                    timeout=_HEALTH_TIMEOUT,
                )
                if health_resp.status_code != 200:
                    continue
                health = health_resp.json()
                gpu_info = health.get("gpu", {})
                # Sum free_mb across all GPUs, convert to bytes
                available_vram = sum(
                    v.get("free_mb", 0)
                    for v in gpu_info.values()
                    if isinstance(v, dict)
                ) * 1024 * 1024

                model_resp = await client.get(
                    f"{endpoint}/v1/models/{model_id}",
                    timeout=_FAST_TIMEOUT,
                )
                if model_resp.status_code != 200:
                    continue
                model_data = model_resp.json()
                model_size = model_data.get("details", {}).get("size", 0) or 0
                # 128 MB overhead for context, KV cache, etc.
                required = model_size + (128 * 1024 * 1024)
                if available_vram >= required:
                    return True
            except Exception:
                continue

        return False

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    async def list_models(self) -> List[Model]:
        """List all available models across all runners, deduplicated by id."""
        seen_ids: set[str] = set()
        all_models: List[Model] = []
        client = self._get_client()

        tasks = []
        for endpoint in self._endpoints:

            async def fetch_models(ep=endpoint):
                try:
                    resp = await client.get(f"{ep}/v1/models")
                    if resp.status_code == 200:
                        return resp.json()
                except Exception as e:
                    logger.warning(f"Failed to list models from {ep}: {e}")
                return []

            tasks.append(fetch_models())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                for model_data in result:
                    mid = model_data.get("id")
                    if mid and mid not in seen_ids:
                        seen_ids.add(mid)
                        all_models.append(Model(**model_data))

        return all_models

    async def model_by_task(self, task: ModelTask) -> Optional[Model]:
        """Find the first model matching the given task across all runners."""
        client = self._get_client()
        for endpoint in self._endpoints:
            try:
                resp = await client.get(
                    f"{endpoint}/v1/models", params={"task": task.value}
                )
                if resp.status_code == 200:
                    models = resp.json()
                    for model_data in models:
                        if model_data.get("task") == task.value:
                            return Model(**model_data)
            except Exception as e:
                logger.warning(f"Failed to query models from {endpoint}: {e}")
                continue
        return None


# Module-level singleton
runner_client = RunnerClient()
