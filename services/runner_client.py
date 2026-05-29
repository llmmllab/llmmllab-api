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
runner, which refuses to start servers when the requested context exceeds
the model's configured context window (returns HTTP 507).
"""

import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from config import CACHE_TIMEOUT_MIN, MODEL_CACHE_REFRESH_SEC, RUNNER_ENDPOINTS
from models import Model, ModelTask
from utils.logging import llmmllogger, _session_id_ctx

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

# Timeouts + circuit-breaker thresholds live in config.py so all env
# knobs are documented and overridable from one place.  See README's
# "Runner / Inference" section.
from config import (
    RUNNER_ACQUIRE_RETRIES as _ACQUIRE_RETRIES,
    RUNNER_ACQUIRE_TIMEOUT_SEC,
    RUNNER_FAST_TIMEOUT_SEC,
    RUNNER_HEALTH_TIMEOUT_SEC,
    RUNNER_MAX_ACQUIRE_FAILURES as _MAX_ACQUIRE_FAILURES,
    RUNNER_UNHEALTHY_WINDOW_SEC as _UNHEALTHY_WINDOW,
)

_HEALTH_TIMEOUT = httpx.Timeout(RUNNER_HEALTH_TIMEOUT_SEC)
_FAST_TIMEOUT = httpx.Timeout(RUNNER_FAST_TIMEOUT_SEC)

# Upper bound on the per-session sticky-pin map.  Sessions are
# transient (one per Claude Code / openclaw conversation, plus one
# per cron run), so this caps memory at ~LIMIT × ~64 bytes ≈ a few MB
# at the high end.  Oldest entries are evicted on overflow, which is
# fine — an evicted session's next acquire just re-pins via the
# ranked path.
_PER_SESSION_PIN_LIMIT = 4096
_ACQUIRE_TIMEOUT = httpx.Timeout(RUNNER_ACQUIRE_TIMEOUT_SEC)


@dataclass(frozen=True)
class ServerHandle:
    """Reference to an allocated llama.cpp server on a runner."""

    base_url: str
    server_id: str
    runner_host: str
    # The model this handle was acquired for.  Used by the sticky-pin
    # path in ``_select_runner`` to detect "the sticky endpoint is
    # already busy serving this same model — try a peer" without an
    # extra network round-trip.  Empty string for legacy handles
    # (e.g. embedding only) that haven't been updated.
    model_id: str = ""


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
        # Per-(endpoint, model_id) tensor_split string, captured from
        # the runner's /v1/models response.  Used by ``_select_runner``
        # to compute *effective* free VRAM for a model — the model is
        # pinned to specific GPUs via tensor_split (e.g. "1,0,0" =>
        # device 0 only), so ranking by total free VRAM across all
        # GPUs on a runner over-counts capacity for pinned models.
        # None means "no pinning, use total".
        self._model_tensor_split: Dict[tuple[str, str], Optional[str]] = {}
        # Pipeline-name → endpoints map, populated alongside the
        # model_map from each runner's /v1/models response by filtering
        # on ``provider == 'in_process'`` and indexing by ``pipeline``.
        # Used by ``_select_pipeline_runner`` to route in-process
        # pipeline calls (rembg, img23d, ...) to whichever runner's
        # yaml declares the corresponding model.
        self._pipeline_map: Dict[str, List[str]] = {}
        self._refresh_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
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
        # Per-endpoint last-seen runner startup_epoch (unix_ms). On change,
        # every handle from that endpoint is dead and must be purged.
        self._runner_epochs: Dict[str, int] = {}
        # Serializes check-and-update of `_runner_epochs` so concurrent
        # callers can't both observe an unchanged epoch, miss the purge,
        # and race on `_active_handles`.
        self._epoch_lock: asyncio.Lock = asyncio.Lock()
        # Sticky model→endpoint pinning.  Once a model is first acquired
        # from an endpoint, future acquires for the same model_id prefer
        # that same endpoint as long as it's still healthy + still hosts
        # the model.  Maximises KV-cache reuse on the runner side (the
        # llama.cpp server stays warm for the same model and benefits
        # from cache_prompt across sessions).
        self._last_endpoint_for_model: Dict[str, str] = {}
        # Per-session sticky pin: ``(session_id, model_id) → endpoint``.
        # When concurrent sessions both want the same model, the global
        # ``_last_endpoint_for_model`` pin would bounce them onto whichever
        # runner most recently acquired, forcing them to alternate and
        # serialise on a parallel=1 slot.  Per-session pins instead let
        # each session settle on whichever runner it first acquired
        # from, so two sessions on the same model converge on two
        # different runners and run concurrently.
        # Bounded via :data:`_PER_SESSION_PIN_LIMIT`; oldest evicted.
        self._last_endpoint_per_session: "OrderedDict[tuple[str, str], str]" = (
            OrderedDict()
        )

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

    def _session_headers(self) -> Dict[str, str]:
        """Return X-Session-ID header from current context, if available."""
        sid = _session_id_ctx.get()
        return {"X-Session-ID": sid} if sid else {}

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
        headers = self._session_headers()

        # Caller's ``timeout`` is both the 503-backoff budget AND the
        # per-request httpx timeout — these used to diverge (httpx hard-
        # coded ``_ACQUIRE_TIMEOUT`` = 150 s regardless of the arg) which
        # broke long-running image-edit calls.  Use the floor of 150 s so
        # short-lived requests still get the historical timeout; honour
        # bigger caller budgets verbatim.
        per_request_timeout = httpx.Timeout(max(timeout, _ACQUIRE_TIMEOUT.read or 150.0))

        async def _send_once() -> httpx.Response:
            """Issue a single upstream request, propagating CancelledError.

            On caller cancellation, the in-flight httpx coroutine is
            cancelled at the await — httpx then closes its in-flight
            stream/connection automatically.  We catch CancelledError
            here only to log and re-raise; we never swallow it and we
            never retry past it.
            """
            try:
                # ``stream`` was removed from ``AsyncClient.request()`` in
                # httpx 0.20; the streaming path uses ``client.stream()``
                # as a context manager instead.  Nothing in the codebase
                # actually passes ``stream=True`` today — chat completions
                # stream directly via a separate code path (per
                # CLAUDE.md → "Runner restart recovery") — so we just
                # drop the kwarg.  Keeping the public ``stream`` arg on
                # ``proxy_request`` itself so we can re-implement it via
                # ``client.stream()`` later without breaking callers.
                return await client.request(
                    method=method,
                    url=url,
                    json=json,
                    headers=headers,
                    timeout=per_request_timeout,
                )
            except asyncio.CancelledError:
                logger.info(
                    "proxy_request cancelled — propagating to upstream httpx call",
                    extra={
                        "server_id": handle.server_id,
                        "method": method,
                        "url": url,
                    },
                )
                raise

        # First attempt (no backoff yet)
        response = await _send_once()

        # Detect stale server handle: 404 from a /v1/server/<id>/... path
        # means the runner has evicted the llama.cpp server we're holding.
        # Convert to StaleServerError so the existing retry layer in
        # CompletionService refreshes the model map and reacquires.
        if self._is_stale_server_response(handle, response):
            logger.warning(
                "proxy_request got 404 for known server handle — raising StaleServerError",
                extra={
                    "server_id": handle.server_id,
                    "url": url,
                    "status_code": response.status_code,
                },
            )
            # Local import to avoid a circular dependency at module load time.
            from graph.errors import StaleServerError

            raise StaleServerError(handle.server_id)

        if response.status_code != 503:
            return response

        last_response = response
        elapsed = 0.0
        attempt = 0

        while elapsed < timeout:
            attempt += 1

            # Opportunistic restart check on 503: if the runner restarted,
            # backing off is pointless — the handle is already dead. Skip
            # straight to StaleServerError so the caller can reacquire.
            epoch_ok = await self._check_runner_epoch(handle.runner_host)
            if not epoch_ok and handle not in self._active_handles:
                logger.warning(
                    "proxy_request 503 + runner restart detected — raising StaleServerError",
                    extra={
                        "server_id": handle.server_id,
                        "runner_host": handle.runner_host,
                    },
                )
                from graph.errors import StaleServerError

                raise StaleServerError(handle.server_id)

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
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                logger.info(
                    "proxy_request cancelled during 503 backoff — aborting retry loop",
                    extra={"server_id": handle.server_id},
                )
                raise
            elapsed += backoff

            response = await _send_once()

            if self._is_stale_server_response(handle, response):
                logger.warning(
                    "proxy_request retry got 404 for known server handle — raising StaleServerError",
                    extra={
                        "server_id": handle.server_id,
                        "url": url,
                    },
                )
                from graph.errors import StaleServerError

                raise StaleServerError(handle.server_id)

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

    # Match runner paths of the form `.../v1/server/<server_id>/...`
    _SERVER_PATH_RE = re.compile(r"/v1/server/([^/]+)(?:/|$)")

    def _is_stale_server_response(
        self, handle: ServerHandle, response: httpx.Response
    ) -> bool:
        """Return True if a 404 indicates the runner no longer knows this server.

        We only convert to a StaleServerError when:
          - the response is 404, AND
          - the request URL targets a /v1/server/<id>/... path, AND
          - the api still believes it holds a valid handle for that
            server_id (it's in ``_active_handles``).

        A 404 from llama.cpp itself (e.g., an unknown completions sub-path)
        will not match the server_id check, so it bubbles up unchanged.
        """
        if response.status_code != 404:
            return False
        request = getattr(response, "request", None)
        url_str = str(request.url) if request is not None else handle.base_url
        m = self._SERVER_PATH_RE.search(url_str)
        if not m:
            return False
        path_server_id = m.group(1)
        if path_server_id != handle.server_id:
            return False
        # Only convert when this api had a live handle for this server_id —
        # otherwise the 404 may be a genuine upstream not-found.
        return any(h.server_id == handle.server_id for h in self._active_handles)

    async def shutdown_all_handles(self) -> None:
        """Shut down all registered server handles on the runner.

        Called during application shutdown to ensure no orphaned llama.cpp
        servers remain running on the runner nodes.
        """
        if not self._active_handles:
            return

        handles_to_shutdown = list(self._active_handles)
        logger.info(f"Shutting down {len(handles_to_shutdown)} active server handle(s)")

        for handle in handles_to_shutdown:
            try:
                await self.shutdown_server(handle)
            except Exception as e:
                logger.warning(
                    f"Failed to shutdown handle {handle.server_id} during cleanup: {e}"
                )

        self._active_handles.clear()

    @staticmethod
    def _is_stale_server_error(response: httpx.Response) -> bool:
        """Check if a 404 response indicates a stale server handle.

        Returns ``True`` when the response is 404 and the body mentions
        a server not being found, meaning the llama.cpp server was
        evicted from the runner.
        """
        if response.status_code != 404:
            return False
        try:
            body = response.text.lower()
            return "server" in body and "not found" in body
        except Exception:
            return False

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

    async def _health_check_loop(self, interval: float = 30.0) -> None:
        """Periodically validate active handles and purge dead ones.

        Runs every *interval* seconds. When a handle fails validation,
        it is removed from tracking and the model map is refreshed so
        subsequent requests get a fresh server.
        """
        while True:
            await asyncio.sleep(interval)
            if not self._active_handles:
                continue
            stale_handles: list[ServerHandle] = []
            for handle in list(self._active_handles):
                if not await self.validate_server_handle(handle):
                    stale_handles.append(handle)
            if stale_handles:
                logger.warning(
                    f"Health check found {len(stale_handles)} stale handle(s) — purging",
                    extra={"server_ids": [h.server_id for h in stale_handles]},
                )
                for handle in stale_handles:
                    self._active_handles.discard(handle)
                    servers = self._active_servers_by_endpoint.get(handle.runner_host)
                    if servers:
                        servers.discard(handle.server_id)
                await self.refresh_model_map()

    def start_health_check(self) -> None:
        """Start the background health check task. Call once during app startup."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def aclose(self) -> None:
        """Close the shared HTTP client and release active servers.  Call during app shutdown."""
        # Shut down all active server handles before closing the client
        await self.shutdown_all_handles()

        if self._health_check_task is not None:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
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
                await client.delete(
                    f"{endpoint}/v1/server/{sid}", timeout=_FAST_TIMEOUT
                )
                logger.info(f"Cleaned up server {sid} on {endpoint}")
            except Exception as e:
                logger.warning(f"Failed to clean up server {sid} on {endpoint}: {e}")

    def _is_connection_error(self, exc: Exception) -> bool:
        """Check if the exception is a connection-level error (disconnect, timeout, etc.)."""
        return isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                ConnectionError,
            ),
        )

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

    # ------------------------------------------------------------------
    # Runner restart detection (startup_epoch tracking)
    # ------------------------------------------------------------------

    # Tight timeout for `/v1/status` — this endpoint is cheap and should
    # never block acquire_server for long. If the runner can't answer
    # within a few seconds, treat it as unreachable (not restarted) and
    # let the normal health / circuit breaker path handle it.
    _STATUS_TIMEOUT = httpx.Timeout(3.0)

    async def _check_runner_epoch(self, endpoint: str) -> bool:
        """Detect runner restart by polling its ``GET /v1/status`` epoch.

        Returns
        -------
        bool
            ``True`` if the runner is reachable and its ``startup_epoch``
            is unchanged (or is being seen for the first time — no
            baseline to compare against).
            ``False`` if the runner is unreachable (single failure — no
            invalidation, to avoid thrash) OR the epoch changed (in
            which case all handles for that endpoint have been purged
            from ``_active_handles`` before this method returns).

        Notes
        -----
        - We update ``_runner_epochs[endpoint]`` ONLY after the purge
          has completed, so a crash mid-purge leaves the old epoch in
          place and the next caller will retry the purge.
        - A check-and-update lock prevents two concurrent callers from
          both observing a stale epoch, missing the purge window, and
          racing on ``_active_handles``.
        """
        client = self._get_client()
        try:
            resp = await client.get(
                f"{endpoint}/v1/status",
                timeout=self._STATUS_TIMEOUT,
                headers=self._session_headers(),
            )
        except Exception as e:
            logger.warning(
                f"Runner {endpoint} /v1/status unreachable, "
                f"skipping epoch check: {e}"
            )
            return False

        if resp.status_code != 200:
            logger.warning(
                f"Runner {endpoint} /v1/status returned "
                f"{resp.status_code}, skipping epoch check"
            )
            return False

        try:
            data = resp.json()
            new_epoch = int(data["startup_epoch"])
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(
                f"Runner {endpoint} /v1/status returned malformed body: {e}"
            )
            return False

        async with self._epoch_lock:
            prev_epoch = self._runner_epochs.get(endpoint)

            # First sighting — record and treat as unchanged. No handles
            # could possibly predate this baseline.
            if prev_epoch is None:
                self._runner_epochs[endpoint] = new_epoch
                logger.debug(
                    f"Runner {endpoint} startup_epoch recorded: {new_epoch}"
                )
                return True

            if prev_epoch == new_epoch:
                return True

            # Epoch changed — runner restarted. Purge every handle
            # belonging to this endpoint BEFORE updating the stored
            # epoch, so a partial failure leaves us in a state where
            # the next caller will retry the purge.
            stale_handles = [
                h for h in self._active_handles if h.runner_host == endpoint
            ]
            for handle in stale_handles:
                self._active_handles.discard(handle)
            self._active_servers_by_endpoint.pop(endpoint, None)

            logger.info(
                f"Runner restart detected for {endpoint} — purged "
                f"{len(stale_handles)} stale handle(s)",
                extra={
                    "endpoint": endpoint,
                    "old_epoch": prev_epoch,
                    "new_epoch": new_epoch,
                    "purged_count": len(stale_handles),
                    "purged_server_ids": [h.server_id for h in stale_handles],
                },
            )

            # Also drop the endpoint from the model map — refresh_model_map
            # will re-discover whatever the freshly-started runner has.
            self._invalidate_model_map_for_endpoint(endpoint)

            # Update only after purge succeeded.
            self._runner_epochs[endpoint] = new_epoch
            return False

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

    async def revalidate_runner_handles(self) -> int:
        """Probe every known runner endpoint's startup_epoch and purge stale handles.

        Used as a recovery trigger when something downstream looks suspicious
        (e.g. an empty LLM response that the chat completion path can't easily
        attribute to a 404, because LangChain's ChatOpenAI calls bypass our
        proxy_request StaleServerError detection).

        Returns the count of handles purged.  A non-zero return means a
        runner restarted; callers should typically raise StaleServerError
        so the upper-layer retry rebuilds workflows with fresh handles.
        """
        before = len(self._active_handles)
        for endpoint in list(self._runner_epochs.keys()):
            await self._check_runner_epoch(endpoint)
        after = len(self._active_handles)
        return max(0, before - after)

    # ------------------------------------------------------------------
    # Runner selection
    # ------------------------------------------------------------------

    async def _select_runner(self, model_id: str) -> Optional[str]:
        """Pick the best endpoint that hosts *model_id*.

        Selection priority (highest first):
          1. **Sticky preference** — if we've previously acquired this
             ``model_id`` from an endpoint and it's still healthy + still
             hosts the model, return it.  Maximises KV-cache reuse on the
             runner: every session targeting the same model lands on the
             same llama.cpp server, sharing prefix cache, slot LRU, and
             saved-KV snapshots.
          2. **Fallback** — among the remaining eligible endpoints,
             rank by ``(-active_handles_here, available_vram)``:
             prefer the endpoint with the FEWEST handles we already
             hold there, breaking ties by most free VRAM.  This still
             spreads first-time-loaded models across runners so they
             don't all pile onto one box.

        An endpoint that doesn't host the model or doesn't respond to
        ``/health`` is skipped entirely.

        The sticky preference DOES NOT lock us in forever — when the
        sticky endpoint becomes unhealthy or loses the model, the next
        call falls back to ranking and resets the pin (in
        ``acquire_server`` on success).
        """
        # Temporary forensic log: emit one line per selection so we can
        # tell from logs exactly which path returns and why. Remove
        # after the runner-fan-out behavior is verified in production.
        logger.info(
            "select_runner: start",
            extra={
                "model_id": model_id,
                "endpoints": list(self._endpoints),
                "global_sticky": self._last_endpoint_for_model.get(model_id),
                "active_handles_for_model": [
                    h.runner_host for h in self._active_handles
                    if h.model_id == model_id
                ],
                "session_id_ctx": _session_id_ctx.get(),
            },
        )
        # --- Per-session sticky path ---------------------------------
        # Each concurrent session keeps its own pin so two sessions on
        # the same model converge on two different runners (instead of
        # alternating onto one and serialising on a parallel=1 slot).
        # Falls back to the global pin below when no per-session pin
        # exists yet — that gives KV-cache reuse benefits to the first
        # acquire by a brand-new session.
        session_id = _session_id_ctx.get() or ""
        if session_id:
            per_session = self._last_endpoint_per_session.get(
                (session_id, model_id)
            )
            if per_session and not self._is_circuit_open(per_session):
                health = await self._health(per_session)
                psession_ids = {
                    m.get("id") if isinstance(m, dict) else m
                    for m in (health.get("models", []) if health else [])
                }
                if health and model_id in psession_ids:
                    # Promote to LRU front and use it.  Don't apply the
                    # busy-escape here: a session continuing its own
                    # conversation SHOULD stay on its existing runner
                    # even while it has an in-flight request there
                    # (the runner's own slot scheduler handles the
                    # turn-level serialisation).
                    self._last_endpoint_per_session.move_to_end(
                        (session_id, model_id)
                    )
                    logger.info(
                        "select_runner: returning per-session sticky",
                        extra={"endpoint": per_session, "session_id": session_id},
                    )
                    return per_session
                # Per-session pin no longer eligible — drop it so we
                # fall through to global / ranked below.
                self._last_endpoint_per_session.pop(
                    (session_id, model_id), None
                )

        # --- Global sticky path --------------------------------------
        sticky = self._last_endpoint_for_model.get(model_id)
        if sticky and not self._is_circuit_open(sticky):
            health = await self._health(sticky)
            # ``health.models`` is a list of dicts ({id,name,task}),
            # not bare ids — normalise before the membership test.
            # Previously this checked ``model_id in [{...}, ...]`` which
            # always returned False, silently disabling sticky pinning.
            sticky_model_ids = {
                m.get("id") if isinstance(m, dict) else m
                for m in (health.get("models", []) if health else [])
            }
            if health and model_id in sticky_model_ids:
                # Don't pin to a sticky endpoint that's already busy
                # serving this same model for another session — that
                # forces concurrent callers to queue on a single
                # parallel=1 slot when an idle peer exists.  When the
                # sticky has at least one handle for this model and an
                # alternative endpoint also hosts it, fall through to
                # the load-aware ranked path.
                sticky_busy_for_model = any(
                    h.runner_host == sticky and h.model_id == model_id
                    for h in self._active_handles
                )
                # Also treat the sticky as busy if the runner itself
                # reports a loaded server for this model that's actively
                # serving (idle_since=None). _active_handles only counts
                # requests this api process knows about; a session
                # holding the slot via another api replica, or a slot
                # that's mid-turn between handle release and the next
                # acquire, looks "free" to _active_handles but is
                # actually pinned on the runner.
                if not sticky_busy_for_model:
                    sticky_loaded = await self._find_loaded_server(
                        sticky, model_id
                    )
                    if sticky_loaded is not None:
                        # Same definition of "busy" Rule 1's `_is_busy`
                        # uses below: loading OR processing OR actively
                        # generating OR last used within CACHE_TIMEOUT_MIN.
                        # A sticky in any of those states risks queuing
                        # the new request behind a session that hasn't
                        # quite released its slot, so fall through to
                        # the load-aware ranked path if an idle peer
                        # exists.
                        if sticky_loaded.get("starting"):
                            sticky_busy_for_model = True
                        elif (sticky_loaded.get("use_count") or 0) > 0:
                            sticky_busy_for_model = True
                        else:
                            idle_since = sticky_loaded.get("idle_since")
                            if idle_since is None:
                                sticky_busy_for_model = True
                            else:
                                try:
                                    sticky_busy_for_model = (
                                        time.time() - float(idle_since)
                                    ) < (CACHE_TIMEOUT_MIN * 60)
                                except (TypeError, ValueError):
                                    # Malformed timestamp — treat as busy to be safe.
                                    sticky_busy_for_model = True
                alternatives_exist = any(
                    e != sticky
                    and not self._is_circuit_open(e)
                    and (e, model_id) in self._model_tensor_split
                    for e in self._endpoints
                )
                logger.info(
                    "select_runner: global-sticky path decision",
                    extra={
                        "sticky": sticky,
                        "sticky_busy_for_model": sticky_busy_for_model,
                        "alternatives_exist": alternatives_exist,
                        "tensor_split_keys_for_model": [
                            k for k in self._model_tensor_split.keys()
                            if k[1] == model_id
                        ],
                    },
                )
                if not (sticky_busy_for_model and alternatives_exist):
                    logger.info(
                        "select_runner: returning global-sticky",
                        extra={"endpoint": sticky},
                    )
                    return sticky
            # Sticky endpoint no longer eligible — drop the pin so the
            # ranking below can choose freshly and ``acquire_server``
            # re-pins on next success.
            self._last_endpoint_for_model.pop(model_id, None)
        elif sticky:
            # Sticky has a tripped circuit; clear the pin so the next
            # ranked path doesn't try it.
            self._last_endpoint_for_model.pop(model_id, None)

        # --- Gather per-endpoint state ------------------------------
        # Collected in one pass so the rule layers below can compare
        # across endpoints without re-fetching.  Each entry:
        #   ep, vram, here_count, my_server (dict from /v1/servers or None)
        # ``my_server`` is the existing llama-server instance on that
        # runner that's already loaded ``model_id``, if any.  ``idle_since``
        # is null when the server is actively serving a request, or a
        # unix timestamp when it went idle.
        # Restrict the candidate scan to the endpoints that the model
        # map says actually host this model, when populated. Falls back
        # to all configured endpoints before the first refresh so the
        # very first request after boot still has something to pick.
        eligible_endpoints = (
            self._model_map.get(model_id) or list(self._endpoints)
        )
        candidates: List[tuple[str, int, int, Optional[dict]]] = []
        for endpoint in eligible_endpoints:
            if self._is_circuit_open(endpoint):
                continue
            health = await self._health(endpoint)
            if not health:
                continue
            model_ids = {
                m.get("id") if isinstance(m, dict) else m
                for m in health.get("models", [])
            }
            if model_id not in model_ids:
                continue
            tensor_split = self._model_tensor_split.get((endpoint, model_id))
            vram = self._effective_free_vram_bytes(health, tensor_split)
            here_count = sum(
                1 for h in self._active_handles
                if h.runner_host == endpoint and h.model_id == model_id
            )
            my_server = await self._find_loaded_server(endpoint, model_id)
            candidates.append((endpoint, vram, here_count, my_server))

        logger.info(
            "select_runner: candidates",
            extra={
                "model_id": model_id,
                "candidates": [
                    {
                        "ep": ep,
                        "vram": vram,
                        "here_count": hc,
                        "srv": (
                            None if srv is None
                            else {
                                "model_id": srv.get("model_id"),
                                "idle_since": srv.get("idle_since"),
                            }
                        ),
                    }
                    for ep, vram, hc, srv in candidates
                ],
            },
        )

        if not candidates:
            logger.info("select_runner: no candidates → None")
            return None

        # --- Rule 1: parallel-spawn over peer-blocked sticky -----------
        # If at least one endpoint has a running server that is in use
        # (idle_since is None or here_count > 0) AND at least one peer
        # endpoint has NO server loaded for this model, prefer the
        # empty peer: spinning up a fresh server there lets the new
        # session run TRULY in parallel instead of queuing behind the
        # busy one on a parallel=1 slot.
        # A server counts as "busy" if it has been used within the
        # last CACHE_TIMEOUT_MIN — that window protects sessions that
        # are merely paused mid-conversation from being preempted.
        # Older idle time is treated as abandoned and handled by
        # Rule 2 below.
        cache_timeout_sec = CACHE_TIMEOUT_MIN * 60
        now = time.time()

        def _is_busy(hc: int, srv: Optional[dict]) -> bool:
            if hc > 0:
                return True
            if srv is None:
                return False
            # A server that's still loading (cold-start, ~30s) is busy too —
            # it can't accept requests yet and the slot is reserved.
            if srv.get("starting"):
                return True
            # The runner-side use_count is the authoritative count of
            # in-flight requests it's processing right now. It catches
            # cases the api-side _active_handles misses (cross-replica,
            # mid-release).
            if (srv.get("use_count") or 0) > 0:
                return True
            idle_since = srv.get("idle_since")
            if idle_since is None:
                return True  # actively serving a request right now
            try:
                return (now - float(idle_since)) < cache_timeout_sec
            except (TypeError, ValueError):
                return True  # malformed timestamp: assume busy to be safe

        busy_endpoints = [
            ep for ep, _v, hc, srv in candidates if _is_busy(hc, srv)
        ]
        # An endpoint is "empty" only if it has neither a loaded server
        # nor any of our active handles — a handle without a visible
        # server (e.g. /v1/servers temporarily unreachable) still means
        # we're using that runner.
        empty_endpoints = [
            (ep, vram) for ep, vram, hc, srv in candidates
            if hc == 0 and srv is None
        ]
        # "Loaded" = has a server for this model, busy OR warm-idle.
        # When a loaded peer exists AND an empty peer exists, we'd
        # rather spawn fresh on the empty one (true parallelism, even
        # if it eats the ~30 s cold-load) than reuse the loaded one
        # (which may share the runner with other busy models).
        loaded_endpoints = [
            ep for ep, _v, hc, srv in candidates
            if srv is not None or hc > 0
        ]
        logger.info(
            "select_runner: rule1 inputs",
            extra={
                "busy_endpoints": busy_endpoints,
                "loaded_endpoints": loaded_endpoints,
                "empty_endpoints": [e for e, _ in empty_endpoints],
            },
        )
        if (busy_endpoints or loaded_endpoints) and empty_endpoints:
            # Pick the empty peer with most effective VRAM (best spawn target).
            empty_endpoints.sort(key=lambda x: x[1], reverse=True)
            logger.info(
                "select_runner: returning rule1 (fan-out to empty peer)",
                extra={"endpoint": empty_endpoints[0][0]},
            )
            return empty_endpoints[0][0]

        # --- Rule 2: prefer a warm idle server over spinning up a new --
        # ``CACHE_TIMEOUT_MIN`` is the threshold below which an idle
        # server *might* belong to a session that's merely paused
        # mid-conversation.  Past that, it's effectively abandoned and
        # safe to commandeer.  Reusing a warm server skips the ~30 s
        # cold-load penalty AND avoids preempting an in-progress session.
        # (``cache_timeout_sec`` and ``now`` were already computed above
        # for the busy-check; reuse them here.)
        warm_idle: List[tuple[str, float]] = []  # (endpoint, idle_seconds)
        for ep, _vram, hc, srv in candidates:
            if hc > 0 or srv is None:
                continue
            idle_since = srv.get("idle_since")
            if idle_since is None:
                continue  # actively serving someone else
            idle_for = now - float(idle_since)
            if idle_for >= cache_timeout_sec:
                warm_idle.append((ep, idle_for))
        if warm_idle:
            # Longest-idle first — least likely to overlap with a
            # paused session, and most likely to be evicted soon if
            # we don't use it.
            warm_idle.sort(key=lambda x: x[1], reverse=True)
            return warm_idle[0][0]

        # --- Rule 3: existing ranked fallback --------------------------
        # ``(-here_count, effective_vram)`` — fewest concurrent handles
        # for this model on the endpoint wins, then most free VRAM.
        best_url: Optional[str] = None
        best_key: Optional[tuple[int, int]] = None
        for ep, vram, here_count, _srv in candidates:
            key = (-here_count, vram)
            if best_key is None or key > best_key:
                best_key = key
                best_url = ep
        return best_url

    async def _find_loaded_server(
        self, endpoint: str, model_id: str
    ) -> Optional[dict]:
        """Return the loaded llama-server entry for ``model_id`` on
        ``endpoint``, or None if no server is loaded for that model.

        Used by ``_select_runner`` to apply the parallel-spawn and
        warm-idle reuse rules above.  Reads from ``/v1/servers`` on
        the runner; failures (network, parse) are non-fatal and just
        return None so selection falls through to ranked.
        """
        try:
            client = self._get_client()
            resp = await client.get(
                f"{endpoint}/v1/servers", timeout=_HEALTH_TIMEOUT
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            for srv in data.get("servers", []):
                if not isinstance(srv, dict):
                    continue
                if srv.get("model_id") == model_id and srv.get("healthy", True):
                    return srv
            return None
        except Exception:
            return None

    @staticmethod
    def _parse_tensor_split(tensor_split: Optional[str]) -> Optional[List[float]]:
        """Parse a llama.cpp ``tensor_split`` string into a weight list.

        ``"1,0,0"`` → ``[1.0, 0.0, 0.0]`` (model only uses device 0).
        Returns None if the string is missing, empty, or malformed —
        callers treat that as "no pinning, use total VRAM".
        """
        if not tensor_split or not isinstance(tensor_split, str):
            return None
        try:
            return [float(x.strip()) for x in tensor_split.split(",") if x.strip()]
        except ValueError:
            return None

    @classmethod
    def _effective_free_vram_bytes(
        cls,
        health: dict,
        tensor_split: Optional[str],
    ) -> int:
        """Sum free VRAM only on GPUs the model will actually land on.

        The runner's ``/health`` returns ``gpu`` as either:
          * a per-GPU dict keyed by gpu id: ``{"0": {"free_mb": ...}, "1": ...}``
          * an aggregate-only dict containing ``available_vram_bytes``.

        For a model with ``tensor_split = "1,0,0"`` we only count device 0,
        even if a runner has 60 GB total split across multiple cards —
        the model can't use VRAM on devices with weight 0.  Falls back to
        aggregate / total when tensor_split is None or the per-GPU view
        is unavailable.
        """
        gpu = health.get("gpu") or {}
        weights = cls._parse_tensor_split(tensor_split)

        # Per-GPU dict path (preferred): keys "0","1","2" → entries with free_mb.
        per_gpu: Dict[int, int] = {}
        if isinstance(gpu, dict):
            for k, v in gpu.items():
                if not isinstance(v, dict) or not str(k).isdigit():
                    continue
                free_mb = v.get("free_mb")
                if free_mb is None:
                    continue
                try:
                    per_gpu[int(k)] = int(float(free_mb) * 1024 * 1024)
                except (TypeError, ValueError):
                    continue

        if per_gpu:
            if weights is not None:
                # Only count GPUs the model is actually pinned to.
                return sum(
                    free for idx, free in per_gpu.items()
                    if idx < len(weights) and weights[idx] > 0
                )
            return sum(per_gpu.values())

        # Fallback: aggregate-only response.
        if isinstance(gpu, dict):
            agg = gpu.get("available_vram_bytes")
            if agg is not None:
                try:
                    return int(agg)
                except (TypeError, ValueError):
                    pass
        return 0

    async def select_pipeline_endpoint(self, pipeline_name: str) -> str:
        """Pick the best endpoint hosting *pipeline_name*.

        Mirrors :meth:`_select_runner` but indexes through
        ``_pipeline_map`` (populated by :meth:`refresh_model_map` from
        models with ``provider == 'in_process'``).  No sticky pinning:
        in-process pipelines are stateless across calls (the cached
        instance lives in the runner's Python process; there's no KV
        cache or slot to preserve like llama.cpp servers have), so
        ranking purely by handle count + free VRAM gives the best
        spread without locking traffic to a slow runner.

        Falls back to the first configured endpoint if no runner has
        refreshed its model map yet — this preserves boot behaviour
        for the first request after startup, when ``refresh_model_map``
        may not have completed.

        Raises :class:`ImageServiceError`-shaped exception via the
        caller if no endpoints are configured at all (caller handles).
        """
        if not self._endpoints:
            raise RuntimeError("No runner endpoints configured")

        # Lazily refresh the map if we have nothing for this pipeline.
        # ``acquire_server`` does the same trick for ``_model_map``.
        if not self._pipeline_map:
            try:
                await self.refresh_model_map()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"refresh_model_map failed during pipeline select: {e}")

        candidates = self._pipeline_map.get(pipeline_name, [])
        if not candidates:
            # No runner declared this pipeline.  Fall back to the first
            # endpoint so we surface the real "pipeline not advertised"
            # error from the runner (404), not a fuzzy api-side guess.
            logger.warning(
                f"Pipeline '{pipeline_name}' not in pipeline_map; "
                f"falling back to first endpoint.  Configured pipelines: "
                f"{sorted(self._pipeline_map.keys())}"
            )
            return self._endpoints[0]

        # Ranked: (-handle_count, available_vram).  No sticky pin (see
        # docstring above for rationale).
        best_url: Optional[str] = None
        best_key: Optional[tuple[int, int]] = None
        for endpoint in candidates:
            if self._is_circuit_open(endpoint):
                continue
            health = await self._health(endpoint)
            if not health:
                continue
            # In-process pipelines don't have a tensor_split — they
            # use whatever device the pipeline's own GPU-select logic
            # chose at load time.  Aggregate free VRAM is the right
            # signal for ranking these.
            vram = self._effective_free_vram_bytes(health, tensor_split=None)
            here_count = sum(
                1 for h in self._active_handles if h.runner_host == endpoint
            )
            key = (-here_count, vram)
            if best_key is None or key > best_key:
                best_key = key
                best_url = endpoint
        if best_url is None:
            # All candidate endpoints unhealthy or circuit-tripped.
            # Pick the first candidate anyway — caller's HTTP error
            # response is more informative than ours would be.
            best_url = candidates[0]
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

    async def acquire_server(
        self, model_id: str, num_ctx: Optional[int] = None, **kwargs
    ) -> ServerHandle:
        """Acquire a new llama.cpp server from a runner.

        Uses the cached model map for fast routing. Falls back to
        health-check scan if the model isn't in the map.

        Parameters
        ----------
        model_id:
            Identifier of the model to load.
        num_ctx:
            Requested context window size. Passed to the runner, which
            refuses the request (HTTP 507) when it exceeds the model's
            configured context window.
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

        # Run the load-aware selector to pick the best endpoint regardless
        # of whether the model map is cached or not. The previous "fast
        # path: just use mapped_endpoints in map order" silently bypassed
        # all of _select_runner's sticky/parallel-spawn/warm-idle logic,
        # which is why every Qwen3_5_9B request landed on whichever
        # endpoint happened to be first in config (main), even when small
        # was idle. The selector itself is fast: it reads cached health
        # state and the in-memory tensor_split + _active_handles maps.
        best = await self._select_runner(model_id)
        if best:
            ordered = [best]
            # Append remaining mapped (or all) endpoints as failover
            # targets in case `best` errors mid-acquire. Prefer the map
            # when populated so we don't dial endpoints that don't host
            # the model.
            mapped = self._model_map.get(model_id) or list(self._endpoints)
            for ep in mapped:
                if ep != best:
                    ordered.append(ep)
        else:
            # Selector found nothing — happens before refresh_model_map
            # ever runs, or if every endpoint failed health. Fall back
            # to the historical "try them all in config order" behavior
            # so the very first request after boot still has a chance.
            mapped_endpoints = self._model_map.get(model_id)
            if mapped_endpoints:
                ordered = list(mapped_endpoints)
            else:
                ordered = list(self._endpoints)

        last_error = None
        skipped_circuit_breaker = []
        for endpoint in ordered:
            # Skip runners with open circuit breaker
            if self._is_circuit_open(endpoint):
                skipped_circuit_breaker.append(
                    f"{endpoint} ({self._acquire_failures.get(endpoint, 0)} failures)"
                )
                logger.warning(
                    f"Skipping {endpoint}: circuit breaker open "
                    f"({self._acquire_failures.get(endpoint, 0)} failures)"
                )

                continue

            # Proactive runner-restart detection: hit /v1/status. If the
            # runner's startup_epoch changed since we last saw it, every
            # handle we hold for this endpoint is dead — purge them now
            # so we don't try to reuse them later. Either way (changed,
            # unchanged, or unreachable) we still proceed to attempt
            # acquire — a fresh server is fine after a purge, and a
            # one-off /v1/status failure shouldn't block real traffic.
            await self._check_runner_epoch(endpoint)

            # Retry connection-level errors per endpoint (configurable via RUNNER_ACQUIRE_RETRIES)
            max_retries = _ACQUIRE_RETRIES
            for attempt in range(max_retries + 1):
                try:
                    client = self._get_client()
                    resp = await client.post(
                        f"{endpoint}/v1/server/create",
                        json=payload,
                        headers=self._session_headers(),
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
                        model_id=model_id,
                    )
                    logger.info(f"Acquired server {handle.server_id} from {endpoint}")
                    self._record_acquire_success(endpoint)
                    self._active_servers_by_endpoint.setdefault(endpoint, set()).add(
                        data["server_id"]
                    )
                    self._active_handles.add(handle)
                    # Pin this model to the endpoint we just used so
                    # follow-up acquires (different sessions, same model)
                    # land here and benefit from the warm KV cache.
                    self._last_endpoint_for_model[model_id] = endpoint
                    # Also pin per-session — this session's next turn
                    # will land on the same endpoint even if a peer
                    # later updates the global pin.  Without this,
                    # concurrent sessions on the same model bounce
                    # between runners every turn.
                    session_id = _session_id_ctx.get() or ""
                    if session_id:
                        key = (session_id, model_id)
                        self._last_endpoint_per_session[key] = endpoint
                        self._last_endpoint_per_session.move_to_end(key)
                        # LRU eviction so the dict stays bounded.
                        while (
                            len(self._last_endpoint_per_session)
                            > _PER_SESSION_PIN_LIMIT
                        ):
                            self._last_endpoint_per_session.popitem(last=False)
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

        # Build a meaningful last_error when all endpoints were skipped
        if last_error is None and skipped_circuit_breaker:
            last_error = (
                f"All {len(skipped_circuit_breaker)} runner(s) skipped "
                f"(circuit breaker open): {', '.join(skipped_circuit_breaker)}"
            )
        elif last_error is None:
            last_error = "No endpoints available"

        raise RuntimeError(
            f"No healthy runner available for model {model_id}. "
            f"Last error: {last_error}"
        )

    async def release_server(self, handle: ServerHandle) -> None:
        """Release an acquired server back to the runner."""
        try:
            client = self._get_client()
            resp = await client.post(
                f"{handle.runner_host}/v1/server/{handle.server_id}/release",
                headers=self._session_headers(),
            )
            resp.raise_for_status()
            logger.info(f"Released server {handle.server_id}")
            self._active_handles.discard(handle)
        except Exception as e:
            # Runner is likely dead — discard handle silently
            logger.warning(f"Failed to release server {handle.server_id} (runner may be down): {e}")
            self._active_handles.discard(handle)
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
                f"{handle.runner_host}/v1/server/{handle.server_id}",
                headers=self._session_headers(),
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
        """Query all runners and build model_id + pipeline_name maps.

        Two indexes are built in a single pass over each runner's
        ``/v1/models`` response:

        * ``_model_map``: ``{model_id -> [endpoints]}`` — used by
          ``acquire_server`` for subprocess-backed models (llama_cpp,
          stable_diffusion_cpp).
        * ``_pipeline_map``: ``{pipeline_name -> [endpoints]}`` —
          built from models with ``provider == 'in_process'`` and a
          declared ``pipeline`` field.  Used by
          ``_select_pipeline_runner`` to route POST /v1/pipelines/<name>/run
          calls to whichever runner advertises a model for that pipeline.

        Because each runner's ``/v1/models`` reflects its own
        ``.models.yaml``, capability splits between runners
        (``llmmllab-runner`` vs ``llmmllab-runner-small``) come from
        yaml alone — no env vars or substring matches in the api.
        """
        new_map: Dict[str, List[str]] = {}
        new_pipeline_map: Dict[str, List[str]] = {}
        new_tensor_split: Dict[tuple[str, str], Optional[str]] = {}
        client = self._get_client()
        tasks = []
        for endpoint in self._endpoints:

            async def fetch_models(ep=endpoint):
                try:
                    resp = await client.get(f"{ep}/v1/models")
                    if resp.status_code == 200:
                        return [(m, ep) for m in resp.json() if isinstance(m, dict) and "id" in m]
                except Exception as e:
                    logger.warning(f"Failed to list models from {ep}: {e}")
                return []

            tasks.append(fetch_models())
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                for model, endpoint in result:
                    model_id = model["id"]
                    new_map.setdefault(model_id, []).append(endpoint)
                    # Capture tensor_split for VRAM accounting in
                    # _select_runner.  ``parameters`` is the nested
                    # ModelParameters object; missing or empty means
                    # "no pinning, use total free VRAM".
                    params = model.get("parameters") or {}
                    new_tensor_split[(endpoint, model_id)] = (
                        params.get("tensor_split") if isinstance(params, dict) else None
                    )
                    if model.get("provider") == "in_process":
                        pipeline_name = model.get("pipeline")
                        if pipeline_name:
                            eps = new_pipeline_map.setdefault(pipeline_name, [])
                            if endpoint not in eps:
                                eps.append(endpoint)
        self._model_map = new_map
        self._model_tensor_split = new_tensor_split
        self._pipeline_map = new_pipeline_map
        logger.info(
            f"Model map refreshed: {len(new_map)} models, "
            f"{len(new_pipeline_map)} pipelines across "
            f"{len(self._endpoints)} endpoints "
            f"(pipelines: {sorted(new_pipeline_map.keys())})"
        )

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

        Returns ``True`` if an existing server has free slots OR sufficient
        VRAM exists on a runner to start a new server, OR if the check is
        inconclusive (fail-open).

        ``False`` is reserved for the case where we POSITIVELY know all
        slots are busy AND no runner has enough free VRAM to start a new
        server.  Even then, the next chunk of work llama.cpp completes
        will free a slot — so a False here just means "wait a moment" not
        "this will never succeed."

        Slots reporting ``is_processing: true`` are still acceptable when
        the busy slot is owned by THIS request's session_id, because
        the slot-pinning LRU on the runner side will route the new
        request onto the same llama.cpp slot which handles its own
        internal queue.  We don't have session_id here, so we treat
        ``processing`` slots as "possibly free for this session" and
        fail-open in ambiguous cases.
        """
        _slots_timeout = httpx.Timeout(3.0)
        client = self._get_client()

        # Case 1: Any active server with at least one slot reported idle?
        # Bail out True on the first such server.
        any_server_responded = False
        for handle in self._active_handles:
            try:
                resp = await client.get(
                    f"{handle.base_url}/slots",
                    timeout=_slots_timeout,
                )
                if resp.status_code == 200:
                    any_server_responded = True
                    slots = resp.json()
                    if slots and any(not s.get("is_processing", False) for s in slots):
                        return True
            except Exception:
                continue

        # Case 2: No idle slot found, but maybe a runner has free VRAM to
        # start a new server for this model.
        endpoints = self._model_map.get(model_id, list(self._endpoints))
        any_endpoint_responded = False
        for endpoint in endpoints:
            try:
                health_resp = await client.get(
                    f"{endpoint}/health",
                    timeout=_HEALTH_TIMEOUT,
                )
                if health_resp.status_code != 200:
                    continue
                any_endpoint_responded = True
                health = health_resp.json()
                gpu_info = health.get("gpu", {})
               # Sum free_mb across all GPUs, convert to bytes
                available_vram = (
                    sum(
                        v.get("free_mb", 0)
                        for v in gpu_info.values()
                        if isinstance(v, dict)
                    )
                    * 1024
                    * 1024
                )

                model_resp = await client.get(
                    f"{endpoint}/v1/models/{model_id}",
                    timeout=_FAST_TIMEOUT,
                )
                if model_resp.status_code != 200:
                    continue
                model_data = model_resp.json()
                model_size = model_data.get("details", {}).get("size", 0) or 0
                required = model_size + (128 * 1024 * 1024)
                if available_vram >= required:
                    return True
            except Exception:
                continue

        # Reached here = (no active server had an idle slot) AND
        # (no runner reported enough free VRAM for a fresh server).
        # If neither cohort even responded to us, we know nothing — fail
        # open so the queue doesn't stall on a transient network blip.
        if not any_server_responded and not any_endpoint_responded:
            logger.warning(
                "check_slot_availability: neither active servers nor "
                "endpoints responded — failing open to avoid queue stall",
                extra={"model_id": model_id},
            )
            return True

        # Positively-known constrained state.  Still return True for now:
        # llama.cpp serializes its own per-slot queue, so admitting one
        # more request just means it'll wait inside llama.cpp instead of
        # in our priority queue.  Blocking here causes 5-minute timeouts
        # in the priority queue when a slot is mid-prefill, which is the
        # bug observed 2026-05-19T22:31.  Keep the throttle for
        # SCHEDULED/SYSTEM sources via the per-model active_counts cap
        # in queue_callbacks._can_proceed.
        return True

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

    async def default_model_by_task(self, task: ModelTask) -> Optional[Model]:
        """Find the default model for the given task across all runners.

        Uses the runner's /v1/models/default endpoint which returns the
        model marked with `default: true` in .models.yaml.
        Falls back to model_by_task() if no default is configured.
        """
        client = self._get_client()
        for endpoint in self._endpoints:
            try:
                resp = await client.get(
                    f"{endpoint}/v1/models/default", params={"task": task.value}
                )
                if resp.status_code == 200:
                    return Model(**resp.json())
            except Exception as e:
                logger.warning(f"Failed to query default model from {endpoint}: {e}")
                continue
        # Fallback: if no default configured, return any model matching the task
        return await self.model_by_task(task)


# Module-level singleton
runner_client = RunnerClient()


@asynccontextmanager
async def server_handle_lease(
    model_id: str,
    num_ctx: Optional[int] = None,
    **kwargs,
) -> AsyncIterator[ServerHandle]:
    """Acquire a ``ServerHandle`` and guarantee release on every exit path.

    Wraps :meth:`RunnerClient.acquire_server` so that the soft refcount
    decrement always fires — on success, exception, and cancellation.
    The underlying llama.cpp process is *not* killed by release; the
    runner's TTL-based reaper handles eviction.  This makes calling
    release after every request safe: a follow-up request for the same
    model will simply re-acquire the (likely still warm) server.

    Example
    -------
    >>> async with server_handle_lease("qwen3", num_ctx=8192) as handle:
    ...     resp = await runner_client.proxy_request(handle, "POST", ...)
    """
    handle = await runner_client.acquire_server(
        model_id, num_ctx=num_ctx, **kwargs
    )
    try:
        yield handle
    finally:
        try:
            await runner_client.release_server(handle)
        except asyncio.CancelledError:
            # Re-raise cancellation but don't swallow it — release_server
            # itself is best-effort and already discards the handle on
            # failure, but if we're being cancelled we want the caller
            # to know.
            raise
        except Exception as e:
            logger.warning(
                "release_server failed in lease cleanup",
                extra={
                    "server_id": handle.server_id,
                    "error": str(e),
                },
            )
