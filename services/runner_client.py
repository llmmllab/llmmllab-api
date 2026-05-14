"""
RunnerClient — HTTP client for the llmmllab-runner service pool.

Routes requests among multiple runner instances based on health and
hardware capability (VRAM). Manages server lifecycle (acquire, release,
shutdown) and model discovery across all runners.

Uses a persistent ``httpx.AsyncClient`` with connection pooling to avoid
the overhead of opening a new TCP connection for every request.

When the runner returns HTTP 500 with a context-related error (e.g., the
runner's own retry loop exhausted all reduced-context attempts), the client
automatically retries with a smaller ``num_ctx`` before giving up. This
provides an additional resilience layer for models that struggle to start
with large context windows on constrained hardware.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

# Timeouts for different request categories
_HEALTH_TIMEOUT = httpx.Timeout(5.0)
_FAST_TIMEOUT = httpx.Timeout(10.0)
_ACQUIRE_TIMEOUT = httpx.Timeout(150.0)


@dataclass
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

    async def aclose(self) -> None:
        """Close the shared HTTP client.  Call during app shutdown."""
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

    async def _health(self, endpoint: str) -> Optional[dict]:
        """Check health of a single runner. Returns health dict or None."""
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
                return None
        except Exception as e:
            logger.warning(f"Runner {endpoint} health check failed: {e}")
            if endpoint in self._healthy:
                self._healthy.remove(endpoint)
            return None

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

    # ------------------------------------------------------------------
    # Context-reduction retry helpers
    # ------------------------------------------------------------------

    _CONTEXT_REDUCTION_RETRIES = 3
    _CONTEXT_REDUCTION_MIN = 2048

    @staticmethod
    def _is_context_error(response: httpx.Response) -> bool:
        """Check if a 500 response indicates a context-window-related failure.

        The runner returns 500 when the llama.cpp server fails to start.
        When the failure is context-related (OOM during context allocation,
        or the runner's own retry loop exhausted all reduced-context
        attempts), the error detail contains keywords like 'context',
        'num_ctx', 'OOM', 'memory', or 'retry attempts'.
        """
        if response.status_code != 500:
            return False
        try:
            detail = response.json().get("detail", "")
        except Exception:
            detail = ""
        detail_lower = str(detail).lower()
        context_keywords = (
            "context",
            "num_ctx",
            "oom",
            "memory",
            "retry attempts",
            "reduced context",
            "cannot start",
        )
        return any(kw in detail_lower for kw in context_keywords)

    def _reduce_context(self, num_ctx: int) -> int:
        """Halve the context size, respecting a minimum floor."""
        reduced = max(num_ctx // 2, self._CONTEXT_REDUCTION_MIN)
        return reduced

    async def _try_acquire(
        self,
        model_id: str,
        payload: dict[str, Any],
        ordered: list[str],
    ) -> tuple[Optional[ServerHandle], Optional[str]]:
        """Attempt to acquire a server from the runner pool.

        Returns (ServerHandle, None) on success, or (None, last_error) on failure.
        """
        last_error: Optional[str] = None
        for endpoint in ordered:
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
                    continue

                resp.raise_for_status()
                data = resp.json()
                handle = ServerHandle(
                    base_url=f"{endpoint}/v1/server/{data['server_id']}",
                    server_id=data["server_id"],
                    runner_host=endpoint,
                )
                logger.info(f"Acquired server {handle.server_id} from {endpoint}")
                self._schedule_refresh()
                return handle, None

            except Exception as e:
                logger.warning(f"Failed to acquire from {endpoint}: {e}")
                last_error = str(e)
                continue

        return None, last_error

    async def acquire_server(self, model_id: str, **kwargs) -> ServerHandle:
        """Acquire a new llama.cpp server from a runner.

        Uses the cached model map for fast routing. Falls back to
        health-check scan if the model isn't in the map.

        When the runner returns a 500 error indicating a context-window
        failure (e.g., the runner's own retry loop exhausted all reduced-
        context attempts), this method automatically retries with a
        progressively smaller ``num_ctx`` before giving up.

        Extra kwargs are forwarded to the runner:
        - ``config_override``: Override runner config (if present)
        - ``num_ctx``: Context window size (if present)

        Returns:
            ServerHandle with connection details for the allocated server.

        Raises:
            RuntimeError: if no runner can satisfy the request.
        """
        payload: dict[str, Any] = {"model_id": model_id}
        config_override = kwargs.get("config_override")
        if config_override:
            payload["config_override"] = config_override
        num_ctx = kwargs.get("num_ctx")
        if num_ctx is not None:
            payload["num_ctx"] = num_ctx

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

        # Try initial acquire
        handle, last_error = await self._try_acquire(model_id, payload, ordered)
        if handle is not None:
            return handle

        # If we have a num_ctx to reduce, retry with progressively smaller
        # context windows. This handles the case where the runner's own
        # retry loop exhausted all reduced-context attempts (e.g., the
        # llama.cpp server cannot start with any context size >= the
        # runner's minimum). By reducing num_ctx further at the API level,
        # we give the runner a chance to start with a smaller context.
        original_num_ctx = payload.get("num_ctx")
        if original_num_ctx is None:
            raise RuntimeError(
                f"No healthy runner available for model {model_id}. "
                f"Last error: {last_error}"
            )

        current_ctx = original_num_ctx
        for attempt in range(1, self._CONTEXT_REDUCTION_RETRIES + 1):
            current_ctx = self._reduce_context(current_ctx)
            logger.warning(
                f"Retrying acquire for {model_id} with reduced context "
                f"num_ctx={current_ctx} (original={original_num_ctx}, "
                f"attempt {attempt}/{self._CONTEXT_REDUCTION_RETRIES})"
            )
            payload["num_ctx"] = current_ctx
            handle, last_error = await self._try_acquire(model_id, payload, ordered)
            if handle is not None:
                logger.info(
                    f"Acquired server {handle.server_id} for {model_id} "
                    f"with reduced context num_ctx={current_ctx} "
                    f"(original={original_num_ctx})"
                )
                return handle

        raise RuntimeError(
            f"No healthy runner available for model {model_id}. "
            f"Failed with num_ctx={original_num_ctx} and all reduced "
            f"context retries exhausted. Last error: {last_error}"
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
        except Exception as e:
            logger.error(f"Failed to release server {handle.server_id}: {e}")
            raise

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
