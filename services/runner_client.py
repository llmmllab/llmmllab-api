"""
RunnerClient — HTTP client for the llmmllab-runner service pool.

Routes requests among multiple runner instances based on health and
hardware capability (VRAM). Manages server lifecycle (acquire, release,
shutdown) and model discovery across all runners.

Uses a persistent ``httpx.AsyncClient`` with connection pooling to avoid
the overhead of opening a new TCP connection for every request.
"""

import asyncio
import logging
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
                self._invalidate_model_map_for_endpoint(endpoint)
                return None
        except Exception as e:
            logger.warning(f"Runner {endpoint} health check failed: {e}")
            if endpoint in self._healthy:
                self._healthy.remove(endpoint)
            self._invalidate_model_map_for_endpoint(endpoint)
            return None

    def _invalidate_model_map_for_endpoint(self, endpoint: str) -> None:
        """Remove an endpoint from the model map when it becomes unhealthy.

        Prevents ``acquire_server()`` from routing to a dead runner.
        """
        for model_id, endpoints in list(self._model_map.items()):
            if endpoint in endpoints:
                endpoints.remove(endpoint)
                if not endpoints:
                    del self._model_map[model_id]
        logger.info(f"Invalidated model map for unhealthy endpoint {endpoint}")

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

    async def acquire_server(self, model_id: str, **kwargs) -> ServerHandle:
        """Acquire a new llama.cpp server from a runner.

        Uses the cached model map for fast routing. Falls back to
        health-check scan if the model isn't in the map.

        Extra kwargs are accepted for forward compatibility with callers
        that pass task/config_override. config_override is forwarded to
        the runner if present.

        Returns:
            ServerHandle with connection details for the allocated server.

        Raises:
            RuntimeError: if no runner can satisfy the request.
        """
        payload: dict[str, Any] = {"model_id": model_id}
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
                return handle

            except Exception as e:
                logger.warning(f"Failed to acquire from {endpoint}: {e}")
                last_error = str(e)
                continue

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
