"""Admin operations against the runner pool.

These are the operations the api exposes under ``/v1/runner/*`` so ops
tooling (e.g. ``scripts/runner_shutdown.sh``) doesn't need a
port-forward to the cluster-internal runner service.  Each call fans
out across every endpoint in :attr:`RunnerClient._endpoints`, aggregates
the results, and returns a shape the router can serialise directly.

Public functions:

* :func:`list_all_runner_servers` — GET /v1/servers on every runner
* :func:`evict_runner_server` — find by ``server_id`` across runners,
  POST /v1/server/<id>/evict on the one that owns it
* :func:`evict_all_runner_servers` — fan-out evict across the world
* :func:`list_runner_pipelines` — GET /v1/pipelines on every runner
* :func:`unload_runner_pipeline` — POST unload on every runner that
  reports the pipeline as loaded

All functions accept an optional ``RunnerClient`` so tests can pass
a stub instead of monkey-patching the global singleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.runner_client import RunnerClient, runner_client as _default_client
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="runner_admin")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RunnerInventory:
    endpoint: str
    active_servers: int
    servers: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class EvictResult:
    endpoint: Optional[str]
    server_id: str
    succeeded: bool
    detail: Optional[str] = None


@dataclass
class PipelineEntry:
    endpoint: str
    name: str
    task: Optional[str]
    loaded: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _http_json(
    cli: RunnerClient,
    method: str,
    url: str,
    timeout: float = 15.0,
) -> Optional[Dict[str, Any]]:
    """Make one runner-API call and return the JSON body or ``None`` on error.

    Errors are logged at WARNING; the caller decides how to surface them
    (typically by marking the endpoint as failed in the aggregated
    response rather than throwing).
    """
    client = cli._get_client()
    try:
        resp = await client.request(
            method,
            url,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"{method} {url} failed: {e}")
        return None
    if resp.status_code >= 400:
        logger.warning(
            f"{method} {url} returned {resp.status_code}: {resp.text[:200]}"
        )
        return None
    try:
        return resp.json()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"{method} {url} body not JSON: {e}")
        return None


# ---------------------------------------------------------------------------
# Health (GPU stats)
# ---------------------------------------------------------------------------


async def list_runner_health(
    client: Optional[RunnerClient] = None,
) -> List[Dict[str, Any]]:
    """Aggregate ``GET /health`` across every configured runner.

    Each entry is the raw runner /health body plus the source
    ``endpoint`` URL, so the shutdown script can print free VRAM per
    GPU per runner.  Failures land as
    ``{"endpoint": ..., "error": "unreachable"}``.
    """
    cli = client or _default_client
    out: List[Dict[str, Any]] = []
    for endpoint in cli._endpoints:
        body = await _http_json(cli, "GET", f"{endpoint}/health")
        if body is None:
            out.append({"endpoint": endpoint, "error": "unreachable"})
            continue
        # Don't echo the (potentially long) model list — the shutdown
        # tooling only cares about GPU + active_servers.
        out.append({
            "endpoint": endpoint,
            "status": body.get("status"),
            "gpu": body.get("gpu") or {},
            "active_servers": body.get("active_servers", 0),
        })
    return out


# ---------------------------------------------------------------------------
# Servers
# ---------------------------------------------------------------------------


async def list_all_runner_servers(
    client: Optional[RunnerClient] = None,
) -> List[RunnerInventory]:
    """Aggregate ``GET /v1/servers`` across every configured runner.

    Returns one :class:`RunnerInventory` per endpoint, including
    endpoints that errored (so the caller can show the failure rather
    than silently omitting an unreachable runner).
    """
    cli = client or _default_client
    inventories: List[RunnerInventory] = []
    for endpoint in cli._endpoints:
        body = await _http_json(cli, "GET", f"{endpoint}/v1/servers")
        if body is None:
            inventories.append(
                RunnerInventory(
                    endpoint=endpoint, active_servers=0, servers=[], error="unreachable"
                )
            )
            continue
        inventories.append(
            RunnerInventory(
                endpoint=endpoint,
                active_servers=int(body.get("active_servers", 0) or 0),
                servers=list(body.get("servers") or []),
            )
        )
    return inventories


async def evict_runner_server(
    server_id: str,
    client: Optional[RunnerClient] = None,
) -> EvictResult:
    """Find ``server_id`` across all configured runners and evict it.

    server_ids are 12-char hex assigned independently by each runner;
    collisions are vanishingly unlikely.  We scan endpoints in order
    and evict on the first match.  Returns a structured result so the
    router can return 404 cleanly when nothing matches.
    """
    cli = client or _default_client
    inventories = await list_all_runner_servers(cli)
    for inv in inventories:
        if inv.error:
            continue
        if not any(s.get("server_id") == server_id for s in inv.servers):
            continue
        # Hit the evict endpoint on the runner that owns the server.
        body = await _http_json(
            cli, "POST", f"{inv.endpoint}/v1/server/{server_id}/evict"
        )
        if body is None:
            return EvictResult(
                endpoint=inv.endpoint,
                server_id=server_id,
                succeeded=False,
                detail="evict POST failed",
            )
        return EvictResult(
            endpoint=inv.endpoint,
            server_id=server_id,
            succeeded=True,
            detail=str(body.get("status") or "evicted"),
        )
    return EvictResult(
        endpoint=None,
        server_id=server_id,
        succeeded=False,
        detail="server_id not found on any runner",
    )


async def evict_all_runner_servers(
    *,
    model_id: Optional[str] = None,
    client: Optional[RunnerClient] = None,
) -> List[EvictResult]:
    """Fan out evict across every runner.

    Evicts both subprocess servers (llama-server, sd-server) and any
    currently-loaded in-process pipelines (rembg, img23d,
    img23d_part).  The in-process unload is what frees VRAM
    between heavy pipeline runs — without it, PyTorch's allocator
    cache holds GB of weights resident even after ``_loaded=False``,
    and the next pipeline OOMs trying to load on a card it thinks
    has free VRAM but actually doesn't.

    If ``model_id`` is provided, only evict subprocess servers
    whose ``model_id`` matches — useful for "kill all
    qwen-image-2512 instances" without touching the LLM serving the
    same chat session.  In-process pipelines are NOT filtered by
    ``model_id`` (the runner reports them as opaque pipeline names,
    not by underlying model id, and there's only one of each per
    runner).
    """
    cli = client or _default_client
    inventories = await list_all_runner_servers(cli)
    results: List[EvictResult] = []
    for inv in inventories:
        if inv.error:
            continue
        for srv in inv.servers:
            srv_id = srv.get("server_id")
            if not srv_id:
                continue
            if model_id is not None and srv.get("model_id") != model_id:
                continue
            body = await _http_json(
                cli, "POST", f"{inv.endpoint}/v1/server/{srv_id}/evict"
            )
            results.append(
                EvictResult(
                    endpoint=inv.endpoint,
                    server_id=srv_id,
                    succeeded=body is not None,
                    detail=(
                        str(body.get("status") or "evicted")
                        if body
                        else "evict POST failed"
                    ),
                )
            )

    # Also unload any loaded in-process pipelines.  Only when no
    # ``model_id`` filter is set (the filter is server-scoped; an
    # in-process pipeline doesn't have a single ``model_id`` field
    # to match against).
    if model_id is None:
        for entry in await list_runner_pipelines(cli):
            if not entry.loaded:
                continue
            body = await _http_json(
                cli,
                "POST",
                f"{entry.endpoint}/v1/pipelines/{entry.name}/unload",
            )
            results.append(
                EvictResult(
                    endpoint=entry.endpoint,
                    server_id=f"pipeline:{entry.name}",
                    succeeded=body is not None,
                    detail=(
                        f"unloaded (loaded={body.get('loaded')})"
                        if body
                        else "unload POST failed"
                    ),
                )
            )
    return results


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


async def list_runner_pipelines(
    client: Optional[RunnerClient] = None,
) -> List[PipelineEntry]:
    """Aggregate ``GET /v1/pipelines`` across every runner.

    Each returned entry is per-runner — the same pipeline name shows up
    once per runner so ops can see exactly which pod has it loaded.
    """
    cli = client or _default_client
    out: List[PipelineEntry] = []
    for endpoint in cli._endpoints:
        body = await _http_json(cli, "GET", f"{endpoint}/v1/pipelines")
        if body is None:
            continue
        for p in body.get("pipelines") or []:
            out.append(
                PipelineEntry(
                    endpoint=endpoint,
                    name=str(p.get("name") or ""),
                    task=p.get("task"),
                    loaded=bool(p.get("loaded", False)),
                )
            )
    return out


async def unload_runner_pipeline(
    name: str,
    *,
    only_loaded: bool = True,
    client: Optional[RunnerClient] = None,
) -> List[Dict[str, Any]]:
    """Fan-out unload for ``name`` on every runner that knows the pipeline.

    By default we only post to runners that report ``loaded=True`` — an
    unload on an unloaded pipeline is harmless but adds a round-trip and
    confuses the audit log.  Pass ``only_loaded=False`` to force-call
    every runner.
    """
    cli = client or _default_client
    pipelines = await list_runner_pipelines(cli)
    results: List[Dict[str, Any]] = []
    for entry in pipelines:
        if entry.name != name:
            continue
        if only_loaded and not entry.loaded:
            results.append(
                {
                    "endpoint": entry.endpoint,
                    "name": name,
                    "skipped": True,
                    "reason": "not loaded",
                }
            )
            continue
        body = await _http_json(
            cli, "POST", f"{entry.endpoint}/v1/pipelines/{name}/unload"
        )
        results.append(
            {
                "endpoint": entry.endpoint,
                "name": name,
                "succeeded": body is not None,
                "detail": body,
            }
        )
    return results
