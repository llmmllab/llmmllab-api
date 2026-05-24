"""Runner-admin endpoints under ``/v1/runner/*``.

Surface the runner's lifecycle operations through the api so ops
tooling doesn't have to ``kubectl port-forward`` to the cluster-internal
runner Service.  Examples:

* ``GET  /v1/runner/servers``              list active servers on every runner
* ``POST /v1/runner/servers/{id}/evict``   force-evict one server by id
* ``POST /v1/runner/servers/evict-all``    fan-out evict; optional ?model=...
* ``GET  /v1/runner/pipelines``            list in-process pipelines
* ``POST /v1/runner/pipelines/{name}/unload``  unload a pipeline

The actual fan-out and per-runner HTTP lives in
``services/runner_admin_service.py``.  This router is a thin wire
layer that just maps requests to service calls and shapes the
response.

Auth: the global auth middleware already gates ``/v1/...`` paths.
Eviction is destructive (cancels in-flight generation, drops VRAM) —
restrict to admin callers via :func:`is_admin`.
"""

from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from middleware.auth import is_admin
from services.runner_admin_service import (
    evict_all_runner_servers,
    evict_runner_server,
    list_all_runner_servers,
    list_runner_health,
    list_runner_pipelines,
    unload_runner_pipeline,
)

router = APIRouter(prefix="/runner", tags=["runner-admin"])


def _require_admin(request: Request) -> None:
    if not is_admin(request):
        raise HTTPException(
            status_code=403,
            detail="Runner admin endpoints require an admin token.",
        )


@router.get("/health")
async def health(request: Request):
    """Per-runner /health fan-out (status, free VRAM per GPU, active count).

    Used by the shutdown script to print before/after VRAM stats.  The
    api's own /health endpoint reports api-side health; this aggregate
    reports the *runner* side which is where VRAM actually lives.
    """
    _require_admin(request)
    return {"runners": await list_runner_health()}


@router.get("/servers")
async def list_servers(request: Request):
    """List every active server across every runner.

    Response::

        {
          "runners": [
            {
              "endpoint": "http://runner-1:8000",
              "active_servers": 2,
              "servers": [{"server_id": ..., "model_id": ..., ...}, ...]
            }, ...
          ]
        }
    """
    _require_admin(request)
    inventories = await list_all_runner_servers()
    return {"runners": [asdict(inv) for inv in inventories]}


@router.post("/servers/{server_id}/evict")
async def evict_server(request: Request, server_id: str):
    """Force-evict the named server.

    Finds the runner that owns ``server_id`` and POSTs
    ``/v1/server/<id>/evict`` to it.  Returns 404 if no runner has the
    server.  Returns 502 if the runner returns an error.
    """
    _require_admin(request)
    result = await evict_runner_server(server_id)
    if not result.succeeded and result.endpoint is None:
        # We searched and nothing matched.
        raise HTTPException(status_code=404, detail=result.detail or "not found")
    if not result.succeeded:
        raise HTTPException(
            status_code=502,
            detail=f"runner {result.endpoint} failed to evict: {result.detail}",
        )
    return asdict(result)


@router.post("/servers/evict-all")
async def evict_all(request: Request, model: Optional[str] = None):
    """Evict every server across every runner.

    Pass ``?model=<id>`` to limit the sweep to servers backing one
    specific model — useful for "kill all sd-server instances" without
    touching the LLM serving the user's chat session.
    """
    _require_admin(request)
    results = await evict_all_runner_servers(model_id=model)
    return {
        "evicted": [asdict(r) for r in results],
        "count": len(results),
    }


@router.get("/pipelines")
async def list_pipelines(request: Request):
    """List in-process pipelines across every runner."""
    _require_admin(request)
    entries = await list_runner_pipelines()
    return {"pipelines": [asdict(e) for e in entries]}


@router.post("/pipelines/{name}/unload")
async def unload_pipeline(request: Request, name: str):
    """Unload an in-process pipeline (e.g. ``img23d``) to free its VRAM.

    Idempotent — runners that don't have the pipeline loaded are
    skipped, and the per-runner result reflects that.
    """
    _require_admin(request)
    results = await unload_runner_pipeline(name)
    return {"results": results, "count": len(results)}
