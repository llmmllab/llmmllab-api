from fastapi import APIRouter, HTTPException
from typing import List

from models.openai.create_embedding_request import CreateEmbeddingRequest
from models.openai.create_embedding_response import CreateEmbeddingResponse, Usage
from models.openai.embedding import Embedding
from services.runner_client import runner_client
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="embeddings_router")

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


def _normalize_inputs(raw) -> List[str]:
    """Coerce the OpenAI `input` field to a list[str].

    OpenAI accepts str | list[str] | list[int] | list[list[int]] (token
    arrays). Our llama.cpp embedding backend embeds text, so we accept
    string forms and reject pre-tokenized integer inputs rather than
    silently mis-embedding them.
    """
    if isinstance(raw, str):
        if not raw:
            raise HTTPException(status_code=400, detail="embeddings input must not be empty")
        return [raw]
    if isinstance(raw, list):
        if not raw:
            raise HTTPException(status_code=400, detail="embeddings input must not be empty")
        if all(isinstance(x, str) for x in raw):
            return raw
        raise HTTPException(
            status_code=400,
            detail="embeddings input must be a string or array of strings; "
            "pre-tokenized integer inputs are not supported by this backend",
        )
    raise HTTPException(status_code=400, detail="invalid embeddings input")


@router.post("")
@router.post("/")
async def createEmbedding(body: CreateEmbeddingRequest) -> CreateEmbeddingResponse:
    """Operation ID: createEmbedding.

    Acquires the requested embedding model on a runner (e.g.
    nomic-embed-text-v2-moe, served with llama.cpp --embedding) and
    proxies to its OpenAI-compatible /v1/embeddings endpoint. The
    llama.cpp response is already OpenAI-shaped; we re-validate it
    through the typed models before returning.
    """
    inputs = _normalize_inputs(body.input)

    handle = await runner_client.acquire_server(model_id=body.model)
    try:
        resp = await runner_client.proxy_request(
            handle,
            "POST",
            "v1/embeddings",
            json={"model": body.model, "input": inputs},
            timeout=120.0,
        )
        if resp.status_code != 200:
            detail = resp.text[:300] if hasattr(resp, "text") else str(resp.status_code)
            logger.warning(
                "embedding runner returned non-200",
                extra={"status": resp.status_code, "model": body.model},
            )
            raise HTTPException(
                status_code=502,
                detail=f"embedding backend error {resp.status_code}: {detail}",
            )
        data = resp.json()
    finally:
        # Release the slot back to the runner; the llama-server stays
        # warm in the runner cache so the next embed call reuses it.
        await runner_client.release_server(handle)

    items = data.get("data") or []
    embeddings = [
        Embedding(
            embedding=item["embedding"],
            index=int(item.get("index", i)),
            object="embedding",
        )
        for i, item in enumerate(items)
    ]
    usage = data.get("usage") or {}
    return CreateEmbeddingResponse(
        data=embeddings,
        model=data.get("model") or body.model,
        object="list",
        usage=Usage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", usage.get("prompt_tokens", 0))),
        ),
    )
