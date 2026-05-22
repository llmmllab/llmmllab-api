"""
Minimal MCP (streamable-http) client for calling the deployed
``mcp-server-web`` instance.

Why hand-rolled instead of pulling in the ``mcp`` / ``fastmcp`` SDK:
the dependency surface is tiny (initialize handshake + tools/call) and
adding a new dependency to the API was disproportionate cost for two
tool calls.  We already have ``httpx`` for HTTP and ``structlog`` for
logging; that's everything we need.

Protocol summary (Streamable HTTP MCP):
  1. POST {base_url}  → JSON-RPC ``initialize``
     server responds with capabilities and an ``mcp-session-id`` header
  2. POST {base_url}  → JSON-RPC notification ``notifications/initialized``
     (no response payload)
  3. POST {base_url}  → JSON-RPC ``tools/call`` (repeat as many times as
     needed for the lifetime of the session)
  4. (optional) DELETE {base_url} to release the session

We treat the session as process-scoped: handshake once on first use,
cache the session ID, and on a 404 (server restart → session expired)
re-handshake once and retry the failing call.

Response Content-Type can be either ``application/json`` (single reply)
or ``text/event-stream`` (one or more SSE events).  Both are handled.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, Optional

import httpx

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="MCPWebClient")


PROTOCOL_VERSION = "2025-06-18"


class MCPCallError(RuntimeError):
    """Raised when the MCP server returns a JSON-RPC error or the
    transport layer fails irrecoverably (after one re-init retry)."""


def _parse_streamable_response(content_type: str, body: str) -> Dict[str, Any]:
    """Decode a fastmcp streamable-http response into a JSON-RPC envelope.

    fastmcp may answer with either ``application/json`` (one envelope) or
    ``text/event-stream`` (one or more SSE ``data:`` lines, each a full
    JSON-RPC envelope).  We accept either and return the first
    non-notification envelope.
    """
    ct = (content_type or "").lower()
    if "text/event-stream" in ct:
        for line in body.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload:
                continue
            try:
                env = json.loads(payload)
            except json.JSONDecodeError:
                continue
            # Skip notifications (no "id") — we want the reply.
            if "id" in env:
                return env
        raise MCPCallError("No JSON-RPC reply in SSE stream")
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise MCPCallError(f"Non-JSON MCP response: {body[:200]}") from e


class MCPWebClient:
    """Streamable-HTTP MCP client for the web-search / fetch_page tools.

    Single instance per process.  Use ``get_default()`` to acquire the
    module-level singleton.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 90.0,
        client_name: str = "llmmllab-api",
    ):
        # fastmcp's streamable-http transport mounts at ``/mcp/``; accept
        # either the bare base URL (``http://.../``) or one that already
        # includes the mount.
        url = base_url.rstrip("/")
        if not url.endswith("/mcp"):
            url = url + "/mcp"
        self.url = url
        self.timeout = timeout
        self.client_name = client_name

        self._http: Optional[httpx.AsyncClient] = None
        self._session_id: Optional[str] = None
        self._init_lock = asyncio.Lock()

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Accept": "application/json, text/event-stream"},
            )
        return self._http

    async def close(self) -> None:
        """Release the cached session and close the HTTP client."""
        if self._session_id and self._http is not None:
            try:
                await self._http.delete(
                    self.url,
                    headers={"mcp-session-id": self._session_id},
                    timeout=5.0,
                )
            except Exception:
                pass  # best effort
            self._session_id = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def _post(
        self,
        payload: Dict[str, Any],
        *,
        session_id: Optional[str] = None,
    ) -> httpx.Response:
        client = await self._client()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if session_id:
            headers["mcp-session-id"] = session_id
        return await client.post(self.url, json=payload, headers=headers)

    async def _ensure_session(self) -> str:
        """Initialise the MCP session (handshake) if not already done."""
        if self._session_id is not None:
            return self._session_id
        async with self._init_lock:
            if self._session_id is not None:
                return self._session_id

            init_req = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": self.client_name, "version": "1.0"},
                },
            }
            resp = await self._post(init_req)
            session_id = resp.headers.get("mcp-session-id") or resp.headers.get(
                "Mcp-Session-Id"
            )
            env = _parse_streamable_response(
                resp.headers.get("content-type", ""), resp.text
            )
            if "error" in env:
                raise MCPCallError(
                    f"MCP initialize failed: {env['error'].get('message')}"
                )
            if not session_id:
                # Stateless servers may omit the session id; we still need
                # something to send, but most fastmcp deployments are
                # session-ful.  Synthesize a placeholder so subsequent
                # requests don't omit the header (some servers reject
                # missing header even in stateless mode).
                session_id = str(uuid.uuid4())

            self._session_id = session_id

            # Required notification — server's contract for a complete
            # handshake.  No reply expected.
            initialized_notify = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            try:
                await self._post(initialized_notify, session_id=session_id)
            except Exception as e:
                # Notification failure isn't fatal for many fastmcp builds.
                logger.debug("initialized notification raised", error=str(e))

            logger.info("MCP session established", session_id=session_id[:8])
            return session_id

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        *,
        _retry: bool = True,
    ) -> str:
        """Call a tool and return its text result.

        On a 404 / "Session not found" reply (server restart), one
        automatic re-init + retry is attempted before raising.
        """
        session_id = await self._ensure_session()
        req = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        resp = await self._post(req, session_id=session_id)

        # Session expired → server restarted; clear cache and try once more.
        if resp.status_code == 404 and _retry:
            logger.info("MCP session expired (404), re-initialising")
            self._session_id = None
            return await self.call_tool(name, arguments, _retry=False)

        if resp.status_code >= 400:
            raise MCPCallError(
                f"MCP HTTP {resp.status_code} from {self.url}: {resp.text[:300]}"
            )

        env = _parse_streamable_response(resp.headers.get("content-type", ""), resp.text)
        if "error" in env:
            err = env["error"]
            raise MCPCallError(
                f"MCP tool '{name}' returned error: {err.get('message')}"
            )

        result = env.get("result") or {}
        content = result.get("content") or []
        # MCP returns content as a list of blocks; concatenate the text ones.
        text_parts = [
            block.get("text", "") for block in content if block.get("type") == "text"
        ]
        return "".join(text_parts)


# Module-level singleton wired from config.MCP_WEB_TOOLS_URL.  Lazy-built
# on first use so import order doesn't matter.
_default_client: Optional[MCPWebClient] = None


def get_default() -> Optional[MCPWebClient]:
    """Return the configured MCP web client, or ``None`` if no URL is set."""
    global _default_client
    if _default_client is not None:
        return _default_client
    import config

    url = getattr(config, "MCP_WEB_TOOLS_URL", None)
    if not url:
        return None
    _default_client = MCPWebClient(url)
    return _default_client


async def shutdown_default() -> None:
    """Close the module-level singleton (call from app shutdown)."""
    global _default_client
    if _default_client is not None:
        await _default_client.close()
        _default_client = None
