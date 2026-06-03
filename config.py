import os

from dotenv import load_dotenv

# Load .env file for local development; in k8s env vars are injected directly.
# .env.local is loaded after .env so it can override values (gitignored).
load_dotenv()
load_dotenv(".env.local")

from utils.logging import llmmllogger

logger = llmmllogger.logger.bind(component="Server")

# ── Logging ──────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")
LOG_FORMAT = os.environ.get("LOG_FORMAT", "console")

# ── Authentication ───────────────────────────────────────────────────
AUTH_ISSUER = os.environ.get("AUTH_ISSUER", "https://auth.longstorymedia.com")
AUTH_AUDIENCE = os.environ.get("AUTH_AUDIENCE", "lsm-client")
AUTH_CLIENT_ID = os.environ.get("AUTH_CLIENT_ID", "lsm-client")
AUTH_CLIENT_SECRET = os.environ.get("AUTH_CLIENT_SECRET", "")
AUTH_JWKS_URI = os.environ.get("AUTH_JWKS_URI", "https://auth.longstorymedia.com/keys")
TEST_USER_ID = os.environ.get("TEST_USER_ID", "")

# ── Database ─────────────────────────────────────────────────────────
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "llmmllab")
DB_SSLMODE = os.environ.get("DB_SSLMODE", "disable")
DB_CONNECTION_STRING = os.environ.get(
    "DB_CONNECTION_STRING",
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={DB_SSLMODE}",
)

if DB_CONNECTION_STRING:
    parts = DB_CONNECTION_STRING.split("@")
    if len(parts) > 1:
        masked_conn = f"***@{parts[1]}"
        logger.debug(f"Database connection string available: {masked_conn}")
else:
    logger.warning("No database connection string available")

# ── Redis ────────────────────────────────────────────────────────────
REDIS_ENABLED = os.environ.get("REDIS_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_CONVERSATION_TTL = int(os.environ.get("REDIS_CONVERSATION_TTL", "360"))
REDIS_MESSAGE_TTL = int(os.environ.get("REDIS_MESSAGE_TTL", "180"))
REDIS_SUMMARY_TTL = int(os.environ.get("REDIS_SUMMARY_TTL", "720"))
REDIS_POOL_SIZE = int(os.environ.get("REDIS_POOL_SIZE", "10"))
REDIS_MIN_IDLE_CONNECTIONS = int(os.environ.get("REDIS_MIN_IDLE_CONNECTIONS", "2"))
REDIS_CONNECT_TIMEOUT = int(os.environ.get("REDIS_CONNECT_TIMEOUT", "5"))

# ── Storage / Paths ──────────────────────────────────────────────────
IMAGE_DIR = os.environ.get("IMAGE_DIR", "/root/images")
IMAGE_RETENTION_HOURS = int(os.environ.get("IMAGE_RETENTION_HOURS", "24"))
CONFIG_DIR = os.environ.get("CONFIG_DIR", "/app/config")
HF_HOME = os.environ.get("HF_HOME", "/root/.cache/huggingface")

# ── Runner / llama.cpp ───────────────────────────────────────────────
# Cache eviction is controlled by the runner via CACHE_TIMEOUT_MIN and
# EVICTION_TIMEOUT_MIN environment variables.

# Mirror of the runner-side ``CACHE_TIMEOUT_MIN`` (minutes).  Used by
# ``runner_client._select_runner`` to decide when a server that has
# been idle for "long enough" can be commandeered by a new session
# without preempting another session that's merely paused mid-turn.
# Should match the runner's value. Default bumped to 10 so that
# multi-turn interactive sessions paused for a few minutes (typing,
# thinking, switching to another app) don't get their slot
# commandeered the moment they leave the keyboard.
CACHE_TIMEOUT_MIN = int(os.environ.get("CACHE_TIMEOUT_MIN", "10"))

# ── External API keys ───────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SEARX_HOST = os.environ.get("SEARX_HOST", "")

# ── Feature flags ───────────────────────────────────────────────────
ENABLE_TOOL_CONTINUATION = (
    os.environ.get("ENABLE_TOOL_CONTINUATION", "true").lower() == "true"
)

# ── Chat / LLM ─────────────────────────────────────────────────────
CHAT_OPENAI_MAX_RETRIES = int(os.environ.get("CHAT_OPENAI_MAX_RETRIES", "2"))

# ── Runner service ─────────────────────────────────────────────────────
RUNNER_ENDPOINTS = os.environ.get("RUNNER_ENDPOINTS", "http://localhost:9000").split(
    ","
)

MODEL_CACHE_REFRESH_SEC = int(os.environ.get("MODEL_CACHE_REFRESH_SEC", "60"))

# ── Server-side tool execution ─────────────────────────────────────────
# Master switch for the server-side execution of "web_search" / "web_fetch"
# style tools.  When false, the API leaves all tools to the client even if
# they match the locally-executable name set — used for tenants that want
# to own their own web access.  Per-request override:
# ``X-Server-Side-Tools: false`` header (Anthropic router).  Per-tool
# override: include ``{"execute": "client"}`` in the tool definition.
SERVER_SIDE_TOOLS_ENABLED = os.environ.get(
    "SERVER_SIDE_TOOLS_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}

# MCP server for web_search / web_fetch.  When set, server-side tool
# execution delegates to this MCP server instead of running the SearxNG
# + Playwright code inline.  Empty falls back to the inline path (handy
# for local dev without a deployed MCP).  Default points at the in-cluster
# Service; override to "" for fully inline behaviour.
MCP_WEB_TOOLS_URL = os.environ.get(
    "MCP_WEB_TOOLS_URL",
    "http://mcp-server-web.llmmllab.svc.cluster.local:8000",
)

# Hard cap on the number of Agent ↔ ServerToolNode loops within a single
# completion.  Prevents a model from spinning the tool loop indefinitely
# when its results don't satisfy the prompt.  Hitting the cap routes the
# graph to END; the assistant message produced on that iteration is what
# the client sees.
SERVER_TOOL_MAX_ITERATIONS = int(
    os.environ.get("SERVER_TOOL_MAX_ITERATIONS", "4")
)

# - Priority Queue -
PRIORITY_QUEUE_ENABLED = (
    os.environ.get("PRIORITY_QUEUE_ENABLED", "true").lower() == "true"
)
PRIORITY_QUEUE_MAX_SIZE = int(os.environ.get("PRIORITY_QUEUE_MAX_SIZE", "100"))
PRIORITY_QUEUE_TIMEOUT_SEC = int(
    os.environ.get("PRIORITY_QUEUE_TIMEOUT_SEC", "300")
)
PRIORITY_QUEUE_AGE_THRESHOLD_SEC = int(
    os.environ.get("PRIORITY_QUEUE_AGE_THRESHOLD_SEC", "60")
)
PRIORITY_QUEUE_MAX_WAIT_MIN_SEC = int(
    os.environ.get("PRIORITY_QUEUE_MAX_WAIT_MIN_SEC", "1")
)
PRIORITY_QUEUE_MAX_WAIT_MAX_SEC = int(
    os.environ.get("PRIORITY_QUEUE_MAX_WAIT_MAX_SEC", "3600")
)
# ── Completion / Retry ─────────────────────────────────────────────────
RUNNER_RETRIES = int(os.environ.get("RUNNER_RETRIES", "2"))
RUNNER_RETRY_BACKOFF_BASE = int(os.environ.get("RUNNER_RETRY_BACKOFF_BASE", "1"))
# Stale server recovery retry count.
# When a llama.cpp server handle is evicted by the runner, the service
# releases the stale handle, refreshes the model map, and retries the
# workflow with a fresh server. Set to 0 to disable retries entirely.
#
# Default is 2 (not 1): the dominant stale-handle cause in production is a
# runner pod being OOMKilled and restarting, which drops every in-memory
# server handle at once.  A single retry can land back on the same runner
# mid-restart (its /server/create then 500s and trips the circuit breaker),
# exhausting the lone retry.  A second retry — after the model-map refresh
# has had a beat to route around the tripped endpoint — recovers cleanly.
STALE_SERVER_RETRIES = int(os.environ.get("STALE_SERVER_RETRIES", "2"))

# Per-request-category HTTP timeouts for the runner client.  Health
# checks are cheap and should fail fast; "fast" covers small-body
# requests like list / status; acquire is the long-pole one because
# llama-server cold-start can take 1-2 minutes for big quantised
# models.
RUNNER_HEALTH_TIMEOUT_SEC = float(
    os.environ.get("RUNNER_HEALTH_TIMEOUT_SEC", "5.0")
)
RUNNER_FAST_TIMEOUT_SEC = float(
    os.environ.get("RUNNER_FAST_TIMEOUT_SEC", "10.0")
)
RUNNER_ACQUIRE_TIMEOUT_SEC = float(
    os.environ.get("RUNNER_ACQUIRE_TIMEOUT_SEC", "150.0")
)

# Circuit-breaker thresholds for the runner pool.  After
# ``RUNNER_MAX_ACQUIRE_FAILURES`` consecutive failures the endpoint
# is marked unhealthy for ``RUNNER_UNHEALTHY_WINDOW_SEC`` seconds.
RUNNER_MAX_ACQUIRE_FAILURES = int(
    os.environ.get("RUNNER_MAX_ACQUIRE_FAILURES", "3")
)
RUNNER_UNHEALTHY_WINDOW_SEC = float(
    os.environ.get("RUNNER_UNHEALTHY_WINDOW_SEC", "60.0")
)
# Per-endpoint connection retries during a single acquire attempt
# (transient network blips between the api and runner Service).
RUNNER_ACQUIRE_RETRIES = int(os.environ.get("RUNNER_ACQUIRE_RETRIES", "2"))

# ── Cold-start (model-loading 503) retry ────────────────────────────────
# A FRESH model server takes ~45-90 s to load on cold start (big quantised
# GGUF + mmproj).  While it's loading, the runner's /v1/server/create
# returns HTTP 503 "Runner busy starting the model …", which acquire_server
# surfaces as a RuntimeError ("No healthy runner available … Last error:
# …503…").  That is *transient* — a short wait then retry succeeds — but it
# is neither a connection error nor a stale-handle 404, so the existing
# retry layers ignored it and the 503 bubbled all the way to the client
# (the agent turn failed).
#
# COLD_START_RETRIES bounds the number of extra attempts; COLD_START_BACKOFF_SEC
# is the (fixed) wait between them.  The wait is intentionally LONGER than the
# generic connection-error backoff because model load is ~45-90 s: 4 retries ×
# 20 s ≈ 80 s of patience covers a typical cold start without hanging a caller
# indefinitely.  Set COLD_START_RETRIES=0 to restore the old surface-immediately
# behaviour.
COLD_START_RETRIES = int(os.environ.get("COLD_START_RETRIES", "4"))
COLD_START_BACKOFF_SEC = float(os.environ.get("COLD_START_BACKOFF_SEC", "20.0"))

# ── Tracing (middleware/tracing.py) ────────────────────────────────────
# OTLP gRPC endpoint for OpenTelemetry trace export.  Points at the
# in-cluster Tempo service by default; override to "" to disable
# tracing entirely (the setup helper no-ops on empty endpoint).
TEMPO_ENDPOINT = os.environ.get(
    "TEMPO_ENDPOINT", "http://tempo.llmmllab.svc.cluster.local:4317"
)

# ── Database maintenance (db/maintenance.py, db/__init__.py) ───────────
# How often the maintenance loop runs (VACUUM ANALYZE + sequence align,
# plus optional REINDEX).  Loop sleeps this many hours between runs.
DB_MAINTENANCE_INTERVAL_HOURS = int(
    os.environ.get("DB_MAINTENANCE_INTERVAL_HOURS", "24")
)
# Delay before the FIRST maintenance run after pod startup, so a new
# pod doesn't compete with warm-up traffic.
DB_MAINTENANCE_INITIAL_DELAY_SECONDS = int(
    os.environ.get("DB_MAINTENANCE_INITIAL_DELAY_SECONDS", "300")
)
# Opt-in REINDEX during maintenance.  Off by default because REINDEX
# CONCURRENTLY can still trigger stale-OID plan errors on live traffic
# under TimescaleDB; enable in low-traffic environments.
DB_REINDEX_ON_MAINTENANCE = os.environ.get(
    "DB_REINDEX_ON_MAINTENANCE", "false"
).lower() in ("1", "true", "yes", "on")

# ── Image server lifecycle (services/image_service.py) ─────────────────
# Tear down sd-server / image-server subprocesses after every image
# request, instead of keeping them warm.  Image servers are 4-12 GB
# resident; for interactive workflows the cold-start cost is worth
# the freed VRAM.  Set to false for benchmarking or batched generation.
IMG_SERVER_AUTO_SHUTDOWN = os.environ.get(
    "IMG_SERVER_AUTO_SHUTDOWN", "true"
).lower() in ("1", "true", "yes", "on")

# ── Vision-token accounting (services/token_counter.py) ────────────────
# Image blocks in multimodal messages cost real vision tokens at the
# runner side (mmproj produces ~700-2500 tokens per image depending on
# resolution), but llama.cpp's text ``/tokenize`` endpoint can't see
# them.  These knobs control how the api estimates those tokens so the
# pre-trim in ``agents/base.py::_ensure_context_fits`` matches what the
# runner actually receives.
#
# Defaults are calibrated for Qwen2/3-VL family (used by Qwen3.6-27B
# with ``clip_model_path: mmproj.gguf``).  Override for other vision
# towers via env.
IMAGE_TOKENS_DEFAULT = int(os.environ.get("IMAGE_TOKENS_DEFAULT", "1500"))
VISION_PATCH_PX = int(os.environ.get("VISION_PATCH_PX", "28"))
VISION_MAX_LONG_EDGE_PX = int(os.environ.get("VISION_MAX_LONG_EDGE_PX", "1280"))

# ── Raw Token Debug ────────────────────────────────────────────────
# When enabled, writes every raw model token (no stripping/modification)
# plus all user messages to <RAW_TOKEN_DEBUG_DIR>/<session_id>.tokens.
# Useful for diagnosing premature stops and context overflow patterns.
RAW_TOKEN_DEBUG = os.environ.get("RAW_TOKEN_DEBUG", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
RAW_TOKEN_DEBUG_DIR = os.environ.get(
    "RAW_TOKEN_DEBUG_DIR", "/tmp/llmmllab_debug"
)
