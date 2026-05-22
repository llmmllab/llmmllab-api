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
    "http://mcp-server-web.llmmllab-mcp.svc.cluster.local:8000",
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
STALE_SERVER_RETRIES = int(os.environ.get("STALE_SERVER_RETRIES", "1"))
