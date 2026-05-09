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
REDIS_ENABLED = os.environ.get("REDIS_ENABLED", "true").lower() == "true"
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

# ── Runner service ─────────────────────────────────────────────────────
RUNNER_ENDPOINTS = os.environ.get("RUNNER_ENDPOINTS", "http://localhost:9000").split(
    ","
)

MODEL_CACHE_REFRESH_SEC = int(os.environ.get("MODEL_CACHE_REFRESH_SEC", "60"))

# ── Completion / Retry ─────────────────────────────────────────────────
RUNNER_RETRIES = int(os.environ.get("RUNNER_RETRIES", "2"))
RUNNER_RETRY_BACKOFF_BASE = int(os.environ.get("RUNNER_RETRY_BACKOFF_BASE", "1"))
