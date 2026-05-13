"""

This FastAPI application provides a comprehensive API for generating images using Stable Diffusion
and text generation with OpenAI-compatible endpoints. The server integrates multiple services:

- Image generation via Stable Diffusion
- Text generation via vLLM with OpenAI-compatible API
- Model management (loading, unloading, listing)
- LoRA adapter management
- Resource monitoring and management

Environment Variables:
- HF_TOKEN: Hugging Face token for model access
- VLLM_MODEL: Model to use for vLLM service (default: "microsoft/DialoGPT-medium")
- PYTORCH_CUDA_ALLOC_CONF: Configured to "expandable_segments:True" to avoid memory fragmentation

Main Components:
- FastAPI application with various routers
- Lifespan context manager for service initialization and cleanup
- Hardware monitoring and memory management
- OpenAI-compatible endpoints (/v1/*)
- Health check endpoint for monitoring system status

Endpoints:
- /: Root endpoint with API information
- /health: Health check endpoint
- /images/*: Image generation endpoints
- /chat/*: Chat completion endpoints
- /models/*: Model management endpoints
- /loras/*: LoRA adapter management endpoints
- /resources/*: System resource endpoints


The application handles initialization and cleanup of all services and provides
detailed logging throughout the startup and shutdown processes.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config import CONFIG_DIR, IMAGE_DIR, TEST_USER_ID
from routers import (
    # images,  # requires PIL (GPU dep, belongs in runner)
    config,
    static,
    websockets,
    users,
    todos,
    model,
    chat,
    conversation,
    db_admin,
    documents,
    ollama,
    api_key,
    metrics as metrics_router,
)
from routers.openai import ROUTERS as OPENAI_ROUTERS
from routers.anthropic import ROUTERS as ANTHROPIC_ROUTERS
from routers.common import ROUTERS as COMMON_ROUTERS
from middleware import (
    AuthMiddleware,
    db_init_middleware,
    MessageValidationMiddleware,
)
from middleware.priority import PriorityMiddleware
from middleware.request_id import RequestIdMiddleware
from middleware.prometheus_metrics import PrometheusMiddleware
from middleware.tracing import setup_tracing, shutdown_tracing
from config import AUTH_JWKS_URI
from services.cleanup_service import cleanup_service
from services.queue_exceptions import QueueFullError, QueueTimeoutError
from db.maintenance import maintenance_service
from utils.logging import llmmllogger
from composer_init import shutdown_composer

logger = llmmllogger.bind(component="app")


# Create required directories if they don't exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Simplified lifespan: start services, optionally init composer, yield, then shutdown."""
    logger.info("Initializing services...")
    cleanup_service.start()

    # Initialize async Redis for durable priority queue
    try:
        from db.redis_client import (  # pylint: disable=import-outside-toplevel
            async_redis,
        )

        await async_redis.connect()
    except Exception as e:
        logger.warning(f"Async Redis init failed (queue will use in-memory): {e}")

    # Initialize database connection and schema if configured
    try:
        from db import storage  # pylint: disable=import-outside-toplevel
        from config import (  # pylint: disable=import-outside-toplevel
            DB_CONNECTION_STRING,
        )

        assert DB_CONNECTION_STRING is not None, "DB_CONNECTION_STRING is not set"

        await storage.initialize(DB_CONNECTION_STRING)
        logger.info("Database schema initialized successfully")

        # Seed local dev user and API key
        if TEST_USER_ID:
            try:
                from db.seed import (
                    seed_test_user_and_api_key,
                )  # pylint: disable=import-outside-toplevel

                assert storage.session_factory is not None
                api_key = await seed_test_user_and_api_key(
                    storage.session_factory, TEST_USER_ID
                )
                if api_key:
                    logger.info(
                        f"Local dev credentials — user_id: {TEST_USER_ID}, "
                        f"api_key: {api_key} (saved to .env.local)"
                    )
            except Exception as e:
                logger.error(f"Failed to seed test user/API key: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")

    # Warm up model-to-runner map so first requests use the fast path
    try:
        from services.runner_client import (
            runner_client,
        )  # pylint: disable=import-outside-toplevel

        await runner_client.refresh_model_map()
        logger.info("Runner model map warmed up")
    except Exception as e:
        logger.warning(f"Runner model map warm-up failed: {e}")

    # Wire up resource-aware priority queue
    try:
        from collections import defaultdict
        from services.runner_client import (
            runner_client as _runner_client,
        )  # pylint: disable=import-outside-toplevel
        from services.priority_queue import (
            priority_queue,
        )  # pylint: disable=import-outside-toplevel

        _active_counts: dict[str, int] = defaultdict(int)
        _SCHEDULED_CAP = 3  # --parallel (4) - 1 reserved for USER

        # Per-session tracking: use centralized registry
        from services.session_registry import (
            get_session_state,
            get_session,
            remove_session,
            SessionState as _SessionState,
        )

        async def _can_proceed(metadata):
            if not metadata.model_id:
                return True
            try:
                from models.request_priority_metadata import RequestSource
            except Exception:
                RequestSource = None
            if (
                RequestSource
                and metadata.source in (RequestSource.SCHEDULED, RequestSource.SYSTEM)
                and _active_counts[metadata.model_id] >= _SCHEDULED_CAP
            ):
                return False
            try:
                return await _runner_client.check_slot_availability(metadata.model_id)
            except Exception:
                return True

        def _on_release(metadata):
            if metadata.model_id:
                _active_counts[metadata.model_id] += 1

            if metadata.session_id:
                import time
                state = get_session(metadata.session_id)
                if state is None:
                    from services.session_registry import get_session_state
                    states = get_session_state()
                    state = _SessionState()
                    state.model_id = metadata.model_id or "unknown"
                    state.source = metadata.source.value if metadata.source else "user"
                    state.start_time = time.monotonic()
                    state.turn_count = 0
                    states[metadata.session_id] = state
                state.turn_count += 1
                active_sessions.labels(
                    model_id=state.model_id,
                    source=state.source,
                ).inc()

        def _on_complete(metadata):
            if metadata.model_id:
                _active_counts[metadata.model_id] -= 1

            if metadata.session_id:
                state = get_session(metadata.session_id)
                if state:
                    active_sessions.labels(
                        model_id=state.model_id,
                        source=state.source,
                    ).dec()

        priority_queue.set_can_proceed_callback(_can_proceed)
        priority_queue.set_session_callbacks(_on_release, _on_complete)
        logger.info("Resource-aware priority queue callback wired up (scheduled cap=%d)", _SCHEDULED_CAP)

        # Import session metrics
        try:
            from middleware.api_metrics import (
                active_sessions,
                session_duration_seconds,
                session_turns_total,
            )
        except ImportError:
            active_sessions = None  # pylint: disable=invalid-name
            session_duration_seconds = None
            session_turns_total = None

        # Background task to detect completed sessions and observe metrics
        async def _cleanup_stale_sessions(stale_timeout=300.0):
            import time
            while True:
                await asyncio.sleep(30)
                now = time.monotonic()
                all_states = get_session_state()
                stale_ids = [
                    sid for sid, state in all_states.items()
                    if now - state.start_time > stale_timeout
                    and state.turn_count > 0
                ]
                for sid in stale_ids:
                    state = remove_session(sid)
                    if state and session_duration_seconds:
                        session_duration_seconds.labels(
                            model_id=state.model_id,
                            source=state.source,
                        ).observe(now - state.start_time)
                    if state and session_turns_total:
                        session_turns_total.labels(
                            model_id=state.model_id,
                            source=state.source,
                        ).observe(state.turn_count)

        asyncio.create_task(_cleanup_stale_sessions())
    except Exception as e:
        logger.warning(f"Failed to wire up queue resource callback: {e}")

    # Wait for at least one runner to be healthy before accepting requests
    try:
        import time as _time  # pylint: disable=import-outside-toplevel
        from services.runner_client import (
            runner_client,
        )  # pylint: disable=import-outside-toplevel

        _start = _time.monotonic()
        _timeout = 120
        while _time.monotonic() - _start < _timeout:
            try:
                models = await runner_client.list_models()
                if models:
                    logger.info(f"Runner ready with {len(models)} models")
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            logger.error(
                "Runner not ready after %ds timeout — accepting requests anyway",
                _timeout,
            )
    except Exception as e:
        logger.warning(f"Runner readiness check failed: {e}")

    try:
        yield  # Application runs here
    finally:
        # Shutdown: clean up resources
        logger.info("Shutting down services...")

        # Stop database maintenance service if running
        try:
            logger.info("Stopping database maintenance service...")
            await maintenance_service.stop_maintenance_schedule()
            logger.info("Database maintenance service stopped")
        except Exception as e:
            logger.info(f"Error stopping database maintenance service: {e}")

        # Stop composer service
        try:
            await shutdown_composer()
            logger.info("Composer service shutdown completed")
        except Exception as e:
            logger.info(f"Error stopping composer service: {e}")

        # Close runner HTTP client pool
        try:
            from services.runner_client import (
                runner_client,
            )  # pylint: disable=import-outside-toplevel

            await runner_client.aclose()
            logger.info("Runner client closed")
        except Exception as e:
            logger.info(f"Error closing runner client: {e}")

        # Close priority queue (stops background recheck task)
        try:
            from services.priority_queue import (
                priority_queue,
            )  # pylint: disable=import-outside-toplevel

            await priority_queue.close()
        except Exception as e:
            logger.info(f"Error closing priority queue: {e}")

        # Shutdown tracing
        shutdown_tracing()

        # Close async Redis
        try:
            from db.redis_client import (  # pylint: disable=import-outside-toplevel
                async_redis,
            )

            await async_redis.close()
        except Exception as e:
            logger.info(f"Error closing async Redis: {e}")

        cleanup_service.shutdown()

logger.info(f"Pre-initializing auth middleware with JWKS URI: {AUTH_JWKS_URI}")
global_auth_middleware = AuthMiddleware(AUTH_JWKS_URI)

# Initialize the FastAPI application with the lifespan context manager
app = FastAPI(
    title="Inference API",
    description="""FastAPI server for inference

## Authentication

This API uses JWT tokens for authentication. To authorize:

1. Click the "Authorize" button in the top right corner of this page
2. Enter your JWT token in the format: `Bearer <your_token>`
3. Click "Authorize" to add it to your session

You can also use API keys via the `X-API-Key` header.
""",
    version="0.1.0",
    redoc_url="/redoc",
    docs_url="/docs",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "images", "description": "Image generation endpoints"},
        {"name": "chat", "description": "Chat completion endpoints"},
        {"name": "models", "description": "Model management endpoints"},
        {"name": "conversation", "description": "Conversation management endpoints"},
        {"name": "users", "description": "User management endpoints"},
        {"name": "config", "description": "Configuration endpoints"},
        {"name": "resources", "description": "System resource endpoints"},
    ],
    # Note: Security schemes are added via event handler below
)


@app.middleware("http")
async def proxy_headers_middleware(request: Request, call_next):
    """Middleware to handle proxy headers for correct scheme detection in redirects"""
    # Trust X-Forwarded-Proto header from reverse proxy
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_proto:
        # Update the request scope to use the forwarded protocol
        request.scope["scheme"] = forwarded_proto

    response = await call_next(request)
    return response


# Store auth middleware in app.state right away
app.state.auth_middleware = global_auth_middleware

# Exception handlers for queue rejections
@app.exception_handler(QueueTimeoutError)
async def queue_timeout_handler(request: Request, exc: QueueTimeoutError):
    return JSONResponse(
        status_code=408,
        content={"error": str(exc)},
        headers={"Retry-After": "5"},
    )


@app.exception_handler(QueueFullError)
async def queue_full_handler(request: Request, exc: QueueFullError):
    return JSONResponse(
        status_code=503,
        content={"error": str(exc) or "Queue is full. Please retry later."},
        headers={"Retry-After": "10"},
    )
# Add message validation middleware to ensure proper response structure
app.add_middleware(MessageValidationMiddleware)
app.middleware("http")(db_init_middleware)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(PriorityMiddleware)
app.add_middleware(PrometheusMiddleware)


# Monkey-patch app.openapi() to add security schemes
def _get_original_openapi(self):
    """Get the original openapi function before patching"""
    return FastAPI.openapi(self)


def _openapi_with_security(self):
    """Wrapper around openapi() that adds security schemes for Swagger UI"""
    # Call the original openapi method
    schema = _get_original_openapi(self)

    # Add security schemes to components
    if "components" not in schema:
        schema["components"] = {}

    schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Bearer token. Example: 'Bearer your_token_here'",
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "scheme": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API Key header. Example: 'X-API-Key: your_key_here'",
        },
    }

    # Set default security for all endpoints
    schema["security"] = [{"bearerAuth": []}, {"apiKeyAuth": []}]

    return schema


# Replace the openapi method for this instance
app.openapi = _openapi_with_security.__get__(app, type(app))


@app.middleware("http")
async def auth_middleware_handler(request: Request, call_next):
    """Authentication middleware to handle token validation and user identification"""
    # Get logger for debugging
    logger.debug(f"Processing request for path: {request.url.path}")

    # Skip auth for public endpoints
    public_paths = [
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/metrics",
        "/static/images/view/",
    ]

    # Check for exact root path or if the path starts with any of the public paths
    if request.url.path == "/" or any(
        request.url.path.startswith(path) for path in public_paths
    ):
        logger.debug(f"Skipping auth for public path: {request.url.path}")
        response = await call_next(request)
        return response

    # Skip auth if middleware is not initialized or disabled
    app_instance = request.app
    if not hasattr(app_instance.state, "auth_middleware"):
        logger.error(
            "Auth middleware not initialized in app state - this should never happen now"
        )
        # Instead of skipping auth, we'll return an error
        return JSONResponse(
            status_code=500,
            content={"error": "Authentication middleware not initialized properly"},
        )

    try:
        # Get the auth middleware from app state
        auth_middleware = app_instance.state.auth_middleware
        logger.debug(f"Authenticating request for path: {request.url.path}")

        # Authenticate the request
        await auth_middleware.authenticate(request)
        logger.debug("Authentication successful")

        # If authentication succeeds, proceed with the request
        response = await call_next(request)

        # Add any auth-related response headers
        if hasattr(request.state, "response_headers"):
            for key, value in request.state.response_headers.items():
                response.headers[key] = value

        return response
    except HTTPException as e:
        # Handle FastAPI HTTP exceptions with proper status code and detail
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except ValueError as e:
        # Handle validation errors
        return JSONResponse(
            status_code=400, content={"error": f"Validation error: {str(e)}"}
        )
    except (ConnectionError, TimeoutError) as e:
        # Handle connection errors
        return JSONResponse(
            status_code=503, content={"error": f"Service unavailable: {str(e)}"}
        )
    except RuntimeError as e:
        # Handle runtime errors
        return JSONResponse(
            status_code=500, content={"error": f"Server error: {str(e)}"}
        )


# Add CORS middleware BEFORE including routers to ensure it's processed in the right order
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include non-versioned routers (for backward compatibility)
# app.include_router(images.router)  # requires PIL (GPU dep)
app.include_router(model.router)
app.include_router(chat.router)
app.include_router(conversation.router)
app.include_router(config.router)
app.include_router(static.router)
app.include_router(websockets.router)
app.include_router(users.router)
app.include_router(todos.router)
app.include_router(documents.router)

# Import and include the internal router
app.include_router(db_admin.router)

# Include Ollama-compatible API endpoints
app.include_router(ollama.router)

# Include auto-generated OpenAI-compatible API endpoints (excluding models and files)
for router in OPENAI_ROUTERS:
    app.include_router(router, prefix="/v1")

# Include auto-generated Anthropic-compatible API endpoints (excluding models and files)
for router in ANTHROPIC_ROUTERS:
    app.include_router(router, prefix="/v1")

# Include common endpoints (models and files)
for router in COMMON_ROUTERS:
    app.include_router(router, prefix="/v1")

# Include API key management endpoints
app.include_router(api_key.router)
app.include_router(metrics_router.router)

# Include session admin endpoints
try:
    from routers import session_admin
    app.include_router(session_admin.router)
except ImportError:
    pass

# Initialize distributed tracing
setup_tracing("llmmllab-api", app)


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    return "OK"
