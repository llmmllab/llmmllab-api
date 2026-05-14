"""Business-level Prometheus metrics for the API service."""

from prometheus_client import Counter, Histogram, Gauge
from middleware.prometheus_metrics import get_metrics_registry

registry = get_metrics_registry()

workflow_completions_total = Counter(
    "workflow_completions_total",
    "Total workflow completions",
    ["workflow_type", "status"],
    registry=registry,
)

workflow_duration_seconds = Histogram(
    "workflow_duration_seconds",
    "Workflow execution duration in seconds",
    ["workflow_type"],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 300.0),
    registry=registry,
)

tool_calls_total = Counter(
    "tool_calls_total",
    "Total tool calls executed",
    ["tool_name"],
    registry=registry,
)

empty_response_retries_total = Counter(
    "empty_response_retries_total",
    "Total empty response retries",
    registry=registry,
)

# --- Queue metrics (migrated from priority_queue.py + new) ---

queue_enqueued_total = Counter(
    "llmmllab_api_queue_enqueued_total",
    "Total requests enqueued by priority",
    ["priority", "source"],
    registry=registry,
)

queue_dequeued_total = Counter(
    "llmmllab_api_queue_dequeued_total",
    "Total requests dequeued by priority",
    ["priority", "source"],
    registry=registry,
)

queue_wait_time_seconds = Histogram(
    "llmmllab_api_queue_wait_time_seconds",
    "Time spent waiting in queue by priority",
    ["priority", "source"],
    registry=registry,
)

queue_size = Gauge(
    "llmmllab_api_queue_size",
    "Current queue size by priority",
    ["priority"],
    registry=registry,
)

queue_aged_total = Counter(
    "llmmllab_api_queue_aged_total",
    "Total requests promoted due to aging",
    ["from_priority", "to_priority"],
    registry=registry,
)

queue_size_by_model = Gauge(
    "llmmllab_api_queue_size_by_model",
    "Current queue size by model",
    ["model_id"],
    registry=registry,
)

queue_size_by_source = Gauge(
    "llmmllab_api_queue_size_by_source",
    "Current queue size by request source",
    ["source"],
    registry=registry,
)

queue_size_by_model_source = Gauge(
    "llmmllab_api_queue_size_by_model_source",
    "Current queue size by model and request source",
    ["model_id", "source"],
    registry=registry,
)

# --- Session metrics ---

active_sessions = Gauge(
    "llmmllab_api_active_sessions",
    "Number of sessions currently in progress",
    ["model_id", "source"],
    registry=registry,
)

session_duration_seconds = Histogram(
    "llmmllab_api_session_duration_seconds",
    "Duration of completed sessions in seconds",
    ["model_id", "source"],
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 200, 300, 600),
    registry=registry,
)

session_turns_total = Histogram(
    "llmmllab_api_session_turns_total",
    "Number of turns per completed session",
    ["model_id", "source"],
    buckets=(1, 2, 3, 5, 10, 20, 50, 100, 200, 500),
    registry=registry,
)
