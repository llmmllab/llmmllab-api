"""Business-level Prometheus metrics for the API service."""

from prometheus_client import Counter, Histogram
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
