"""OpenTelemetry tracing setup for distributed trace collection."""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # pylint: disable=import-no-side-effects,unused-import
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor  # pylint: disable=import-no-side-effects,unused-import
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from config import TEMPO_ENDPOINT


def setup_tracing(service_name: str, app) -> None:
    """Configure OpenTelemetry tracing for the given FastAPI app."""
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    exporter = OTLPSpanExporter(endpoint=TEMPO_ENDPOINT, insecure=True)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()


def shutdown_tracing() -> None:
    """Disable instrumentation on shutdown."""
    try:
        FastAPIInstrumentor.uninstrument()
    except Exception:  # pragma: no cover
        pass
    try:
        HTTPXClientInstrumentor().uninstrument()
    except Exception:  # pragma: no cover
        pass
