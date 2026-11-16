"""OpenTelemetry setup. Spans go to Phoenix (Arize) or any OTLP collector when configured."""
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

_configured = False


def setup_tracing(service_name: str = "document-ingest") -> None:
    """Idempotent. Exporter choice (env):
    PHOENIX_COLLECTOR_ENDPOINT -> Phoenix; OTEL_EXPORTER_OTLP_ENDPOINT -> OTLP;
    INGEST_OTEL_CONSOLE=1 -> console. Otherwise spans are created but not exported."""
    global _configured
    if _configured:
        return
    _configured = True

    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if phoenix_endpoint:
        from phoenix.otel import register

        register(
            project_name=os.getenv("PHOENIX_PROJECT_NAME", service_name),
            endpoint=_phoenix_traces_url(phoenix_endpoint),
            batch=True,
        )
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    if os.getenv("INGEST_OTEL_CONSOLE") == "1":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


def _phoenix_traces_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/v1/traces") else endpoint + "/v1/traces"


def get_tracer():
    return trace.get_tracer("ingest")
