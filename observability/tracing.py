from __future__ import annotations

import logging
import os


def configure_tracing(service_name: str = 'medigenius') -> None:
    """
    Optional tracing setup. Prefers OTLP exporter when available,
    falls back to console exporter, then no-op if OTel is missing entirely.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({'service.name': service_name})
        provider = TracerProvider(resource=resource)

        exporter = None
        otlp_endpoint = os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                logging.getLogger(__name__).info(
                    'OTLP tracing configured',
                    extra={'node': 'startup', 'endpoint': otlp_endpoint},
                )
            except ImportError:
                pass

        if exporter is None:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            exporter = ConsoleSpanExporter()
            logging.getLogger(__name__).info(
                'Console tracing configured (set OTEL_EXPORTER_OTLP_ENDPOINT for production)',
                extra={'node': 'startup'},
            )

        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    except Exception as exc:  # pragma: no cover - optional dependency path
        logging.getLogger(__name__).warning(
            'Tracing disabled; OpenTelemetry unavailable',
            extra={'node': 'startup', 'error_code': 'otel_unavailable'},
            exc_info=exc,
        )
