"""Publishers: write judge scores back to LangSmith (run feedback) and Phoenix (span annotations)."""
from typing import Optional

KEY_PREFIX = "agent_"  # keeps agent scores separate from the quality gate's own feedback keys


class TraceLangSmithPublisher:
    """Writes the diagnosis as feedback on the root run of a LangSmith trace."""

    def __init__(self, client=None):
        if client is None:
            from langsmith import Client

            client = Client()
        self.client = client

    def publish(self, trace, diagnosis) -> Optional[str]:
        if trace.source != "langsmith":
            return "LangSmith: skipped, trace did not come from LangSmith"
        self.client.create_feedback(
            run_id=trace.root_id,
            key=f"{KEY_PREFIX}trace_diagnosis",
            value=diagnosis.category,
            comment=_diagnosis_text(diagnosis),
            source_info={"agent": "trace"},
            session_id=trace.session_id,
        )
        return None


class TracePhoenixPublisher:
    """Writes the diagnosis as a note on the root span of a Phoenix trace."""

    def __init__(self, client=None):
        if client is None:
            from phoenix.client import Client

            client = Client()
        self.client = client

    def publish(self, trace, diagnosis) -> Optional[str]:
        if trace.source != "phoenix":
            return "Phoenix: skipped, trace did not come from Phoenix"
        self.client.spans.add_span_note(span_id=trace.root_id, note=_diagnosis_text(diagnosis))
        return None


def _diagnosis_text(diagnosis) -> str:
    if diagnosis.category == "healthy":
        return "healthy: no problems detected by automated checks"
    return (f"[{diagnosis.category}, {diagnosis.confidence} confidence] {diagnosis.root_cause} "
            f"Fix: {diagnosis.suggested_fix}")[:1000]


class LangSmithPublisher:
    def __init__(self, client=None):
        if client is None:
            from langsmith import Client

            client = Client()
        self.client = client

    def publish(self, item, verdict) -> Optional[str]:
        if not item.run_id:
            return "LangSmith: skipped, item has no run_id"
        for metric, score in verdict.scores.items():
            self.client.create_feedback(
                run_id=item.run_id,
                key=f"{KEY_PREFIX}{metric}",
                score=score,
                comment=verdict.reason or None,
                source_info={"agent": "judge"},
                session_id=item.session_id,
            )
        return None


class PhoenixPublisher:
    def __init__(self, client=None):
        if client is None:
            from phoenix.client import Client

            client = Client()
        self.client = client

    def publish(self, item, verdict) -> Optional[str]:
        if not item.span_id:
            return "Phoenix: skipped, item has no span_id"
        for metric, score in verdict.scores.items():
            self.client.spans.add_span_annotation(
                span_id=item.span_id,
                annotation_name=f"{KEY_PREFIX}{metric}",
                annotator_kind="LLM",
                score=score,
                explanation=verdict.reason or None,
                sync=True,
            )
        return None
