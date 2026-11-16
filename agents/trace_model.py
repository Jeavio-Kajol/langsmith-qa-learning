"""Backend-neutral trace model plus loaders for LangSmith, Phoenix and JSON files."""
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

PREVIEW_CHARS = 300


@dataclass
class TraceSpan:
    span_id: str
    name: str
    parent_id: Optional[str] = None
    kind: str = ""
    error: Optional[str] = None  # set when the span failed
    latency_ms: Optional[float] = None
    input: str = ""
    output: str = ""
    output_empty: bool = False
    depth: int = 0


@dataclass
class Trace:
    source: str  # langsmith | phoenix | file
    trace_id: str
    root_id: Optional[str]
    spans: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)  # existing judge/eval scores, e.g. agent_faithfulness
    session_id: Optional[str] = None

    @property
    def root(self) -> Optional[TraceSpan]:
        return next((s for s in self.spans if s.span_id == self.root_id), None)


def is_empty(value) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, (list, tuple, dict)):
        return all(is_empty(v) for v in (value.values() if isinstance(value, dict) else value))
    return False


def preview(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…[truncated]"


def _latency_ms(start, end) -> Optional[float]:
    if start is None or end is None:
        return None
    return (end - start).total_seconds() * 1000


def _assign_depths(spans: list) -> list:
    by_id = {s.span_id: s for s in spans}
    for span in spans:
        depth, parent, seen = 0, span.parent_id, set()
        while parent in by_id and parent not in seen:
            seen.add(parent)
            depth += 1
            parent = by_id[parent].parent_id
        span.depth = depth
    return spans


def trace_from_langsmith(client, run_id: str) -> Trace:
    root = client.read_run(run_id, load_child_runs=True)
    spans = []

    def walk(run, parent_id):
        spans.append(
            TraceSpan(
                span_id=str(run.id),
                name=run.name,
                parent_id=parent_id,
                kind=str(run.run_type or ""),
                error=str(run.error) if run.error else None,
                latency_ms=_latency_ms(run.start_time, run.end_time),
                input=preview(run.inputs or {}),
                output=preview(run.outputs or {}),
                output_empty=is_empty(run.outputs),
            )
        )
        for child in sorted(run.child_runs or [], key=lambda r: r.start_time or datetime.min):
            walk(child, str(run.id))

    walk(root, None)
    scores = {}
    for feedback in client.list_feedback(run_ids=[root.id]):
        if feedback.score is not None:
            scores[feedback.key] = feedback.score
    return Trace(
        "langsmith", str(root.trace_id or root.id), str(root.id), _assign_depths(spans), scores,
        str(root.session_id) if getattr(root, "session_id", None) else None,
    )


def _attr(attributes: dict, key: str):
    if key in attributes:
        return attributes[key]
    node = attributes
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def trace_from_phoenix(client, project: str, trace_id: str) -> Trace:
    raw = client.spans.get_spans(project_identifier=project, trace_ids=[trace_id], limit=1000)
    spans = []
    for item in sorted(raw, key=lambda s: s.get("start_time") or ""):
        attributes = item.get("attributes") or {}
        start, end = item.get("start_time"), item.get("end_time")
        start = datetime.fromisoformat(start) if isinstance(start, str) else start
        end = datetime.fromisoformat(end) if isinstance(end, str) else end
        output = _attr(attributes, "output.value")
        has_documents = any(str(k).startswith("retrieval.documents") for k in attributes)
        failed = str(item.get("status_code", "")).upper() == "ERROR"
        spans.append(
            TraceSpan(
                span_id=item["context"]["span_id"],
                name=item["name"],
                parent_id=item.get("parent_id"),
                kind=str(item.get("span_kind", "")),
                error=(item.get("status_message") or "span reported ERROR status") if failed else None,
                latency_ms=_latency_ms(start, end),
                input=preview(_attr(attributes, "input.value") or ""),
                output=preview(output or ""),
                output_empty=is_empty(output) and not has_documents,
            )
        )
    root = next((s for s in spans if not s.parent_id), None)
    scores = {}
    if root:
        for annotation in client.spans.get_span_annotations(span_ids=[root.span_id], project_identifier=project):
            score = (annotation.get("result") or {}).get("score")
            if score is not None:
                scores[annotation["name"]] = score
    return Trace("phoenix", trace_id, root.span_id if root else None, _assign_depths(spans), scores)


def trace_from_file(path: str) -> Trace:
    """JSON: {"trace_id", "scores": {}, "spans": [{span_id, name, parent_id, kind, error,
    latency_ms, input, output, output_empty}]}"""
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    spans = [
        TraceSpan(
            span_id=str(s["span_id"]), name=s["name"], parent_id=s.get("parent_id"), kind=s.get("kind", ""),
            error=s.get("error"), latency_ms=s.get("latency_ms"), input=preview(s.get("input", "")),
            output=preview(s.get("output", "")), output_empty=bool(s.get("output_empty", is_empty(s.get("output")))),
        )
        for s in data["spans"]
    ]
    root = next((s for s in spans if not s.parent_id), None)
    return Trace("file", str(data.get("trace_id", "file")), root.span_id if root else None,
                 _assign_depths(spans), data.get("scores", {}))
