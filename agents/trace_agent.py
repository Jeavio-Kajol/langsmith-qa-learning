"""Trace agent: explains why a RAG/LLM run went wrong.

Graph: diagnose -(no findings)-> END (healthy, no LLM call)
                 \\-> explain -> verify -(invalid, retry once)-> explain
                                       \\-> publish -> END

Deterministic checks find the problems; the LLM only explains them, and its explanation is
rejected if it cites spans or checks that are not in the trace.
"""
import time
from dataclasses import dataclass, field
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.judge_agent import build_judge_llm, extract_json, invoke_llm, is_provider_quota_error
from agents.trace_model import Trace
from ingest.telemetry import get_tracer

CATEGORIES = ("retrieval", "generation", "provider", "performance", "other")
CONFIDENCE = ("low", "medium", "high")
MAX_ATTEMPTS = 2
MAX_SPANS_IN_PROMPT = 60
SLOW_SHARE = 0.6  # a leaf span using >= 60% of total time ...
MIN_SLOW_MS = 3000  # ... and at least 3 s is flagged as slow
LOW_SCORE = 0.5  # judge scores worse than this are flagged (hallucination_rate: above this)


@dataclass
class Finding:
    code: str
    span: str
    detail: str


@dataclass
class Diagnosis:
    trace_id: str
    findings: list = field(default_factory=list)
    category: str = "healthy"
    root_cause: str = ""
    evidence: list = field(default_factory=list)
    suggested_fix: str = ""
    confidence: str = ""
    llm_used: bool = False
    attempts: int = 0
    error: Optional[str] = None
    publish_errors: list = field(default_factory=list)
    publish_notes: list = field(default_factory=list)
    fatal: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.publish_errors


class TraceState(TypedDict, total=False):
    trace: Trace
    diagnosis: Diagnosis
    prompt: str
    raw: str
    parse_error: str
    done: bool
    route: str


def diagnose(trace: Trace) -> list:
    findings = []
    root = trace.root
    has_children = {s.parent_id for s in trace.spans}
    for span in trace.spans:
        if span.error:
            findings.append(Finding("span_error", span.name, span.error[:300]))
        is_retriever = span.kind.lower() == "retriever" or "retriev" in span.name.lower()
        is_generation = span.kind.lower() == "llm" or "generat" in span.name.lower()
        if span.output_empty and not span.error:
            if is_retriever:
                findings.append(Finding("empty_retrieval", span.name, "retrieval returned no documents"))
            elif is_generation or span is root:
                findings.append(Finding("empty_output", span.name, "span produced an empty output"))
    root_ms = root.latency_ms if root else None
    if root_ms:
        for span in trace.spans:
            if span.span_id in has_children or span is root or span.latency_ms is None:
                continue
            if span.latency_ms >= MIN_SLOW_MS and span.latency_ms >= SLOW_SHARE * root_ms:
                share = span.latency_ms / root_ms
                findings.append(Finding("slow_span", span.name, f"{span.latency_ms:.0f} ms ({share:.0%} of total)"))
    for key, score in trace.scores.items():
        bad = score > LOW_SCORE if key.endswith("hallucination_rate") else score < LOW_SCORE
        if bad:
            findings.append(Finding("low_score", key, f"{key}={score:.2f}"))
    return findings


def build_prompt(trace: Trace, findings: list) -> str:
    lines = []
    for span in trace.spans[:MAX_SPANS_IN_PROMPT]:
        latency = f"{span.latency_ms:.0f}ms" if span.latency_ms is not None else "?ms"
        status = f"ERROR: {span.error[:200]}" if span.error else "ok"
        lines.append(f"{'  ' * span.depth}- {span.name} [{span.kind}] {status} {latency}")
        lines.append(f"{'  ' * span.depth}    in: {span.input}")
        lines.append(f"{'  ' * span.depth}    out: {span.output}")
    if len(trace.spans) > MAX_SPANS_IN_PROMPT:
        lines.append(f"(... {len(trace.spans) - MAX_SPANS_IN_PROMPT} more spans omitted)")
    finding_lines = "\n".join(f"- {f.code} @ {f.span}: {f.detail}" for f in findings)
    return f"""You are a debugging assistant for a Retrieval-Augmented Generation (RAG) application.
Automated checks flagged problems in the trace below. Explain the most likely root cause.
Use only the evidence shown; do not invent spans or facts.

TRACE (indented tree):
{chr(10).join(lines)}

AUTOMATED FINDINGS:
{finding_lines}

Return only JSON:
{{"category": one of {list(CATEGORIES)},
  "root_cause": "1-2 sentences",
  "evidence": ["span names or finding codes (e.g. rag-retrieval, empty_retrieval), copied exactly"],
  "suggested_fix": "1-2 concrete sentences",
  "confidence": one of {list(CONFIDENCE)}}}"""


def validate_explanation(parsed: dict, trace: Trace, findings: list) -> dict:
    if parsed.get("category") not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}")
    if parsed.get("confidence") not in CONFIDENCE:
        raise ValueError(f"confidence must be one of {CONFIDENCE}")
    for key in ("root_cause", "suggested_fix"):
        if not isinstance(parsed.get(key), str) or not parsed[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    evidence = parsed.get("evidence")
    if not isinstance(evidence, list) or not evidence or not all(isinstance(e, str) for e in evidence):
        raise ValueError("evidence must be a non-empty list of strings")
    known = {s.name.lower() for s in trace.spans} | {f.code for f in findings} | {f.span.lower() for f in findings}
    known |= {f"{f.code} @ {f.span}".lower() for f in findings}
    unknown = [e for e in evidence if e.strip().lower() not in known]
    if unknown:
        raise ValueError(f"evidence cites things not in the trace: {unknown}")
    return parsed


class TraceAgent:
    def __init__(self, llm=None, publishers=(), sleep=time.sleep, max_attempts=5):
        self._llm = llm
        self.publishers = list(publishers)
        self._sleep = sleep
        self._max_attempts = max_attempts
        self.quota_exhausted: Optional[str] = None
        self.graph = self._build_graph()

    @property
    def llm(self):
        if self._llm is None:
            self._llm = build_judge_llm()
        return self._llm

    def _build_graph(self):
        graph = StateGraph(TraceState)
        graph.add_node("diagnose", self._diagnose)
        graph.add_node("explain", self._explain)
        graph.add_node("verify", self._verify)
        graph.add_node("publish", self._publish)
        graph.set_entry_point("diagnose")
        graph.add_conditional_edges("diagnose", lambda s: s["route"], {"explain": "explain", "publish": "publish", "end": END})
        graph.add_edge("explain", "verify")
        graph.add_conditional_edges(
            "verify",
            lambda s: END if s.get("done") else ("explain" if s.get("parse_error") else "publish"),
            {END: END, "explain": "explain", "publish": "publish"},
        )
        graph.add_edge("publish", END)
        return graph.compile()

    def _diagnose(self, state: TraceState) -> TraceState:
        trace = state["trace"]
        diagnosis = Diagnosis(trace_id=trace.trace_id, findings=diagnose(trace))
        if not trace.spans:
            diagnosis.error = "trace has no spans"
            return {"diagnosis": diagnosis, "route": "end"}
        if not diagnosis.findings:
            diagnosis.root_cause = "No problems detected by the automated checks."
            return {"diagnosis": diagnosis, "route": "publish"}  # healthy: no LLM call
        if self.quota_exhausted:
            diagnosis.error = f"skipped: provider quota exhausted ({self.quota_exhausted})"
            diagnosis.fatal = True
            return {"diagnosis": diagnosis, "route": "end"}
        return {"diagnosis": diagnosis, "prompt": build_prompt(trace, diagnosis.findings), "route": "explain"}

    def _explain(self, state: TraceState) -> TraceState:
        diagnosis = state["diagnosis"]
        diagnosis.attempts += 1
        diagnosis.llm_used = True
        try:
            response = invoke_llm(self.llm, state["prompt"], self._sleep, self._max_attempts)
        except Exception as error:
            diagnosis.error = f"explanation call failed: {error}"
            if is_provider_quota_error(error):
                diagnosis.fatal = True
                self.quota_exhausted = str(error)[:200]
            return {"diagnosis": diagnosis, "done": True, "parse_error": ""}
        return {"diagnosis": diagnosis, "raw": str(response.content)}

    def _verify(self, state: TraceState) -> TraceState:
        diagnosis = state["diagnosis"]
        if state.get("done"):
            return {}
        try:
            parsed = validate_explanation(extract_json(state["raw"]), state["trace"], diagnosis.findings)
        except (ValueError, TypeError) as error:
            if diagnosis.attempts < MAX_ATTEMPTS:
                return {"parse_error": str(error)}
            diagnosis.error = f"invalid explanation after {diagnosis.attempts} attempts: {error}"
            return {"diagnosis": diagnosis, "done": True, "parse_error": ""}
        diagnosis.category = parsed["category"]
        diagnosis.root_cause = parsed["root_cause"].strip()
        diagnosis.evidence = parsed["evidence"]
        diagnosis.suggested_fix = parsed["suggested_fix"].strip()
        diagnosis.confidence = parsed["confidence"]
        return {"diagnosis": diagnosis, "parse_error": ""}

    def _publish(self, state: TraceState) -> TraceState:
        diagnosis, trace = state["diagnosis"], state["trace"]
        for publisher in self.publishers:
            try:
                note = publisher.publish(trace, diagnosis)
                if note:
                    diagnosis.publish_notes.append(note)
            except Exception as error:
                diagnosis.publish_errors.append(f"{type(publisher).__name__}: {error}")
        return {"diagnosis": diagnosis}

    def analyze(self, trace: Trace) -> Diagnosis:
        with get_tracer().start_as_current_span("trace_agent.analyze") as span:
            diagnosis = self.graph.invoke({"trace": trace})["diagnosis"]
            span.set_attribute("trace_agent.category", diagnosis.category)
            span.set_attribute("trace_agent.finding_count", len(diagnosis.findings))
            span.set_attribute("trace_agent.ok", diagnosis.ok)
            return diagnosis
