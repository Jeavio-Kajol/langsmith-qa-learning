"""Report agent: compares two evaluation runs and flags regressions.

Graph: compare -(no regressions or no LLM)-> render -> END
              \\-> summarize -> verify -(invalid, retry once)-> summarize
                                      \\-> render -> END

Every number and every regression flag comes from code. The LLM only writes a short summary, and it
is rejected if it cites metrics or examples that are not in the comparison.
"""
import json
import time
from dataclasses import dataclass, field
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.judge_agent import build_judge_llm, extract_json, invoke_llm, is_provider_quota_error
from ingest.telemetry import get_tracer

LOWER_IS_BETTER = ("hallucination_rate",)
METRIC_TOLERANCE = 0.05  # a metric mean worse by more than this is a regression
EXAMPLE_DROP = 0.3  # an example whose score got worse by at least this is listed
MAX_EXAMPLES_IN_REPORT = 10
MAX_ATTEMPTS = 2

# Same limits as the quality gate in rag/rag_eval_runner.py (a test keeps them in sync).
GATE_THRESHOLDS = {
    "context_recall": 0.85,
    "context_precision": 0.80,
    "faithfulness": 0.90,
    "hallucination_rate": 0.05,
    "no_answer_rate": 0.95,
    "answer_relevance": 0.80,
}


@dataclass
class RunResults:
    """Per-example scores of one evaluation run: {example_key: {metric: score}}."""

    name: str
    examples: dict
    source: str = "file"


@dataclass
class MetricChange:
    metric: str
    baseline: float
    candidate: float
    delta: float  # positive = better, negative = worse (direction already applied)
    status: str  # regression | improvement | unchanged
    gate_change: str = ""  # "pass->fail" or "fail->pass" against GATE_THRESHOLDS


@dataclass
class ExampleChange:
    example: str
    metric: str
    baseline: float
    candidate: float


@dataclass
class Report:
    baseline: str
    candidate: str
    metrics: list = field(default_factory=list)
    worst_examples: list = field(default_factory=list)
    compared: int = 0
    only_in_baseline: list = field(default_factory=list)
    only_in_candidate: list = field(default_factory=list)
    summary: str = ""
    next_step: str = ""
    evidence: list = field(default_factory=list)
    llm_used: bool = False
    attempts: int = 0
    error: Optional[str] = None
    fatal: bool = False

    @property
    def regressions(self) -> list:
        return [m for m in self.metrics if m.status == "regression" or m.gate_change == "pass->fail"]

    @property
    def coverage_problem(self) -> bool:
        return bool(self.only_in_baseline) or self.compared == 0

    @property
    def ok(self) -> bool:
        return self.error is None and not self.regressions and not self.coverage_problem


class ReportState(TypedDict, total=False):
    baseline: RunResults
    candidate: RunResults
    report: Report
    prompt: str
    raw: str
    parse_error: str
    done: bool
    route: str


def passes_gate(metric: str, value: float) -> Optional[bool]:
    limit = GATE_THRESHOLDS.get(metric)
    if limit is None:
        return None
    return value < limit if metric in LOWER_IS_BETTER else value > limit  # strict, like the gate


def _mean(values: list) -> float:
    return sum(values) / len(values)


def _better_delta(metric: str, baseline: float, candidate: float) -> float:
    delta = candidate - baseline
    return -delta if metric in LOWER_IS_BETTER else delta


def compare(baseline: RunResults, candidate: RunResults) -> Report:
    report = Report(baseline=baseline.name, candidate=candidate.name)
    common = sorted(set(baseline.examples) & set(candidate.examples))
    report.compared = len(common)
    report.only_in_baseline = sorted(set(baseline.examples) - set(candidate.examples))
    report.only_in_candidate = sorted(set(candidate.examples) - set(baseline.examples))
    metrics = sorted({m for k in common for m in baseline.examples[k]} & {m for k in common for m in candidate.examples[k]})
    worst = []
    for metric in metrics:
        pairs = [(k, baseline.examples[k][metric], candidate.examples[k][metric])
                 for k in common if metric in baseline.examples[k] and metric in candidate.examples[k]]
        if not pairs:
            continue
        base_mean, cand_mean = _mean([p[1] for p in pairs]), _mean([p[2] for p in pairs])
        delta = _better_delta(metric, base_mean, cand_mean)
        status = "regression" if delta < -METRIC_TOLERANCE else "improvement" if delta > METRIC_TOLERANCE else "unchanged"
        before, after = passes_gate(metric, base_mean), passes_gate(metric, cand_mean)
        gate_change = ""
        if before is True and after is False:
            gate_change = "pass->fail"
        elif before is False and after is True:
            gate_change = "fail->pass"
        report.metrics.append(MetricChange(metric, base_mean, cand_mean, delta, status, gate_change))
        for key, b, c in pairs:
            if _better_delta(metric, b, c) <= -EXAMPLE_DROP:
                worst.append(ExampleChange(key, metric, b, c))
    worst.sort(key=lambda e: _better_delta(e.metric, e.baseline, e.candidate))
    report.worst_examples = worst[:MAX_EXAMPLES_IN_REPORT]
    return report


def build_prompt(report: Report) -> str:
    metric_lines = "\n".join(
        f"- {m.metric}: {m.baseline:.3f} -> {m.candidate:.3f} ({m.status}"
        f"{', gate ' + m.gate_change if m.gate_change else ''})" for m in report.metrics)
    example_lines = "\n".join(
        f"- example {e.example}: {e.metric} {e.baseline:.2f} -> {e.candidate:.2f}" for e in report.worst_examples) or "- none"
    return f"""You are summarizing an evaluation comparison for a RAG application.
Baseline run: {report.baseline}. Candidate run: {report.candidate}. Examples compared: {report.compared}.
Use only the numbers below; do not invent causes or numbers.

METRICS (baseline -> candidate; hallucination_rate lower is better):
{metric_lines}

WORST EXAMPLES:
{example_lines}

Return only JSON:
{{"summary": "2-3 sentences on what regressed and by how much",
  "evidence": ["metric names or bare example ids (e.g. faithfulness, q0), copied exactly"],
  "next_step": "1 concrete sentence"}}"""


def validate_summary(parsed: dict, report: Report) -> dict:
    for key in ("summary", "next_step"):
        if not isinstance(parsed.get(key), str) or not parsed[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    evidence = parsed.get("evidence")
    if not isinstance(evidence, list) or not evidence or not all(isinstance(e, str) for e in evidence):
        raise ValueError("evidence must be a non-empty list of strings")
    known = {m.metric for m in report.metrics} | {e.example for e in report.worst_examples}
    cleaned = [e.strip()[len("example "):] if e.strip().lower().startswith("example ") else e.strip() for e in evidence]
    unknown = [e for e in cleaned if e not in known]
    if unknown:
        raise ValueError(f"evidence cites things not in the comparison: {unknown}")
    return parsed


def render_markdown(report: Report) -> str:
    lines = [f"# Evaluation comparison: {report.baseline} vs {report.candidate}", "",
             f"Examples compared: {report.compared}", ""]
    verdict = "REGRESSIONS FOUND" if report.regressions else "no regressions"
    lines += [f"**Result: {verdict}**", ""]
    if report.summary:
        lines += [report.summary, "", f"Next step: {report.next_step}", ""]
    if report.error:
        lines += [f"> Summary unavailable: {report.error}", ""]
    lines += ["| Metric | Baseline | Candidate | Change | Status |", "|---|---|---|---|---|"]
    for m in report.metrics:
        status = m.status + (f" (gate {m.gate_change})" if m.gate_change else "")
        lines.append(f"| {m.metric} | {m.baseline:.3f} | {m.candidate:.3f} | {m.delta:+.3f} | {status} |")
    if report.worst_examples:
        lines += ["", "## Worst examples", ""]
        lines += [f"- `{e.example}` {e.metric}: {e.baseline:.2f} -> {e.candidate:.2f}" for e in report.worst_examples]
    if report.only_in_baseline:
        lines += ["", f"Missing from candidate ({len(report.only_in_baseline)}): " + ", ".join(report.only_in_baseline[:10])]
    if report.only_in_candidate:
        lines += ["", f"New in candidate ({len(report.only_in_candidate)}): " + ", ".join(report.only_in_candidate[:10])]
    return "\n".join(lines) + "\n"


class ReportAgent:
    def __init__(self, llm=None, use_llm=True, sleep=time.sleep, max_attempts=5):
        self._llm = llm
        self.use_llm = use_llm
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
        graph = StateGraph(ReportState)
        graph.add_node("compare", self._compare)
        graph.add_node("summarize", self._summarize)
        graph.add_node("verify", self._verify)
        graph.set_entry_point("compare")
        graph.add_conditional_edges("compare", lambda s: s["route"], {"summarize": "summarize", "end": END})
        graph.add_edge("summarize", "verify")
        graph.add_conditional_edges(
            "verify",
            lambda s: END if s.get("done") or not s.get("parse_error") else "summarize",
            {END: END, "summarize": "summarize"},
        )
        return graph.compile()

    def _compare(self, state: ReportState) -> ReportState:
        report = compare(state["baseline"], state["candidate"])
        if not report.metrics:
            report.error = "no metrics in common between the two runs"
            return {"report": report, "route": "end"}
        if not (self.use_llm and report.regressions):
            return {"report": report, "route": "end"}
        if self.quota_exhausted:
            report.error = f"summary skipped: provider quota exhausted ({self.quota_exhausted})"
            report.fatal = True
            return {"report": report, "route": "end"}
        return {"report": report, "prompt": build_prompt(report), "route": "summarize"}

    def _summarize(self, state: ReportState) -> ReportState:
        report = state["report"]
        report.attempts += 1
        report.llm_used = True
        try:
            response = invoke_llm(self.llm, state["prompt"], self._sleep, self._max_attempts)
        except Exception as error:
            report.error = f"summary call failed: {error}"
            if is_provider_quota_error(error):
                report.fatal = True
                self.quota_exhausted = str(error)[:200]
            return {"report": report, "done": True, "parse_error": ""}
        return {"report": report, "raw": str(response.content)}

    def _verify(self, state: ReportState) -> ReportState:
        report = state["report"]
        if state.get("done"):
            return {}
        try:
            parsed = validate_summary(extract_json(state["raw"]), report)
        except (ValueError, TypeError) as error:
            if report.attempts < MAX_ATTEMPTS:
                return {"parse_error": str(error)}
            report.error = f"invalid summary after {report.attempts} attempts: {error}"
            return {"report": report, "done": True, "parse_error": ""}
        report.summary = parsed["summary"].strip()
        report.next_step = parsed["next_step"].strip()
        report.evidence = parsed["evidence"]
        return {"report": report, "parse_error": ""}

    def run(self, baseline: RunResults, candidate: RunResults) -> Report:
        with get_tracer().start_as_current_span("report_agent.run") as span:
            report = self.graph.invoke({"baseline": baseline, "candidate": candidate})["report"]
            span.set_attribute("report_agent.regression_count", len(report.regressions))
            span.set_attribute("report_agent.compared", report.compared)
            span.set_attribute("report_agent.ok", report.ok)
            return report


def results_from_file(path: str) -> RunResults:
    """JSON: {"name": "...", "examples": {"<example id>": {"<metric>": <score>, ...}, ...}}"""
    with open(path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    examples = data.get("examples")
    if not isinstance(examples, dict) or not examples:
        raise ValueError(f"{path}: 'examples' must be a non-empty object")
    clean = {}
    for key, scores in examples.items():
        if not isinstance(scores, dict):
            raise ValueError(f"{path}: scores for example {key} must be an object")
        for metric, value in scores.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{path}: score {metric} of example {key} must be a number")
        clean[str(key)] = {str(m): float(v) for m, v in scores.items()}
    return RunResults(name=str(data.get("name") or path), examples=clean, source="file")


def results_from_langsmith(client, project: str) -> RunResults:
    """Reads an experiment: one root run per dataset example, scores from its feedback."""
    runs = {str(r.id): r for r in client.list_runs(project_name=project, is_root=True)}
    if not runs:
        raise ValueError(f"LangSmith project {project!r} has no runs")
    examples: dict = {}
    for feedback in client.list_feedback(run_ids=list(runs)):
        if feedback.score is None:
            continue
        run = runs.get(str(feedback.run_id))
        if run is None or run.reference_example_id is None:
            continue
        examples.setdefault(str(run.reference_example_id), {})[feedback.key] = float(feedback.score)
    if not examples:
        raise ValueError(f"LangSmith project {project!r} has no scored runs linked to dataset examples")
    return RunResults(name=project, examples=examples, source="langsmith")
