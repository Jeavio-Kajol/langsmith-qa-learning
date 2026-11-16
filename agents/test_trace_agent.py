"""Offline tests for the trace agent (fake LLM, fake clients, no network).

    python -m unittest agents.test_trace_agent -v
"""
import json
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

from agents.publishers import TraceLangSmithPublisher, TracePhoenixPublisher
from agents.trace_agent import TraceAgent, diagnose
from agents.trace_model import Trace, TraceSpan, trace_from_langsmith, trace_from_phoenix


class FakeLLM:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), 0

    def invoke(self, prompt):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=response if isinstance(response, str) else json.dumps(response))


def span(name, parent="root", **kw):
    return TraceSpan(span_id=name, name=name, parent_id=parent, **kw)


def rag_trace(**scores):
    return Trace(
        "file", "t1", "root",
        [
            TraceSpan("root", "rag-question-answering", latency_ms=1000, output="{...}"),
            span("rag-retrieval", kind="retriever", output_empty=True, latency_ms=200),
            span("rag-generation", kind="llm", output="I don't know", latency_ms=700),
        ],
        scores,
    )


EXPLANATION = {
    "category": "retrieval",
    "root_cause": "Retrieval returned no documents so the answer had no grounding.",
    "evidence": ["rag-retrieval", "empty_retrieval"],
    "suggested_fix": "Check the index and the query embedding.",
    "confidence": "high",
}


class DiagnoseTests(unittest.TestCase):
    def test_flags_empty_retrieval_error_slow_and_low_scores(self):
        trace = Trace("file", "t", "root", [
            TraceSpan("root", "app", latency_ms=10000),
            span("fetch", kind="retriever", output_empty=True),
            span("llm-call", kind="llm", error="429 rate limit", latency_ms=8000),
        ], {"agent_faithfulness": 0.2, "agent_hallucination_rate": 0.8, "agent_answer_relevance": 0.9})
        codes = sorted((f.code, f.span) for f in diagnose(trace))
        self.assertEqual(codes, [
            ("empty_retrieval", "fetch"), ("low_score", "agent_faithfulness"),
            ("low_score", "agent_hallucination_rate"), ("slow_span", "llm-call"), ("span_error", "llm-call"),
        ])

    def test_healthy_trace_has_no_findings(self):
        trace = Trace("file", "t", "root", [TraceSpan("root", "app", latency_ms=500, output="x"),
                                            span("gen", kind="llm", output="ok", latency_ms=400)],
                      {"agent_faithfulness": 0.9, "agent_hallucination_rate": 0.1})
        self.assertEqual(diagnose(trace), [])


class AgentTests(unittest.TestCase):
    def test_healthy_trace_makes_no_llm_call(self):
        llm = FakeLLM()
        trace = Trace("file", "t", "root", [TraceSpan("root", "app", output="x")])
        diagnosis = TraceAgent(llm).analyze(trace)
        self.assertTrue(diagnosis.ok)
        self.assertEqual((diagnosis.category, llm.calls), ("healthy", 0))

    def test_explains_problem_and_publishes(self):
        published = []
        publisher = SimpleNamespace(publish=lambda t, d: published.append(d.category))
        diagnosis = TraceAgent(FakeLLM(EXPLANATION), [publisher]).analyze(rag_trace())
        self.assertTrue(diagnosis.ok)
        self.assertEqual((diagnosis.category, published), ("retrieval", ["retrieval"]))

    def test_invented_evidence_is_rejected_then_retry_succeeds(self):
        bad = {**EXPLANATION, "evidence": ["vector-database-crash"]}
        llm = FakeLLM(bad, EXPLANATION)
        diagnosis = TraceAgent(llm).analyze(rag_trace())
        self.assertEqual((diagnosis.ok, diagnosis.attempts, llm.calls), (True, 2, 2))

    def test_evidence_in_finding_format_is_accepted(self):
        explanation = {**EXPLANATION, "evidence": ["empty_retrieval @ rag-retrieval"]}
        self.assertTrue(TraceAgent(FakeLLM(explanation)).analyze(rag_trace()).ok)

    def test_persistently_invalid_output_fails_without_publishing(self):
        published = []
        publisher = SimpleNamespace(publish=lambda t, d: published.append(1))
        diagnosis = TraceAgent(FakeLLM("nope", {**EXPLANATION, "category": "healthy"}), [publisher]).analyze(rag_trace())
        self.assertFalse(diagnosis.ok)
        self.assertIn("invalid explanation", diagnosis.error)
        self.assertEqual(published, [])

    def test_quota_exhaustion_stops_further_calls(self):
        llm = FakeLLM(Exception("tokens per day limit reached"))
        agent = TraceAgent(llm)
        first, second = agent.analyze(rag_trace()), agent.analyze(rag_trace())
        self.assertTrue(first.fatal and second.fatal)
        self.assertEqual(llm.calls, 1)

    def test_empty_trace_is_an_error(self):
        self.assertIn("no spans", TraceAgent(FakeLLM()).analyze(Trace("file", "t", None, [])).error)


class LoaderAndPublisherTests(unittest.TestCase):
    def test_langsmith_tree_is_flattened_with_errors_scores_and_latency(self):
        t0 = datetime(2026, 1, 1)
        child = SimpleNamespace(id="c1", name="rag-retrieval", run_type="retriever", error=None, start_time=t0,
                                end_time=t0 + timedelta(seconds=1), inputs={"q": "x"}, outputs={"output": []},
                                child_runs=[])
        root = SimpleNamespace(id="r1", name="app", run_type="chain", error="boom", start_time=t0,
                               end_time=t0 + timedelta(seconds=2), inputs={}, outputs={"a": "b"},
                               child_runs=[child], trace_id="tr1", session_id="s1")
        client = SimpleNamespace(read_run=lambda rid, load_child_runs: root,
                                 list_feedback=lambda run_ids: [SimpleNamespace(key="agent_faithfulness", score=0.3)])
        trace = trace_from_langsmith(client, "r1")
        self.assertEqual([(s.name, s.depth) for s in trace.spans], [("app", 0), ("rag-retrieval", 1)])
        self.assertTrue(trace.spans[1].output_empty)
        self.assertEqual((trace.spans[0].error, trace.spans[0].latency_ms), ("boom", 2000.0))
        self.assertEqual(trace.scores, {"agent_faithfulness": 0.3})
        self.assertEqual({f.code for f in diagnose(trace)}, {"span_error", "empty_retrieval", "low_score"})

    def test_phoenix_spans_are_loaded(self):
        raw = [
            {"context": {"span_id": "p1"}, "name": "app", "parent_id": None, "span_kind": "CHAIN",
             "start_time": "2026-01-01T00:00:00", "end_time": "2026-01-01T00:00:01", "status_code": "OK",
             "attributes": {"output.value": "x"}},
            {"context": {"span_id": "p2"}, "name": "retrieve", "parent_id": "p1", "span_kind": "RETRIEVER",
             "start_time": "2026-01-01T00:00:00", "end_time": "2026-01-01T00:00:01", "status_code": "ERROR",
             "status_message": "index down", "attributes": {}},
        ]
        client = SimpleNamespace(spans=SimpleNamespace(
            get_spans=lambda **kw: raw,
            get_span_annotations=lambda **kw: [{"name": "agent_faithfulness", "result": {"score": 0.1}}]))
        trace = trace_from_phoenix(client, "proj", "t")
        self.assertEqual((trace.root_id, trace.spans[1].error, trace.scores), ("p1", "index down", {"agent_faithfulness": 0.1}))

    def test_publishers_target_their_own_backend_only(self):
        diagnosis = TraceAgent(FakeLLM(EXPLANATION)).analyze(rag_trace())
        feedback, notes = [], []
        ls = TraceLangSmithPublisher(SimpleNamespace(create_feedback=lambda **kw: feedback.append(kw)))
        px = TracePhoenixPublisher(SimpleNamespace(spans=SimpleNamespace(add_span_note=lambda **kw: notes.append(kw))))
        self.assertIn("skipped", ls.publish(rag_trace(), diagnosis))
        self.assertIn("skipped", px.publish(rag_trace(), diagnosis))
        ls_trace = Trace("langsmith", "t", "r1", [TraceSpan("r1", "app")], session_id="s1")
        px_trace = Trace("phoenix", "t", "p1", [TraceSpan("p1", "app")])
        self.assertIsNone(ls.publish(ls_trace, diagnosis))
        self.assertIsNone(px.publish(px_trace, diagnosis))
        self.assertEqual((feedback[0]["run_id"], feedback[0]["value"], notes[0]["span_id"]), ("r1", "retrieval", "p1"))


if __name__ == "__main__":
    unittest.main()
