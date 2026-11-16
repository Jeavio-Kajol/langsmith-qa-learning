"""Offline tests for the report agent (no network)."""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from agents.report_agent import (
    GATE_THRESHOLDS, ReportAgent, RunResults, compare, passes_gate, render_markdown, results_from_file,
    results_from_langsmith,
)


class FakeLLM:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), 0

    def invoke(self, prompt):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=response if isinstance(response, str) else json.dumps(response))


def run(name, **examples):
    return RunResults(name=name, examples=examples)


BASE = run("base", a={"faithfulness": 1.0, "hallucination_rate": 0.0}, b={"faithfulness": 1.0, "hallucination_rate": 0.0})
BAD = run("new", a={"faithfulness": 0.2, "hallucination_rate": 0.8}, b={"faithfulness": 1.0, "hallucination_rate": 0.0})
GOOD_SUMMARY = {"summary": "Faithfulness dropped.", "evidence": ["faithfulness", "a"], "next_step": "Check example a."}


class CompareTests(unittest.TestCase):
    def test_flags_regression_and_direction_of_lower_is_better_metrics(self):
        report = compare(BASE, BAD)
        by_metric = {m.metric: m for m in report.metrics}
        self.assertEqual(by_metric["faithfulness"].status, "regression")
        self.assertEqual(by_metric["hallucination_rate"].status, "regression")  # rose, so worse
        self.assertLess(by_metric["hallucination_rate"].delta, 0)
        self.assertEqual({e.example for e in report.worst_examples}, {"a"})
        self.assertFalse(report.ok)

    def test_improvement_and_no_change_are_not_regressions(self):
        report = compare(BAD, BASE)
        self.assertTrue(all(m.status == "improvement" for m in report.metrics))
        self.assertTrue(compare(BASE, BASE).ok)

    def test_gate_crossing_is_flagged_even_for_small_drops(self):
        base = run("b", a={"faithfulness": 0.93})
        cand = run("c", a={"faithfulness": 0.89})
        change = compare(base, cand).metrics[0]
        self.assertEqual(change.status, "unchanged")
        self.assertEqual(change.gate_change, "pass->fail")
        self.assertFalse(compare(base, cand).ok)

    def test_gate_boundaries_are_strict(self):
        self.assertFalse(passes_gate("faithfulness", 0.90))
        self.assertFalse(passes_gate("hallucination_rate", 0.05))
        self.assertIsNone(passes_gate("unknown_metric", 1.0))

    def test_missing_examples_are_a_coverage_problem(self):
        report = compare(BASE, run("new", a={"faithfulness": 1.0, "hallucination_rate": 0.0}))
        self.assertEqual(report.only_in_baseline, ["b"])
        self.assertTrue(report.coverage_problem)
        self.assertFalse(report.ok)
        self.assertTrue(compare(run("x", z={"faithfulness": 1.0}), BASE).coverage_problem)

    def test_gate_thresholds_match_the_quality_gate(self):
        from rag.rag_eval_runner import THRESHOLDS

        self.assertEqual(GATE_THRESHOLDS, THRESHOLDS)


class AgentTests(unittest.TestCase):
    def test_no_regression_makes_no_llm_call(self):
        llm = FakeLLM()
        report = ReportAgent(llm=llm).run(BASE, BASE)
        self.assertEqual(llm.calls, 0)
        self.assertTrue(report.ok)

    def test_summary_is_attached_to_regressions(self):
        llm = FakeLLM(GOOD_SUMMARY)
        report = ReportAgent(llm=llm).run(BASE, BAD)
        self.assertEqual(report.summary, "Faithfulness dropped.")
        self.assertIn("REGRESSIONS FOUND", render_markdown(report))
        self.assertFalse(report.ok)

    def test_no_llm_flag_skips_the_call(self):
        llm = FakeLLM()
        report = ReportAgent(llm=llm, use_llm=False).run(BASE, BAD)
        self.assertEqual(llm.calls, 0)
        self.assertTrue(report.regressions)

    def test_invented_evidence_is_rejected_then_retried(self):
        bad = dict(GOOD_SUMMARY, evidence=["made_up_metric"])
        llm = FakeLLM(bad, GOOD_SUMMARY)
        report = ReportAgent(llm=llm).run(BASE, BAD)
        self.assertEqual((llm.calls, report.attempts), (2, 2))
        self.assertIsNone(report.error)

    def test_example_prefix_in_evidence_is_accepted(self):
        llm = FakeLLM(dict(GOOD_SUMMARY, evidence=["example a", "faithfulness"]))
        self.assertIsNone(ReportAgent(llm=llm).run(BASE, BAD).error)

    def test_invalid_twice_reports_error_but_keeps_the_numbers(self):
        bad = dict(GOOD_SUMMARY, evidence=["made_up_metric"])
        report = ReportAgent(llm=FakeLLM(bad, bad)).run(BASE, BAD)
        self.assertIn("invalid summary", report.error)
        self.assertTrue(report.regressions)
        self.assertIn("Summary unavailable", render_markdown(report))

    def test_quota_error_stops_later_calls(self):
        agent = ReportAgent(llm=FakeLLM(RuntimeError("rate_limit_exceeded: tokens per day (TPD) limit reached")),
                            sleep=lambda s: None, max_attempts=1)
        first = agent.run(BASE, BAD)
        second = agent.run(BASE, BAD)
        self.assertTrue(first.fatal)
        self.assertIn("quota exhausted", second.error)

    def test_no_common_metrics_is_an_error(self):
        report = ReportAgent(llm=FakeLLM()).run(run("a", x={"m1": 1.0}), run("b", x={"m2": 1.0}))
        self.assertIn("no metrics in common", report.error)


class LoaderTests(unittest.TestCase):
    def test_file_loader_validates_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, bad = os.path.join(tmp, "g.json"), os.path.join(tmp, "b.json")
            with open(good, "w", encoding="utf-8") as f:
                json.dump({"name": "g", "examples": {"1": {"faithfulness": 1}}}, f)
            with open(bad, "w", encoding="utf-8") as f:
                json.dump({"examples": {"1": {"faithfulness": "high"}}}, f)
            self.assertEqual(results_from_file(good).examples, {"1": {"faithfulness": 1.0}})
            with self.assertRaises(ValueError):
                results_from_file(bad)

    def test_langsmith_loader_groups_feedback_by_dataset_example(self):
        runs = [SimpleNamespace(id="r1", reference_example_id="ex1"), SimpleNamespace(id="r2", reference_example_id=None)]
        feedback = [
            SimpleNamespace(run_id="r1", key="faithfulness", score=0.9),
            SimpleNamespace(run_id="r1", key="note", score=None),
            SimpleNamespace(run_id="r2", key="faithfulness", score=0.1),
        ]
        client = SimpleNamespace(list_runs=lambda **kw: runs, list_feedback=lambda **kw: feedback)
        self.assertEqual(results_from_langsmith(client, "exp").examples, {"ex1": {"faithfulness": 0.9}})
        empty = SimpleNamespace(list_runs=lambda **kw: [], list_feedback=lambda **kw: [])
        with self.assertRaises(ValueError):
            results_from_langsmith(empty, "exp")


if __name__ == "__main__":
    unittest.main()
