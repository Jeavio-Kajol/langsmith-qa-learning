"""Offline tests for the judge agent (fake LLM and fake publishers, no network).

    python -m unittest agents.test_judge_agent -v
"""
import json
import unittest
from types import SimpleNamespace

from agents.judge import items_from_langsmith
from agents.judge_agent import MAX_FIELD_CHARS, JudgeAgent, JudgeItem
from agents.publishers import LangSmithPublisher, PhoenixPublisher


class FakeLLM:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=response if isinstance(response, str) else json.dumps(response))


GOOD = {
    "claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}],
    "chunk_relevant": [True, False, True, True],
    "answer_relevance": 0.9,
    "reason": "one unsupported claim",
}


def item(**overrides):
    base = dict(question="What is X?", answer="X is Y.", contexts=["c1", "c2", "c3", "c4"], run_id="run-1", span_id="span-1")
    base.update(overrides)
    return JudgeItem(**base)


class Recorder:
    def __init__(self):
        self.published = []

    def publish(self, it, verdict):
        self.published.append(verdict.scores)


class JudgeAgentTests(unittest.TestCase):
    def test_scores_are_computed_from_labels_and_published(self):
        recorder = Recorder()
        verdict = JudgeAgent(FakeLLM(GOOD), [recorder]).evaluate(item())
        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.scores["faithfulness"], 0.5)
        self.assertEqual(verdict.scores["hallucination_rate"], 0.5)
        self.assertEqual(verdict.scores["context_precision"], 0.75)
        self.assertEqual(verdict.scores["answer_relevance"], 0.9)
        self.assertEqual(recorder.published, [verdict.scores])

    def test_no_claims_means_no_hallucination(self):
        data = {**GOOD, "claims": []}
        verdict = JudgeAgent(FakeLLM(data)).evaluate(item())
        self.assertEqual((verdict.scores["faithfulness"], verdict.scores["hallucination_rate"]), (1.0, 0.0))

    def test_invalid_output_retries_once_then_fails_without_publishing(self):
        recorder = Recorder()
        llm = FakeLLM("not json", {**GOOD, "answer_relevance": 1.5})
        verdict = JudgeAgent(llm, [recorder]).evaluate(item())
        self.assertFalse(verdict.ok)
        self.assertEqual(llm.calls, 2)
        self.assertIn("invalid output", verdict.error)
        self.assertEqual(recorder.published, [])

    def test_invalid_then_valid_recovers(self):
        verdict = JudgeAgent(FakeLLM({**GOOD, "chunk_relevant": [True]}, GOOD)).evaluate(item())
        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.attempts, 2)

    def test_rate_limit_retries_but_quota_stops_all_further_calls(self):
        sleeps = []
        agent = JudgeAgent(FakeLLM(Exception("429 Too Many Requests"), GOOD), sleep=sleeps.append)
        self.assertTrue(agent.evaluate(item()).ok)
        self.assertEqual(sleeps, [15])

        llm = FakeLLM(Exception("Error: tokens per day limit reached"))
        agent = JudgeAgent(llm)
        verdicts = agent.evaluate_many([item(), item()])
        self.assertTrue(all(not v.ok for v in verdicts))
        self.assertTrue(verdicts[0].fatal and verdicts[1].fatal)
        self.assertEqual(llm.calls, 1)

    def test_oversized_and_empty_inputs_are_rejected_not_truncated(self):
        llm = FakeLLM()
        agent = JudgeAgent(llm)
        self.assertIn("exceeds", agent.evaluate(item(answer="x" * (MAX_FIELD_CHARS + 1))).error)
        self.assertIn("required", agent.evaluate(item(question="  ")).error)
        self.assertEqual(llm.calls, 0)

    def test_publish_failure_is_reported(self):
        class Broken:
            def publish(self, it, verdict):
                raise RuntimeError("down")

        verdict = JudgeAgent(FakeLLM(GOOD), [Broken()]).evaluate(item())
        self.assertFalse(verdict.ok)
        self.assertIn("down", verdict.publish_errors[0])


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.verdict = JudgeAgent(FakeLLM(GOOD)).evaluate(item())

    def test_langsmith_feedback_keys_are_prefixed(self):
        calls = []
        client = SimpleNamespace(create_feedback=lambda **kw: calls.append(kw))
        LangSmithPublisher(client).publish(item(), self.verdict)
        self.assertEqual({c["key"] for c in calls}, {f"agent_{m}" for m in self.verdict.scores})
        self.assertTrue(all(c["run_id"] == "run-1" for c in calls))

    def test_phoenix_annotations_and_missing_ids_are_noted(self):
        calls = []
        client = SimpleNamespace(spans=SimpleNamespace(add_span_annotation=lambda **kw: calls.append(kw)))
        publisher = PhoenixPublisher(client)
        self.assertIsNone(publisher.publish(item(), self.verdict))
        self.assertEqual(len(calls), 4)
        self.assertIn("skipped", publisher.publish(item(span_id=None), self.verdict))
        self.assertEqual(len(calls), 4)


class LangSmithSourceTests(unittest.TestCase):
    def test_runs_without_question_or_answer_are_skipped(self):
        runs = [
            SimpleNamespace(id="1", inputs={"question": "q"}, outputs={"answer": "a", "retrieved_sources": ["c"]}),
            SimpleNamespace(id="2", inputs={"question": "q"}, outputs={}),
        ]
        client = SimpleNamespace(list_runs=lambda **kw: runs)
        items, skipped = items_from_langsmith("proj", 5, "rag-question-answering", client)
        self.assertEqual(([i.run_id for i in items], skipped), (["1"], 1))
        self.assertEqual(items[0].contexts, ["c"])


if __name__ == "__main__":
    unittest.main()
