import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

with patch.dict(os.environ, {"GROQ_API_KEY": "offline-test", "LANGSMITH_TRACING": "false"}):
    import day10_eval_runner as runner
    import day8_rag_app as app

from day8_rag_index import retrieve_documents


class RAGEvalRunnerTests(unittest.TestCase):
    def setUp(self):
        runner.metric_scores.clear()
        runner.judge_errors.clear()
        runner.target_errors.clear()
        runner.provider_error = None

    def test_retrieval_drops_distant_chunks(self):
        store = Mock()
        store.similarity_search_with_score.return_value = [
            ("relevant", 0.85), ("unrelated", 0.60), ("unrelated", 0.50),
        ]
        self.assertEqual(retrieve_documents(store, "test"), [("relevant", 0.85)])
        store.similarity_search_with_score.assert_called_once_with("test", k=3)

    def test_retrieval_preserves_close_comparison_topics(self):
        store = Mock()
        store.similarity_search_with_score.return_value = [
            ("manual", 0.75), ("automation", 0.70), ("unrelated", 0.30),
        ]
        self.assertEqual(
            retrieve_documents(store, "compare manual and automation"),
            [("manual", 0.75), ("automation", 0.70)],
        )

    def test_retrieval_handles_empty_index(self):
        store = Mock()
        store.similarity_search_with_score.return_value = []
        self.assertEqual(retrieve_documents(store, "test"), [])

    def test_retrieval_excludes_unnamed_neighboring_topic(self):
        regression = SimpleNamespace(metadata={"section": "Regression Testing"})
        sanity = SimpleNamespace(metadata={"section": "Sanity Testing"})
        store = Mock()
        store.similarity_search_with_score.return_value = [(regression, 0.70), (sanity, 0.68)]
        self.assertEqual(
            retrieve_documents(store, "What is regression testing?"),
            [(regression, 0.70)],
        )

    def test_retrieval_preserves_explicit_three_topic_comparison(self):
        sanity = SimpleNamespace(metadata={"section": "Sanity Testing"})
        regression = SimpleNamespace(metadata={"section": "Regression Testing"})
        smoke = SimpleNamespace(metadata={"section": "Smoke Testing"})
        results = [(sanity, 0.80), (regression, 0.60), (smoke, 0.55)]
        store = Mock()
        store.similarity_search_with_score.return_value = results
        self.assertEqual(
            retrieve_documents(store, "How do regression testing, smoke testing, and sanity testing differ?"),
            results,
        )

    def test_retrieval_tightens_gap_for_unnamed_topics(self):
        store = Mock()
        store.similarity_search_with_score.return_value = [("relevant", 0.85), ("weaker", 0.75)]
        self.assertEqual(retrieve_documents(store, "test"), [("relevant", 0.85)])

    def test_negated_topic_does_not_override_semantic_matches(self):
        automation = SimpleNamespace(metadata={"section": "Automation Testing"})
        manual = SimpleNamespace(metadata={"section": "Manual Testing"})
        results = [(automation, 0.80), (manual, 0.78)]
        store = Mock()
        store.similarity_search_with_score.return_value = results
        self.assertEqual(
            retrieve_documents(store, "Which testing approach compares results without automation scripts?"),
            results,
        )

    def test_daily_quota_does_not_retry(self):
        for quota in ("tokens per day", "requests per day"):
            with self.subTest(quota=quota):
                model = Mock()
                model.invoke.side_effect = RuntimeError(f"429 rate limit: {quota}")
                with patch.object(app.time, "sleep") as sleep:
                    with self.assertRaisesRegex(RuntimeError, "daily quota exhausted"):
                        app.invoke_with_retry(model, "test")
                model.invoke.assert_called_once_with("test")
                sleep.assert_not_called()

    def test_transient_limit_still_retries(self):
        model = Mock()
        model.invoke.side_effect = [RuntimeError("429 tokens per minute"), "ok"]
        with patch.object(app.time, "sleep") as sleep, redirect_stdout(io.StringIO()):
            self.assertEqual(app.invoke_with_retry(model, "test"), "ok")
        self.assertEqual(model.invoke.call_count, 2)
        sleep.assert_called_once_with(15)

    def test_non_rate_error_does_not_retry(self):
        model = Mock()
        model.invoke.side_effect = ValueError("invalid request")
        with patch.object(app.time, "sleep") as sleep:
            with self.assertRaises(ValueError):
                app.invoke_with_retry(model, "test")
        self.assertEqual(model.invoke.call_count, 1)
        sleep.assert_not_called()

    def test_missing_answer_skips_judge(self):
        with patch.object(runner, "invoke_with_retry") as invoke:
            result = runner.combined_llm_judge_evaluator({}, {}, {})
        self.assertEqual(result["key"], "judge_error")
        self.assertTrue(runner.judge_errors)
        invoke.assert_not_called()

    def test_valid_judge_scores(self):
        response = SimpleNamespace(content='{"context_precision": 0.9, "faithfulness": 1, "hallucination_rate": 0, "answer_relevance": 1}')
        with patch.object(runner, "invoke_with_retry", return_value=response):
            results = runner.combined_llm_judge_evaluator(
                {"question": "What is regression testing?"},
                {"answer": "Checks existing functionality.", "retrieved_sources": ["Checks existing functionality."]},
                {},
            )
        self.assertEqual(len(results), 4)
        self.assertEqual(runner.metric_scores["hallucination_rate"], [0.0])
        self.assertFalse(runner.judge_errors)

    def test_empty_judge_response_retries_once(self):
        responses = [
            SimpleNamespace(content=""),
            SimpleNamespace(content='{"context_precision": 0.9, "faithfulness": 1, "hallucination_rate": 0, "answer_relevance": 1}'),
        ]
        with patch.object(runner, "invoke_with_retry", side_effect=responses) as invoke, redirect_stdout(io.StringIO()):
            results = runner.combined_llm_judge_evaluator(
                {"question": "test"}, {"answer": "test"}, {},
            )
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(len(results), 4)
        self.assertEqual(runner.metric_scores["faithfulness"], [1.0])
        self.assertFalse(runner.judge_errors)

    def test_invalid_judge_response_still_blocks_gate(self):
        with patch.object(runner, "invoke_with_retry", return_value=SimpleNamespace(content="not JSON")) as invoke, redirect_stdout(io.StringIO()):
            result = runner.combined_llm_judge_evaluator(
                {"question": "test"}, {"answer": "test"}, {},
            )
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(result["key"], "judge_error")
        self.assertEqual(len(runner.judge_errors), 1)
        self.assertFalse(runner.metric_scores)

    def test_gate_thresholds_and_state_reset(self):
        for score, hallucination, expected in (
            (1.0, 0.0, 0),
            (0.0, 0.0, 1),
            (1.0, 0.05, 1),
            (None, None, 1),
        ):
            with self.subTest(score=score, hallucination=hallucination):
                runner.metric_scores["faithfulness"].append(1.0)
                runner.judge_errors.append("stale")
                runner.target_errors.append("stale")
                runner.provider_error = "stale"

                def evaluate(*args, **kwargs):
                    self.assertFalse(runner.metric_scores)
                    self.assertFalse(runner.judge_errors)
                    self.assertFalse(runner.target_errors)
                    self.assertIsNone(runner.provider_error)
                    if score is not None:
                        for metric in runner.THRESHOLDS:
                            runner.record_score(metric, hallucination if metric == "hallucination_rate" else score)
                    return []

                with patch.object(runner, "evaluate", side_effect=evaluate), patch.object(runner, "build_knowledge_index"), redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(), expected)

    def test_quota_circuit_breaker_blocks_remaining_calls(self):
        def evaluate(target, **kwargs):
            for _ in range(2):
                with self.assertRaises(RuntimeError):
                    target({"question": "test"})
                runner.combined_llm_judge_evaluator({}, {}, {})
            return []

        with patch.object(runner, "evaluate", side_effect=evaluate), patch.object(runner, "build_knowledge_index"), patch.object(runner, "run_rag", side_effect=RuntimeError("429 tokens per day")) as generate, patch.object(runner, "invoke_with_retry") as judge, redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 1)
        self.assertEqual(generate.call_count, 1)
        judge.assert_not_called()

    def test_target_error_blocks_otherwise_passing_scores(self):
        def evaluate(target, **kwargs):
            with self.assertRaises(ValueError):
                target({"question": "test"})
            for metric in runner.THRESHOLDS:
                runner.record_score(metric, 0.0 if metric == "hallucination_rate" else 1.0)
            return []

        with patch.object(runner, "evaluate", side_effect=evaluate), patch.object(runner, "build_knowledge_index"), patch.object(runner, "run_rag", side_effect=ValueError("target failed")), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 1)

    def test_diagnostic_limit(self):
        examples = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
        with patch.object(runner, "Client") as client, patch.object(runner, "evaluate", return_value=[]) as evaluate, patch.object(runner, "build_knowledge_index"), redirect_stdout(io.StringIO()):
            client.return_value.list_examples.return_value = iter(examples)
            self.assertEqual(runner.main(limit=1), 1)
        self.assertEqual(evaluate.call_args.kwargs["data"], examples[:1])

    def test_invalid_limit(self):
        with patch.object(runner, "build_knowledge_index") as build:
            with self.assertRaises(ValueError):
                runner.main(limit=0)
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()