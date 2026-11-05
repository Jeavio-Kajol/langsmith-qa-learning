import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

with patch.dict(os.environ, {"GROQ_API_KEY": "offline-test", "GROQ_MODEL": "openai/gpt-oss-20b", "GROQ_JUDGE_MODEL": "openai/gpt-oss-20b", "LANGSMITH_TRACING": "false"}):
    from rag import rag_eval_runner as runner
    from rag import rag_app as app
    from chat import chat_app
    from chat import chat_eval_runner as starter

from rag.rag_index import load_documents, retrieve_documents


class RAGEvalRunnerTests(unittest.TestCase):
    def setUp(self):
        runner.metric_scores.clear()
        runner.judge_errors.clear()
        runner.target_errors.clear()
        runner.provider_error = None
        runner.judge_feedback_cache.clear()

    def test_retrieval_drops_distant_chunks(self):
        store = Mock()
        store.similarity_search_with_score.return_value = [
            ("relevant", 0.85), ("unrelated", 0.60), ("unrelated", 0.50),
        ]
        self.assertEqual(retrieve_documents(store, "test"), [("relevant", 0.85)])
        store.similarity_search_with_score.assert_called_once_with("test", k=3)

    def test_knowledge_notes_can_be_loaded(self):
        documents = load_documents()
        self.assertEqual(len(documents), 1)
        self.assertIn("Regression Testing", documents[0].page_content)
        self.assertEqual(documents[0].metadata["source"], "qa_knowledge.txt")

    def test_all_model_calls_use_configured_groq_model(self):
        for model in (app.llm, runner.JUDGE_LLM, chat_app.llm, starter.judge_llm):
            self.assertEqual(model.model_name, "openai/gpt-oss-20b")

    def test_chat_evaluation_calls_actual_app_with_only_question(self):
        inputs = {"question": "What is regression testing?", "answer": "REFERENCE MUST NOT REACH APP"}
        with patch.object(starter, "chat", return_value="generated answer") as chat:
            self.assertEqual(starter.run_app(inputs), {"output": "generated answer"})
        chat.assert_called_once_with(inputs["question"])

    def test_chat_evaluation_preserves_interactive_app_prompt(self):
        with patch.object(starter, "chat", side_effect=chat_app.chat.__wrapped__), patch.object(chat_app, "llm") as model:
            model.invoke.return_value = SimpleNamespace(content="answer")
            result = starter.run_app({"question": "test question", "reference": "HIDDEN EXPECTED ANSWER"})
        self.assertEqual(result, {"output": "answer"})
        messages = model.invoke.call_args.args[0]
        self.assertEqual([message.type for message in messages], ["system", "human"])
        self.assertEqual(messages[1].content, "test question")
        self.assertNotIn("HIDDEN EXPECTED ANSWER", " ".join(message.content for message in messages))
        self.assertEqual(model.invoke.call_args.kwargs["config"]["metadata"]["session_id"], "chat-session")

    def test_chat_evaluation_uses_uploaded_golden_dataset(self):
        with patch.object(starter, "client") as client, patch.object(starter, "evaluate") as evaluate, redirect_stdout(io.StringIO()):
            client.has_dataset.return_value = True
            self.assertIs(starter.main(), evaluate.return_value)
        client.has_dataset.assert_called_once_with(dataset_name="chat-golden-v1")
        self.assertIs(evaluate.call_args.args[0], starter.run_app)
        self.assertEqual(evaluate.call_args.kwargs["data"], "chat-golden-v1")
        self.assertEqual(evaluate.call_args.kwargs["max_concurrency"], 1)
        self.assertEqual(evaluate.call_args.kwargs["evaluators"], [starter.keyword_coverage, starter.hallucination_check, starter.llm_relevance])
        client.create_dataset.assert_not_called()

    def test_chat_evaluation_requires_upload_without_creating_starter_data(self):
        with patch.object(starter, "client") as client, patch.object(starter, "evaluate") as evaluate:
            client.has_dataset.return_value = False
            with self.assertRaisesRegex(ValueError, "python -m chat.upload_chat_dataset"):
                starter.main()
        client.create_dataset.assert_not_called()
        evaluate.assert_not_called()

    def test_rag_rejects_empty_and_oversized_input_before_retrieval(self):
        for question in ("", "   ", None, "test " * 3000):
            with self.subTest(question_type=type(question).__name__):
                with patch.object(app, "get_relevant_context") as retrieve:
                    with self.assertRaises(ValueError):
                        app.run_rag.__wrapped__(question, Mock())
                retrieve.assert_not_called()

    def test_rag_uses_separate_system_and_user_messages(self):
        with patch.object(app, "invoke_with_retry", return_value=SimpleNamespace(content="answer")) as invoke:
            self.assertEqual(app.generate_answer.__wrapped__("question", [{"content": "notes"}]), "answer")
        messages = invoke.call_args.args[1]
        self.assertEqual([message.type for message in messages], ["system", "human"])
        self.assertIn("not instructions", messages[0].content)

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
        for quota in ("tokens per day", "requests per day", "insufficient_quota", "exceeded your current quota"):
            with self.subTest(quota=quota):
                model = Mock()
                model.invoke.side_effect = RuntimeError(f"429 rate limit: {quota}")
                with patch.object(app.time, "sleep") as sleep:
                    with self.assertRaisesRegex(RuntimeError, "Provider quota exhausted"):
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

    def test_refusal_with_fabricated_answer_does_not_pass(self):
        example = SimpleNamespace(metadata={"category": "out_of_scope"})
        for answer, expected in (
            ("I don't know based on the available knowledge.", 1.0),
            ("I don't know, but the salary is 100000 per month.", 0.0),
            ("The salary is 100000 per month.", 0.0),
        ):
            self.assertEqual(runner.no_answer_evaluator({}, {"answer": answer}, {}, example)["score"], expected)

    def test_starter_keyword_calibration_examples(self):
        example = SimpleNamespace(outputs={"required_keywords": ["existing functionality", "code changes"]})
        for answer, expected in (
            ("Checks existing functionality after code changes.", 1.0),
            ("Only checks new features.", 0.0),
            ("Checks existing functionality after modifications.", 0.5),
        ):
            result = starter.keyword_coverage(SimpleNamespace(outputs={"output": answer}), example)
            self.assertEqual(result["score"], expected)

    def test_judge_cache_does_not_merge_distinct_examples(self):
        response = SimpleNamespace(content='{"context_precision": 1, "faithfulness": 1, "hallucination_rate": 0, "answer_relevance": 1}')
        with patch.object(runner, "invoke_with_retry", return_value=response) as invoke:
            for case_id in ("one", "two"):
                runner.faithfulness_evaluator({}, {"answer": "same answer"}, {}, SimpleNamespace(id=case_id))
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(runner.metric_scores["faithfulness"], [1.0, 1.0])

    def test_diagnostic_subset_cannot_report_full_pass(self):
        def evaluate(*args, **kwargs):
            for metric in runner.THRESHOLDS:
                runner.record_score(metric, 0.0 if metric == "hallucination_rate" else 1.0)
            return []
        with patch.object(runner, "Client") as client, patch.object(runner, "evaluate", side_effect=evaluate), patch.object(runner, "build_knowledge_index"), redirect_stdout(io.StringIO()):
            client.return_value.list_examples.return_value = iter([SimpleNamespace(id="one")])
            self.assertEqual(runner.main(limit=1), 1)

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

    def test_core_evaluators_share_one_judge_call(self):
        response = SimpleNamespace(content='{"context_precision": 0.9, "faithfulness": 1, "hallucination_rate": 0, "answer_relevance": 1}')
        inputs, outputs = {"question": "test"}, {"answer": "test"}
        with patch.object(runner, "invoke_with_retry", return_value=response) as invoke:
            for evaluator in (runner.faithfulness_evaluator, runner.hallucination_evaluator, runner.answer_relevance_evaluator):
                evaluator(inputs, outputs, {}, SimpleNamespace(id="example-1"))
        invoke.assert_called_once()
        self.assertEqual(runner.metric_scores["faithfulness"], [1.0])

    def test_failed_output_does_not_pollute_quality_scores(self):
        self.assertIsNone(runner.context_recall_evaluator({}, {}, {"expected_context_contains": ["test"]})["score"])
        self.assertIsNone(runner.no_answer_evaluator({}, {}, {}, SimpleNamespace(metadata={"category": "out_of_scope"}))["score"])
        self.assertFalse(runner.metric_scores)

    def test_strict_threshold_boundaries_fail(self):
        for boundary_metric, boundary in runner.THRESHOLDS.items():
            with self.subTest(metric=boundary_metric):
                def evaluate(*args, **kwargs):
                    for metric in runner.THRESHOLDS:
                        score = 0.0 if metric == "hallucination_rate" else 1.0
                        runner.record_score(metric, boundary if metric == boundary_metric else score)
                    return []
                with patch.object(runner, "evaluate", side_effect=evaluate), patch.object(runner, "build_knowledge_index"), redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(), 1)

    def test_reusable_runner_accepts_configuration(self):
        target = Mock(return_value={"answer": "test"})
        config = runner.EvalConfig(dataset_name="custom", experiment_prefix="reviewed", app_function=target)
        def evaluate(function, **kwargs):
            self.assertEqual(function({"question": "test"}), {"answer": "test"})
            self.assertEqual(kwargs["data"], "custom")
            self.assertEqual(kwargs["experiment_prefix"], "reviewed")
            self.assertEqual(len(kwargs["evaluators"]), 5)
            for metric in runner.THRESHOLDS:
                runner.record_score(metric, 0.0 if metric == "hallucination_rate" else 1.0)
            return []
        with patch.object(runner, "evaluate", side_effect=evaluate), patch.object(runner, "build_knowledge_index") as build, redirect_stdout(io.StringIO()):
            summary = runner.RAGEvalRunner(config).run()
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["scores"]["faithfulness"], 1.0)
        build.assert_not_called()
        with self.assertRaises(SystemExit):
            runner.RAGEvalRunner().assert_pass({"passed": True, "diagnostic": True})

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