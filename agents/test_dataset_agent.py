"""Offline tests for the dataset agent (no network)."""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from agents.dataset_agent import (
    DatasetAgent, Section, assign_ids, candidates_to_json, check_candidate, load_approved, parse_sections,
    upload_approved,
)

KNOWLEDGE = """# QA Knowledge Base

## Regression Testing
Regression testing verifies that existing functionality still works after code changes. It helps identify unintended side effects.

## Smoke Testing
Smoke testing is a preliminary set of checks used to determine whether a build is stable enough for further testing.
"""
SECTIONS = parse_sections(KNOWLEDGE)

GOOD = {
    "question": "Why run regression tests after code changes?",
    "answer": "Regression testing verifies that existing functionality still works after code changes and helps identify unintended side effects.",
    "required_keywords": ["existing functionality", "code changes"],
    "expected_context_contains": ["Regression Testing", "unintended side effects"],
    "category": "grounded_answer",
    "difficulty": "medium",
}


class FakeLLM:
    def __init__(self, *responses):
        self.responses, self.calls, self.prompts = list(responses), 0, []

    def invoke(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=response if isinstance(response, str) else json.dumps(response))


def variant(**changes):
    return dict(GOOD, **changes)


class CheckTests(unittest.TestCase):
    def check(self, raw, known=()):
        return check_candidate(raw, SECTIONS[:1], list(known))

    def test_parse_sections_splits_on_headings(self):
        self.assertEqual([s.title for s in SECTIONS], ["Regression Testing", "Smoke Testing"])

    def test_valid_candidate_passes(self):
        self.assertEqual(self.check(GOOD)["category"], "grounded_answer")

    def test_phrases_must_come_from_source(self):
        with self.assertRaisesRegex(ValueError, "not found in the source"):
            self.check(variant(required_keywords=["existing functionality", "load balancing"]))
        with self.assertRaisesRegex(ValueError, "not found in the source"):
            self.check(variant(expected_context_contains=["Mutation Testing"]))

    def test_heading_markers_are_stripped_from_phrases(self):
        fields = self.check(variant(expected_context_contains=["## Regression Testing"]))
        self.assertEqual(fields["expected_context_contains"], ["Regression Testing"])

    def test_keywords_must_be_in_the_answer(self):
        with self.assertRaisesRegex(ValueError, "not present in the answer"):
            self.check(variant(answer="It helps identify unintended side effects."))

    def test_ungrounded_answer_is_rejected(self):
        answer = "Regression testing verifies existing functionality after code changes using kubernetes clusters, docker containers, and terraform pipelines."
        with self.assertRaisesRegex(ValueError, "grounded"):
            self.check(variant(answer=answer))

    def test_duplicates_and_bad_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.check(GOOD, known=["Why run regression tests after code changes"])
        for bad in (variant(category="out_of_scope"), variant(difficulty="insane"), variant(question="  "),
                    variant(required_keywords=[]), variant(question="x" * 400)):
            with self.assertRaises(ValueError):
                self.check(bad)

    def test_multi_topic_needs_two_sections(self):
        with self.assertRaisesRegex(ValueError, "two sections"):
            self.check(variant(category="multi_topic_retrieval"))
        check_candidate(variant(category="multi_topic_retrieval"), SECTIONS, [])


class AgentTests(unittest.TestCase):
    def test_keeps_valid_and_drops_invalid_candidates(self):
        llm = FakeLLM({"examples": [GOOD, variant(question="Another?", required_keywords=["nonsense"])]})
        result = DatasetAgent(llm=llm).generate_for(SECTIONS[:1], 2, [])
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(len(result.rejected), 1)
        self.assertTrue(result.ok)
        self.assertFalse(result.accepted[0].approved)

    def test_retries_once_with_the_rejection_reason(self):
        llm = FakeLLM({"examples": [variant(category="bogus")]}, {"examples": [GOOD]})
        result = DatasetAgent(llm=llm).generate_for(SECTIONS[:1], 1, [])
        self.assertEqual((llm.calls, result.attempts, len(result.accepted)), (2, 2, 1))
        self.assertIn("rejected", llm.prompts[1])

    def test_fails_after_two_invalid_attempts(self):
        llm = FakeLLM("not json", {"examples": []})
        result = DatasetAgent(llm=llm).generate_for(SECTIONS[:1], 1, [])
        self.assertFalse(result.ok)
        self.assertIn("after 2 attempts", result.error)

    def test_existing_questions_are_not_repeated(self):
        llm = FakeLLM({"examples": [GOOD]}, {"examples": [GOOD]})
        result = DatasetAgent(llm=llm).generate_for(SECTIONS[:1], 1, ["Why run regression tests after code changes?"])
        self.assertFalse(result.ok)

    def test_quota_error_stops_later_calls(self):
        agent = DatasetAgent(llm=FakeLLM(RuntimeError("rate_limit_exceeded: tokens per day (TPD) limit reached")),
                             sleep=lambda s: None, max_attempts=1)
        first = agent.generate_for(SECTIONS[:1], 1, [])
        second = agent.generate_for(SECTIONS[:1], 1, [])
        self.assertTrue(first.fatal)
        self.assertIn("quota exhausted", second.error)


class UploadTests(unittest.TestCase):
    def make_file(self, tmp, approved):
        result = DatasetAgent(llm=FakeLLM({"examples": [GOOD]})).generate_for(SECTIONS[:1], 1, [])
        assign_ids(result.accepted)
        data = candidates_to_json(result.accepted, "k.txt")
        data["candidates"][0]["approved"] = approved
        path = os.path.join(tmp, "c.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    def test_only_approved_examples_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_approved(self.make_file(tmp, False)), [])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_approved(self.make_file(tmp, True))[0]["id"], "RAG-GEN-001")

    def test_upload_marks_human_verified_and_skips_existing_ids(self):
        created = []
        client = SimpleNamespace(
            has_dataset=lambda **kw: True,
            read_dataset=lambda **kw: SimpleNamespace(id="d1"),
            list_examples=lambda **kw: [SimpleNamespace(metadata={"case_id": "RAG-GEN-002"})],
            create_example=lambda **kw: created.append(kw),
        )
        items = [dict(GOOD, id="RAG-GEN-001"), dict(GOOD, id="RAG-GEN-002")]
        self.assertEqual(upload_approved(client, items), (1, 1))
        self.assertTrue(created[0]["metadata"]["human_verified"])
        self.assertEqual(created[0]["inputs"], {"question": GOOD["question"]})


if __name__ == "__main__":
    unittest.main()
