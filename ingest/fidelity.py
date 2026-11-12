"""DeepEval metric: does the converted Markdown still contain the facts of the source?

Deterministic (no LLM call), so it can gate CI offline.
"""
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class ContentFidelityMetric(BaseMetric):
    """Score = fraction of expected phrases found (case/whitespace-insensitive) in
    the converted text. test_case.actual_output = converted Markdown,
    test_case.context = list of phrases that must survive conversion."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""
        self.async_mode = False
        self.strict_mode = False
        self.evaluation_model = None
        self.error = None

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join((text or "").lower().split())

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        expected = list(test_case.context or [])
        if not expected:
            self.score, self.success, self.reason = 0.0, False, "no expected phrases provided"
            return self.score
        output = self._norm(test_case.actual_output)
        missing = [p for p in expected if self._norm(p) not in output]
        self.score = (len(expected) - len(missing)) / len(expected)
        self.success = self.score >= self.threshold
        self.reason = "all expected content preserved" if not missing else f"missing: {missing}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Content Fidelity"
