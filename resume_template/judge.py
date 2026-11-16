"""Optional DeepEval LLM-judge for the fuzzy part of a conversion (meaning of summary and bullets).

Deterministic checks in checks.py cover facts (names, dates, companies). A judge is only
needed for wording: did the summary/bullets keep the source's meaning, numbers and claims?
Uses Groq as the judge model, so no OpenAI key is needed.
"""
import json
import os

from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from resume_template.template import ResumeTemplate, render_markdown

CRITERIA = (
    "The converted resume must preserve the meaning of the source resume. Penalise any added claim, "
    "changed number, changed employer/role, or dropped achievement in the summary and experience bullets. "
    "Reformatting, reordering, and normalised dates are fine."
)


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, llm=None, name: str = ""):
        self._llm = llm
        self._name = name or os.getenv("GROQ_JUDGE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))

    def load_model(self):
        if self._llm is None:
            from dotenv import load_dotenv
            from langchain_groq import ChatGroq

            load_dotenv()
            self._llm = ChatGroq(model=self._name, temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
        return self._llm

    def generate(self, prompt: str, schema=None):
        text = self.load_model().invoke(prompt + "\nRespond with a single JSON object.").content
        if schema is not None:
            return schema.model_validate(json.loads(text))
        return text

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self._name


def judge_fidelity(source: str, template: ResumeTemplate, model: DeepEvalBaseLLM, threshold: float = 0.7):
    """Returns (passed, score, reason)."""
    metric = GEval(
        name="Resume meaning preserved",
        criteria=CRITERIA,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=model,
        threshold=threshold,
        async_mode=False,
    )
    metric.measure(LLMTestCase(input=source, actual_output=render_markdown(template)))
    return metric.is_successful(), metric.score, metric.reason
