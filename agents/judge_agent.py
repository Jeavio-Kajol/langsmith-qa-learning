"""Judge agent: a small LangGraph workflow that scores a RAG answer and publishes the scores.

Graph: prepare -> score -> verify -(invalid, retry once)-> score
                              \\-(ok)-> publish -> END

The LLM only labels claims and context chunks; the numeric scores are computed here so
they are consistent and auditable. Errors are reported on the verdict, never hidden.
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from ingest.telemetry import get_tracer

MAX_FIELD_CHARS = 10000
MAX_TOTAL_CHARS = 40000
MAX_SCORE_ATTEMPTS = 2
METRICS = ("faithfulness", "hallucination_rate", "context_precision", "answer_relevance")


@dataclass
class JudgeItem:
    question: str
    answer: str
    contexts: list
    run_id: Optional[str] = None  # LangSmith run to attach feedback to
    span_id: Optional[str] = None  # Phoenix span to attach annotations to
    item_id: Optional[str] = None
    session_id: Optional[str] = None  # LangSmith project (session) id of the run


@dataclass
class JudgeVerdict:
    item_id: Optional[str]
    scores: dict = field(default_factory=dict)
    reason: str = ""
    claims: list = field(default_factory=list)
    error: Optional[str] = None
    publish_errors: list = field(default_factory=list)
    publish_notes: list = field(default_factory=list)
    attempts: int = 0
    fatal: bool = False  # provider quota exhausted: stop making model calls

    @property
    def ok(self) -> bool:
        return self.error is None and not self.publish_errors and set(self.scores) == set(METRICS)


class JudgeState(TypedDict, total=False):
    item: JudgeItem
    prompt: str
    raw: str
    verdict: JudgeVerdict
    parse_error: str
    done: bool


def is_provider_quota_error(error) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "insufficient_quota",
            "exceeded your current quota",
            "billing hard limit",
            "tokens per day",
            "requests per day",
        )
    )


def is_rate_limit_error(error) -> bool:
    message = str(error).lower()
    return any(m in message for m in ("rate limit", "429", "too many requests", "tokens per minute"))


def invoke_llm(llm, prompt, sleep=time.sleep, max_attempts=5):
    """Call the LLM, retrying temporary rate limits; provider quota exhaustion is raised at once."""
    for attempt in range(max_attempts):
        try:
            return llm.invoke(prompt)
        except Exception as error:
            if is_provider_quota_error(error):
                raise RuntimeError(f"Provider quota exhausted: {error}") from error
            if not is_rate_limit_error(error) or attempt == max_attempts - 1:
                raise
            sleep(15 * (attempt + 1))


def build_judge_llm():
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.getenv("GROQ_JUDGE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")),
        temperature=0,
        max_tokens=2048,
        max_retries=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def build_prompt(item: JudgeItem) -> str:
    chunks = "\n".join(f"[{i}] {text}" for i, text in enumerate(item.contexts)) or "(no context retrieved)"
    return f"""You are a strict evaluator of a Retrieval-Augmented Generation (RAG) application.
Use ONLY the question, retrieved context chunks and answer below. Do not use outside knowledge.

QUESTION:
{item.question}

RETRIEVED CONTEXT CHUNKS (numbered):
{chunks}

ANSWER:
{item.answer}

Tasks:
1. Break the answer into its individual factual claims. For each claim, say whether the
   retrieved context supports it. An answer that correctly says it does not know is not
   penalised and has no claims.
2. For every context chunk, say whether it is relevant to the question (true/false), in order.
3. Give answer_relevance from 0 to 1: how directly the answer addresses the question.
4. Give a one-sentence reason.

Return only JSON in this shape:
{{"claims": [{{"claim": "...", "supported": true}}],
  "chunk_relevant": [true],
  "answer_relevance": 0.0,
  "reason": "..."}}"""


def extract_json(text: str) -> dict:
    text = str(text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"judge did not return JSON: {text[:200]}")
        return json.loads(match.group(0))


def compute_scores(parsed: dict, context_count: int) -> tuple[dict, list]:
    """Validate the judge's labels and turn them into scores. Raises ValueError if invalid."""
    claims = parsed.get("claims")
    chunk_relevant = parsed.get("chunk_relevant")
    relevance = parsed.get("answer_relevance")
    if not isinstance(claims, list) or not all(
        isinstance(c, dict) and isinstance(c.get("supported"), bool) for c in claims
    ):
        raise ValueError("claims must be a list of {claim, supported: bool}")
    if not isinstance(chunk_relevant, list) or not all(isinstance(v, bool) for v in chunk_relevant):
        raise ValueError("chunk_relevant must be a list of booleans")
    if len(chunk_relevant) != context_count:
        raise ValueError(f"chunk_relevant has {len(chunk_relevant)} entries for {context_count} chunks")
    if isinstance(relevance, bool) or not isinstance(relevance, (int, float)) or not 0.0 <= relevance <= 1.0:
        raise ValueError(f"answer_relevance must be a number between 0 and 1, got {relevance!r}")

    supported = sum(1 for c in claims if c["supported"])
    scores = {
        "faithfulness": supported / len(claims) if claims else 1.0,
        "hallucination_rate": (len(claims) - supported) / len(claims) if claims else 0.0,
        "context_precision": sum(chunk_relevant) / context_count if context_count else 0.0,
        "answer_relevance": float(relevance),
    }
    return scores, claims


class JudgeAgent:
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
        graph = StateGraph(JudgeState)
        graph.add_node("prepare", self._prepare)
        graph.add_node("score", self._score)
        graph.add_node("verify", self._verify)
        graph.add_node("publish", self._publish)
        graph.set_entry_point("prepare")
        graph.add_conditional_edges("prepare", lambda s: END if s.get("done") else "score")
        graph.add_edge("score", "verify")
        graph.add_conditional_edges(
            "verify",
            lambda s: "publish" if not s.get("done") and not s.get("parse_error") else
            ("score" if s.get("parse_error") else END),
            {"publish": "publish", "score": "score", END: END},
        )
        graph.add_edge("publish", END)
        return graph.compile()

    def _prepare(self, state: JudgeState) -> JudgeState:
        item = state["item"]
        verdict = JudgeVerdict(item_id=item.item_id)
        if self.quota_exhausted:
            verdict.error, verdict.fatal = f"skipped: provider quota exhausted ({self.quota_exhausted})", True
            return {"verdict": verdict, "done": True}
        question, answer = str(item.question or ""), str(item.answer or "")
        contexts = [str(c) for c in (item.contexts or [])]
        if not question.strip() or not answer.strip():
            verdict.error = "question and answer are required"
        elif len(question) > MAX_FIELD_CHARS or len(answer) > MAX_FIELD_CHARS:
            verdict.error = f"question or answer exceeds {MAX_FIELD_CHARS} characters (not truncated)"
        elif len(question) + len(answer) + sum(map(len, contexts)) > MAX_TOTAL_CHARS:
            verdict.error = f"input exceeds {MAX_TOTAL_CHARS} characters (not truncated)"
        if verdict.error:
            return {"verdict": verdict, "done": True}
        item = JudgeItem(question, answer, contexts, item.run_id, item.span_id, item.item_id, item.session_id)
        return {"item": item, "prompt": build_prompt(item), "verdict": verdict}

    def _invoke(self, prompt: str):
        return invoke_llm(self.llm, prompt, self._sleep, self._max_attempts)

    def _score(self, state: JudgeState) -> JudgeState:
        verdict = state["verdict"]
        verdict.attempts += 1
        try:
            response = self._invoke(state["prompt"])
        except Exception as error:
            verdict.error = f"judge call failed: {error}"
            if is_provider_quota_error(error):
                verdict.fatal = True
                self.quota_exhausted = str(error)[:200]
            return {"verdict": verdict, "raw": "", "done": True, "parse_error": ""}
        return {"verdict": verdict, "raw": str(response.content)}

    def _verify(self, state: JudgeState) -> JudgeState:
        verdict = state["verdict"]
        if state.get("done"):
            return {}
        try:
            parsed = extract_json(state["raw"])
            scores, claims = compute_scores(parsed, len(state["item"].contexts))
        except (ValueError, KeyError, TypeError) as error:
            if verdict.attempts < MAX_SCORE_ATTEMPTS:
                return {"parse_error": str(error)}
            verdict.error = f"judge returned invalid output after {verdict.attempts} attempts: {error}"
            return {"verdict": verdict, "done": True, "parse_error": ""}
        verdict.scores, verdict.claims = scores, claims
        verdict.reason = str(parsed.get("reason", ""))[:500]
        return {"verdict": verdict, "parse_error": ""}

    def _publish(self, state: JudgeState) -> JudgeState:
        verdict, item = state["verdict"], state["item"]
        for publisher in self.publishers:
            try:
                note = publisher.publish(item, verdict)
                if note:
                    verdict.publish_notes.append(note)
            except Exception as error:
                verdict.publish_errors.append(f"{type(publisher).__name__}: {error}")
        return {"verdict": verdict}

    def evaluate(self, item: JudgeItem) -> JudgeVerdict:
        with get_tracer().start_as_current_span("judge.evaluate") as span:
            verdict = self.graph.invoke({"item": item})["verdict"]
            span.set_attribute("judge.ok", verdict.ok)
            for metric, score in verdict.scores.items():
                span.set_attribute(f"judge.{metric}", score)
            if verdict.error:
                span.set_attribute("judge.error", verdict.error[:500])
            return verdict

    def evaluate_many(self, items) -> list:
        return [self.evaluate(item) for item in items]
