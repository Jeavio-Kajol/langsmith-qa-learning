"""Dataset agent: drafts new RAG test questions from the knowledge base for a human to review.

Graph: generate -> verify -(invalid, retry once)-> generate
                       \\-> END

The LLM proposes questions; code decides what is kept. A candidate is dropped unless every keyword and
context phrase appears in its source section, most of its answer's words come from that section, and the
question is not a near-duplicate. Candidates are written with approved=false; nothing reaches LangSmith
until a human sets approved=true and runs the upload step.
"""
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.judge_agent import build_judge_llm, extract_json, invoke_llm, is_provider_quota_error
from ingest.telemetry import get_tracer

KNOWLEDGE_FILE = "documents/qa_knowledge.txt"
CATEGORIES = ("retrieval_accuracy", "context_relevance", "grounded_answer", "multi_topic_retrieval")
DIFFICULTIES = ("easy", "medium", "hard")
MAX_ATTEMPTS = 2
MAX_QUESTION_CHARS = 300
MAX_EXISTING_IN_PROMPT = 100
DUPLICATE_SIMILARITY = 0.8  # token overlap at or above this counts as a duplicate question
MIN_ANSWER_GROUNDING = 0.8  # share of the answer's content words that must appear in the sections
DEFAULT_UPLOAD_DATASET = "rag-golden-candidates-v1"  # deliberately not the quality-gate dataset


@dataclass
class Section:
    title: str
    text: str  # includes the title line


@dataclass
class Candidate:
    id: str
    question: str
    answer: str
    required_keywords: list
    expected_context_contains: list
    category: str
    difficulty: str
    source_sections: list
    min_retrieved_docs: int = 1
    approved: bool = False


@dataclass
class SectionResult:
    sections: list
    accepted: list = field(default_factory=list)
    rejected: list = field(default_factory=list)  # (question, reason)
    attempts: int = 0
    error: Optional[str] = None
    fatal: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.accepted)


class DatasetState(TypedDict, total=False):
    sections: list
    count: int
    existing: list
    result: SectionResult
    prompt: str
    raw: str
    parse_error: str
    done: bool


def parse_sections(text: str) -> list:
    sections = []
    for block in re.split(r"(?m)^## ", text)[1:]:
        title, _, body = block.partition("\n")
        if body.strip():
            sections.append(Section(title=title.strip(), text=f"{title.strip()}\n{body.strip()}"))
    return sections


def _words(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def similarity(a: str, b: str) -> float:
    wa, wb = set(_words(a)), set(_words(b))
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def is_duplicate(question: str, others: list) -> bool:
    return any(similarity(question, other) >= DUPLICATE_SIMILARITY for other in others)


def answer_grounding(answer: str, source: str) -> float:
    content = [w for w in _words(answer) if len(w) > 3]
    if not content:
        return 0.0
    source_words = set(_words(source))
    return sum(w in source_words for w in content) / len(content)


def _string_list(value, name: str) -> list:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v.strip() for v in value):
        raise ValueError(f"{name} must be a non-empty list of strings")
    return [v.strip() for v in value]


def check_candidate(raw: dict, sections: list, known_questions: list) -> dict:
    """Returns the cleaned fields or raises ValueError explaining why the candidate is rejected."""
    if not isinstance(raw, dict):
        raise ValueError("candidate must be an object")
    for key in ("question", "answer"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    question, answer = raw["question"].strip(), raw["answer"].strip()
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"question is longer than {MAX_QUESTION_CHARS} characters")
    if raw.get("category") not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES}")
    if raw.get("difficulty") not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if raw["category"] == "multi_topic_retrieval" and len(sections) < 2:
        raise ValueError("multi_topic_retrieval needs at least two sections")
    keywords = [p.lstrip("# ").strip() for p in _string_list(raw.get("required_keywords"), "required_keywords")]
    phrases = [p.lstrip("# ").strip() for p in _string_list(raw.get("expected_context_contains"), "expected_context_contains")]
    source = "\n".join(s.text for s in sections).lower()
    missing = [p for p in keywords + phrases if p.lower() not in source]
    if missing:
        raise ValueError(f"phrases not found in the source text: {missing}")
    missing = [k for k in keywords if k.lower() not in answer.lower()]
    if missing:
        raise ValueError(f"required_keywords not present in the answer: {missing}")
    grounding = answer_grounding(answer, source)
    if grounding < MIN_ANSWER_GROUNDING:
        raise ValueError(f"answer is only {grounding:.0%} grounded in the source text")
    if is_duplicate(question, known_questions):
        raise ValueError("near-duplicate of an existing question")
    return {"question": question, "answer": answer, "required_keywords": keywords,
            "expected_context_contains": phrases, "category": raw["category"], "difficulty": raw["difficulty"]}


def build_prompt(sections: list, count: int, existing: list, feedback: str = "") -> str:
    source = "\n\n".join(f"## {s.text}" for s in sections)
    known = "\n".join(f"- {q}" for q in existing[:MAX_EXISTING_IN_PROMPT]) or "- none"
    retry = f"\nYour previous attempt was rejected: {feedback}\nFix this.\n" if feedback else ""
    multi = ("These are several topics: include at least one multi_topic_retrieval question that needs "
             "more than one of them.\n") if len(sections) > 1 else ""
    return f"""You write test questions for a Retrieval-Augmented Generation (RAG) quality dataset.
Write {count} NEW questions that the SOURCE TEXT below fully answers. Use only facts in the source text.
{multi}{retry}
SOURCE TEXT:
{source}

EXISTING QUESTIONS (do not repeat or rephrase these):
{known}

Return only JSON: {{"examples": [{{
  "question": "...",
  "answer": "2-3 sentences copied or closely paraphrased from the source text",
  "required_keywords": ["2-3 words or phrases that appear exactly in BOTH the answer and the source text"],
  "expected_context_contains": ["phrases copied exactly from the source text, e.g. the section title"],
  "category": one of {list(CATEGORIES)},
  "difficulty": one of {list(DIFFICULTIES)}}}]}}"""


class DatasetAgent:
    def __init__(self, llm=None, sleep=time.sleep, max_attempts=5):
        self._llm = llm
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
        graph = StateGraph(DatasetState)
        graph.add_node("generate", self._generate)
        graph.add_node("verify", self._verify)
        graph.set_entry_point("generate")
        graph.add_edge("generate", "verify")
        graph.add_conditional_edges(
            "verify",
            lambda s: END if s.get("done") or not s.get("parse_error") else "generate",
            {END: END, "generate": "generate"},
        )
        return graph.compile()

    def _generate(self, state: DatasetState) -> DatasetState:
        result = state["result"]
        result.attempts += 1
        prompt = build_prompt(state["sections"], state["count"], state["existing"], state.get("parse_error", ""))
        try:
            response = invoke_llm(self.llm, prompt, self._sleep, self._max_attempts)
        except Exception as error:
            result.error = f"generation call failed: {error}"
            if is_provider_quota_error(error):
                result.fatal = True
                self.quota_exhausted = str(error)[:200]
            return {"result": result, "done": True, "parse_error": ""}
        return {"result": result, "raw": str(response.content)}

    def _verify(self, state: DatasetState) -> DatasetState:
        result = state["result"]
        if state.get("done"):
            return {}
        try:
            examples = extract_json(state["raw"]).get("examples")
            if not isinstance(examples, list) or not examples:
                raise ValueError("'examples' must be a non-empty list")
        except (ValueError, TypeError, AttributeError) as error:
            return self._retry_or_fail(result, str(error))
        known = list(state["existing"]) + [c.question for c in result.accepted]
        result.rejected = []
        for raw in examples:
            try:
                fields = check_candidate(raw, state["sections"], known)
            except ValueError as error:
                question = raw.get("question", "?") if isinstance(raw, dict) else "?"
                result.rejected.append((str(question)[:120], str(error)))
                continue
            known.append(fields["question"])
            result.accepted.append(Candidate(id="", source_sections=[s.title for s in state["sections"]], **fields))
        if not result.accepted:
            reasons = "; ".join(sorted({reason for _, reason in result.rejected}))[:400]
            return self._retry_or_fail(result, f"no valid examples ({reasons})")
        return {"result": result, "parse_error": ""}

    def _retry_or_fail(self, result: SectionResult, reason: str) -> DatasetState:
        if result.attempts < MAX_ATTEMPTS:
            return {"result": result, "parse_error": reason}
        result.error = f"no valid examples after {result.attempts} attempts: {reason}"
        return {"result": result, "done": True, "parse_error": ""}

    def generate_for(self, sections: list, count: int, existing: list) -> SectionResult:
        result = SectionResult(sections=[s.title for s in sections])
        if self.quota_exhausted:
            result.error = f"skipped: provider quota exhausted ({self.quota_exhausted})"
            result.fatal = True
            return result
        with get_tracer().start_as_current_span("dataset_agent.generate") as span:
            result = self.graph.invoke({"sections": sections, "count": count, "existing": existing,
                                        "result": result})["result"]
            span.set_attribute("dataset_agent.sections", ",".join(result.sections))
            span.set_attribute("dataset_agent.accepted", len(result.accepted))
            span.set_attribute("dataset_agent.rejected", len(result.rejected))
            return result


def assign_ids(candidates: list, start: int = 1) -> None:
    for offset, candidate in enumerate(candidates):
        candidate.id = f"RAG-GEN-{start + offset:03d}"


def candidates_to_json(candidates: list, source: str) -> dict:
    return {"source": source, "note": "Review each example, then set approved to true. Nothing is uploaded until you do.",
            "candidates": [c.__dict__ for c in candidates]}


def load_approved(path: str) -> list:
    with open(path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    approved = []
    for item in data.get("candidates", []):
        if item.get("approved") is True:
            check_fields = {k: item.get(k) for k in
                            ("question", "answer", "required_keywords", "expected_context_contains", "category", "difficulty")}
            _string_list(check_fields["required_keywords"], "required_keywords")
            _string_list(check_fields["expected_context_contains"], "expected_context_contains")
            if not item.get("id") or not str(check_fields["question"] or "").strip() or not str(check_fields["answer"] or "").strip():
                raise ValueError(f"approved candidate {item.get('id')!r} is missing id, question or answer")
            approved.append(item)
    return approved


def upload_approved(client, items: list, dataset_name: str = DEFAULT_UPLOAD_DATASET) -> tuple:
    """Uploads human-approved candidates; skips case ids already present. Returns (uploaded, skipped)."""
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Agent-drafted RAG examples, each approved by a human before upload.",
        )
    existing = {(e.metadata or {}).get("case_id") for e in client.list_examples(dataset_id=dataset.id)}
    uploaded = skipped = 0
    for item in items:
        if item["id"] in existing:
            skipped += 1
            continue
        client.create_example(
            inputs={"question": item["question"]},
            outputs={"answer": item["answer"], "required_keywords": item["required_keywords"],
                     "expected_context_contains": item["expected_context_contains"],
                     "min_retrieved_docs": int(item.get("min_retrieved_docs", 1))},
            metadata={"case_id": item["id"], "category": item["category"], "difficulty": item["difficulty"],
                      "source": "agent-generated, human-approved", "human_verified": True,
                      "evaluation_type": item["category"]},
            dataset_id=dataset.id,
        )
        uploaded += 1
    return uploaded, skipped


def load_sections(path: str = KNOWLEDGE_FILE) -> list:
    sections = parse_sections(Path(path).read_text(encoding="utf-8-sig"))
    if not sections:
        raise ValueError(f"{path}: no '## ' sections found")
    return sections
