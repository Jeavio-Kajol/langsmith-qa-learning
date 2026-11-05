import argparse
import json
import os
import re
from collections import defaultdict
from itertools import islice

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langsmith import Client, evaluate

from day8_rag_app import (
    build_knowledge_index,
    invoke_with_retry,
    is_daily_quota_error,
    run_rag,
)


load_dotenv()

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_NAME = "rag-golden-v1"
EXPERIMENT_PREFIX = "day10-rag-eval-fixed"

JUDGE_LLM = ChatGroq(
    model=os.getenv(
        "GROQ_JUDGE_MODEL",
        os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    ),
    temperature=0,
    max_tokens=1024,
    max_retries=0,
)

THRESHOLDS = {
    "context_recall": 0.85,
    "context_precision": 0.80,
    "faithfulness": 0.90,
    "hallucination_rate": 0.05,
    "no_answer_rate": 0.95,
    "answer_relevance": 0.80,
}

# Scores are collected only when an evaluation is valid.
metric_scores = defaultdict(list)

judge_errors = []
target_errors = []
provider_error = None


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def normalize_text(value):
    """Normalize text for consistent phrase matching."""

    value = str(value).lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", "", value)

    return value.strip()


def get_sources(outputs):
    """Get retrieved source chunks from RAG output."""

    sources = outputs.get("retrieved_sources", [])

    if isinstance(sources, str):
        return [sources]

    if not isinstance(sources, list):
        return []

    return [str(source) for source in sources]


def get_category(example):
    """Read the dataset category from example metadata."""

    if example is None:
        return ""

    metadata = getattr(example, "metadata", {}) or {}

    return str(metadata.get("category", "")).lower()


def get_expected_phrases(reference_outputs):
    phrases = reference_outputs.get(
        "expected_context_contains",
        [],
    )

    if isinstance(phrases, str):
        phrases = [phrases]

    if not isinstance(phrases, list):
        return []

    return [
        str(phrase).strip()
        for phrase in phrases
        if str(phrase).strip()
    ]


def record_score(metric_name, score):
    metric_scores[metric_name].append(float(score))


def extract_json(response_text):
    """Extract JSON even if the model adds surrounding text."""

    response_text = str(response_text).strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        response_text,
        re.DOTALL,
    )

    if not match:
        raise ValueError(
            f"Judge did not return valid JSON: {response_text}"
        )

    return json.loads(match.group(0))


def validate_score(value, metric_name):
    """Ensure a judge score is a valid number between 0 and 1."""

    try:
        score = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid score for {metric_name}: {value}"
        )

    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"Score outside 0-1 range for {metric_name}: {score}"
        )

    return score


# --------------------------------------------------
# EVALUATOR 1: CONTEXT RECALL
# --------------------------------------------------

def context_recall_evaluator(
    inputs,
    outputs,
    reference_outputs,
    example=None,
):
    expected_phrases = get_expected_phrases(
        reference_outputs
    )

    # No expected context means recall is not applicable.
    if not expected_phrases:
        return {
            "key": "context_recall_not_applicable",
            "score": 1.0,
            "comment": "No expected context phrases were provided.",
        }

    sources = get_sources(outputs)

    retrieved_text = normalize_text(
        " ".join(sources)
    )

    matched_phrases = []

    for phrase in expected_phrases:
        normalized_phrase = normalize_text(phrase)

        if normalized_phrase in retrieved_text:
            matched_phrases.append(phrase)

    score = len(matched_phrases) / len(expected_phrases)

    record_score("context_recall", score)

    return {
        "key": "context_recall",
        "score": score,
        "comment": (
            f"Matched {len(matched_phrases)} "
            f"out of {len(expected_phrases)} expected phrases."
        ),
    }


# --------------------------------------------------
# EVALUATOR 2: COMBINED LLM JUDGE
# --------------------------------------------------

def combined_llm_judge_evaluator(
    inputs,
    outputs,
    reference_outputs,
    example=None,
):
    global provider_error

    if provider_error or not str(outputs.get("answer", "")).strip():
        error_message = provider_error or "RAG target returned no answer."
        judge_errors.append(error_message)
        return {
            "key": "judge_error",
            "score": 0.0,
            "comment": error_message[:1000],
        }

    question = inputs.get("question", "")

    answer = outputs.get("answer", "")

    sources = get_sources(outputs)

    context = "\n\n".join(sources)

    prompt = f"""
You are an expert evaluator of a Retrieval-Augmented Generation
(RAG) application used for software testing knowledge.

Evaluate the answer using the question and retrieved context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Evaluate these four metrics independently.

1. context_precision:
How relevant are the retrieved context chunks to the question?
Score from 0 to 1.
1 means all retrieved context is relevant.
0 means none of the retrieved context is relevant.

2. faithfulness:
Is the generated answer supported by the retrieved context?
Score from 0 to 1.
1 means all factual claims are supported.
0 means the answer is entirely unsupported.

3. hallucination_rate:
What proportion of factual claims in the answer are fabricated
or unsupported by the retrieved context?
Score from 0 to 1.
0 means no hallucination.
1 means all factual claims are hallucinated.
If the answer contains no factual claims, use 0.

4. answer_relevance:
Does the answer directly address the user's question?
Score from 0 to 1.
1 means fully relevant.
0 means completely irrelevant.

Important:
- Do not use outside knowledge.
- Judge only using the supplied question, context and answer.
- An answer that correctly says it does not know should not be
  penalized for lacking information that is absent from context.
- Return only valid JSON.
- Do not include Markdown or explanations outside JSON.

Required JSON format:
{{
    "context_precision": 0.0,
    "faithfulness": 0.0,
    "hallucination_rate": 0.0,
    "answer_relevance": 0.0
}}
"""

    try:
        response = invoke_with_retry(
            JUDGE_LLM,
            prompt,
        )

        result = extract_json(response.content)

        scores = {
            "context_precision": validate_score(
                result["context_precision"],
                "context_precision",
            ),
            "faithfulness": validate_score(
                result["faithfulness"],
                "faithfulness",
            ),
            "hallucination_rate": validate_score(
                result["hallucination_rate"],
                "hallucination_rate",
            ),
            "answer_relevance": validate_score(
                result["answer_relevance"],
                "answer_relevance",
            ),
        }

        for metric_name, score in scores.items():
            record_score(metric_name, score)

        return [
            {
                "key": metric_name,
                "score": score,
                "comment": "Scored by the combined Groq LLM judge.",
            }
            for metric_name, score in scores.items()
        ]

    except Exception as error:
        error_message = str(error)

        if is_daily_quota_error(error):
            provider_error = error_message

        judge_errors.append(error_message)

        print(f"\nLLM judge evaluation failed: {error_message}")

        # Return a valid result so LangSmith does not reject [].
        # Do not include this score in metric averages.
        return {
            "key": "judge_error",
            "score": 0.0,
            "comment": error_message[:1000],
        }


# --------------------------------------------------
# EVALUATOR 3: NO-ANSWER RATE
# --------------------------------------------------

def no_answer_evaluator(
    inputs,
    outputs,
    reference_outputs,
    example=None,
):
    category = get_category(example)

    answer = normalize_text(
        outputs.get("answer", "")
    )

    # This metric applies only to out-of-scope examples.
    if category != "out_of_scope":
        return {
            "key": "no_answer_not_applicable",
            "score": 1.0,
            "comment": "This example is not categorized as out_of_scope.",
        }

    abstention_phrases = [
        "i dont know based on the available knowledge",
        "i dont know",
        "not available in the provided knowledge",
        "not enough information",
        "cannot answer based on the available knowledge",
        "information is not available",
    ]

    correctly_abstained = any(
        phrase in answer
        for phrase in abstention_phrases
    )

    score = 1.0 if correctly_abstained else 0.0

    record_score("no_answer_rate", score)

    return {
        "key": "no_answer_rate",
        "score": score,
        "comment": (
            "Correctly refused an out-of-scope question."
            if score == 1.0
            else "The answer did not clearly refuse an out-of-scope question."
        ),
    }


# --------------------------------------------------
# RAG TARGET FUNCTION
# --------------------------------------------------

def main(dataset_name=DATASET_NAME, limit=None):
    global provider_error

    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")

    metric_scores.clear()
    judge_errors.clear()
    target_errors.clear()
    provider_error = None

    data = dataset_name
    if limit is not None:
        data = list(islice(
            Client().list_examples(dataset_name=dataset_name),
            limit,
        ))
        if not data:
            raise ValueError(f"Dataset has no examples: {dataset_name}")
        print(f"Diagnostic subset: {len(data)} examples (not a full CI run).")

    print("\nBuilding RAG knowledge index...")

    vector_store = build_knowledge_index()

    print("\nKnowledge index ready.")

    def rag_target(inputs):
        global provider_error

        if provider_error:
            raise RuntimeError(provider_error)

        try:
            return run_rag(inputs["question"], vector_store)
        except Exception as error:
            target_errors.append(str(error))
            if is_daily_quota_error(error):
                provider_error = str(error)
            raise

    # --------------------------------------------------
    # RUN LANGSMITH EVALUATION
    # --------------------------------------------------

    print("\nStarting Day 10 RAG evaluation...")
    print(f"Dataset: {dataset_name}")
    print(f"Experiment: {EXPERIMENT_PREFIX}")

    results = evaluate(
        rag_target,
        data=data,
        evaluators=[
            context_recall_evaluator,
            combined_llm_judge_evaluator,
            no_answer_evaluator,
        ],
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=1,
    )

    # --------------------------------------------------
    # EVALUATION SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("DAY 10 EVALUATION SUMMARY")
    print("=" * 55)

    metric_names = [
        "context_recall",
        "context_precision",
        "faithfulness",
        "hallucination_rate",
        "no_answer_rate",
        "answer_relevance",
    ]

    summary = {}

    for metric_name in metric_names:
        scores = metric_scores.get(metric_name, [])

        if scores:
            average_score = sum(scores) / len(scores)

            summary[metric_name] = average_score

            print(
                f"{metric_name}: {average_score:.4f} "
                f"| valid samples: {len(scores)}"
            )

        else:
            summary[metric_name] = None

            print(
                f"{metric_name}: N/A "
                "| valid samples: 0"
            )

    print("\n" + "=" * 55)
    print("EVALUATION ERROR SUMMARY")
    print("=" * 55)

    print(f"LLM judge errors: {len(judge_errors)}")
    print(f"RAG target errors: {len(target_errors)}")

    if target_errors:
        print(f"First target error: {target_errors[0][:1000]}")

    if provider_error:
        print(f"Provider quota error: {provider_error[:1000]}")

    if judge_errors:
        for index, error in enumerate(judge_errors, start=1):
            print(f"{index}. {error[:300]}")

    # --------------------------------------------------
    # CI THRESHOLD CHECK
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("CI THRESHOLD CHECKS")
    print("=" * 55)

    all_passed = True

    for metric_name, threshold in THRESHOLDS.items():
        actual = summary[metric_name]

        if actual is None:
            passed = False
            status = "NO VALID DATA"

        elif metric_name == "hallucination_rate":
            passed = actual < threshold
            status = "PASS" if passed else "FAIL"

        else:
            passed = actual >= threshold
            status = "PASS" if passed else "FAIL"

        if not passed:
            all_passed = False

        actual_text = (
            f"{actual:.4f}"
            if actual is not None
            else "N/A"
        )

        comparison = (
            "<"
            if metric_name == "hallucination_rate"
            else ">="
        )

        print(
            f"{metric_name}: {status} "
            f"| actual={actual_text} "
            f"| target {comparison} {threshold}"
        )

    # Judge failures also block the CI gate.
    if judge_errors or target_errors or provider_error:
        all_passed = False

    print("\n" + "=" * 55)

    if all_passed:
        print("EVALUATION PASSED — CI gate is successful.")
    else:
        print("EVALUATION FAILED — CI gate is blocking this run.")

    print("=" * 55)

    return 0 if all_passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 10 RAG evaluation gate")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--limit", type=int, help="Run only N examples for diagnosis")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    raise SystemExit(main(dataset_name=args.dataset, limit=args.limit))