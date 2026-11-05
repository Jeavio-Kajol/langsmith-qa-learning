import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langsmith import Client
from langsmith.evaluation import evaluate

from chat.chat_app import chat

# Load configuration and API keys
load_dotenv()

DATASET_NAME = "chat-golden-v1"

judge_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

client = Client()


def run_app(inputs: dict) -> dict:
    """
    The application under test.

    LangSmith passes each dataset example's inputs to this function.
    It must return a dictionary containing the generated answer.
    """

    question = inputs["question"]

    return {
        "output": chat(question)
    }


def keyword_coverage(run, example) -> dict:
    """
    Check what proportion of required keywords appear in the answer.

    Note: keyword presence does not guarantee that the answer is correct.
    """

    output = str(
        run.outputs.get("output", "")
    ).lower()

    keywords = example.outputs.get("required_keywords", [])

    hits = [
        keyword
        for keyword in keywords
        if keyword.lower() in output
    ]

    score = len(hits) / len(keywords) if keywords else 1.0

    return {
        "key": "keyword_coverage",
        "score": score,
        "comment": (
            f"Found {len(hits)}/{len(keywords)} required keywords. "
            f"Missing: {[kw for kw in keywords if kw not in hits]}"
        )
    }


def hallucination_check(run, example) -> dict:
    """
    Flag an answer if it contains any configured forbidden phrase.

    This is a simple phrase check, not a complete hallucination detector.
    """

    answer = str(
        run.outputs.get("output", "")
    ).lower()

    forbidden = [
        keyword.lower()
        for keyword in example.outputs.get("forbidden_keywords", [])
    ]

    violations = [
        keyword
        for keyword in forbidden
        if keyword in answer
    ]

    return {
        "key": "hallucination_check",
        "score": 0.0 if violations else 1.0,
        "comment": (
            f"Forbidden phrases found: {violations}"
            if violations
            else "No configured forbidden phrases found"
        )
    }


def llm_relevance(run, example) -> dict:
    """
    Ask a model to score how relevant the answer is to the question.

    The model should return one decimal number between 0.0 and 1.0.
    """

    question = example.inputs["question"]

    answer = str(
        run.outputs.get("output", "")
    )

    prompt = f"""
Rate how relevant the answer is to the question.

Use this scale:
0.0 = completely irrelevant
0.5 = partially relevant
1.0 = directly relevant and addresses the question

Return ONLY one decimal number between 0.0 and 1.0.

Question:
{question}

Answer:
{answer}
"""

    response = judge_llm.invoke(prompt)
    result = response.content.strip()

    # Accept only a numeric score, rather than extracting an arbitrary
    # number from a potentially longer explanation.
    if not re.fullmatch(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", result):
        return {
            "key": "llm_relevance",
            "score": 0.0,
            "comment": f"Invalid judge response: {result!r}"
        }

    score = float(result)

    return {
        "key": "llm_relevance",
        "score": score,
        "comment": "Score provided by the LLM relevance judge"
    }


def main():
    if not client.has_dataset(dataset_name=DATASET_NAME):
        raise ValueError(
            f"Dataset {DATASET_NAME!r} is missing. Upload the reviewed questions "
            "first with: python -m chat.upload_chat_dataset"
        )

    print(f"Evaluating the chat app using dataset: {DATASET_NAME}")

    results = evaluate(
        run_app,
        data=DATASET_NAME,
        evaluators=[
            keyword_coverage,
            hallucination_check,
            llm_relevance
        ],
        experiment_prefix="chat-golden-eval",
        max_concurrency=1,
    )

    print("\nEvaluation finished.")
    print(results)
    return results


if __name__ == "__main__":
    main()