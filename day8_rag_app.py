import os
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langsmith import traceable

from day8_rag_index import (
    load_documents,
    split_documents,
    create_vector_store,
    retrieve_documents,
)

load_dotenv()

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    temperature=0,
    max_tokens=1024,
    max_retries=0,
)


def is_daily_quota_error(error):
    message = str(error).lower()
    return "tokens per day" in message or "requests per day" in message


def invoke_with_retry(model, prompt, max_attempts=5):
    """Retry Groq requests when the rate limit is reached."""

    for attempt in range(max_attempts):
        try:
            return model.invoke(prompt)

        except Exception as error:
            if is_daily_quota_error(error):
                raise RuntimeError(
                    "Groq daily quota exhausted. Wait for quota recovery, "
                    "use a model with available quota via GROQ_MODEL / "
                    "GROQ_JUDGE_MODEL, or upgrade your Groq plan. "
                    f"Provider details: {error}"
                ) from error

            error_message = str(error).lower()

            is_rate_limit = (
                "rate limit" in error_message
                or "429" in error_message
                or "too many requests" in error_message
                or "tokens per minute" in error_message
            )

            if not is_rate_limit or attempt == max_attempts - 1:
                raise

            wait_seconds = 15 * (attempt + 1)

            print(
                f"Groq rate limit reached. "
                f"Retrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)


@traceable(name="rag-indexing")
def build_knowledge_index():
    documents = load_documents()
    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    return vector_store


@traceable(name="rag-retrieval")
def get_relevant_context(vector_store, question):
    results = retrieve_documents(
        vector_store,
        question,
        k=3,
    )

    return [
        {
            "content": document.page_content,
            "source": document.metadata.get(
                "source",
                "unknown",
            ),
            "similarity_score": float(score),
        }
        for document, score in results
    ]


@traceable(name="rag-generation")
def generate_answer(question, retrieved_sources):
    context = "\n\n".join(
        item["content"]
        for item in retrieved_sources
    )

    prompt = f"""
You are a QA knowledge assistant.

Answer the user's question using ONLY the knowledge provided below.

If the answer is not available in the provided knowledge,
respond exactly:

I don't know based on the available knowledge.

Do not invent facts.
Do not use outside knowledge.
Keep the answer clear and relevant.

Knowledge:
{context}

Question:
{question}

Answer:
"""

    response = invoke_with_retry(llm, prompt)

    return str(response.content).strip()


@traceable(name="rag-question-answering")
def run_rag(question, vector_store):
    retrieved_sources = get_relevant_context(
        vector_store,
        question,
    )

    answer = generate_answer(
        question,
        retrieved_sources,
    )

    return {
        "answer": answer,
        "retrieved_sources": [
            item["content"]
            for item in retrieved_sources
        ],
        "retrieved_count": len(retrieved_sources),
    }