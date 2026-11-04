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
    model="openai/gpt-oss-20b",
    temperature=0,
)


@traceable(name="rag-indexing")
def build_knowledge_index():
    """Load, split, embed and index the QA knowledge base."""
    documents = load_documents()
    chunks = split_documents(documents)
    return create_vector_store(chunks)


@traceable(name="rag-retrieval")
def get_relevant_context(vector_store, question):
    """Return retrieved chunk text and similarity scores."""
    results = retrieve_documents(vector_store, question, k=3)

    return [
        {
            "content": document.page_content,
            "source": document.metadata.get("source", "unknown"),
            "similarity_score": float(score),
        }
        for document, score in results
    ]


@traceable(name="rag-generation")
def generate_answer(question, retrieved_sources):
    """Generate an answer using only retrieved knowledge context."""
    context = "\n\n".join(
        item["content"] for item in retrieved_sources
    )

    prompt = f"""
You are a QA knowledge assistant.

Answer the user's question using ONLY the provided knowledge context.
If the answer is not available in the context, clearly say:
"I don't know based on the available knowledge."

Do not invent information.

Knowledge Context:
{context}

User Question:
{question}

Answer:
"""
    response = llm.invoke(prompt)
    return str(response.content)


@traceable(name="rag-question-answering")
def run_rag(question, vector_store):
    """Run retrieval and generation; expose both outputs for evaluation."""
    retrieved_sources = get_relevant_context(vector_store, question)
    answer = generate_answer(question, retrieved_sources)

    return {
        "answer": answer,
        "retrieved_sources": [
            item["content"] for item in retrieved_sources
        ],
        "retrieved_count": len(retrieved_sources),
    }


if __name__ == "__main__":
    print("Building RAG knowledge index...")
    vector_store = build_knowledge_index()
    print("\nRAG application is ready!")

    while True:
        question = input(
            "\nAsk a QA question (or type exit): "
        ).strip()

        if question.lower() == "exit":
            print("Closing RAG application.")
            break

        if not question:
            print("Please enter a question.")
            continue

        result = run_rag(question, vector_store)
        print("\nRAG Answer:")
        print(result["answer"])
        print("\nRetrieved chunks:", result["retrieved_count"])
