from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parents[1]

DOCUMENT_PATH = (
    BASE_DIR / "documents" / "qa_knowledge.txt"
)


def load_documents():
    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {DOCUMENT_PATH}"
        )

    text = DOCUMENT_PATH.read_text(
        encoding="utf-8"
    )

    if not text.strip():
        raise ValueError(
            "Knowledge file is empty."
        )

    return [
        Document(
            page_content=text,
            metadata={
                "source": "qa_knowledge.txt"
            },
        )
    ]


def split_documents(documents):

    headers_to_split_on = [
        ("##", "section")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    all_chunks = []

    for document in documents:

        chunks = markdown_splitter.split_text(
            document.page_content
        )

        for chunk in chunks:

            chunk.metadata["source"] = (
                document.metadata.get(
                    "source",
                    "unknown",
                )
            )

            all_chunks.append(chunk)

    print(
        f"Total topic-based chunks created: {len(all_chunks)}"
    )

    for index, chunk in enumerate(all_chunks, start=1):
        print(
            f"Chunk {index}: "
            f"{chunk.metadata.get('section', 'Unknown section')}"
        )

    return all_chunks


def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = InMemoryVectorStore(
        embeddings
    )

    vector_store.add_documents(chunks)

    print("Embeddings created successfully.")
    print("Documents added to vector store.")

    return vector_store


def retrieve_documents(
    vector_store,
    question,
    k=3,
    max_score_gap=0.05,
):

    results = vector_store.similarity_search_with_score(
        question,
        k=k,
    )

    if not results:
        return []

    topic_text = re.sub(r"\bwithout\s+\w+", "", question.lower())
    question_terms = set(re.findall(r"\w+", topic_text))
    named_results = []
    for document, score in results:
        section = getattr(document, "metadata", {}).get("section", "")
        section_terms = set(re.findall(r"\w+", section.lower())) - {"testing", "and"}
        if section_terms and section_terms <= question_terms:
            named_results.append((document, score))

    if named_results:
        return named_results

    minimum_score = float(results[0][1]) - max_score_gap

    return [
        (document, score)
        for document, score in results
        if float(score) >= minimum_score
    ]