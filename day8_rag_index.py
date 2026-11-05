from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent

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
):

    return vector_store.similarity_search_with_score(
        question,
        k=k,
    )