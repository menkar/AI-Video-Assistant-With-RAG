import os
import shutil
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR      = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Set RENDER=true in render.yaml env vars to switch to Mistral embeddings API.
# This avoids loading the 400 MB sentence-transformers model on low-RAM hosts.
_CLOUD_MODE = os.getenv("RENDER", "").lower() in ("1", "true", "yes")


def get_embeddings():
    """
    Return the appropriate embedding model.

    - Cloud / Render  → MistralAIEmbeddings (API call, ~0 local RAM)
    - Local           → HuggingFaceEmbeddings all-MiniLM-L6-v2 (local model, ~400 MB)
    """
    if _CLOUD_MODE:
        from langchain_mistralai import MistralAIEmbeddings
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
        return MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=api_key,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"show_progress_bar": True, "batch_size": 32},
        )


def build_vector_store(transcript: str) -> Chroma:
    print(f"Building vector store (cloud_mode={_CLOUD_MODE})")

    # Remove any stale collection/lock files from a previous run.
    # On Windows, leftover ChromaDB files can cause a silent hang.
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(transcript)
    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    print(f"Embedding {len(docs)} chunks — this may take a minute on CPU…")
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print("Vector store built successfully.")
    return vector_store


def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
