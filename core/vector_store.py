import os
import shutil
import tempfile
from typing import Callable, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_EMBED_BATCH    = 32

# Set RENDER=true in render.yaml env vars to switch to Mistral embeddings API.
_CLOUD_MODE = os.getenv("RENDER", "").lower() in ("1", "true", "yes")

# Tracks the temp directory used by the most recent build so load_vector_store
# can reopen it within the same Streamlit session.
_current_chroma_dir: Optional[str] = None
_current_chroma_client: Optional[chromadb.PersistentClient] = None


def get_embeddings():
    """
    Return the appropriate embedding model.
    - Cloud / Render  → MistralAIEmbeddings (API call, zero local RAM)
    - Local           → HuggingFaceEmbeddings all-MiniLM-L6-v2 (~90 MB)
    """
    if _CLOUD_MODE:
        from langchain_mistralai import MistralAIEmbeddings
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set in environment / .env")
        return MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": _EMBED_BATCH},
        )


def build_vector_store(
    transcript: str,
    progress_fn: Optional[Callable[[int, int], None]] = None,
) -> Chroma:
    """
    Build a ChromaDB vector store from a transcript.

    progress_fn(done, total) is called after each embedding batch so callers
    (e.g. the Streamlit UI) can update a progress indicator in real time.
    """
    global _current_chroma_dir, _current_chroma_client

    print(f"Building vector store (cloud_mode={_CLOUD_MODE})")

    # ── Split transcript into chunks ──────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts     = splitter.split_text(transcript)
    metadatas = [{"chunk_index": i} for i in range(len(texts))]
    ids       = [str(i)             for i in range(len(texts))]
    print(f"Embedding {len(texts)} chunks…")

    # ── Compute embeddings in batches (enables real-time progress reporting) ──
    embedding_fn  = get_embeddings()
    all_embeddings: list = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch      = texts[i : i + _EMBED_BATCH]
        batch_embs = embedding_fn.embed_documents(batch)
        all_embeddings.extend(batch_embs)
        done = min(i + _EMBED_BATCH, len(texts))
        print(f"  Embedded {done}/{len(texts)} chunks")
        if progress_fn:
            progress_fn(done, len(texts))

    # ── Write to a fresh temp directory (eliminates all SQLite lock issues) ───
    if _current_chroma_dir and os.path.exists(_current_chroma_dir):
        try:
            shutil.rmtree(_current_chroma_dir)
        except Exception:
            pass

    _current_chroma_dir    = tempfile.mkdtemp(prefix="menkarai_vdb_")
    _current_chroma_client = chromadb.PersistentClient(path=_current_chroma_dir)
    collection = _current_chroma_client.get_or_create_collection(COLLECTION_NAME)
    collection.add(
        embeddings=all_embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids,
    )

    vector_store = Chroma(
        client=_current_chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )
    print(f"Vector store ready — {len(texts)} chunks indexed.")
    return vector_store


def load_vector_store() -> Chroma:
    """Reopen the vector store built in the current session."""
    if not _current_chroma_client:
        raise RuntimeError("No vector store found. Please run the pipeline first.")
    return Chroma(
        client=_current_chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
