"""
embedder.py — Embed document chunks and store them in Chroma or FAISS vector store.
"""
from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import Chroma, FAISS
from src.utils.config import (
    VECTOR_STORE, CHROMA_PERSIST_DIR, FAISS_INDEX_PATH,
    LLM_PROVIDER, OPENAI_API_KEY
)


def _get_embeddings():
    """Return the embedding model based on config."""
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def embed_and_store(chunks: List[Document]):
    """
    Embed chunks and persist to the configured vector store.

    Args:
        chunks: Chunked Document objects.

    Returns:
        The vector store instance.
    """
    embeddings = _get_embeddings()

    if VECTOR_STORE == "chroma":
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        vectorstore.persist()
        print(f"[Embedder] Stored {len(chunks)} chunks in Chroma at '{CHROMA_PERSIST_DIR}'")

    elif VECTOR_STORE == "faiss":
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"[Embedder] Stored {len(chunks)} chunks in FAISS at '{FAISS_INDEX_PATH}'")

    else:
        raise ValueError(f"Unknown vector store: '{VECTOR_STORE}'")

    return vectorstore


def load_vectorstore():
    """Load an existing vector store from disk."""
    embeddings = _get_embeddings()

    if VECTOR_STORE == "chroma":
        return Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )
    elif VECTOR_STORE == "faiss":
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings)
    else:
        raise ValueError(f"Unknown vector store: '{VECTOR_STORE}'")
