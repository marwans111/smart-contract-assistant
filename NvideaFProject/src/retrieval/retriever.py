"""
retriever.py — Semantic similarity retriever from the vector store.
"""
from langchain.schema.vectorstore import VectorStoreRetriever
from src.utils.config import TOP_K


def get_retriever(vectorstore) -> VectorStoreRetriever:
    """
    Build a retriever from the vector store.

    Args:
        vectorstore: A LangChain VectorStore instance.

    Returns:
        A retriever configured for top-k similarity search.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    print(f"[Retriever] Ready — top_k={TOP_K}")
    return retriever
