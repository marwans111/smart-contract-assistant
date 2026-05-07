"""
loader.py — Load PDF and DOCX files using LangChain document loaders.
"""
import os
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.schema import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def load_document(file_path: str) -> List[Document]:
    """
    Load a PDF or DOCX file and return a list of LangChain Documents.

    Args:
        file_path: Path to the uploaded file.

    Returns:
        List of Document objects with page_content and metadata.

    Raises:
        ValueError: If the file format is not supported.
    """
    ext = os.path.splitext(file_path)[-1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: '{ext}'. Supported: {SUPPORTED_EXTENSIONS}")

    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)

    docs = loader.load()
    print(f"[Loader] Loaded {len(docs)} page(s) from '{os.path.basename(file_path)}'")
    return docs
