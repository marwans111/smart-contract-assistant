"""
tests/test_pipeline.py — Unit tests for ingestion, guardrails, and retrieval.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document

from src.ingestion.chunker import chunk_documents
from src.llm.guardrails import check_input, apply_output_guardrail


# ── Chunker ──────────────────────────────────────────────────────────────────

class TestChunker:
    def test_chunks_long_document(self):
        docs = [Document(page_content="word " * 300, metadata={"page": 1})]
        chunks = chunk_documents(docs)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        docs = [Document(page_content="word " * 300, metadata={})]
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert len(chunk.page_content) <= 600  # chunk_size + small overhead

    def test_empty_document_returns_empty(self):
        docs = [Document(page_content="", metadata={})]
        chunks = chunk_documents(docs)
        assert chunks == []


# ── Guardrails ────────────────────────────────────────────────────────────────

class TestGuardrails:
    def test_clean_query_passes(self):
        assert check_input("What are the payment terms?") is None

    def test_blocked_topic_detected(self):
        result = check_input("How to hack the password?")
        assert result is not None
        assert "⛔" in result

    def test_short_query_blocked(self):
        result = check_input("hi")
        assert result is not None

    def test_output_includes_disclaimer(self):
        source_docs = [Document(page_content="text", metadata={"page": 2})]
        output = apply_output_guardrail("The answer is X.", source_docs)
        assert "legal advice" in output.lower()
        assert "page" in output.lower()

    def test_output_no_sources_warns(self):
        output = apply_output_guardrail("Answer.", [])
        assert "unreliable" in output.lower()


# ── Loader (mocked) ───────────────────────────────────────────────────────────

class TestLoader:
    @patch("src.ingestion.loader.PyMuPDFLoader")
    def test_pdf_loads(self, mock_loader):
        mock_loader.return_value.load.return_value = [
            Document(page_content="Contract text", metadata={"page": 1})
        ]
        from src.ingestion.loader import load_document
        docs = load_document("contract.pdf")
        assert len(docs) == 1

    def test_unsupported_format_raises(self):
        from src.ingestion.loader import load_document
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document("file.txt")
