"""
server.py — FastAPI + LangServe microservice backend.
Exposes the QA chain as a REST endpoint.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langserve import add_routes

from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_and_store, load_vectorstore
from src.retrieval.retriever import get_retriever
from src.llm.qa_chain import build_qa_chain, ask
from src.llm.guardrails import check_input, apply_output_guardrail

app = FastAPI(
    title="Smart Contract Assistant API",
    description="RAG pipeline for legal document Q&A",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory chain state (single-user demo)
_chain = None


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str


@app.post("/ingest")
async def ingest(file_path: str):
    """Load, chunk, embed a document and prepare the QA chain."""
    global _chain
    try:
        docs = load_document(file_path)
        chunks = chunk_documents(docs)
        vectorstore = embed_and_store(chunks)
        retriever = get_retriever(vectorstore)
        _chain = build_qa_chain(retriever)
        return {"status": "ok", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(req: QuestionRequest):
    """Answer a question about the ingested document."""
    global _chain
    if _chain is None:
        raise HTTPException(status_code=400, detail="No document ingested yet. Call /ingest first.")

    guard_msg = check_input(req.question)
    if guard_msg:
        raise HTTPException(status_code=422, detail=guard_msg)

    result = ask(_chain, req.question)
    answer = apply_output_guardrail(result["answer"], result.get("source_documents", []))
    return {"answer": answer}


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
