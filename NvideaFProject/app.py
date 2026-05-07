"""
app.py — Gradio UI for the Smart Contract Assistant.
Tabs: Upload & Process | Chat | Summarize
"""
import gradio as gr
from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import embed_and_store
from src.retrieval.retriever import get_retriever
from src.llm.qa_chain import build_qa_chain, ask
from src.llm.summarizer import summarize_document
from src.llm.guardrails import check_input, apply_output_guardrail

# Global state
_state = {"chain": None, "chunks": None}


def process_file(file):
    """Ingest uploaded file and build the QA chain."""
    if file is None:
        return "⚠️ Please upload a file first."
    try:
        docs = load_document(file.name)
        chunks = chunk_documents(docs)
        vectorstore = embed_and_store(chunks)
        retriever = get_retriever(vectorstore)
        _state["chain"] = build_qa_chain(retriever)
        _state["chunks"] = chunks
        return f"✅ Document processed — {len(chunks)} chunks ready. You can now chat!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def chat_fn(message, history):
    """Handle a chat message."""
    if _state["chain"] is None:
        return "⚠️ Please upload and process a document first (go to the Upload tab)."

    guard_msg = check_input(message)
    if guard_msg:
        return guard_msg

    try:
        result = ask(_state["chain"], message)
        answer = result["answer"]
        source_docs = result.get("source_documents", [])
        return apply_output_guardrail(answer, source_docs)
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"


def summarize_fn():
    """Summarize the uploaded document."""
    if _state["chunks"] is None:
        return "⚠️ Please upload and process a document first."
    try:
        summary = summarize_document(_state["chunks"])
        return summary
    except Exception as e:
        return f"❌ Error summarizing: {str(e)}"


# --- Build UI ---
with gr.Blocks(title="Smart Contract Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 Smart Contract Summary & Q&A Assistant")
    gr.Markdown("Upload a legal contract (PDF or DOCX), then ask questions or get a summary.")

    with gr.Tab("📁 Upload & Process"):
        file_input = gr.File(label="Upload PDF or DOCX", file_types=[".pdf", ".docx"])
        process_btn = gr.Button("Process Document", variant="primary")
        process_output = gr.Textbox(label="Status", interactive=False)
        process_btn.click(process_file, inputs=file_input, outputs=process_output)

    with gr.Tab("💬 Chat"):
        chatbot = gr.ChatInterface(
            fn=chat_fn,
            chatbot=gr.Chatbot(height=450),
            textbox=gr.Textbox(placeholder="Ask a question about your contract..."),
            examples=[
                "What are the payment terms?",
                "Who are the parties in this contract?",
                "What are the termination conditions?",
                "Are there any penalties or indemnification clauses?",
            ],
        )

    with gr.Tab("📝 Summarize"):
        summarize_btn = gr.Button("Generate Summary", variant="primary")
        summary_output = gr.Textbox(label="Contract Summary", lines=15, interactive=False)
        summarize_btn.click(summarize_fn, outputs=summary_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
