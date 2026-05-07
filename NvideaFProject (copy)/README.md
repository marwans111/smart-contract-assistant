# 📄 Smart Contract Summary & Q&A Assistant

A modular RAG (Retrieval Augmented Generation) pipeline for querying legal contracts and documents via a conversational chat interface.

---

## 🚀 Features

- Upload PDF / DOCX contracts
- Automatic chunking & embedding
- Semantic search with vector store (Chroma / FAISS)
- LangChain-powered Q&A with source citations
- Guardrails for factuality & safety
- Optional document summarization
- Clean Gradio UI

---

## 🗂️ Project Structure

```
smart-contract-assistant/
├── src/
│   ├── ingestion/          # File parsing, chunking, embedding
│   │   ├── __init__.py
│   │   ├── loader.py       # PDF/DOCX loader
│   │   ├── chunker.py      # Text chunking logic
│   │   └── embedder.py     # Embedding + vector store
│   ├── retrieval/          # Semantic search
│   │   ├── __init__.py
│   │   └── retriever.py    # Vector store retriever
│   ├── llm/                # LLM pipeline + guardrails
│   │   ├── __init__.py
│   │   ├── qa_chain.py     # LangChain QA chain
│   │   ├── summarizer.py   # Summarization chain
│   │   └── guardrails.py   # Input/output safety checks
│   └── utils/
│       ├── __init__.py
│       └── config.py       # App configuration
├── app.py                  # Gradio UI entry point
├── server.py               # FastAPI + LangServe server
├── tests/                  # Unit tests
├── evaluation/             # Eval metrics & scripts
├── notebooks/              # Jupyter exploration notebooks
├── data/sample_contracts/  # Sample test files
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/smart-contract-assistant.git
cd smart-contract-assistant
```

### 2. Create & activate virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key (or leave blank to use local HuggingFace model)
```

### 5. Run the app

```bash
# Option A: Gradio UI only
python app.py

# Option B: FastAPI + LangServe backend
python server.py
```

Open your browser at `http://localhost:7860` (Gradio) or `http://localhost:8000` (FastAPI)

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📊 Evaluation

```bash
python evaluation/run_eval.py
```

Generates metrics: Faithfulness, Answer Relevance, Context Recall.

---

## 🔧 Configuration

Edit `src/utils/config.py` or set these in your `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (optional) |
| `LLM_PROVIDER` | `openai` | `openai` or `huggingface` |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `VECTOR_STORE` | `chroma` | `chroma` or `faiss` |
| `TOP_K` | `4` | Number of retrieved chunks |

---

## 🛡️ Guardrails

- Input: blocks off-topic or harmful queries
- Output: enforces answer grounded in retrieved context only
- Disclaimer added to all responses

---

## 📦 Tech Stack

- **LangChain** — pipeline orchestration
- **Gradio** — UI
- **FastAPI + LangServe** — microservice backend
- **Chroma / FAISS** — vector store
- **SentenceTransformers / OpenAI** — embeddings
- **PyMuPDF, python-docx** — file parsing

---

## ⚠️ Disclaimer

This tool is for informational and educational purposes only. It does not constitute legal advice.
