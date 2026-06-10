# K-12 Question Generation System

A locally-hosted, privacy-preserving system that uses RAG (Retrieval-Augmented Generation) to automatically generate grade-appropriate, Bloom's taxonomy-aligned questions from K-12 curriculum documents. Built with Streamlit + LangChain + Ollama + FAISS. **No API keys required — runs entirely on-device.**

---

## Features

- **Question Generation** — generates MCQ, Short Answer, and True/False questions from uploaded curriculum content
- **Grade Band Targeting** — K-2, 3-5, 6-8, 9-12 with vocabulary and complexity adapted accordingly
- **Bloom's Taxonomy Alignment** — target Remember, Understand, Apply, or Analyze cognitive levels
- **Interactive Quiz UI** — students select answers and get instant feedback with explanations
- **Export** — download generated questions as JSON or CSV
- **Conversational Q&A** — ask free-form questions about the content with source citations
- **FAISS Persistence** — index is saved to disk; no re-indexing needed on restart
- **Bundled Curriculum Support** — place PDFs in `data/` folder for automatic indexing on startup
- **Fully Local** — Ollama handles all inference; no data leaves your machine

---

## Quick Start (macOS)

### 1. Create and activate virtual environment

```bash
cd rag-tutor
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Apple Silicon (M1/M2/M3):** If `faiss-cpu` fails, try `brew install faiss` first, then re-run pip install.

### 3. Install and start Ollama

```bash
brew install ollama
ollama serve
```

Download required models:

```bash
ollama pull llama3.2          # chat model
ollama pull nomic-embed-text  # embedding model
```

### 4. Add curriculum documents (optional but recommended)

Place your K-12 PDF/TXT/MD files in the `data/` folder. The app will index them automatically on first launch.

### 5. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Usage

### Generating Questions

1. Place curriculum files in `data/` (or upload via the UI)
2. The app auto-indexes on startup — wait for the confirmation message
3. Go to **Section 2: Generate K-12 Questions**
4. Choose grade band, question type, Bloom's level, and number of questions
5. Click **Generate Questions**
6. Answer interactively — correct/incorrect feedback shown instantly
7. Export as JSON or CSV for further use

### Q&A Mode

- Use **Section 3** to ask free-form questions grounded in the curriculum content
- Source citations show exactly which chunks informed the answer
- Supports multi-turn conversation with memory

---

## Project Structure

```
rag-tutor/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── data/                         # Place curriculum PDFs/TXTs here
│   └── README.txt                # Instructions for adding files
├── faiss_index/                  # Auto-created: persisted FAISS index
├── RAG_WORKFLOW_EXPLANATION.md   # Technical architecture documentation
└── FUTURE_ENHANCEMENTS.md        # Code snippets for optional extensions
```

---

## System Architecture

```
Curriculum Files (data/ or upload)
         ↓
Text Extraction (pypdf / UTF-8)
         ↓
Recursive Text Chunking (LangChain)
         ↓
Embedding Generation (Ollama: nomic-embed-text)
         ↓
FAISS Vector Store (saved to disk)
         ↓
      ┌──────────────────────────────┐
      │  Question Generation Mode    │
      │  vs.similarity_search()      │
      │  → Prompt with grade/Bloom's │
      │  → LLM (Ollama)              │
      │  → JSON questions parsed     │
      │  → Interactive quiz UI       │
      └──────────────────────────────┘
      ┌──────────────────────────────┐
      │  Q&A / Tutor Mode            │
      │  ConversationalRetrievalChain│
      │  → Retriever (top-k chunks)  │
      │  → LLM answer + citations    │
      └──────────────────────────────┘
```

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Disk (models) | 4 GB (llama3.2 + nomic) | 8 GB |
| OS | macOS 12+ / Linux | macOS 13+ |
| Python | 3.8+ | 3.11+ |

---

## Troubleshooting

**FAISS install fails on Apple Silicon:**
```bash
brew install faiss
pip install faiss-cpu==1.7.4
```

**Ollama not running:**
```bash
curl http://localhost:11434/api/tags  # check status
ollama serve                           # start manually
```

**Model not found:**
```bash
ollama list
ollama pull llama3.2
ollama pull nomic-embed-text
```

**Questions fail to parse (JSON error):**
Lower the temperature to 0.1–0.2 in the sidebar. Larger models (llama3.1, mistral) tend to follow JSON instructions more reliably.

**Clear stuck index:**
Click **Clear Saved Index** in the sidebar, then re-index.

---

## Research Context

This project demonstrates that a fully functional K-12 question generation system can be built and run locally without cloud services or API costs. Key research properties:

- **Privacy-preserving** — student data and curriculum content never leave the device
- **Curriculum-grounded** — questions are generated from actual document content, not model memory
- **Pedagogically structured** — Bloom's taxonomy and grade band controls align with educational assessment standards
- **Reproducible** — fixed Ollama model versions ensure consistent outputs for evaluation

---

## License

Demo project for educational research purposes.
