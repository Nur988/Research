import io
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import question_engine

DATA_DIR = Path(__file__).parent / "data"


def _fetch_ollama_models(base_url: str):
    """Return (chat_models, embed_models) sorted newest-first from the live Ollama instance.

    Falls back to hardcoded defaults if Ollama is unreachable.
    """
    import urllib.request
    _EMBED_KEYWORDS = ("embed", "nomic", "bge", "e5", "minilm")
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as r:
            data = json.loads(r.read())
        models = sorted(
            data.get("models", []),
            key=lambda m: m.get("modified_at", ""),
            reverse=True,
        )
        names = [m["name"] for m in models]
        chat   = [n for n in names if not any(k in n.lower() for k in _EMBED_KEYWORDS)] or ["llama3.2"]
        embed  = [n for n in names if     any(k in n.lower() for k in _EMBED_KEYWORDS)] or ["nomic-embed-text"]
        return chat, embed
    except Exception:
        return ["llama3.2", "llama3", "mistral"], ["nomic-embed-text"]
FAISS_INDEX_PATH = "faiss_index"

# ── Document helpers ────────────────────────────────────────────────────────

def extract_text_from_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def _file_to_documents(name: str, data: bytes, chunk_size: int, chunk_overlap: int) -> List[Document]:
    ext = (name.split(".")[-1] or "").lower()
    if ext == "pdf":
        content = extract_text_from_pdf(io.BytesIO(data))
    elif ext in ("txt", "md"):
        content = data.decode("utf-8", errors="ignore")
    else:
        try:
            content = data.decode("utf-8", errors="ignore")
        except Exception:
            content = ""

    if not content.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return [
        Document(page_content=c, metadata={"source": name, "chunk": i})
        for i, c in enumerate(splitter.split_text(content))
    ]


def to_documents(
    uploaded_files,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    docs = []
    for uf in uploaded_files:
        docs.extend(_file_to_documents(uf.name, uf.read(), chunk_size, chunk_overlap))
    return docs


def load_bundled_docs(chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    docs = []
    for path in DATA_DIR.iterdir():
        if path.suffix.lower() in (".pdf", ".txt", ".md"):
            docs.extend(_file_to_documents(path.name, path.read_bytes(), chunk_size, chunk_overlap))
    return docs

# ── Vector store ────────────────────────────────────────────────────────────

def build_vectorstore(docs: List[Document], embedding_model: str, base_url: str) -> FAISS:
    embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(FAISS_INDEX_PATH)
    return vs


def load_vectorstore(embedding_model: str, base_url: str) -> Optional[FAISS]:
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    try:
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.warning(f"Could not load saved index: {e}")
        return None

# ── RAG chain ───────────────────────────────────────────────────────────────

def build_cr_chain(vs: FAISS, llm_model: str, temperature: float, k: int, base_url: str):
    llm = ChatOllama(model=llm_model, temperature=temperature, base_url=base_url)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )

# ── UI helpers ───────────────────────────────────────────────────────────────

def show_sources(src_docs: List[Document]) -> None:
    if not src_docs:
        return
    with st.expander("Sources used in this answer"):
        for i, d in enumerate(src_docs, 1):
            src = d.metadata.get("source", "uploaded_file")
            ch = d.metadata.get("chunk", "?")
            st.markdown(f"- **{i}. {src}** (chunk {ch})")


def show_generated_questions(questions: List[dict], q_type: str) -> None:
    if not questions:
        return

    st.markdown("---")
    st.subheader("Generated Questions")

    for idx, q in enumerate(questions, 1):
        with st.expander(f"Q{idx}: {q.get('question', '(no question)')}", expanded=True):
            if q_type == "MCQ":
                options = q.get("options", {})
                correct = q.get("correct", "").lower()
                user_key = f"q_{idx}_answer"
                choice = st.radio(
                    "Select your answer:",
                    options=list(options.keys()),
                    format_func=lambda k, opts=options: f"{k.upper()}) {opts[k]}",
                    key=user_key,
                    index=None,
                )
                if choice is not None:
                    if choice.lower() == correct:
                        st.success(f"Correct! The answer is **{correct.upper()}**.")
                    else:
                        st.error(f"Incorrect. The correct answer is **{correct.upper()}**.")
                    if q.get("explanation"):
                        st.info(q["explanation"])
            elif q_type == "True/False":
                correct = q.get("correct", "")
                user_key = f"q_{idx}_tf"
                choice = st.radio("Your answer:", ["True", "False"], key=user_key, index=None)
                if choice is not None:
                    if choice == correct:
                        st.success(f"Correct! The answer is **{correct}**.")
                    else:
                        st.error(f"Incorrect. The correct answer is **{correct}**.")
                    if q.get("explanation"):
                        st.info(q["explanation"])
            else:
                with st.expander("Show sample answer"):
                    st.write(q.get("sample_answer", ""))
                    if q.get("explanation"):
                        st.caption(q["explanation"])


def questions_to_download_bytes(questions: List[dict], fmt: str) -> bytes:
    if fmt == "JSON":
        return json.dumps(questions, indent=2).encode()
    df = pd.DataFrame(questions)
    return df.to_csv(index=False).encode()


# ── App layout ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="K-12 Question Generator", page_icon="🎓", layout="wide")
st.title("K-12 Question Generation System")
st.caption("Upload curriculum documents → index → generate grade-appropriate questions or ask questions. Runs locally with Ollama.")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Settings")
    st.info("Make sure Ollama is running: `ollama serve`")

    ollama_base_url = st.text_input(
        "Ollama Base URL",
        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    os.environ["OLLAMA_BASE_URL"] = ollama_base_url

    _chat_models, _embed_models = _fetch_ollama_models(
        os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    model = st.selectbox("Chat model", _chat_models, index=0)
    embed_model = st.selectbox("Embedding model", _embed_models, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    k = st.slider("Retriever top-k", 1, 10, 4, 1)

    st.markdown("---")

    if st.button("Clear Saved Index", help="Deletes the saved FAISS index from disk and resets the session."):
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH)
        st.session_state.vectorstore = None
        st.session_state.chain = None
        st.session_state.chat_log = []
        st.session_state.generated_questions = []
        st.success("Index cleared. Re-index to continue.")
        st.rerun()

    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown("- Place curriculum PDFs in the `data/` folder for auto-indexing.")
    st.markdown("- `ollama pull llama3.2` and `ollama pull nomic-embed-text`")

# ── Session state init ───────────────────────────────────────────────────────

for key, default in [
    ("vectorstore", None),
    ("chain", None),
    ("chat_log", []),
    ("generated_questions", []),
    ("gen_q_type", "MCQ"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Auto-load: saved FAISS index or bundled data/ docs ───────────────────────

if st.session_state.vectorstore is None:
    saved_vs = load_vectorstore(embed_model, ollama_base_url)
    if saved_vs is not None:
        st.session_state.vectorstore = saved_vs
        st.session_state.chain = build_cr_chain(saved_vs, model, temperature, k, ollama_base_url)
        st.info("Loaded existing index from disk. Ready to use.")
    else:
        bundled = load_bundled_docs()
        if bundled:
            with st.spinner(f"Auto-indexing {len(bundled)} chunks from bundled curriculum files…"):
                try:
                    vs = build_vectorstore(bundled, embed_model, ollama_base_url)
                    st.session_state.vectorstore = vs
                    st.session_state.chain = build_cr_chain(vs, model, temperature, k, ollama_base_url)
                    st.success(f"Auto-indexed {len(bundled)} chunks from `data/` folder.")
                except Exception as e:
                    st.warning(f"Auto-indexing failed: {e}. Upload files manually below.")

# ── Section 1: Document upload ────────────────────────────────────────────────

st.markdown("### 1) Upload Additional Curriculum Files (optional)")
st.caption("Files in the `data/` folder are indexed automatically. Use this to add extra documents.")

uploaded = st.file_uploader(
    "Drop PDFs / TXT / MD (multiple allowed).",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

col_a, col_b = st.columns(2)
with col_a:
    chunk_size = st.number_input("Chunk size", min_value=300, max_value=3000, value=1000, step=100)
with col_b:
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=150, step=10)

if st.button("Index Uploaded Documents", type="primary", use_container_width=True):
    if not uploaded:
        st.error("Please upload at least one file.")
    else:
        with st.spinner("Indexing…"):
            try:
                docs = to_documents(uploaded, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not docs:
                    st.warning("No text extracted. Ensure your files contain selectable text.")
                else:
                    vs = build_vectorstore(docs, embed_model, ollama_base_url)
                    st.session_state.vectorstore = vs
                    st.session_state.chain = build_cr_chain(vs, model, temperature, k, ollama_base_url)
                    st.success(f"Indexed {len(docs)} chunks across {len(uploaded)} file(s). Index saved to disk.")
            except Exception as e:
                st.error(f"Error during indexing: {e}")

# ── Section 2: Question generation ───────────────────────────────────────────

st.markdown("---")
st.markdown("### 2) Generate K-12 Questions")

if st.session_state.vectorstore is None:
    st.info("Index documents first (upload above or place files in the `data/` folder).")
else:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        grade_band = st.selectbox("Grade band", ["K-2", "3-5", "6-8", "9-12"], index=2)
    with col2:
        q_type = st.selectbox("Question type", ["MCQ", "Short Answer", "True/False"])
        st.session_state.gen_q_type = q_type
    with col3:
        bloom_level = st.selectbox("Bloom's level", ["Remember", "Understand", "Apply", "Analyze"], index=1)
    with col4:
        n_questions = st.slider("Number of questions", 3, 10, 5)

    if st.button("Generate Questions", type="primary", use_container_width=True):
        with st.spinner("Generating questions… (local models may take 30–90 seconds)"):
            try:
                result = question_engine.generate_questions(
                    grade_band=grade_band,
                    bloom_level=bloom_level,
                    question_type=q_type,
                    n=n_questions,
                    retrieval=True,
                    model=model,
                    ollama_url=ollama_base_url,
                    temperature=temperature,
                    vs=st.session_state.vectorstore,
                )
                if result["parse_failed"]:
                    st.warning(
                        "Question generation failed to produce valid JSON after 2 attempts. "
                        "Check the application logs or try a different model / lower temperature."
                    )
                st.session_state.generated_questions = result["questions"]
            except Exception as e:
                st.error(f"Error generating questions: {e}")

    if st.session_state.generated_questions:
        questions = st.session_state.generated_questions
        show_generated_questions(questions, st.session_state.gen_q_type)

        st.markdown("---")
        st.markdown("**Export questions**")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Download as JSON",
                data=questions_to_download_bytes(questions, "JSON"),
                file_name="generated_questions.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_dl2:
            st.download_button(
                "Download as CSV",
                data=questions_to_download_bytes(questions, "CSV"),
                file_name="generated_questions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Quick diversity summary
        with st.expander("Question coverage summary"):
            bloom_counts = {}
            for q in questions:
                bloom_counts[bloom_level] = bloom_counts.get(bloom_level, 0) + 1
            summary_df = pd.DataFrame([
                {"Grade Band": grade_band, "Question Type": st.session_state.gen_q_type,
                 "Bloom's Level": bloom_level, "Count": len(questions)}
            ])
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

# ── Section 3: Q&A chat ───────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### 3) Ask Questions About the Content")

if st.session_state.chain is None:
    st.info("Index documents first to enable Q&A.")
else:
    question = st.text_input(
        "Your question",
        placeholder="e.g., Explain how photosynthesis works for a Year-6 student…",
    )

    col_ask, col_clear = st.columns(2)
    with col_ask:
        ask_btn = st.button("Ask", use_container_width=True)
    with col_clear:
        clear_btn = st.button("Clear chat", use_container_width=True)

    if clear_btn:
        st.session_state.chat_log = []
        if st.session_state.chain and hasattr(st.session_state.chain.memory, "clear"):
            st.session_state.chain.memory.clear()
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

    if ask_btn:
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Generating grounded answer…"):
                try:
                    result = st.session_state.chain.invoke({"question": question})
                    answer = result.get("answer", "")
                    src_docs = result.get("source_documents", [])
                    st.session_state.chat_log.append(("user", question))
                    st.session_state.chat_log.append(("ai", answer, src_docs))
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

    if st.session_state.chat_log:
        st.markdown("---")
        st.subheader("Conversation")
        for role, *rest in st.session_state.chat_log:
            if role == "user":
                st.markdown(f"**You:** {rest[0]}")
            else:
                answer, src_docs = rest
                st.markdown(f"**Tutor:** {answer}")
                show_sources(src_docs)

st.markdown("---")
st.caption("Locally-hosted K-12 question generation system. No API keys required. All processing runs on-device via Ollama.")
