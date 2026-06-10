# Future Enhancements - Code Snippets

This file contains ready-to-use code snippets for adding quiz mode and FAISS persistence.

## 1. Quiz Mode - Generate 3 MCQs After Each Answer

### Add this function to `app.py`:

````python
import json
from langchain.prompts import ChatPromptTemplate

def generate_quiz(answer: str, source_docs: List[Document], llm: ChatOpenAI) -> List[dict]:
    """Generate 3 multiple-choice questions based on the answer and source context."""
    context = "\n\n".join([doc.page_content[:500] for doc in source_docs[:3]])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an educational quiz generator. Generate exactly 3 multiple-choice questions based on the provided answer and context. Return ONLY valid JSON, no markdown."),
        ("user", """Based on this answer and context, generate 3 multiple-choice questions suitable for students.

Answer: {answer}

Context:
{context}

Return a JSON array with this exact structure:
[
  {{
    "question": "Question text here?",
    "options": {{
      "a": "Option A text",
      "b": "Option B text",
      "c": "Option C text",
      "d": "Option D text"
    }},
    "correct": "a"
  }},
  ...
]

JSON only, no explanation.""")
    ])

    chain = prompt_template | llm
    response = chain.invoke({"answer": answer, "context": context})
    content = response.content.strip()

    # Clean up if wrapped in markdown code blocks
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        quiz = json.loads(content)
        return quiz if isinstance(quiz, list) else [quiz]
    except json.JSONDecodeError:
        st.warning("Failed to parse quiz. LLM response: " + content[:200])
        return []

def show_quiz(quiz_items: List[dict]):
    """Display quiz questions in Streamlit."""
    if not quiz_items:
        return
    with st.expander("📝 Quiz - Test Your Understanding", expanded=True):
        for idx, q in enumerate(quiz_items, 1):
            st.markdown(f"**Q{idx}: {q.get('question', '')}**")
            options = q.get('options', {})
            correct = q.get('correct', '').lower()
            for letter, text in options.items():
                st.markdown(f"- **{letter})** {text}")
            # Don't show correct answer initially, or use checkbox/radio
            st.caption(f"✓ Correct answer: {correct.upper()}")
            st.markdown("---")
````

### Update the answer display section in `app.py`:

Replace this section:

```python
if st.session_state.chat_log:
    st.markdown("---")
    st.subheader("💬 Conversation")
    for role, *rest in st.session_state.chat_log:
        if role == "user":
            st.markdown(f"**You:** {rest[0]}")
        else:
            answer, src_docs = rest
            st.markdown(f"**Tutor:** {answer}")
            show_sources(src_docs)
```

With:

```python
if st.session_state.chat_log:
    st.markdown("---")
    st.subheader("💬 Conversation")
    for role, *rest in st.session_state.chat_log:
        if role == "user":
            st.markdown(f"**You:** {rest[0]}")
        else:
            answer, src_docs = rest
            st.markdown(f"**Tutor:** {answer}")
            show_sources(src_docs)

            # Generate and show quiz if enabled
            quiz_enabled = st.session_state.get("quiz_mode", False)
            if quiz_enabled and st.session_state.chain:
                with st.spinner("Generating quiz questions…"):
                    llm = st.session_state.chain.llm
                    quiz_items = generate_quiz(answer, src_docs, llm)
                    if quiz_items:
                        show_quiz(quiz_items)
```

### Add quiz toggle in sidebar:

Add this in the sidebar section:

```python
quiz_mode = st.checkbox("Enable Quiz Mode", value=st.session_state.get("quiz_mode", False))
st.session_state.quiz_mode = quiz_mode
```

## 2. Persist FAISS Index to Disk

### Update `build_vectorstore` function:

```python
def build_vectorstore(docs: List[Document], embedding_model: str, index_path: str = "faiss_index") -> FAISS:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vs = FAISS.from_documents(docs, embeddings)
    # Save to disk
    vs.save_local(index_path)
    return vs
```

### Add load function:

```python
def load_vectorstore(embedding_model: str, index_path: str = "faiss_index") -> FAISS:
    """Load existing FAISS index from disk."""
    if not os.path.exists(index_path):
        return None
    try:
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vs = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vs
    except Exception as e:
        st.warning(f"Failed to load index: {e}")
        return None
```

### Update initialization section (after session state setup):

Add this right after the session state initialization:

```python
# Try to load existing index on startup
if st.session_state.vectorstore is None and os.getenv("OPENAI_API_KEY"):
    loaded_vs = load_vectorstore(embed_model)
    if loaded_vs:
        st.session_state.vectorstore = loaded_vs
        st.session_state.chain = build_cr_chain(loaded_vs, model, temperature=temperature, k=k)
        st.info("📂 Loaded existing index from disk. You can ask questions or re-index to update.")
```

### Update indexing button:

Change the indexing success section to:

```python
vs = build_vectorstore(docs, embed_model, index_path="faiss_index")
st.session_state.vectorstore = vs
st.session_state.chain = build_cr_chain(vs, model, temperature=temperature, k=k)
st.success(f"Indexed {len(docs)} chunks across {len(uploaded)} file(s). Index saved to 'faiss_index/' folder.")
```

### Add clear index button (optional):

In the sidebar or main area:

```python
if st.button("🗑️ Clear Saved Index", help="Deletes the saved FAISS index from disk"):
    import shutil
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
        st.session_state.vectorstore = None
        st.session_state.chain = None
        st.success("Index cleared!")
        st.rerun()
```

## 3. Combined: Quiz + Persistence

If you want both features, integrate all the snippets above. The index will persist across sessions, and you can enable quiz mode in the sidebar.

## Notes

- **FAISS Persistence Warning:** The `allow_dangerous_deserialization=True` flag is required because FAISS indexes can execute code. Only load indexes you created yourself!
- **Quiz Generation:** The LLM might occasionally fail to return valid JSON. Consider adding retry logic or fallback prompts.
- **Performance:** Saving large indexes can take a few seconds. Consider showing a progress indicator.
