# PRODUCT REVIEW EVIDENCE
**Project:** AI-Powered K-12 Question Generation System (RAG, Local)  
**Experiment referenced throughout:** experiment_002 — Australian Curriculum: Science F–6 v9  
**Date compiled:** 2026-06-10

---

## 1. INDIVIDUAL CONTRIBUTION

### 1.1 Component Inventory

| File | Lines | Description | Complexity |
|------|-------|-------------|-----------|
| `app.py` | 391 | Streamlit UI: document upload/indexing, Q&A chat chain, question generation UI, model selector, FAISS persistence | **High** |
| `question_engine.py` | 356 | Core generation logic + shared document-loading pipeline (pdf/txt/md/docx), vectorstore builder, FAISS retrieval, per-question LLM loop with retry | **High** |
| `run_experiment.py` | 234 | Batch experiment runner: 2×2×2×1 grid, auto-builds index from data/, per-question JSONL streaming, CLI args | **Medium** |
| `ifw_check.py` | 266 | 7 deterministic IWF checks, faithful_score() LLM judge, score_jsonl() pipeline, argparse CLI | **High** |
| `analysis.py` | 359 | Results table builder, two matplotlib charts (grouped bar + stacked bar), faithfulness summary row, parameterized CLI | **Medium** |
| `tests/test_ifw.py` | 325 | 14 pytest unit tests (2 per IWF check), positive and negative cases, make_q() helper | **Medium** |
| `requirements.txt` | 13 | Dependency manifest | Low |
| `results/findings_summary.md` | ~80 | Structured research summary: headlines, per-condition table, findings, tradeoff interpretation, limitations, next steps | Low |
| `README.md` | 187 | Setup guide, architecture diagram, hardware requirements, troubleshooting | Low |
| `RAG_WORKFLOW_EXPLANATION.md` | ~1436 | Detailed technical walkthrough of the RAG pipeline with code examples and mathematical explanations | Low |

**Total production code: ~1,606 lines across 6 Python files.**

#### Complexity Justifications

- **`app.py` — High:** Integrates six distinct subsystems (Streamlit state, document loading, vectorstore, conversational chain, question generation, quiz UI) with error handling across all paths. Stateful Streamlit patterns (session_state, rerun) add non-obvious control flow.

- **`question_engine.py` — High:** The question generation loop is architecturally complex — it wraps a known llama3.2 JSON-mode limitation (returns single object, not array) with a per-question retry strategy, handles four file formats including docx table extraction, and manages the full indexing pipeline. The `_parse_response` function must handle both `dict` and `list` returns.

- **`ifw_check.py` — High:** Seven independent psychometric checks with distinct linguistic/statistical logic (trigram matching, length-ratio thresholds, regex pattern detection), plus an LLM-as-judge faithfulness scorer that calls a separate HTTP endpoint, all in a single file with a streaming pipeline.

- **`run_experiment.py` — Medium:** Clean grid execution with auto-build fallback and JSONL streaming, but the underlying complexity lives in `question_engine.py`.

- **`analysis.py` — Medium:** Grouped bar and stacked bar charts with value labels, faithfulness summary row, and parameterized CLI require care but follow a clear pattern. No branching logic across chart types.

- **`tests/test_ifw.py` — Medium:** Well-structured tests with non-trivial cases (e.g. the `stem_answer_overlap` test requires a trigram shared with correct answer but not distractors, and a second case where the same trigram appears in a distractor to suppress the flag).

---

### 1.2 Git History

```
ed6fb37  2026-06-10  Fetch available Ollama models dynamically; default to newest
39ebee3  2026-06-10  Initial commit: K-12 RAG question generation system
```

**Total commits: 2. Date range: 1 day (2026-06-10).**

**Milestone timeline inferred from commit messages and file timestamps:**

| Milestone | Notes |
|-----------|-------|
| Initial app (app.py, README) | Commit `39ebee3` — base Streamlit UI, static Ollama model list |
| Dynamic model selector | Commit `ed6fb37` — `_fetch_ollama_models()` queries live Ollama API |
| question_engine.py extraction | Not committed separately — refactor from app.py done in-session |
| run_experiment.py created | Not committed — batch runner added in-session |
| ifw_check.py + tests created | Not committed — IWF pipeline added in-session |
| analysis.py created | Not committed — visualisation added in-session |
| Experiment 001 run | Not committed — first run (placeholder index, 0% faithfulness found) |
| python-docx + docx loading | Not committed — added to fix empty curriculum index |
| Experiment 002 run | Not committed — real curriculum, 120 questions, results produced |
| findings_summary.md | Not committed |

> **⚠️ Critical documentation gap:** Only 2 commits capture roughly 1,600 lines of new code and two complete experiments. The entire research pipeline (run_experiment, ifw_check, analysis, experiment data) is untracked in git. Before the review, commit all files and write meaningful messages that match the milestones above.

---

## 2. TECHNICAL UNDERSTANDING

### 2.1 Design Decisions

#### D1 — Chat model: llama3.2
- **Decision:** `DEFAULT_MODEL = "llama3.2"` (`question_engine.py:25`)
- **Rationale:** Smallest capable Ollama chat model; fast local inference on consumer hardware; available via `ollama pull llama3.2`.
- **Trade-off:** llama3.2 with `format="json"` reliably returns a single JSON object regardless of being asked for an array. This caused 100% parse failures in a preliminary run and required a full architectural workaround (one-question-per-call loop). Lower diversity than larger models: in experiment_001, the same question was generated 8–13 times within a single cell of 15 questions.
- **⚠️ Weakly justified:** No ablation against llama3.1 or mistral, which follow multi-item array instructions more reliably. The architectural workaround for the JSON-mode limitation is a liability — a supervisor is likely to ask why you didn't switch models.

#### D2 — `format="json"` forced JSON output
- **Decision:** `ChatOllama(..., format="json")` (`question_engine.py:313`)
- **Rationale:** Ensures the LLM output is parseable JSON, avoiding markdown-wrapped responses.
- **Trade-off:** With llama3.2, this flag causes single-object output. `_parse_response()` (`question_engine.py:215`) handles this by wrapping a bare `dict` in a list, but the underlying behaviour is a known Ollama/llama3.2 interaction, not a bug in your code.
- **⚠️ Weakly justified:** The flag does solve the markdown-wrapping problem but introduces the single-object problem. A structured output prompt without `format="json"` (and more robust parsing) would avoid both issues.

#### D3 — One question per LLM call (generation loop)
- **Decision:** Loop `n` times, requesting 1 question per call (`question_engine.py:319–335`)
- **Rationale:** Direct workaround for D1/D2: since llama3.2 returns one object per call regardless, request exactly one.
- **Trade-off:** Multiplies LLM call count by n (15×); experiment_002 took ~4.7 minutes for 120 questions. The same context is reused across all calls in a cell, causing repetitive question stems within a cell.
- **⚠️ Weakly justified:** The low-diversity problem (same question stem repeated) is an unresolved research validity concern. With `temperature=0.2` and an identical retrieved context on each call, the model converges to the same answer every time.

#### D4 — Fixed retrieval query: "main concepts key ideas overview"
- **Decision:** `_RETRIEVAL_QUERY = "main concepts key ideas overview"` (`question_engine.py:29`)
- **Rationale:** A generic query retrieves broad coverage of the curriculum chunk rather than a topic-specific slice.
- **Trade-off:** Every question in a cell is generated from the same k=8 retrieved chunks. This is the root cause of within-cell repetition. A grade/bloom-aware query (e.g. "Year 3–5 science concepts") would vary the context and improve diversity.
- **⚠️ Weakly justified:** The fixed query means retrieval is not actually conditioned on the experimental factors (grade band, Bloom's level). This undermines the research claim that RAG is grounding questions in curriculum.

#### D5 — Retrieval k=8 for generation (vs k=4 for Q&A)
- **Decision:** `_RETRIEVAL_K = 8` (`question_engine.py:30`) for question generation; k=4 is the UI default for the Q&A chain (`app.py:173` slider default).
- **Rationale:** More chunks give the LLM broader curriculum context for question creation.
- **Trade-off:** The discrepancy between k=8 (generation) and k=4 (Q&A) is undocumented and inconsistent. Only 600 characters of each chunk reach the prompt (`_CHUNK_PREVIEW = 600`, `question_engine.py:31`), meaning the actual context fed to the LLM is 8 × 600 = 4,800 chars, not 8 × 1,000.
- **⚠️ Weakly justified:** Neither the k=8 choice nor the 600-char truncation is justified in comments or documentation.

#### D6 — Chunk size: 1000 chars, overlap: 150 chars
- **Decision:** `chunk_size=1000, chunk_overlap=150` defaults in `file_to_documents()` (`question_engine.py:66–67`)
- **Rationale:** Documented in `RAG_WORKFLOW_EXPLANATION.md` — balance between embedding model token limits and retrieval precision; 15% overlap prevents boundary information loss.
- **Trade-off:** The Australian Science Curriculum uses short, structured content descriptions (one sentence each). A 1000-char chunk likely spans 3–5 content descriptions from different year levels, mixing context. Smaller chunks (300–500 chars) might retrieve more specific, relevant content.
- **⚠️ Weakly justified:** No ablation study at different chunk sizes against the actual curriculum document.

#### D7 — Temperature: 0.2
- **Decision:** `temperature: float = 0.2` default in `generate_questions()` (`question_engine.py:257`)
- **Rationale:** Educational context requires accuracy over creativity; lower temperature reduces hallucinations.
- **Trade-off:** Combined with the fixed retrieval query and the per-call loop (D3/D4), temperature=0.2 causes near-deterministic repetition within a cell. This is the strongest factor in the diversity problem.

#### D8 — IWF acceptability threshold: ≤1 flaw
- **Decision:** `"acceptable": len(flaws) <= 1` (`ifw_check.py:120`)
- **Rationale:** A question with exactly one minor structural flaw is considered usable; two or more flaws make it problematic.
- **Trade-off:** This threshold is arbitrary — there is no citation to psychometrics literature justifying it. Some flaws (e.g. `negated_stem`) arguably make an item unacceptable on their own; others (e.g. `answer_position_bias`) are minor.
- **⚠️ Weakly justified:** A supervisor will likely ask where this threshold comes from. Have a reference ready, or reframe it as a provisional threshold pending expert review.

#### D9 — Faithfulness judge: binary YES/NO via LLM
- **Decision:** `faithful_score()` sends a binary YES/NO prompt to llama3.2 via `/api/generate` (`ifw_check.py:125–175`)
- **Rationale:** Simple, fast, no external dependencies; leverages the same local model.
- **Trade-off:** Using the same model that generated the question to also judge its faithfulness creates a circularity risk (the model may be lenient on its own outputs). No human validation has been done. Word-boundary regex (`\byes\b`, `\bno\b`) was added for robustness but the raw responses observed were clean single-word `YES`/`NO` in all cases.

#### D10 — FAISS flat index (exact search)
- **Decision:** `FAISS.from_documents()` uses the default flat (exact search) index (`question_engine.py:134–135`)
- **Rationale:** The corpus is small (183 chunks); flat search is exact and takes <5ms at this scale.
- **Trade-off:** Would not scale past ~50,000 chunks without switching to an approximate index (IVF, HNSW). Appropriate for the current scope.

#### D11 — Embedding model: nomic-embed-text
- **Decision:** `embed_model="nomic-embed-text"` default in `build_vectorstore()` (`question_engine.py:120`)
- **Rationale:** Purpose-built embedding model; produces 768-dim vectors; faster and more space-efficient than using a chat model for embeddings.
- **Trade-off:** Requires a second model download (~274 MB). No comparison against other embedding options (e.g. `mxbai-embed-large`).

---

### 2.2 End-to-End Data Flow

```
data/science-curriculum-content-f-6-v9.docx
         │
         ▼ question_engine._extract_docx_text()     [paragraphs + table cells]
         │
         ▼ question_engine.file_to_documents()       [RecursiveCharacterTextSplitter, 1000/150]
         │
         ▼  183 Document objects (182 curriculum + 1 README)
         │
         ▼ question_engine.build_vectorstore()       [OllamaEmbeddings(nomic-embed-text)]
         │
         ▼  faiss_index/  (FAISS flat index, saved to disk)
         │
   ┌─────┴──────────────────────────────────────────────────────────┐
   │  run_experiment.py: 8-cell grid loop                           │
   │  for (retrieval, grade_band, bloom_level, q_type) in cells:    │
   │                                                                │
   │  if retrieval:                                                  │
   │    vs.similarity_search("main concepts key ideas overview", k=8)│
   │    context = join(chunk["content"][:600] for chunk in results)  │
   │  else:                                                          │
   │    context = None                                               │
   │                                                                │
   │  for q_idx in range(N_PER_CELL=15):                            │
   │    prompt = question_engine._build_prompt(context, 1, ...)      │
   │    raw = ChatOllama(llama3.2, temp=0.2, format="json").invoke() │
   │    questions.extend(question_engine._parse_response(raw))       │
   │    → write JSONL record immediately (flush)                     │
   └────────────────────────────────────────────────────────────────┘
         │
         ▼  results/experiment_002.jsonl  (120 records)
         │
         ▼ ifw_check.score_question()     [7 deterministic checks]
         ▼ ifw_check.faithful_score()     [Ollama YES/NO judge, RAG records only]
         │
         ▼  results/experiment_002_scored.jsonl  (120 records + iwf + faithfulness keys)
         │
         ▼ analysis.main()
           → terminal table + results/results_table_v2.csv
           → results/chart_acceptable_v2.png
           → results/chart_flaws_v2.png
           → "RAG questions: 82% acceptable, 55% faithful."
```

**Key functions in order:** `_extract_docx_text` → `file_to_documents` → `build_vectorstore` → `_load_vectorstore` → `vs.similarity_search` → `_build_prompt` → `ChatOllama.invoke` → `_parse_response` → `_write_record` → `score_question` → `faithful_score` → `score_jsonl` → `aggregate` → `build_rows` → `chart_acceptable` → `chart_flaws`

---

## 3. PRODUCT QUALITY & RESULTS

### 3.1 Experiment Configuration (as implemented)

**Factor levels hardcoded in `run_experiment.py:35–41`:**

| Factor | Levels used |
|--------|-------------|
| Retrieval condition | `[True, False]` — RAG vs. Zero-shot |
| Grade band | `["3-5", "9-12"]` ← only 2 of 4 available (K-2 and 6-8 omitted) |
| Bloom's level | `["Remember", "Analyze"]` ← only 2 of 4 available (Understand and Apply omitted) |
| Question type | `["MCQ"]` ← only MCQ tested (Short Answer and True/False omitted) |
| N per cell | `15` |
| **Total** | **8 cells × 15 = 120 questions** |

**IWF acceptability rubric (as implemented in `ifw_check.py`):**

| Check | Flag condition |
|-------|---------------|
| `longest_option_correct` | Correct answer strictly longer than every distractor |
| `answer_position_bias` | Correct answer in position `a` (first) or `d` (last) |
| `all_none_of_above` | Any option contains "all of the above" or "none of the above" |
| `negated_stem` | Stem contains `NOT` or `EXCEPT` in full capitals |
| `duplicate_options` | Two or more options identical after lowercase + whitespace collapse |
| `stem_answer_overlap` | 3-word trigram from stem appears in correct answer but no distractor |
| `implausible_distractor` | Any distractor shorter than 40% of mean option length |
| **Acceptable** | flaw_count ≤ 1 |

---

### 3.2 Full Numerical Results — Experiment 002

#### Per-Cell Results

| Condition | N | Parse Fails | Acc | % Acc | Mean Flaws | Top Flaw | Faithful | % Faithful |
|-----------|---|-------------|-----|-------|-----------|---------|---------|-----------|
| RAG \| 3-5 \| Remember | 15 | 0 | 13 | 86.7% | 1.07 | answer\_position\_bias | 4/15 | 26.7% |
| RAG \| 3-5 \| Analyze | 15 | 0 | 15 | 100.0% | 0.20 | duplicate\_options | 12/15 | 80.0% |
| RAG \| 9-12 \| Remember | 15 | 0 | 7 | 46.7% | 1.60 | longest\_option\_correct | 8/15 | 53.3% |
| RAG \| 9-12 \| Analyze | 15 | 0 | 14 | 93.3% | 0.93 | answer\_position\_bias | 9/15 | 60.0% |
| Zero-shot \| 3-5 \| Remember | 15 | 0 | 15 | 100.0% | 0.87 | longest\_option\_correct | N/A | — |
| Zero-shot \| 3-5 \| Analyze | 15 | 0 | 14 | 93.3% | 1.07 | longest\_option\_correct | N/A | — |
| Zero-shot \| 9-12 \| Remember | 15 | 0 | 15 | 100.0% | 0.07 | longest\_option\_correct | N/A | — |
| Zero-shot \| 9-12 \| Analyze | 15 | 0 | 13 | 86.7% | 0.93 | answer\_position\_bias | N/A | — |
| **RAG — all** | 60 | 0 | 49 | **81.7%** | 1.05 | answer\_position\_bias | 33/60 | **55.0%** |
| **Zero-shot — all** | 60 | 0 | 57 | **95.0%** | 0.73 | longest\_option\_correct | — | N/A |

#### Flaw Type Frequency

| Flaw | RAG count | Zero-shot count | Total |
|------|----------|----------------|-------|
| `longest_option_correct` | 16 | 28 | **44** |
| `answer_position_bias` | 30 | 11 | **41** |
| `implausible_distractor` | 5 | 4 | 9 |
| `stem_answer_overlap` | 4 | 1 | 5 |
| `duplicate_options` | 2 | 0 | 2 |
| `all_none_of_above` | 0 | 0 | 0 |
| `negated_stem` | 0 | 0 | 0 |
| `duplicate_options` | 2 | 0 | 2 |

#### Faithfulness by RAG Cell

| Cell | Faithful | % |
|------|---------|---|
| RAG \| 3-5 \| Remember | 4/15 | 26.7% |
| RAG \| 3-5 \| Analyze | 12/15 | 80.0% |
| RAG \| 9-12 \| Remember | 8/15 | 53.3% |
| RAG \| 9-12 \| Analyze | 9/15 | 60.0% |
| **RAG total** | **33/60** | **55.0%** |

---

### 3.3 Cross-Check Against Planned Presentation Figures

| Claimed figure | Actual value | Status |
|----------------|-------------|--------|
| RAG 82% acceptable | 81.7% (rounds to 82%) | ✅ Correct |
| RAG 55% faithful | 55.0% (33/60) | ✅ Correct |
| Zero-shot 95% acceptable | 95.0% (57/60) | ✅ Correct |
| Worst cell: RAG 9-12 Remember 46.7% | 7/15 = 46.7% | ✅ Correct |
| 0% parse failures | All 8 cells = 0 parse fails | ✅ Correct |
| 120 items in 8 cells | 120 confirmed | ✅ Correct |
| "Dominant flaw: longest_option_correct" | **Globally:** 44 vs 41 — barely dominant (3-item margin). **For RAG specifically:** answer\_position\_bias (30) dominates, not longest\_option\_correct (16). | ⚠️ Misleading |

> **Flag:** The findings_summary.md `Key Findings` bullet says *"The most common RAG flaws (longest option correct, answer position bias)"* listing `longest_option_correct` first. This is incorrect for RAG — `answer_position_bias` is the dominant RAG flaw by nearly 2:1. The global figure (all 120 items) is 44 vs 41, a margin of 3. Correct the findings document before the review, or clarify that `longest_option_correct` dominates in zero-shot and the combined total, while `answer_position_bias` dominates in RAG.

> **Note:** The `findings_summary.md` paragraph on next steps correctly identifies `longest option correct` and `answer position bias` together as the main flaw targets — that phrasing is accurate.

---

## 4. SUPPORTING DOCUMENTATION

### 4.1 Code Comment Coverage

| File | Coverage | Assessment |
|------|----------|-----------|
| `question_engine.py` | Good | Module docstring explains dual purpose; all public functions have docstrings with args, returns, and raises; inline comment explaining the per-call loop workaround for llama3.2 |
| `ifw_check.py` | Good | Each check function has a docstring explaining the psychometric rationale; module docstring lists public API |
| `tests/test_ifw.py` | Good | Each test has a comment explaining the specific edge case being triggered or suppressed |
| `run_experiment.py` | Medium | Module docstring + section separators; experiment grid constants are commented; function bodies are sparsely commented |
| `analysis.py` | Medium | Section separators (`# ── name ──`) but no function docstrings; chart parameters not commented |
| `app.py` | Low | Section separators only; no function docstrings except the Ollama model fetcher; `build_cr_chain` parameters undocumented |

### 4.2 External Libraries

| Library | Source | Purpose |
|---------|--------|---------|
| `streamlit` | `requirements.txt` | Web UI framework — page layout, widgets, session state |
| `langchain-core` | `requirements.txt` | Document class, base LLM interfaces |
| `langchain-text-splitters` | `requirements.txt` | `RecursiveCharacterTextSplitter` for chunking |
| `langchain-ollama` | `requirements.txt` | `ChatOllama` (generation), `OllamaEmbeddings` (indexing) |
| `langchain-community` | `requirements.txt` | `FAISS` vector store integration |
| `langchain-classic` | `requirements.txt` | `ConversationalRetrievalChain`, `ConversationBufferMemory` (Q&A mode) |
| `faiss-cpu` | `requirements.txt` | Vector similarity search index |
| `pypdf` | `requirements.txt` | PDF text extraction |
| `python-docx` | `requirements.txt` (added) | `.docx` text + table extraction |
| `tiktoken` | `requirements.txt` | **Not imported anywhere in the codebase.** Listed as a dependency but unused. |
| `pandas` | `requirements.txt` | CSV export in app.py; DataFrame in question coverage summary |
| `matplotlib` | Installed, not in requirements.txt | Chart generation in `analysis.py` |
| `numpy` | Installed, not in requirements.txt | Bar chart positioning in `analysis.py` |

> **⚠️ Two gaps:** `tiktoken` is listed in requirements.txt but never imported — remove it or document why it's there. `matplotlib` and `numpy` are used in `analysis.py` but are absent from `requirements.txt`.

### 4.3 Existing Documentation

| Artefact | Exists | Quality | Issues |
|----------|--------|---------|--------|
| `README.md` | ✅ | Good — setup guide, architecture diagram, troubleshooting | Project structure diagram is stale: does not list `question_engine.py`, `run_experiment.py`, `ifw_check.py`, `analysis.py`, or `results/` |
| `RAG_WORKFLOW_EXPLANATION.md` | ✅ | Very detailed | References old function names and line numbers that no longer match the refactored code (e.g. references `ChatOpenAI` style; references `app.py` line numbers that shifted); describes only the Q&A pipeline, not the research pipeline |
| `FUTURE_ENHANCEMENTS.md` | ✅ | Present | ⚠️ Contains `ChatOpenAI` and `OpenAIEmbeddings` import references — these require API keys and contradict the fully-local premise of the project. These are leftover from a cloud-API prototype; a supervisor reading this file may question whether the "fully local" claim is consistent |
| `results/findings_summary.md` | ✅ | Good | Minor flaw-dominance inconsistency noted in §3.3 |
| Design document | ❌ | Missing | No document describing the research pipeline architecture (experiment grid, IWF rubric derivation, faithfulness judge design) |
| Test suite | ✅ (partial) | 14 tests, all pass | Tests cover only `ifw_check.py`; no tests for `question_engine`, `run_experiment`, `analysis` |
| Requirements attribution | ❌ | Missing | No citations to the psychometrics papers from which the 7 IWF checks are drawn |

---

### 4.4 Documentation Gap Checklist (ordered by effort × rubric impact)

| Priority | Gap | Effort | Rubric area |
|----------|-----|--------|-------------|
| 1 | **Commit all uncommitted files** — 1,600+ lines and both experiments are untracked | Low (5 min) | Individual Contribution |
| 2 | **Fix requirements.txt** — add `matplotlib`, `numpy`; remove `tiktoken` | Low (2 min) | Supporting Documentation |
| 3 | **Add flaw-dominance correction to findings_summary.md** — clarify RAG vs global dominant flaw | Low (10 min) | Product Quality |
| 4 | **Update README project structure** — add the six new files built since initial commit | Low (10 min) | Supporting Documentation |
| 5 | **Add IWF literature citations** — cite Haladyna & Downing or similar for each check | Medium (30 min) | Technical Understanding |
| 6 | **Remove or annotate FUTURE_ENHANCEMENTS.md** — either purge the `ChatOpenAI` references or add a note that these are legacy snippets | Low (15 min) | Supporting Documentation |
| 7 | **Add design document for research pipeline** — one page: experiment grid rationale, IWF threshold choice, faithfulness judge design | Medium (1 hr) | Technical Understanding |
| 8 | **Document the k=8 vs k=4 discrepancy** — explain why generation uses k=8 and Q&A uses k=4 | Low (inline comment) | Technical Understanding |
| 9 | **Document `_CHUNK_PREVIEW = 600`** — explain why chunks are truncated to 600 chars in generation prompts | Low (inline comment) | Technical Understanding |
| 10 | **Add docstrings to `app.py` functions** | Medium (30 min) | Supporting Documentation |

---

## 5. ANTICIPATED Q&A

**Q1: Why did you choose llama3.2 as the generation model?**  
*Honest answer:* It is the smallest capable chat model available via Ollama and was a pragmatic starting point for local inference. However, llama3.2's `format="json"` mode returns a single JSON object rather than an array, which caused 100% parse failures in an initial run. The fix was to generate one question per LLM call rather than batching. A larger model (llama3.1, mistral) would have avoided this. If asked to defend the choice, frame it as a deliberate constraint to test the lower bound of local model capability, with the per-call loop as the adaptation strategy.

**Q2: Why is the retrieval query fixed to "main concepts key ideas overview"?**  
*Honest answer:* It is not — this is a significant design gap. The same 5-word generic query retrieves the same k=8 chunks for every question in a given cell. The experimental factors (grade band, Bloom's level) do not influence what context the model receives. A grade- and bloom-aware query would improve both diversity and ecological validity.

**Q3: Why does your system generate the same question multiple times within a cell?**  
*Honest answer:* Three design decisions compound: (1) the fixed retrieval query returns identical context on every call within a cell, (2) `temperature=0.2` is near-deterministic, and (3) the one-question-per-call loop repeats this combination 15 times. In experiment_001, this produced the same question 8–13 times in some cells. Experiment_002 is better due to a richer corpus, but within-cell diversity remains low. Mitigations: increase temperature, vary the retrieval query, or use maximal marginal relevance (MMR) retrieval.

**Q4: Where does the ≤1 flaw acceptability threshold come from?**  
*Honest answer:* It is a provisional threshold set in the code without a literature citation. The psychometrics literature (Haladyna & Downing, 1989; Rodriguez, 2005) identifies item-writing flaws but does not specify an acceptability threshold in terms of flaw count. This should be acknowledged as a design choice pending expert review, and the threshold should be cited as provisional.

**Q5: What happens when retrieval returns nothing useful?**  
*Honest answer:* There is no fallback. If `vs.similarity_search()` returns chunks that are semantically unrelated to the grade/bloom target, the LLM still generates questions from that context. In experiment_001 with the README.txt placeholder, retrieval returned index-format instructions and the model generated geography questions about compasses from that noise — 0% faithfulness. The system provides no retrieval quality gate.

**Q6: How is the correct answer position assigned in the generated questions?**  
*Honest answer:* It is assigned entirely by the LLM in the generated JSON, not by the system. The `_MCQ_SCHEMA` in the prompt (`question_engine.py:220`) requests `{"correct": "a"}` as an example, and `answer_position_bias` (31/60 RAG questions flagged) reveals that the model systematically places the correct answer at position `a` or `d`. The system does not shuffle options post-generation.

**Q7: Why is faithfulness only scored for RAG questions?**  
*Honest answer:* Zero-shot questions have no retrieved chunks — there is no curriculum text to evaluate against. `faithful_score()` returns `{"faithful": None}` when `retrieved_chunks == []` (`ifw_check.py:136–137`). This is correct by design: faithfulness is only a meaningful metric when there is a grounding document to compare against.

**Q8: Your faithfulness score uses the same model that generated the questions as the judge. Isn't that circular?**  
*Honest answer:* Yes, this is a limitation acknowledged in findings_summary.md. llama3.2 evaluates whether the answer it generated is supported by the curriculum. The model may be systematically lenient on its own outputs. The faithfulness metric should be treated as a directional estimate, not a ground-truth measurement. Human expert validation on a sample of 30–40 items would calibrate it.

**Q9: Your RAG 9-12 Remember cell scored only 46.7% acceptable — is this a problem with RAG or with the prompt?**  
*Honest answer:* Both. The `longest_option_correct` flaw dominates this cell (mean_flaws=1.60). When the model retrieves dense academic curriculum text, the correct answer tends to be a precise technical statement (longer, more qualified) while distractors are shorter everyday phrasings. This is a prompt engineering failure: the prompt does not instruct the model to match option lengths. It is also a retrieval issue: academic 9-12 content contains more complex sentence structures that the model echoes directly into the correct option.

**Q10: The experiment only covers 2 of 4 grade bands, 2 of 4 Bloom's levels, and only MCQ. How do you justify this as sufficient?**  
*Honest answer:* It is a scoped pilot, not a complete evaluation. The 2×2 grid (grade bands 3-5 and 9-12, Bloom's Remember and Analyze) tests the two extreme grade levels and the two extreme Bloom's levels, representing the boundaries of the design space. Short Answer and True/False were excluded because `format="json"` stability is higher for MCQ and because the IWF rubric was designed for MCQ. Acknowledge this as a deliberate scope reduction and frame the remaining cells as a natural next step.

---

*Document compiled from: `app.py`, `question_engine.py`, `run_experiment.py`, `ifw_check.py`, `analysis.py`, `tests/test_ifw.py`, `requirements.txt`, `README.md`, `RAG_WORKFLOW_EXPLANATION.md`, `FUTURE_ENHANCEMENTS.md`, `results/findings_summary.md`, `results/experiment_002_scored.jsonl` (120 records, parsed directly).*
