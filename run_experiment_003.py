#!/usr/bin/env python3
"""
run_experiment_003.py — Experiment 003 runner (improved retrieval + model comparison).

Sub-experiment 003a (same model as 002, better retrieval):
  - Dual grade-band FAISS indices: F-6 (curriculum-f-6 + glossary) and
    7-10 (curriculum-7-10 + glossary).  Saved to faiss_index_f6/ and
    faiss_index_7_10/ so they can be reused by 003b without rebuilding.
  - Dynamic retrieval query built from grade band + Bloom's level each call.
  - Post-generation option shuffling so the correct answer is not
    systematically placed in any fixed position.
  - Length-balance retry: if the correct answer is noticeably longer than
    all distractors, one more LLM call is made with an explicit instruction
    to balance option lengths.
  - Same model, temperature, and 8-cell grid as experiment_002.

Sub-experiment 003b (preferred model on top of 003a retrieval):
  - Checks available Ollama models and selects from this preference list:
    llama3.1:8b > mistral:7b > gemma2:9b; falls back to llama3.2.
  - Raises temperature to 0.7 to improve within-cell diversity.
  - Reuses dual indices from 003a exactly.
  - Same 8-cell grid, 15 questions per cell, 120 total.

Both sub-experiments write to results/experiment_003/.

Usage:
    python run_experiment_003.py
    python run_experiment_003.py --ollama-url http://localhost:11434
    python run_experiment_003.py --embed-model nomic-embed-text
    python run_experiment_003.py --skip-003a   # if 003a already done
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Resolve paths so the script can be run from anywhere ─────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from question_engine import (
    DEFAULT_OLLAMA_URL,
    _build_prompt,
    _parse_response,
    build_vectorstore,
    file_to_documents,
)

# ── Output directories ────────────────────────────────────────────────────────
OUT_DIR           = HERE / "results" / "experiment_003"
FAISS_F6_PATH     = HERE / "faiss_index_f6"
FAISS_7_10_PATH   = HERE / "faiss_index_7_10"

# ── Experiment grid (identical to experiment_002) ─────────────────────────────
RETRIEVAL_CONDITIONS = [True, False]
GRADE_BANDS          = ["3-5", "9-12"]
BLOOM_LEVELS         = ["Remember", "Analyze"]
QUESTION_TYPES       = ["MCQ"]
N_PER_CELL           = 15

# ── Model constants ───────────────────────────────────────────────────────────
MODEL_003A       = "llama3.2"        # same as experiment_002
TEMPERATURE_003A = 0.2
TEMPERATURE_003B = 0.7

# Preference order for 003b.  The first model found on the running Ollama
# instance is used; llama3.2 is the final fallback.
MODEL_PREFERENCE_003B = ["llama3.1:8b", "mistral:7b", "gemma2:9b"]
MODEL_FALLBACK_003B   = "llama3.2"

# ── Retrieval constants ───────────────────────────────────────────────────────
RETRIEVAL_K     = 8
CHUNK_PREVIEW   = 600

# Threshold: correct answer is flagged as "too long" if it is more than
# 40 % longer than the mean option length.  Matches the spirit of the
# implausible_distractor check in ifw_check.py.
LENGTH_BIAS_THRESHOLD = 1.4


# ── Index routing ─────────────────────────────────────────────────────────────

def _grade_to_index(grade_band: str) -> str:
    """Return which index key a grade band maps to.

    Grade bands "K-2", "3-5", "6-8" → "f6"  (F-6 curriculum)
    Grade band  "9-12"               → "7_10" (7-10 curriculum)
    """
    return "7_10" if grade_band == "9-12" else "f6"


# ── Dynamic retrieval query ───────────────────────────────────────────────────

def build_retrieval_query(grade_band: str, bloom_level: str) -> str:
    """Build a context-specific FAISS search query.

    Instead of the fixed generic phrase used in experiment_002, the query
    now encodes the target year level and the cognitive demand so retrieved
    chunks are relevant to the exact experimental cell being generated.

    Args:
        grade_band:  One of "K-2", "3-5", "6-8", "9-12".
        bloom_level: One of "Remember", "Understand", "Apply", "Analyze".

    Returns:
        A descriptive query string for FAISS similarity_search().
    """
    year_part = {
        "K-2":  "Foundation Year 1 Year 2 early primary",
        "3-5":  "Year 3 Year 4 Year 5 upper primary",
        "6-8":  "Year 6 Year 7 Year 8 lower secondary",
        "9-12": "Year 9 Year 10 upper secondary",
    }[grade_band]

    bloom_part = {
        "Remember":   "recall identify name define key facts concepts",
        "Understand": "explain describe summarise interpret concepts",
        "Apply":      "apply use demonstrate solve calculate processes",
        "Analyze":    "compare contrast differentiate analyse cause effect",
    }[bloom_level]

    return f"science {year_part} curriculum {bloom_part}"


# ── Option shuffling ──────────────────────────────────────────────────────────

def shuffle_options(q: Dict[str, Any]) -> Dict[str, Any]:
    """Randomly reassign option labels (a/b/c/d) so the correct answer
    lands in a uniformly random position.

    The logical content of every option is preserved — only the letter
    labels are reassigned.  The "correct" field is updated to match.

    Args:
        q: Question dict with "options" ({"a":…,"b":…,"c":…,"d":…})
           and "correct" (the letter of the right option).

    Returns:
        A new dict with shuffled options and updated "correct" key.
    """
    options     = q.get("options", {})
    correct_key = q.get("correct", "").lower()

    if not options or not correct_key:
        return q

    keys        = sorted(options.keys())          # ["a","b","c","d"]
    values      = [options[k] for k in keys]
    random.shuffle(values)

    new_options = {k: v for k, v in zip(keys, values)}

    # Find which new key holds the original correct text
    correct_text = options.get(correct_key, "")
    new_correct  = correct_key  # fallback: unchanged
    for k, v in new_options.items():
        if v == correct_text:
            new_correct = k
            break

    return {**q, "options": new_options, "correct": new_correct}


# ── Length-balance check ──────────────────────────────────────────────────────

def _correct_is_longest(q: Dict[str, Any]) -> bool:
    """Return True when the correct answer is longer than every distractor.

    Uses raw character length (consistent with ifw_check._longest_option_correct).

    Args:
        q: Question dict with "options" and "correct".

    Returns:
        True if the correct option is strictly the longest of all options.
    """
    options     = q.get("options", {})
    correct_key = q.get("correct", "").lower()
    correct_txt = options.get(correct_key, "")
    distractors = [v for k, v in options.items() if k.lower() != correct_key]

    if not correct_txt or not distractors:
        return False
    return all(len(correct_txt) > len(d) for d in distractors)

_LENGTH_BALANCE_SUFFIX = (
    "\n\nIMPORTANT: All four answer options (a, b, c, d) must be roughly "
    "the same length. The correct answer must NOT be noticeably longer than "
    "the distractors. Write all options in a similar number of words."
)


# ── Model selection for 003b ──────────────────────────────────────────────────

def select_model_003b(ollama_url: str) -> Tuple[str, str]:
    """Probe the live Ollama instance and return (model_name, note).

    Tries MODEL_PREFERENCE_003B in order; falls back to MODEL_FALLBACK_003B.
    The returned note is written to every output file so results are
    unambiguous when comparing experiments later.

    Args:
        ollama_url: Base URL of the Ollama service.

    Returns:
        Tuple of (model_name, human-readable note).
    """
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5) as r:
            data     = json.loads(r.read())
        available = {m["name"] for m in data.get("models", [])}
        # Normalise: "llama3.2:latest" matches "llama3.2"
        available_base = {n.split(":")[0] for n in available}
    except Exception as exc:
        note = f"Could not reach Ollama ({exc}); falling back to {MODEL_FALLBACK_003B}"
        return MODEL_FALLBACK_003B, note

    for pref in MODEL_PREFERENCE_003B:
        base = pref.split(":")[0]
        if pref in available or base in available_base:
            return pref, f"Selected preferred model: {pref}"

    note = (
        f"None of {MODEL_PREFERENCE_003B} found on Ollama. "
        f"Available chat models: {sorted(available_base - {'nomic-embed-text', 'nomic', 'bge', 'e5', 'minilm'})}. "
        f"Falling back to {MODEL_FALLBACK_003B}."
    )
    return MODEL_FALLBACK_003B, note


# ── Index building ────────────────────────────────────────────────────────────

def build_dual_indices(embed_model: str, ollama_url: str) -> Dict[str, Any]:
    """Build (or load) the two grade-band-specific FAISS indices.

    F-6 index:   science-curriculum-content-f-6-v9.docx + science-glossary-f-10-v9.docx
    7-10 index:  science-curriculum-content-7-10-v9.docx + science-glossary-f-10-v9.docx

    Both indices are saved to disk so 003b can reuse them without re-embedding.

    Args:
        embed_model: Ollama embedding model name (e.g. "nomic-embed-text").
        ollama_url:  Base URL of the Ollama service.

    Returns:
        Dict {"f6": FAISS, "7_10": FAISS}
    """
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_url)

    indices: Dict[str, Any] = {}

    specs = {
        "f6": {
            "path":  FAISS_F6_PATH,
            "files": [
                "science-curriculum-content-f-6-v9.docx",
                "science-glossary-f-10-v9.docx",
            ],
            "label": "F-6 curriculum + glossary",
        },
        "7_10": {
            "path":  FAISS_7_10_PATH,
            "files": [
                "science-curriculum-content-7-10-v9.docx",
                "science-glossary-f-10-v9.docx",
            ],
            "label": "7-10 curriculum + glossary",
        },
    }

    for key, spec in specs.items():
        if spec["path"].exists():
            print(f"  Loading {spec['label']} index from {spec['path']} …", flush=True)
            vs = FAISS.load_local(
                str(spec["path"]),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            print(f"  Loaded ({spec['label']}).", flush=True)
        else:
            print(f"  Building {spec['label']} index …", flush=True)
            docs: List = []
            for fname in spec["files"]:
                fpath = HERE / "data" / fname
                if not fpath.exists():
                    print(f"  WARNING: {fpath} not found — skipping.", flush=True)
                    continue
                raw  = fpath.read_bytes()
                docs.extend(file_to_documents(fname, raw))
            if not docs:
                print(f"  ERROR: No documents loaded for {spec['label']}. Exiting.")
                sys.exit(1)
            print(f"  {len(docs)} chunks loaded. Embedding …", flush=True)
            vs = build_vectorstore(docs, embed_model=embed_model,
                                   ollama_url=ollama_url, save_path=spec["path"])
            print(f"  Index built and saved ({spec['label']}).", flush=True)
        indices[key] = vs

    return indices


# ── JSONL record writer ───────────────────────────────────────────────────────

def write_record(
    f,
    q:                    Dict[str, Any],
    retrieved_chunks:     List[dict],
    prompt:               str,
    model:                str,
    retrieval:            bool,
    grade_band:           str,
    bloom_level:          str,
    parse_failed:         bool,
    experiment_id:        str,
    retrieval_query:      str,
    length_balance_retry: bool,
    index_used:           str,
) -> None:
    """Serialise one question record to a JSONL line and flush immediately.

    All new experiment_003 fields are included so no data is lost when
    comparing across experiments.

    Args:
        f:                    Open file handle.
        q:                    Parsed question dict (empty dict on parse failure).
        retrieved_chunks:     List of chunk dicts used as context ([] for zero-shot).
        prompt:               The prompt text sent on the last LLM call.
        model:                Ollama model name actually used.
        retrieval:            True = RAG condition, False = zero-shot.
        grade_band:           e.g. "3-5".
        bloom_level:          e.g. "Remember".
        parse_failed:         True if both parse attempts exhausted.
        experiment_id:        "003a" or "003b".
        retrieval_query:      Dynamic query string sent to FAISS (empty for zero-shot).
        length_balance_retry: True if a length-balance retry was attempted.
        index_used:           "f6", "7_10", or "none".
    """
    record = {
        "experiment_id":        experiment_id,
        "question":             q.get("question", ""),
        "options":              q.get("options", {}),
        "answer":               q.get("correct", ""),
        "retrieved_chunks":     retrieved_chunks,
        "prompt":               prompt,
        "model":                model,
        "retrieval":            retrieval,
        "grade_band":           grade_band,
        "bloom_level":          bloom_level,
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "parse_failed":         parse_failed,
        "retrieval_query":      retrieval_query,
        "length_balance_retry": length_balance_retry,
        "index_used":           index_used,
    }
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()


# ── Per-question generation ───────────────────────────────────────────────────

def generate_one_question(
    llm,
    base_prompt:    str,
    grade_band:     str,
    bloom_level:    str,
    question_type:  str,
    verbose_prefix: str = "",
) -> Tuple[Optional[Dict[str, Any]], bool, bool]:
    """Generate and post-process a single question.

    Two attempts are made: first with base_prompt, second with a JSON-format
    clarification suffix appended.  After a successful parse, option labels
    are shuffled, and a length-balance retry is triggered if needed.

    Args:
        llm:           ChatOllama instance.
        base_prompt:   The generation prompt (single-question variant).
        grade_band:    For context in the length-balance retry.
        bloom_level:   For context in the length-balance retry.
        question_type: "MCQ", "Short Answer", or "True/False".
        verbose_prefix: Short string prepended to progress log lines.

    Returns:
        Tuple of (question_dict_or_None, parse_failed, length_balance_retry)
        question_dict has been shuffled and is ready to write.
    """
    _RETRY_SUFFIX = (
        "\n\nIMPORTANT: Your previous response could not be parsed as JSON. "
        "Return ONLY a raw JSON object. No markdown fences, no extra keys."
    )

    def _try_parse(raw: str) -> Optional[Dict[str, Any]]:
        try:
            items = _parse_response(raw)
            return items[0] if items else None
        except (json.JSONDecodeError, ValueError):
            return None

    # Attempt 1
    raw1  = llm.invoke(base_prompt).content.strip()
    q     = _try_parse(raw1)

    # Attempt 2 if needed
    if q is None:
        raw2 = llm.invoke(base_prompt + _RETRY_SUFFIX).content.strip()
        q    = _try_parse(raw2)
        if q is None:
            print(f"{verbose_prefix}  PARSE FAILED (both attempts)", flush=True)
            return None, True, False

    # ── Length-balance check ─────────────────────────────────────────────────
    length_retry = False
    if question_type == "MCQ" and _correct_is_longest(q):
        length_retry = True
        lb_prompt    = base_prompt + _LENGTH_BALANCE_SUFFIX
        raw_lb       = llm.invoke(lb_prompt).content.strip()
        q_lb         = _try_parse(raw_lb)
        if q_lb is not None:
            q = q_lb
        # If retry fails to parse, keep original q; length_retry still True

    # ── Shuffle options ──────────────────────────────────────────────────────
    if question_type == "MCQ":
        q = shuffle_options(q)

    return q, False, length_retry


# ── Core experiment runner ────────────────────────────────────────────────────

def run_sub_experiment(
    experiment_id:  str,
    model:          str,
    temperature:    float,
    indices:        Dict[str, Any],
    output_path:    Path,
    ollama_url:     str,
    model_note:     str = "",
) -> Dict[str, Any]:
    """Run one sub-experiment (003a or 003b) and write a raw JSONL file.

    Iterates over the 8-cell grid, generates N_PER_CELL questions per cell,
    writes each record to disk immediately after generation.

    Args:
        experiment_id: "003a" or "003b".
        model:         Ollama chat model to use.
        temperature:   Sampling temperature.
        indices:       Dict {"f6": FAISS, "7_10": FAISS}.
        output_path:   Path to write the raw JSONL file.
        ollama_url:    Ollama base URL.
        model_note:    Human-readable note about model selection for 003b.

    Returns:
        Summary dict with total_written, parse_failures, retry_counts.
    """
    from itertools import product
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=model,
        temperature=temperature,
        base_url=ollama_url,
        format="json",
    )

    cells        = list(product(RETRIEVAL_CONDITIONS, GRADE_BANDS,
                                BLOOM_LEVELS, QUESTION_TYPES))
    total_cells  = len(cells)
    total_expect = total_cells * N_PER_CELL

    print("=" * 68, flush=True)
    print(f"K-12 RAG Experiment — {experiment_id}", flush=True)
    if model_note:
        print(f"  Model note : {model_note}", flush=True)
    print(f"  Model      : {model}  temperature={temperature}", flush=True)
    print(f"  Grid       : {total_cells} cells × {N_PER_CELL} = {total_expect} questions", flush=True)
    print(f"  Output     : {output_path}", flush=True)
    print("=" * 68, flush=True)

    t_start           = time.time()
    total_written     = 0
    total_parse_fails = 0
    total_lb_retries  = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for cell_idx, (retrieval, grade_band, bloom_level, q_type) in enumerate(cells, 1):
            cond  = "RAG" if retrieval else "ZeroShot"
            label = f"{cond} | grade={grade_band} | bloom={bloom_level}"
            print(f"\n[Cell {cell_idx}/{total_cells}] {label}", flush=True)

            # ── Retrieval setup ───────────────────────────────────────────────
            retrieved_chunks: List[dict] = []
            context:          Optional[str] = None
            retrieval_query   = ""
            index_used        = "none"

            if retrieval:
                index_key     = _grade_to_index(grade_band)
                vs            = indices[index_key]
                retrieval_query = build_retrieval_query(grade_band, bloom_level)
                index_used    = index_key
                from langchain_core.documents import Document
                docs: List[Document] = vs.similarity_search(retrieval_query, k=RETRIEVAL_K)
                for d in docs:
                    retrieved_chunks.append({
                        "source":  d.metadata.get("source", ""),
                        "chunk":   d.metadata.get("chunk", ""),
                        "content": d.page_content[:CHUNK_PREVIEW],
                    })
                context = "\n\n".join(c["content"] for c in retrieved_chunks)

            # ── Build prompt ──────────────────────────────────────────────────
            base_prompt = _build_prompt(context, 1, q_type, grade_band, bloom_level)

            # ── Per-question loop ─────────────────────────────────────────────
            for q_idx in range(1, N_PER_CELL + 1):
                prefix = f"  Q{q_idx:02d}"
                q, parse_failed, lb_retry = generate_one_question(
                    llm, base_prompt, grade_band, bloom_level, q_type, prefix
                )

                if lb_retry:
                    total_lb_retries += 1

                if parse_failed:
                    total_parse_fails += 1
                    write_record(
                        fout, {}, retrieved_chunks, base_prompt, model,
                        retrieval, grade_band, bloom_level, True,
                        experiment_id, retrieval_query, False, index_used,
                    )
                    print(f"{prefix} FAIL (parse)", flush=True)
                    continue

                write_record(
                    fout, q, retrieved_chunks, base_prompt, model,
                    retrieval, grade_band, bloom_level, False,
                    experiment_id, retrieval_query, lb_retry, index_used,
                )
                total_written += 1
                lb_tag  = " [LB-retry]" if lb_retry else ""
                preview = q.get("question", "")[:65]
                print(f"{prefix} OK{lb_tag}  {preview}", flush=True)

    elapsed = time.time() - t_start
    print("\n" + "=" * 68, flush=True)
    print(f"Experiment {experiment_id} complete", flush=True)
    print(f"  Written       : {total_written} / {total_expect}", flush=True)
    print(f"  Parse fails   : {total_parse_fails}", flush=True)
    print(f"  LB retries    : {total_lb_retries}", flush=True)
    print(f"  Elapsed       : {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    print("=" * 68, flush=True)

    return {
        "total_written":     total_written,
        "total_expected":    total_expect,
        "parse_failures":    total_parse_fails,
        "lb_retries":        total_lb_retries,
        "elapsed_s":         elapsed,
    }


# ── Verification ──────────────────────────────────────────────────────────────

def verify_raw_output(jsonl_path: Path, experiment_id: str, log_lines: List[str]) -> bool:
    """Verify raw JSONL before scoring.

    Checks:
    1. Exactly 120 records.
    2. Every non-failed record has all required fields.
    3. Flags cells where >3 questions share the same 8-word prefix.

    Args:
        jsonl_path:    Path to the raw JSONL file.
        experiment_id: "003a" or "003b" for log labelling.
        log_lines:     List to append log text to (written later to file).

    Returns:
        True if no critical errors (count mismatch or missing fields).
    """
    REQUIRED_FIELDS = [
        "experiment_id", "question", "options", "answer",
        "retrieved_chunks", "model", "retrieval", "grade_band",
        "bloom_level", "parse_failed", "retrieval_query",
        "length_balance_retry", "index_used",
    ]

    records  = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log_lines.append(f"[{experiment_id}] JSONL parse error: {e}")

    log_lines.append(f"\n{'='*60}")
    log_lines.append(f"Verification — {experiment_id}  ({jsonl_path.name})")
    log_lines.append(f"{'='*60}")
    log_lines.append(f"Records found : {len(records)}  (expected 120)")

    ok = True

    if len(records) != 120:
        log_lines.append(f"CRITICAL: record count mismatch — expected 120, got {len(records)}")
        ok = False
    else:
        log_lines.append("Record count  : OK (120)")

    # Field check on non-failed records
    missing_field_count = 0
    for i, r in enumerate(records):
        if r.get("parse_failed"):
            continue
        for field in REQUIRED_FIELDS:
            if field not in r:
                log_lines.append(f"  Record {i}: missing field '{field}'")
                missing_field_count += 1
    if missing_field_count == 0:
        log_lines.append("Required fields: OK (all present)")
    else:
        log_lines.append(f"CRITICAL: {missing_field_count} missing fields across records")
        ok = False

    # Diversity check per cell
    from collections import defaultdict
    by_cell: Dict[str, List] = defaultdict(list)
    for r in records:
        if r.get("parse_failed"):
            continue
        cond = "RAG" if r["retrieval"] else "ZeroShot"
        key  = f"{cond}|{r['grade_band']}|{r['bloom_level']}"
        by_cell[key].append(r)

    log_lines.append("\nDiversity check (8-word prefix uniqueness per cell):")
    for cell_key, recs in sorted(by_cell.items()):
        stems   = [" ".join(r["question"].split()[:8]).lower() for r in recs]
        counter = Counter(stems)
        unique  = len(set(stems))
        dupes   = {s: c for s, c in counter.items() if c > 3}
        status  = f"{unique}/{len(recs)} unique"
        if dupes:
            flag = "  ⚠ >3 sharing same prefix"
            for s, c in dupes.items():
                log_lines.append(f"    {cell_key}: '{s}' × {c}")
        else:
            flag = ""
        log_lines.append(f"  {cell_key}: {status}{flag}")

    return ok


def verify_scored_output(jsonl_path: Path, experiment_id: str, log_lines: List[str]) -> None:
    """Verify scored JSONL after ifw_check.score_jsonl().

    Checks:
    - Every RAG record has a faithfulness judgement.
    - Every record has a flaw list (even if empty).

    Args:
        jsonl_path:    Path to the scored JSONL file.
        experiment_id: Label for log output.
        log_lines:     List to append log text to.
    """
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    log_lines.append(f"\n--- Post-scoring verification: {experiment_id} ---")
    missing_faith = 0
    missing_flaws = 0
    for i, r in enumerate(records):
        if r.get("parse_failed"):
            continue
        if r.get("retrieval") and (
            "faithfulness" not in r or r["faithfulness"].get("faithful") is None
        ):
            # Zero-shot will have faithful=None by design — only flag RAG records
            # where faithful is None (not scored) rather than intentionally absent
            if r.get("retrieval") and r.get("faithfulness", {}).get("faithful") is None:
                # This is unexpected for RAG
                missing_faith += 1
                log_lines.append(f"  Record {i}: RAG record missing faithfulness score")
        if "iwf" not in r or "flaws" not in r.get("iwf", {}):
            missing_flaws += 1
            log_lines.append(f"  Record {i}: missing IWF flaw list")

    if missing_faith == 0:
        log_lines.append("  Faithfulness: OK (all RAG records scored)")
    else:
        log_lines.append(f"  ANOMALY: {missing_faith} RAG records without faithfulness score")
    if missing_flaws == 0:
        log_lines.append("  IWF flaws: OK (all records have flaw list)")
    else:
        log_lines.append(f"  ANOMALY: {missing_flaws} records missing flaw list")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run K-12 RAG Experiment 003 (003a and 003b)."
    )
    parser.add_argument("--embed-model",  default="nomic-embed-text",
                        help="Ollama embedding model (default: nomic-embed-text)")
    parser.add_argument("--ollama-url",   default=DEFAULT_OLLAMA_URL,
                        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})")
    parser.add_argument("--skip-003a",    action="store_true",
                        help="Skip 003a if already run (reuse existing raw JSONL)")
    parser.add_argument("--skip-indices", action="store_true",
                        help="Skip index building if indices already exist on disk")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines: List[str] = [
        f"Experiment 003 Verification Log",
        f"Generated: {datetime.now().isoformat()}",
    ]

    # ── Step 1: Build / load dual indices ────────────────────────────────────
    print("\n[Step 1/6] Building dual FAISS indices …", flush=True)
    indices = build_dual_indices(args.embed_model, args.ollama_url)

    # ── Step 2: Select model for 003b ────────────────────────────────────────
    print("\n[Step 2/6] Selecting model for 003b …", flush=True)
    model_003b, model_note = select_model_003b(args.ollama_url)
    print(f"  {model_note}", flush=True)
    log_lines.append(f"\n003b model selection: {model_note}")

    # Write model selection to a small config file in the output dir
    model_cfg = {
        "experiment_id": "003b",
        "model_selected": model_003b,
        "model_note": model_note,
        "preference_list": MODEL_PREFERENCE_003B,
        "fallback": MODEL_FALLBACK_003B,
        "timestamp": datetime.now().isoformat(),
    }
    (OUT_DIR / "model_selection_003b.json").write_text(
        json.dumps(model_cfg, indent=2), encoding="utf-8"
    )

    # ── Step 3: Run 003a ─────────────────────────────────────────────────────
    path_003a_raw    = OUT_DIR / "experiment_003a.jsonl"
    path_003a_scored = OUT_DIR / "experiment_003a_scored.jsonl"

    if args.skip_003a and path_003a_raw.exists():
        print("\n[Step 3/6] Skipping 003a (--skip-003a flag, file exists).", flush=True)
    else:
        print("\n[Step 3/6] Running experiment 003a …", flush=True)
        run_sub_experiment(
            experiment_id="003a",
            model=MODEL_003A,
            temperature=TEMPERATURE_003A,
            indices=indices,
            output_path=path_003a_raw,
            ollama_url=args.ollama_url,
        )

    # ── Step 4: Verify and score 003a ────────────────────────────────────────
    print("\n[Step 4/6] Verifying and scoring 003a …", flush=True)
    ok_003a = verify_raw_output(path_003a_raw, "003a", log_lines)
    if not ok_003a:
        print("WARNING: 003a verification flagged critical issues. See verification_log.txt.", flush=True)

    from ifw_check import score_jsonl
    print("  Scoring 003a with IWF + faithfulness …", flush=True)
    score_jsonl(
        input_path=path_003a_raw,
        output_path=path_003a_scored,
        ollama_url=args.ollama_url,
        model=MODEL_003A,
    )
    verify_scored_output(path_003a_scored, "003a", log_lines)

    # ── Step 5: Run 003b ─────────────────────────────────────────────────────
    path_003b_raw    = OUT_DIR / "experiment_003b.jsonl"
    path_003b_scored = OUT_DIR / "experiment_003b_scored.jsonl"

    print("\n[Step 5/6] Running experiment 003b …", flush=True)
    run_sub_experiment(
        experiment_id="003b",
        model=model_003b,
        temperature=TEMPERATURE_003B,
        indices=indices,
        output_path=path_003b_raw,
        ollama_url=args.ollama_url,
        model_note=model_note,
    )

    print("\n[Step 5/6] Verifying and scoring 003b …", flush=True)
    ok_003b = verify_raw_output(path_003b_raw, "003b", log_lines)
    if not ok_003b:
        print("WARNING: 003b verification flagged critical issues. See verification_log.txt.", flush=True)

    print("  Scoring 003b with IWF + faithfulness …", flush=True)
    score_jsonl(
        input_path=path_003b_raw,
        output_path=path_003b_scored,
        ollama_url=args.ollama_url,
        model=model_003b,
    )
    verify_scored_output(path_003b_scored, "003b", log_lines)

    # ── Write verification log ────────────────────────────────────────────────
    log_path = OUT_DIR / "verification_log.txt"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"\nVerification log written to {log_path}", flush=True)

    # ── Step 6: Run analysis ─────────────────────────────────────────────────
    print("\n[Step 6/6] Running analysis_003.py …", flush=True)
    import analysis_003
    analysis_003.run_analysis(
        path_003a=path_003a_scored,
        path_003b=path_003b_scored,
        model_003b=model_003b,
        model_note=model_note,
        out_dir=OUT_DIR,
    )

    print("\nAll done. Output files in:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
