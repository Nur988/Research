#!/usr/bin/env python3
"""
run_experiment.py — batch experiment runner for the K-12 RAG question generation study.

Runs a 2×2×2×1 grid (retrieval × grade_band × bloom_level × question_type),
writes one JSONL record per question immediately to the output file,
and prints a live progress + final summary.

If no FAISS index exists, automatically builds one from data/ using the same
document-loading pipeline as app.py (supports .pdf, .txt, .md, .docx).

Usage (from rag-tutor/ directory):
    python run_experiment.py
    python run_experiment.py --output results/experiment_002.jsonl
    python run_experiment.py --model llama3 --embed-model nomic-embed-text
    python run_experiment.py --ollama-url http://localhost:11434
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

# ── Resolve paths relative to this script so it can be run from anywhere ────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))  # ensure question_engine is importable

FAISS_INDEX_PATH = HERE / "faiss_index"
RESULTS_DIR      = HERE / "results"

# ── Fixed experiment grid ────────────────────────────────────────────────────
RETRIEVAL_CONDITIONS = [True, False]
GRADE_BANDS          = ["3-5", "9-12"]
BLOOM_LEVELS         = ["Remember", "Analyze"]
QUESTION_TYPES       = ["MCQ"]
N_PER_CELL           = 15
# 2 × 2 × 2 × 1 = 8 cells × 15 = 120 questions total


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_vectorstore(embed_model: str, ollama_url: str):
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_url)
    return FAISS.load_local(
        str(FAISS_INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _build_vectorstore_from_data(embed_model: str, ollama_url: str):
    """Build and save a FAISS index from data/ using the shared loading pipeline."""
    from question_engine import load_docs_from_data_dir, build_vectorstore
    print("Loading documents from data/ …", flush=True)
    docs = load_docs_from_data_dir()
    if not docs:
        print("ERROR: No documents found in data/. Add curriculum files (.pdf, .txt, .md, .docx) and retry.")
        sys.exit(1)
    print(f"Loaded {len(docs)} chunks. Building FAISS index …", flush=True)
    vs = build_vectorstore(docs, embed_model=embed_model, ollama_url=ollama_url, save_path=FAISS_INDEX_PATH)
    print(f"Index built and saved to {FAISS_INDEX_PATH}\n", flush=True)
    return vs


def _cell_label(retrieval: bool, grade_band: str, bloom_level: str, question_type: str) -> str:
    cond = "RAG" if retrieval else "ZeroShot"
    return f"{cond} | grade={grade_band} | bloom={bloom_level} | type={question_type}"


def _write_record(
    f,
    q: dict,
    retrieved_chunks: list,
    prompt: str,
    model: str,
    retrieval: bool,
    grade_band: str,
    bloom_level: str,
    parse_failed: bool,
) -> None:
    """Serialise one question to a JSONL line and flush immediately."""
    record = {
        "question":         q.get("question", ""),
        "options":          q.get("options", {}),
        "answer":           q.get("correct", ""),
        "retrieved_chunks": retrieved_chunks,
        "prompt":           prompt,
        "model":            model,
        "retrieval":        retrieval,
        "grade_band":       grade_band,
        "bloom_level":      bloom_level,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "parse_failed":     parse_failed,
    }
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run K-12 RAG question generation experiment and write results to JSONL."
    )
    parser.add_argument("--model",       default="llama3.2",
                        help="Ollama chat model (default: llama3.2)")
    parser.add_argument("--embed-model", default="nomic-embed-text",
                        help="Ollama embedding model (default: nomic-embed-text)")
    parser.add_argument("--ollama-url",  default="http://localhost:11434",
                        help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--output",      default=str(RESULTS_DIR / "experiment_001.jsonl"),
                        help="Output JSONL path (default: results/experiment_001.jsonl)")
    args = parser.parse_args()

    from question_engine import generate_questions

    OUTPUT_FILE = Path(args.output)

    # ── Pre-flight checks ────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    vs = None
    if True in RETRIEVAL_CONDITIONS:
        if not FAISS_INDEX_PATH.exists():
            print("No FAISS index found — building from data/ …", flush=True)
            try:
                vs = _build_vectorstore_from_data(args.embed_model, args.ollama_url)
            except Exception as exc:
                print(f"ERROR: Could not build index: {exc}")
                sys.exit(1)
        else:
            print(f"Loading FAISS index from {FAISS_INDEX_PATH} …", flush=True)
            try:
                vs = _load_vectorstore(args.embed_model, args.ollama_url)
            except Exception as exc:
                print(f"ERROR: Could not load FAISS index: {exc}")
                sys.exit(1)
            print("Index loaded.\n", flush=True)

    # ── Experiment header ────────────────────────────────────────────────────
    cells = list(product(RETRIEVAL_CONDITIONS, GRADE_BANDS, BLOOM_LEVELS, QUESTION_TYPES))
    total_cells    = len(cells)
    total_expected = total_cells * N_PER_CELL
    exp_name       = OUTPUT_FILE.stem  # e.g. "experiment_002"

    print("=" * 64)
    print(f"K-12 RAG Experiment — {exp_name}")
    print(f"  Grid    : {total_cells} cells × {N_PER_CELL} questions = {total_expected} expected")
    print(f"  Model   : {args.model}")
    print(f"  Embed   : {args.embed_model}")
    print(f"  Output  : {OUTPUT_FILE}")
    print("=" * 64)

    # ── Run ──────────────────────────────────────────────────────────────────
    t_start           = time.time()
    total_written     = 0
    total_parse_fails = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for cell_idx, (retrieval, grade_band, bloom_level, question_type) in enumerate(cells, 1):
            label = _cell_label(retrieval, grade_band, bloom_level, question_type)
            print(f"\n[Cell {cell_idx}/{total_cells}] {label}", flush=True)

            # Call generate_questions with its exact current signature
            try:
                result = generate_questions(
                    grade_band=grade_band,
                    bloom_level=bloom_level,
                    question_type=question_type,
                    n=N_PER_CELL,
                    retrieval=retrieval,
                    model=args.model,
                    ollama_url=args.ollama_url,
                    vs=vs if retrieval else None,
                )
            except Exception as exc:
                total_parse_fails += 1
                print(f"  CELL ERROR: {exc}", flush=True)
                continue

            if result["parse_failed"]:
                total_parse_fails += 1
                # Write one failure record so the cell is represented in the JSONL
                _write_record(
                    f,
                    q={},
                    retrieved_chunks=result["retrieved_chunks"],
                    prompt=result["raw_prompt"],
                    model=result["model"],
                    retrieval=retrieval,
                    grade_band=grade_band,
                    bloom_level=bloom_level,
                    parse_failed=True,
                )
                print("  Q-- PARSE FAILED (both attempts exhausted)", flush=True)
                continue

            # Write each question individually and flush immediately
            for q_idx, q in enumerate(result["questions"], 1):
                _write_record(
                    f,
                    q=q,
                    retrieved_chunks=result["retrieved_chunks"],
                    prompt=result["raw_prompt"],
                    model=result["model"],
                    retrieval=retrieval,
                    grade_band=grade_band,
                    bloom_level=bloom_level,
                    parse_failed=False,
                )
                total_written += 1
                preview = q.get("question", "")[:70]
                print(f"  Q{q_idx:02d} PASS  {preview}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 64)
    print("Experiment complete")
    print(f"  Questions written  : {total_written} / {total_expected}")
    print(f"  Parse failures     : {total_parse_fails}")
    print(f"  Time elapsed       : {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Output file        : {OUTPUT_FILE}")
    print("=" * 64)


if __name__ == "__main__":
    main()
