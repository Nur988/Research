"""
ifw_check.py — Item-Writing Flaw (IWF) checker + faithfulness scorer for MCQ questions.

score_question(q)         → {"flaw_count": int, "acceptable": bool, "flaws": [str]}
faithful_score(...)       → {"faithful": bool|None, "raw": str}
score_jsonl(in, out, ...) → adds "iwf" and "faithfulness" keys to every JSONL line.

Input schema expected in q:
    q["question"]  str  — the stem
    q["options"]   dict — {"a": "...", "b": "...", "c": "...", "d": "..."}
    q["answer"]    str  — correct option letter, e.g. "a"

CLI usage:
    python ifw_check.py --input results/experiment_001.jsonl \\
                        --output results/experiment_001_scored_v2.jsonl \\
                        --model llama3.2 \\
                        --ollama-url http://localhost:11434
"""

import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MODEL      = "llama3.2"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _trigrams(words: List[str]) -> List[str]:
    return [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]


# ── 7 IWF check functions ─────────────────────────────────────────────────────

def _longest_option_correct(correct_text: str, distractor_texts: List[str]) -> bool:
    if not correct_text or not distractor_texts:
        return False
    return all(len(correct_text) > len(d) for d in distractor_texts)


def _answer_position_bias(answer: str, sorted_keys: List[str]) -> bool:
    if len(sorted_keys) < 2:
        return False
    return answer in (sorted_keys[0], sorted_keys[-1])


def _all_none_of_above(all_texts: List[str]) -> bool:
    phrases = ("all of the above", "none of the above")
    return any(p in text.lower() for text in all_texts for p in phrases)


def _negated_stem(stem: str) -> bool:
    return bool(re.search(r"\bNOT\b|\bEXCEPT\b", stem))


def _duplicate_options(all_texts: List[str]) -> bool:
    normed = [_normalise(t) for t in all_texts]
    return len(normed) != len(set(normed))


def _stem_answer_overlap(
    stem: str, correct_text: str, distractor_texts: List[str]
) -> bool:
    words = stem.split()
    if len(words) < 3 or not correct_text:
        return False
    correct_lower      = correct_text.lower()
    distractor_lowers  = [d.lower() for d in distractor_texts]
    for tg in _trigrams(words):
        tg_lower = tg.lower()
        if tg_lower in correct_lower and not any(tg_lower in d for d in distractor_lowers):
            return True
    return False


def _implausible_distractor(all_texts: List[str], distractor_texts: List[str]) -> bool:
    if not all_texts or not distractor_texts:
        return False
    mean_len  = sum(len(t) for t in all_texts) / len(all_texts)
    threshold = 0.4 * mean_len
    return any(len(d) < threshold for d in distractor_texts)


# ── Public API — IWF ──────────────────────────────────────────────────────────

def score_question(q: Dict[str, Any]) -> Dict[str, Any]:
    """Run all 7 IWF checks. Returns {"flaw_count", "acceptable", "flaws"}."""
    stem    = q.get("question", "")
    options = q.get("options", {})
    answer  = q.get("answer", "").lower()

    sorted_keys      = sorted(options.keys())
    correct_text     = options.get(answer, "")
    distractor_texts = [options[k] for k in sorted_keys if k != answer]
    all_texts        = [options[k] for k in sorted_keys]

    flaws: List[str] = []
    if _longest_option_correct(correct_text, distractor_texts):
        flaws.append("longest_option_correct")
    if _answer_position_bias(answer, sorted_keys):
        flaws.append("answer_position_bias")
    if _all_none_of_above(all_texts):
        flaws.append("all_none_of_above")
    if _negated_stem(stem):
        flaws.append("negated_stem")
    if _duplicate_options(all_texts):
        flaws.append("duplicate_options")
    if _stem_answer_overlap(stem, correct_text, distractor_texts):
        flaws.append("stem_answer_overlap")
    if _implausible_distractor(all_texts, distractor_texts):
        flaws.append("implausible_distractor")

    return {"flaw_count": len(flaws), "acceptable": len(flaws) <= 1, "flaws": flaws}


# ── Public API — Faithfulness ─────────────────────────────────────────────────

def faithful_score(
    question: str,
    answer: str,
    retrieved_chunks: List[dict],
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Ask the LLM whether the correct answer is grounded in the retrieved chunks.

    Returns {"faithful": bool|None, "raw": str}.
    faithful is None (not False) when retrieved_chunks is empty (zero-shot
    condition) — there is no curriculum text to evaluate against.
    """
    if not retrieved_chunks:
        return {"faithful": None, "raw": ""}

    chunks_text = "\n\n".join(
        c["content"] for c in retrieved_chunks if "content" in c
    )

    prompt = (
        "You are evaluating whether a quiz question and its correct answer "
        "are grounded in the provided curriculum text.\n\n"
        f"Curriculum text:\n{chunks_text}\n\n"
        f"Question: {question}\n"
        f"Correct answer: {answer}\n\n"
        "Is the correct answer supported by the curriculum text above?\n"
        "Reply with only one word: YES or NO"
    )

    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        raw = data.get("response", "").strip()
        # Word-boundary search handles: yes, Yes, YES, "YES", Yes., YES!
        # If both words appear somehow, YES wins (conservative for faithfulness).
        if re.search(r"\byes\b", raw, re.IGNORECASE):
            faithful = True
        elif re.search(r"\bno\b", raw, re.IGNORECASE):
            faithful = False
        else:
            faithful = False  # unparseable → treat as not faithful
    except Exception as exc:
        raw      = f"ERROR: {exc}"
        faithful = False

    return {"faithful": faithful, "raw": raw}


# ── Public API — JSONL pipeline ───────────────────────────────────────────────

def score_jsonl(
    input_path: "str | Path",
    output_path: "str | Path",
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
) -> None:
    """
    Read experiment JSONL, add "iwf" and "faithfulness" keys, write new JSONL.

    IWF is deterministic and requires no LLM call.
    Faithfulness calls Ollama for RAG records; zero-shot records receive
    {"faithful": None, "raw": ""} without any LLM call.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    with input_path.open(encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        total = sum(1 for line in input_path.open(encoding="utf-8") if line.strip())
        idx   = 0

        fin.seek(0)
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            idx += 1
            record["iwf"] = score_question(record)

            # Resolve answer text for the faithfulness prompt
            options     = record.get("options", {})
            answer_key  = record.get("answer", "").lower()
            answer_text = options.get(answer_key, answer_key)

            faith = faithful_score(
                question=record.get("question", ""),
                answer=answer_text,
                retrieved_chunks=record.get("retrieved_chunks", []),
                ollama_url=ollama_url,
                model=model,
            )
            record["faithfulness"] = faith

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            tag  = f"faithful={faith['faithful']}"
            prev = record.get("question", "")[:55]
            print(f"  [{idx:3d}/{total}] {tag:<16}  {prev}", flush=True)

    print(f"\nWrote {idx} records → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Score IWF + faithfulness for a JSONL experiment file."
    )
    parser.add_argument("--input",       required=True,
                        help="Input JSONL path (e.g. results/experiment_001.jsonl)")
    parser.add_argument("--output",      required=True,
                        help="Output JSONL path")
    parser.add_argument("--model",       default=DEFAULT_MODEL,
                        help=f"Ollama chat model (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama-url",  default=DEFAULT_OLLAMA_URL,
                        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})")
    args = parser.parse_args()

    score_jsonl(
        input_path=args.input,
        output_path=args.output,
        ollama_url=args.ollama_url,
        model=args.model,
    )
