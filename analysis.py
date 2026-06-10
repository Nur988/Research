#!/usr/bin/env python3
"""
analysis.py — summarise results/experiment_001_scored.jsonl

Produces:
    results/results_table.csv
    results/chart_acceptable.png
    results/chart_flaws.png

Prints the results table and a one-sentence summary to stdout.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np

HERE        = Path(__file__).parent
SCORED_PATH = HERE / "results" / "experiment_001_scored_v2.jsonl"
RESULTS_DIR = HERE / "results"

# ── Flaw metadata ─────────────────────────────────────────────────────────────
# Ordered for consistent stacking in charts; unused flaws are silently skipped.
FLAW_ORDER = [
    "longest_option_correct",
    "answer_position_bias",
    "stem_answer_overlap",
    "implausible_distractor",
    "all_none_of_above",
    "negated_stem",
    "duplicate_options",
]

FLAW_LABELS = {
    "longest_option_correct": "Longest opt. correct",
    "answer_position_bias":   "Answer position bias",
    "stem_answer_overlap":    "Stem–answer overlap",
    "implausible_distractor": "Implausible distractor",
    "all_none_of_above":      "All/none of the above",
    "negated_stem":           "Negated stem",
    "duplicate_options":      "Duplicate options",
}

# Colours for flaw stack slices
FLAW_COLORS = [
    "#2563EB", "#F59E0B", "#10B981", "#EF4444",
    "#8B5CF6", "#F97316", "#6B7280",
]

# Cell order for the table
CELL_ORDER: List[Tuple[str, str, str]] = [
    ("RAG",       "3-5",  "Remember"),
    ("RAG",       "3-5",  "Analyze"),
    ("RAG",       "9-12", "Remember"),
    ("RAG",       "9-12", "Analyze"),
    ("Zero-shot", "3-5",  "Remember"),
    ("Zero-shot", "3-5",  "Analyze"),
    ("Zero-shot", "9-12", "Remember"),
    ("Zero-shot", "9-12", "Analyze"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_records() -> List[dict]:
    records = []
    with SCORED_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def condition_key(r: dict) -> Tuple[str, str, str]:
    cond = "RAG" if r["retrieval"] else "Zero-shot"
    return (cond, r["grade_band"], r["bloom_level"])


def aggregate(records: List[dict]) -> dict:
    """Compute summary stats for a list of scored question records."""
    n = len(records)
    if n == 0:
        return {
            "n": 0, "parse_fail_pct": 0.0, "mean_flaws": 0.0,
            "pct_acceptable": 0.0, "top_flaw": "—",
            "flaw_counter": Counter(),
        }
    parse_fails  = sum(1 for r in records if r.get("parse_failed", False))
    total_flaws  = sum(r["iwf"]["flaw_count"] for r in records)
    n_acceptable = sum(1 for r in records if r["iwf"]["acceptable"])
    flaw_counter = Counter(f for r in records for f in r["iwf"]["flaws"])
    top_flaw     = flaw_counter.most_common(1)[0][0] if flaw_counter else "—"
    return {
        "n":              n,
        "parse_fail_pct": parse_fails / n * 100,
        "mean_flaws":     total_flaws / n,
        "pct_acceptable": n_acceptable / n * 100,
        "top_flaw":       top_flaw,
        "flaw_counter":   flaw_counter,
    }


# ── 1. Results table ──────────────────────────────────────────────────────────

def build_rows(records: List[dict]) -> List[dict]:
    by_cell = defaultdict(list)
    for r in records:
        by_cell[condition_key(r)].append(r)

    rows = []

    # One row per cell
    for cond, grade, bloom in CELL_ORDER:
        agg = aggregate(by_cell[(cond, grade, bloom)])
        rows.append({
            "Condition":    f"{cond} | {grade} | {bloom}",
            "N":            agg["n"],
            "Parse Fail %": f"{agg['parse_fail_pct']:.1f}",
            "Mean Flaws":   f"{agg['mean_flaws']:.2f}",
            "% Acceptable": f"{agg['pct_acceptable']:.1f}",
            "Top Flaw":     agg["top_flaw"],
        })

    # Two summary rows
    for label, flag in [("RAG", True), ("Zero-shot", False)]:
        subset = [r for r in records if bool(r["retrieval"]) is flag]
        agg    = aggregate(subset)
        rows.append({
            "Condition":    f"{label} — all conditions",
            "N":            agg["n"],
            "Parse Fail %": f"{agg['parse_fail_pct']:.1f}",
            "Mean Flaws":   f"{agg['mean_flaws']:.2f}",
            "% Acceptable": f"{agg['pct_acceptable']:.1f}",
            "Top Flaw":     agg["top_flaw"],
        })

    # Faithfulness summary row (RAG only — zero-shot has no retrieved chunks)
    rag_recs = [r for r in records if r["retrieval"]]
    n_rag    = len(rag_recs)
    n_faithful = sum(
        1 for r in rag_recs
        if r.get("faithfulness", {}) and r["faithfulness"].get("faithful") is True
    )
    rag_faith_pct = n_faithful / n_rag * 100 if n_rag else 0.0

    rows.append({
        "Condition":    f"RAG faithfulness: {rag_faith_pct:.1f}% | Zero-shot: N/A",
        "N":            n_rag,
        "Parse Fail %": "—",
        "Mean Flaws":   "—",
        "% Acceptable": "—",
        "Top Flaw":     "—",
    })

    return rows


COLUMNS = ["Condition", "N", "Parse Fail %", "Mean Flaws", "% Acceptable", "Top Flaw"]


def print_table(rows: List[dict]) -> None:
    widths = {
        col: max(len(col), max(len(str(r[col])) for r in rows))
        for col in COLUMNS
    }

    def fmt(row: dict) -> str:
        return "  ".join(str(row[c]).ljust(widths[c]) for c in COLUMNS)

    sep = "  ".join("-" * widths[c] for c in COLUMNS)
    print(fmt({c: c for c in COLUMNS}))
    print(sep)
    for i, row in enumerate(rows):
        if i == len(CELL_ORDER):        # separator before the two summary rows
            print()
            print(sep)
        elif i == len(CELL_ORDER) + 2:  # separator before the faithfulness row
            print()
            print(sep)
        print(fmt(row))


def save_csv(rows: List[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


# ── 2. chart_acceptable.png ───────────────────────────────────────────────────

def chart_acceptable(records: List[dict], path: Path) -> None:
    by_cell  = defaultdict(list)
    for r in records:
        by_cell[condition_key(r)].append(r)

    combos = [("3-5", "Remember"), ("3-5", "Analyze"),
              ("9-12", "Remember"), ("9-12", "Analyze")]
    x_labels = [f"Grade {g}\n{b}" for g, b in combos]

    rag_pcts  = [aggregate(by_cell[("RAG",       g, b)])["pct_acceptable"] for g, b in combos]
    zero_pcts = [aggregate(by_cell[("Zero-shot", g, b)])["pct_acceptable"] for g, b in combos]

    x     = np.arange(len(combos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    b_rag  = ax.bar(x - width / 2, rag_pcts,  width, label="RAG",       color="#2563EB", zorder=3)
    b_zero = ax.bar(x + width / 2, zero_pcts, width, label="Zero-shot", color="#F59E0B", zorder=3)

    # Value labels above bars
    for bar in list(b_rag) + list(b_zero):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 1.5,
            f"{h:.0f}%", ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 118)
    ax.set_ylabel("% Acceptable Questions", labelpad=8)
    ax.set_title("RAG vs Zero-Shot: % Acceptable Questions by Condition", pad=14)
    ax.legend(frameon=False, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── 3. chart_flaws.png ────────────────────────────────────────────────────────

def chart_flaws(records: List[dict], path: Path) -> None:
    rag_records  = [r for r in records if r["retrieval"]]
    zero_records = [r for r in records if not r["retrieval"]]

    rag_counter  = Counter(f for r in rag_records  for f in r["iwf"]["flaws"])
    zero_counter = Counter(f for r in zero_records for f in r["iwf"]["flaws"])

    # Only render flaws that appear at least once in either group
    active = [f for f in FLAW_ORDER if rag_counter[f] + zero_counter[f] > 0]

    n_rag  = len(rag_records)
    n_zero = len(zero_records)

    x      = np.arange(2)
    labels = ["RAG", "Zero-shot"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = np.zeros(2)

    for i, flaw in enumerate(active):
        vals = np.array([
            rag_counter[flaw]  / n_rag  * 100,
            zero_counter[flaw] / n_zero * 100,
        ])
        ax.bar(
            x, vals,
            bottom=bottoms,
            color=FLAW_COLORS[i % len(FLAW_COLORS)],
            label=FLAW_LABELS[flaw],
            zorder=3,
        )
        # Label slice only if tall enough to read
        for j in range(2):
            if vals[j] >= 3:
                ax.text(
                    x[j], bottoms[j] + vals[j] / 2,
                    f"{vals[j]:.0f}%",
                    ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold",
                )
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of Questions Triggering Flaw", labelpad=8)
    ax.set_title("Item-Writing Flaw Breakdown: RAG vs Zero-Shot", pad=14)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyse a scored experiment JSONL file.")
    parser.add_argument(
        "--scored-input", default=str(HERE / "results" / "experiment_001_scored_v2.jsonl"),
        help="Path to the scored JSONL file (default: results/experiment_001_scored_v2.jsonl)",
    )
    parser.add_argument(
        "--output-suffix", default="",
        help="Suffix appended to output filenames, e.g. '_v2' → results_table_v2.csv",
    )
    args = parser.parse_args()

    # Override module-level constant with CLI value so load_records() uses it
    global SCORED_PATH
    SCORED_PATH = Path(args.scored_input)
    suffix      = args.output_suffix
    RESULTS_DIR.mkdir(exist_ok=True)

    records = load_records()
    print(f"Loaded {len(records)} records from {SCORED_PATH.name}\n")

    # Table
    rows = build_rows(records)
    print_table(rows)
    save_csv(rows, RESULTS_DIR / f"results_table{suffix}.csv")

    print()

    # Charts
    chart_acceptable(records, RESULTS_DIR / f"chart_acceptable{suffix}.png")
    chart_flaws(records,      RESULTS_DIR / f"chart_flaws{suffix}.png")

    # Summary sentence
    rag_recs    = [r for r in records if r["retrieval"]]
    zero_recs   = [r for r in records if not r["retrieval"]]
    rag_pct     = sum(1 for r in rag_recs  if r["iwf"]["acceptable"]) / len(rag_recs)  * 100
    zero_pct    = sum(1 for r in zero_recs if r["iwf"]["acceptable"]) / len(zero_recs) * 100
    rag_faith   = sum(
        1 for r in rag_recs
        if r.get("faithfulness", {}) and r["faithfulness"].get("faithful") is True
    )
    rag_faith_pct = rag_faith / len(rag_recs) * 100 if rag_recs else 0.0

    print(
        f'\nRAG questions: {rag_pct:.0f}% acceptable, {rag_faith_pct:.0f}% faithful.\n'
        f'Zero-shot questions: {zero_pct:.0f}% acceptable, faithfulness not applicable.'
    )


if __name__ == "__main__":
    main()
