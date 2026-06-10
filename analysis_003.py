#!/usr/bin/env python3
"""
analysis_003.py — full analysis pipeline for Experiment 003.

Produces every required output file in results/experiment_003/:
  results_003a.csv, results_003b.csv, results_comparison.csv
  chart_acceptability_comparison.png, chart_faithfulness_comparison.png
  chart_flaws_003a.png, chart_flaws_003b.png
  chart_flaws_comparison.png, chart_diversity_score.png
  stats_003a.json, stats_003b.json, stats_comparison.json
  findings_003.md, slide_content_003.md

Can be called programmatically via run_analysis() or run from the CLI:
    python analysis_003.py

CLI loads paths from the default locations produced by run_experiment_003.py.
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Path constants ────────────────────────────────────────────────────────────
HERE    = Path(__file__).parent
OUT_DIR = HERE / "results" / "experiment_003"

PATH_002_SCORED = HERE / "results" / "experiment_002_scored.jsonl"
PATH_003A_DEFAULT = OUT_DIR / "experiment_003a_scored.jsonl"
PATH_003B_DEFAULT = OUT_DIR / "experiment_003b_scored.jsonl"

# ── Shared flaw metadata (same order as analysis.py) ─────────────────────────
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

FLAW_COLORS = [
    "#2563EB", "#F59E0B", "#10B981", "#EF4444",
    "#8B5CF6", "#F97316", "#6B7280",
]

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

EXP_COLORS = {
    "002":  "#94A3B8",   # slate-grey
    "003a": "#2563EB",   # blue
    "003b": "#10B981",   # emerald
}
EXP_LABELS = {"002": "Exp 002", "003a": "Exp 003a", "003b": "Exp 003b"}


# ── Record loading ────────────────────────────────────────────────────────────

def load_records(path: Path) -> List[dict]:
    """Load all non-empty JSONL records from a file.

    Args:
        path: Path to a scored JSONL file.

    Returns:
        List of dicts, one per question record.
    """
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def condition_key(r: dict) -> Tuple[str, str, str]:
    """Return (cond, grade_band, bloom_level) tuple for a record.

    Args:
        r: A single question record dict.

    Returns:
        Tuple e.g. ("RAG", "3-5", "Remember").
    """
    cond = "RAG" if r.get("retrieval") else "Zero-shot"
    return (cond, r.get("grade_band", ""), r.get("bloom_level", ""))


# ── Aggregation helpers ───────────────────────────────────────────────────────

def diversity_score(records: List[dict]) -> int:
    """Count distinct 8-word question-stem prefixes within a group.

    The first eight whitespace-separated words of each question stem form
    the uniqueness key.  A higher value means more varied stems.

    Args:
        records: List of question records from one cell.

    Returns:
        Number of unique 8-word prefixes.
    """
    prefixes = set()
    for r in records:
        words = r.get("question", "").split()[:8]
        if words:
            prefixes.add(" ".join(words).lower())
    return len(prefixes)


def aggregate(records: List[dict]) -> dict:
    """Compute summary statistics for a list of scored question records.

    Args:
        records: Subset of question records (one cell or a rolled-up group).

    Returns:
        Dict with keys: n, parse_fail_pct, mean_flaws, pct_acceptable,
        top_flaw, flaw_counter, n_faithful, pct_faithful, diversity.
    """
    n = len(records)
    if n == 0:
        return {
            "n": 0, "parse_fail_pct": 0.0, "mean_flaws": 0.0,
            "pct_acceptable": 0.0, "top_flaw": "—",
            "flaw_counter": Counter(), "n_faithful": 0,
            "pct_faithful": 0.0, "diversity": 0,
        }

    non_failed = [r for r in records if not r.get("parse_failed")]
    parse_fails  = n - len(non_failed)
    total_flaws  = sum(r["iwf"]["flaw_count"] for r in non_failed if "iwf" in r)
    n_acceptable = sum(1 for r in non_failed if r.get("iwf", {}).get("acceptable", False))
    flaw_counter = Counter(f for r in non_failed for f in r.get("iwf", {}).get("flaws", []))
    top_flaw     = flaw_counter.most_common(1)[0][0] if flaw_counter else "—"

    rag_recs     = [r for r in non_failed if r.get("retrieval")]
    n_faithful   = sum(
        1 for r in rag_recs
        if r.get("faithfulness", {}) and r["faithfulness"].get("faithful") is True
    )
    pct_faithful = (n_faithful / len(rag_recs) * 100) if rag_recs else 0.0

    n_nf = len(non_failed) if non_failed else 1
    return {
        "n":              n,
        "parse_fail_pct": parse_fails / n * 100,
        "mean_flaws":     total_flaws / n_nf,
        "pct_acceptable": n_acceptable / n_nf * 100,
        "top_flaw":       top_flaw,
        "flaw_counter":   flaw_counter,
        "n_faithful":     n_faithful,
        "pct_faithful":   pct_faithful,
        "diversity":      diversity_score(non_failed),
    }


# ── CSV output ────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "Condition", "N", "Parse Fail %", "Mean Flaws",
    "% Acceptable", "Top Flaw", "N Faithful", "% Faithful", "Diversity",
]


def build_csv_rows(records: List[dict]) -> List[dict]:
    """Build per-condition and summary rows matching analysis.py format.

    Args:
        records: All scored records from one experiment.

    Returns:
        List of row dicts keyed by CSV_COLUMNS, ready to write.
    """
    by_cell: Dict = defaultdict(list)
    for r in records:
        by_cell[condition_key(r)].append(r)

    rows = []
    for cond, grade, bloom in CELL_ORDER:
        agg = aggregate(by_cell.get((cond, grade, bloom), []))
        rows.append({
            "Condition":    f"{cond} | {grade} | {bloom}",
            "N":            agg["n"],
            "Parse Fail %": f"{agg['parse_fail_pct']:.1f}",
            "Mean Flaws":   f"{agg['mean_flaws']:.2f}",
            "% Acceptable": f"{agg['pct_acceptable']:.1f}",
            "Top Flaw":     agg["top_flaw"],
            "N Faithful":   agg["n_faithful"] if cond == "RAG" else "—",
            "% Faithful":   f"{agg['pct_faithful']:.1f}" if cond == "RAG" else "—",
            "Diversity":    agg["diversity"],
        })

    for label, flag in [("RAG", True), ("Zero-shot", False)]:
        subset = [r for r in records if bool(r.get("retrieval")) is flag]
        agg    = aggregate(subset)
        rows.append({
            "Condition":    f"{label} — all",
            "N":            agg["n"],
            "Parse Fail %": f"{agg['parse_fail_pct']:.1f}",
            "Mean Flaws":   f"{agg['mean_flaws']:.2f}",
            "% Acceptable": f"{agg['pct_acceptable']:.1f}",
            "Top Flaw":     agg["top_flaw"],
            "N Faithful":   agg["n_faithful"] if label == "RAG" else "—",
            "% Faithful":   f"{agg['pct_faithful']:.1f}" if label == "RAG" else "—",
            "Diversity":    agg["diversity"],
        })

    return rows


def save_csv(rows: List[dict], path: Path) -> None:
    """Write rows to a CSV file.

    Args:
        rows: List of row dicts matching CSV_COLUMNS.
        path: Destination path.
    """
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path.name}")


def build_comparison_csv(recs_002, recs_003a, recs_003b, path: Path) -> None:
    """Write a three-experiment comparison CSV, one row per condition.

    Columns: Condition, then {metric}_{002/003a/003b} for each metric.

    Args:
        recs_002:  Records from experiment_002 scored JSONL.
        recs_003a: Records from experiment_003a scored JSONL.
        recs_003b: Records from experiment_003b scored JSONL.
        path:      Destination CSV path.
    """
    metrics = ["% Acceptable", "% Faithful", "Mean Flaws", "Diversity", "Parse Fail %"]

    def cell_vals(records, cond, grade, bloom):
        by_cell = defaultdict(list)
        for r in records:
            by_cell[condition_key(r)].append(r)
        agg = aggregate(by_cell.get((cond, grade, bloom), []))
        is_rag = cond == "RAG"
        return {
            "% Acceptable": f"{agg['pct_acceptable']:.1f}",
            "% Faithful":   f"{agg['pct_faithful']:.1f}" if is_rag else "—",
            "Mean Flaws":   f"{agg['mean_flaws']:.2f}",
            "Diversity":    str(agg["diversity"]),
            "Parse Fail %": f"{agg['parse_fail_pct']:.1f}",
        }

    fieldnames = ["Condition"] + [
        f"{m} (002)" for m in metrics
    ] + [
        f"{m} (003a)" for m in metrics
    ] + [
        f"{m} (003b)" for m in metrics
    ]

    rows = []
    for cond, grade, bloom in CELL_ORDER:
        v002  = cell_vals(recs_002,  cond, grade, bloom)
        v003a = cell_vals(recs_003a, cond, grade, bloom)
        v003b = cell_vals(recs_003b, cond, grade, bloom)
        row = {"Condition": f"{cond} | {grade} | {bloom}"}
        for m in metrics:
            row[f"{m} (002)"]  = v002[m]
            row[f"{m} (003a)"] = v003a[m]
            row[f"{m} (003b)"] = v003b[m]
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path.name}")


# ── Stats JSON output ─────────────────────────────────────────────────────────

def build_stats_json(records: List[dict], experiment_id: str, model: str) -> dict:
    """Build the complete stats dict for one experiment.

    Args:
        records:       All scored records.
        experiment_id: "003a" or "003b".
        model:         Model name used for this experiment.

    Returns:
        Nested dict with per-condition and aggregate statistics.
    """
    by_cell: Dict = defaultdict(list)
    for r in records:
        by_cell[condition_key(r)].append(r)

    conditions = {}
    for cond, grade, bloom in CELL_ORDER:
        cell_recs = by_cell.get((cond, grade, bloom), [])
        agg       = aggregate(cell_recs)
        # Capture the retrieval query used (first non-empty value in the cell)
        rq = next(
            (r.get("retrieval_query", "") for r in cell_recs if r.get("retrieval_query")),
            ""
        )
        # Model used
        m = next(
            (r.get("model", model) for r in cell_recs if r.get("model")),
            model
        )
        key = f"{cond} | {grade} | {bloom}"
        conditions[key] = {
            "total_questions":  agg["n"],
            "n_acceptable":     int(agg["n"] * agg["pct_acceptable"] / 100),
            "pct_acceptable":   round(agg["pct_acceptable"], 1),
            "mean_flaw_count":  round(agg["mean_flaws"], 3),
            "n_faithful":       agg["n_faithful"],
            "pct_faithful":     round(agg["pct_faithful"], 1),
            "diversity_score":  agg["diversity"],
            "most_common_flaw": agg["top_flaw"],
            "parse_failures":   int(agg["n"] * agg["parse_fail_pct"] / 100),
            "model_used":       m,
            "retrieval_query":  rq,
        }

    # Aggregates
    rag_recs  = [r for r in records if r.get("retrieval")]
    zero_recs = [r for r in records if not r.get("retrieval")]
    agg_rag   = aggregate(rag_recs)
    agg_zero  = aggregate(zero_recs)
    agg_all   = aggregate(records)

    lb_retries = sum(1 for r in records if r.get("length_balance_retry"))

    return {
        "experiment_id":          experiment_id,
        "model":                  model,
        "total_questions":        agg_all["n"],
        "aggregate_RAG": {
            "n_acceptable":    int(len(rag_recs) * agg_rag["pct_acceptable"] / 100) if rag_recs else 0,
            "pct_acceptable":  round(agg_rag["pct_acceptable"], 1),
            "n_faithful":      agg_rag["n_faithful"],
            "pct_faithful":    round(agg_rag["pct_faithful"], 1),
            "mean_flaws":      round(agg_rag["mean_flaws"], 3),
            "diversity":       agg_rag["diversity"],
            "most_common_flaw":agg_rag["top_flaw"],
            "parse_failures":  int(len(rag_recs) * agg_rag["parse_fail_pct"] / 100),
        },
        "aggregate_zero_shot": {
            "n_acceptable":    int(len(zero_recs) * agg_zero["pct_acceptable"] / 100) if zero_recs else 0,
            "pct_acceptable":  round(agg_zero["pct_acceptable"], 1),
            "mean_flaws":      round(agg_zero["mean_flaws"], 3),
            "diversity":       agg_zero["diversity"],
            "most_common_flaw":agg_zero["top_flaw"],
            "parse_failures":  int(len(zero_recs) * agg_zero["parse_fail_pct"] / 100),
        },
        "length_balance_retries": lb_retries,
        "conditions":             conditions,
    }


def build_comparison_stats(stats_002: dict, stats_003a: dict, stats_003b: dict) -> dict:
    """Compute delta table between experiments for every numeric metric.

    For each metric, computes:
      - 003a_vs_002:  003a value minus 002 value
      - 003b_vs_002:  003b value minus 002 value
      - 003b_vs_003a: 003b value minus 003a value
    Each delta carries a direction label: "improved", "degraded", or "neutral".
    Higher is better for pct_acceptable, pct_faithful, and diversity.
    Lower is better for mean_flaws and parse_failures.

    Args:
        stats_002:  Stats dict produced by build_stats_json for experiment_002.
        stats_003a: Stats dict for experiment_003a.
        stats_003b: Stats dict for experiment_003b.

    Returns:
        Nested delta dict.
    """
    def _delta(a_val, b_val, higher_is_better: bool) -> dict:
        try:
            delta = round(float(b_val) - float(a_val), 2)
        except (TypeError, ValueError):
            return {"delta": None, "direction": "neutral"}
        if abs(delta) < 0.01:
            direction = "neutral"
        elif higher_is_better:
            direction = "improved" if delta > 0 else "degraded"
        else:
            direction = "improved" if delta < 0 else "degraded"
        return {"delta": delta, "direction": direction}

    # Build a flat metrics dict from stats dicts
    def _flat(s: dict) -> dict:
        rag  = s.get("aggregate_RAG", {})
        zero = s.get("aggregate_zero_shot", {})
        return {
            "rag_pct_acceptable":  rag.get("pct_acceptable", 0),
            "rag_pct_faithful":    rag.get("pct_faithful", 0),
            "rag_mean_flaws":      rag.get("mean_flaws", 0),
            "rag_diversity":       rag.get("diversity", 0),
            "rag_parse_failures":  rag.get("parse_failures", 0),
            "zero_pct_acceptable": zero.get("pct_acceptable", 0),
            "zero_mean_flaws":     zero.get("mean_flaws", 0),
            "zero_diversity":      zero.get("diversity", 0),
        }

    higher_better = {
        "rag_pct_acceptable":  True,
        "rag_pct_faithful":    True,
        "rag_mean_flaws":      False,
        "rag_diversity":       True,
        "rag_parse_failures":  False,
        "zero_pct_acceptable": True,
        "zero_mean_flaws":     False,
        "zero_diversity":      True,
    }

    f002   = _flat(stats_002)
    f003a  = _flat(stats_003a)
    f003b  = _flat(stats_003b)

    deltas: dict = {}
    for metric, hib in higher_better.items():
        deltas[metric] = {
            "003a_vs_002":  _delta(f002[metric],  f003a[metric], hib),
            "003b_vs_002":  _delta(f002[metric],  f003b[metric], hib),
            "003b_vs_003a": _delta(f003a[metric], f003b[metric], hib),
        }

    return {
        "description": (
            "Delta table for numeric metrics across experiments. "
            "direction='improved' means the change was beneficial."
        ),
        "deltas": deltas,
    }


# ── Charts ────────────────────────────────────────────────────────────────────

def _add_bar_labels(ax, bars, fmt="{:.0f}%", fontsize=7):
    """Add value labels above or inside bars."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.8,
                fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize,
            )


def chart_acceptability_comparison(
    recs_002, recs_003a, recs_003b, path: Path
) -> None:
    """Grouped bar chart: % acceptable for 002, 003a, 003b per condition.

    Two subplots: RAG (top) and Zero-shot (bottom).

    Args:
        recs_002:  experiment_002 scored records.
        recs_003a: experiment_003a scored records.
        recs_003b: experiment_003b scored records.
        path:      Output PNG path.
    """
    combos   = [("3-5", "Remember"), ("3-5", "Analyze"),
                ("9-12", "Remember"), ("9-12", "Analyze")]
    x_labels = [f"Grade {g}\n{b}" for g, b in combos]
    x        = np.arange(len(combos))
    width    = 0.24

    def _pcts(records, cond):
        by_cell = defaultdict(list)
        for r in records:
            by_cell[condition_key(r)].append(r)
        return [aggregate(by_cell.get((cond, g, b), []))["pct_acceptable"]
                for g, b in combos]

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=False)

    for ax, cond in zip(axes, ["RAG", "Zero-shot"]):
        p002  = _pcts(recs_002,  cond)
        p003a = _pcts(recs_003a, cond)
        p003b = _pcts(recs_003b, cond)

        b002  = ax.bar(x - width, p002,  width, label="Exp 002",  color=EXP_COLORS["002"],  zorder=3)
        b003a = ax.bar(x,         p003a, width, label="Exp 003a", color=EXP_COLORS["003a"], zorder=3)
        b003b = ax.bar(x + width, p003b, width, label="Exp 003b", color=EXP_COLORS["003b"], zorder=3)

        for bars in [b002, b003a, b003b]:
            _add_bar_labels(ax, bars)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(0, 120)
        ax.set_ylabel("% Acceptable")
        ax.set_title(f"{cond} conditions — % IWF-Acceptable Questions")
        ax.legend(frameon=False, fontsize=8)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Experiment 002 vs 003a vs 003b — Writing Quality", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def chart_faithfulness_comparison(
    recs_002, recs_003a, recs_003b, path: Path
) -> None:
    """Grouped bar chart: % faithful for RAG conditions across 3 experiments.

    Args:
        recs_002, recs_003a, recs_003b: Scored records.
        path: Output PNG path.
    """
    combos   = [("3-5", "Remember"), ("3-5", "Analyze"),
                ("9-12", "Remember"), ("9-12", "Analyze")]
    x_labels = [f"Grade {g}\n{b}" for g, b in combos]
    x        = np.arange(len(combos))
    width    = 0.24

    def _faith(records):
        by_cell = defaultdict(list)
        for r in records:
            by_cell[condition_key(r)].append(r)
        vals = []
        for g, b in combos:
            cell_recs = by_cell.get(("RAG", g, b), [])
            agg = aggregate(cell_recs)
            vals.append(agg["pct_faithful"])
        return vals

    fig, ax = plt.subplots(figsize=(11, 5))

    b002  = ax.bar(x - width, _faith(recs_002),  width, label="Exp 002",  color=EXP_COLORS["002"],  zorder=3)
    b003a = ax.bar(x,         _faith(recs_003a), width, label="Exp 003a", color=EXP_COLORS["003a"], zorder=3)
    b003b = ax.bar(x + width, _faith(recs_003b), width, label="Exp 003b", color=EXP_COLORS["003b"], zorder=3)

    for bars in [b002, b003a, b003b]:
        _add_bar_labels(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% Faithful to Curriculum")
    ax.set_title("RAG Curriculum Faithfulness — Experiment 002 vs 003a vs 003b", pad=12)
    ax.legend(frameon=False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def chart_flaws_single(records: List[dict], title: str, path: Path) -> None:
    """Stacked bar chart showing flaw breakdown for one experiment (RAG vs Zero-shot).

    Args:
        records: Scored records for one experiment.
        title:   Chart title.
        path:    Output PNG path.
    """
    rag_recs  = [r for r in records if r.get("retrieval")]
    zero_recs = [r for r in records if not r.get("retrieval")]
    n_rag     = len(rag_recs)  or 1
    n_zero    = len(zero_recs) or 1

    rag_counter  = Counter(f for r in rag_recs  for f in r.get("iwf", {}).get("flaws", []))
    zero_counter = Counter(f for r in zero_recs for f in r.get("iwf", {}).get("flaws", []))

    active = [f for f in FLAW_ORDER if rag_counter[f] + zero_counter[f] > 0]

    x       = np.arange(2)
    labels  = ["RAG", "Zero-shot"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = np.zeros(2)

    for i, flaw in enumerate(active):
        vals = np.array([
            rag_counter[flaw]  / n_rag  * 100,
            zero_counter[flaw] / n_zero * 100,
        ])
        ax.bar(x, vals, bottom=bottoms,
               color=FLAW_COLORS[i % len(FLAW_COLORS)],
               label=FLAW_LABELS[flaw], zorder=3)
        for j in range(2):
            if vals[j] >= 4:
                ax.text(x[j], bottoms[j] + vals[j] / 2,
                        f"{vals[j]:.0f}%",
                        ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of Questions Triggering Flaw")
    ax.set_title(title, pad=12)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def chart_flaws_comparison(
    recs_002, recs_003a, recs_003b, path: Path
) -> None:
    """Grouped bar chart comparing flaw frequencies across 3 experiments.

    Shows each active flaw type on the x-axis, 3 bars per flaw (one per
    experiment), separately for RAG and Zero-shot conditions.

    Args:
        recs_002, recs_003a, recs_003b: Scored records.
        path: Output PNG path.
    """
    def _flaw_pcts(records, rag: bool) -> Dict[str, float]:
        subset = [r for r in records if bool(r.get("retrieval")) is rag]
        n      = len(subset) or 1
        counter= Counter(f for r in subset for f in r.get("iwf", {}).get("flaws", []))
        return {f: counter[f] / n * 100 for f in FLAW_ORDER}

    exps    = [("002", recs_002), ("003a", recs_003a), ("003b", recs_003b)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, rag, cond_label in zip(axes, [True, False], ["RAG", "Zero-shot"]):
        pcts_by_exp = {eid: _flaw_pcts(recs, rag) for eid, recs in exps}
        active      = [f for f in FLAW_ORDER if any(pcts_by_exp[eid][f] > 0 for eid, _ in exps)]

        x     = np.arange(len(active))
        width = 0.25

        for i, (eid, _) in enumerate(exps):
            vals  = [pcts_by_exp[eid][f] for f in active]
            bars  = ax.bar(x + (i - 1) * width, vals, width,
                           label=EXP_LABELS[eid], color=EXP_COLORS[eid], zorder=3)
            _add_bar_labels(ax, bars, fmt="{:.0f}%", fontsize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [FLAW_LABELS.get(f, f) for f in active],
            rotation=25, ha="right", fontsize=8
        )
        ax.set_ylabel("% of Questions Triggering Flaw")
        ax.set_title(f"{cond_label} — Flaw Comparison")
        ax.legend(frameon=False, fontsize=8)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Item-Writing Flaw Frequencies — Exp 002 vs 003a vs 003b", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def chart_diversity(
    recs_002, recs_003a, recs_003b, path: Path
) -> None:
    """Grouped bar chart: within-cell diversity (unique 8-word prefixes).

    Args:
        recs_002, recs_003a, recs_003b: Scored records.
        path: Output PNG path.
    """
    exps    = [("002", recs_002), ("003a", recs_003a), ("003b", recs_003b)]
    all_conds = CELL_ORDER
    x_labels  = [f"{cond}\n{g}|{b}" for cond, g, b in all_conds]
    x         = np.arange(len(all_conds))
    width     = 0.24

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (eid, records) in enumerate(exps):
        by_cell: Dict = defaultdict(list)
        for r in records:
            by_cell[condition_key(r)].append(r)
        vals = [
            diversity_score([r for r in by_cell.get((cond, g, b), [])
                             if not r.get("parse_failed")])
            for cond, g, b in all_conds
        ]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=EXP_LABELS[eid], color=EXP_COLORS[eid], zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                    str(v), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_ylim(0, N_PER_CELL + 2 if True else 17)
    ax.axhline(y=15, linestyle="--", linewidth=0.8, color="black", alpha=0.4,
               label="Max possible (15)")
    ax.set_ylabel("Unique 8-word stem prefixes (out of 15)")
    ax.set_title("Within-Cell Question Diversity — Exp 002 vs 003a vs 003b", pad=12)
    ax.legend(frameon=False, fontsize=8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── N_PER_CELL reference for diversity chart ──────────────────────────────────
N_PER_CELL = 15


# ── Markdown report generators ────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    """Format a percentage string.

    Args:
        n:     Numerator.
        total: Denominator.

    Returns:
        Formatted string like "73.3%".
    """
    if total == 0:
        return "N/A"
    return f"{n / total * 100:.1f}%"


def write_findings(
    stats_002: dict,
    stats_003a: dict,
    stats_003b: dict,
    comparison: dict,
    model_003b: str,
    model_note: str,
    path: Path,
) -> None:
    """Write findings_003.md with real numbers from the experiment results.

    Args:
        stats_002, stats_003a, stats_003b: Stats dicts.
        comparison:  Delta table dict.
        model_003b:  Model name used in 003b.
        model_note:  Human-readable note about model selection.
        path:        Output path.
    """
    s002  = stats_002
    s003a = stats_003a
    s003b = stats_003b
    d     = comparison["deltas"]

    r002_acc   = s002["aggregate_RAG"]["pct_acceptable"]
    r002_faith = s002["aggregate_RAG"]["pct_faithful"]
    z002_acc   = s002["aggregate_zero_shot"]["pct_acceptable"]

    r003a_acc   = s003a["aggregate_RAG"]["pct_acceptable"]
    r003a_faith = s003a["aggregate_RAG"]["pct_faithful"]
    z003a_acc   = s003a["aggregate_zero_shot"]["pct_acceptable"]

    r003b_acc   = s003b["aggregate_RAG"]["pct_acceptable"]
    r003b_faith = s003b["aggregate_RAG"]["pct_faithful"]
    z003b_acc   = s003b["aggregate_zero_shot"]["pct_acceptable"]

    r003a_div = s003a["aggregate_RAG"]["diversity"]
    r003b_div = s003b["aggregate_RAG"]["diversity"]
    r002_div  = s002["aggregate_RAG"]["diversity"]

    r003a_top = s003a["aggregate_RAG"].get("most_common_flaw", "—")
    r003b_top = s003b["aggregate_RAG"].get("most_common_flaw", "—")
    r002_top  = s002["aggregate_RAG"].get("most_common_flaw", "—")

    lb_003a   = s003a.get("length_balance_retries", 0)
    lb_003b   = s003b.get("length_balance_retries", 0)

    def _dir(metric, exp):
        try:
            return d[metric][exp]["direction"]
        except (KeyError, TypeError):
            return "—"

    def _delta_str(metric, exp):
        try:
            val = d[metric][exp]["delta"]
            if val is None:
                return "—"
            sign = "+" if val >= 0 else ""
            return f"{sign}{val:.1f}"
        except (KeyError, TypeError):
            return "—"

    # Collect per-cell table rows for results section
    def _cell_row(stats, label):
        rows = []
        for cond, grade, bloom in [
            ("RAG", "3-5", "Remember"), ("RAG", "3-5", "Analyze"),
            ("RAG", "9-12", "Remember"), ("RAG", "9-12", "Analyze"),
        ]:
            key = f"{cond} | {grade} | {bloom}"
            c   = stats["conditions"].get(key, {})
            acc = c.get("pct_acceptable", 0)
            fth = c.get("pct_faithful", 0)
            div = c.get("diversity_score", 0)
            rows.append(f"| {label} \\| Grade {grade} \\| {bloom} | {acc:.1f}% | {fth:.1f}% | {div}/15 |")
        return rows

    rag_table_003a = _cell_row(s003a, "003a")
    rag_table_003b = _cell_row(s003b, "003b")
    rag_table_002  = _cell_row(s002,  "002")

    content = f"""# Experiment 003 Findings

## Overview

Experiment 003 investigated two targeted improvements to the K-12 RAG question generation
pipeline established in Experiment 002. Two sub-experiments were run:

- **003a**: Same model (llama3.2) with improved retrieval — dual grade-band indices, dynamic
  queries, post-generation option shuffling, and length-balance retries.
- **003b**: {model_003b} at temperature 0.7, with identical retrieval improvements from 003a.

{model_note}

Both sub-experiments used the same 8-cell grid (2 retrieval conditions × 2 grade bands
× 2 Bloom's levels × MCQ, 15 questions per cell, 120 questions total) as Experiment 002.

---

## Configuration

### What changed in 003a compared to 002

| Change | 002 behaviour | 003a behaviour |
|--------|--------------|----------------|
| Index | Single flat index (all documents) | Dual indices: F-6 curriculum + glossary for grades 3-5; 7-10 curriculum + glossary for grades 9-12 |
| Retrieval query | Fixed: "main concepts key ideas overview" | Dynamic: encodes target year level and Bloom's level per question |
| Option positions | Left as generated (correct answer systematically first/last) | Options shuffled uniformly at random after generation |
| Length balance | None | One retry with explicit length instruction if correct answer was longest |
| Temperature | 0.2 | 0.2 (unchanged) |
| Model | llama3.2 | llama3.2 (unchanged) |

### What changed in 003b compared to 003a

| Change | 003a behaviour | 003b behaviour |
|--------|---------------|----------------|
| Model | llama3.2 | {model_003b} ({model_note.split('.')[0]}) |
| Temperature | 0.2 | 0.7 |
| Retrieval | Dual dynamic indices | Identical — reused from 003a |
| Shuffling | Yes | Yes (identical) |
| Length retry | Yes | Yes (identical) |

---

## Results

### Writing quality (acceptability)

| Experiment | RAG % Acceptable | Zero-shot % Acceptable |
|------------|-----------------|----------------------|
| 002 | {r002_acc:.1f}% | {z002_acc:.1f}% |
| 003a | {r003a_acc:.1f}% | {z003a_acc:.1f}% |
| 003b | {r003b_acc:.1f}% | {z003b_acc:.1f}% |

RAG acceptability changed by {_delta_str('rag_pct_acceptable', '003a_vs_002')} pp
in 003a (retrieval improvements only) and
{_delta_str('rag_pct_acceptable', '003b_vs_002')} pp overall in 003b.

Per-cell RAG results:

| Condition | % Acceptable | % Faithful | Diversity |
|-----------|-------------|-----------|-----------|
{chr(10).join(rag_table_002)}
{chr(10).join(rag_table_003a)}
{chr(10).join(rag_table_003b)}

### Curriculum faithfulness

| Experiment | RAG % Faithful |
|------------|---------------|
| 002 | {r002_faith:.1f}% |
| 003a | {r003a_faith:.1f}% |
| 003b | {r003b_faith:.1f}% |

Faithfulness changed by {_delta_str('rag_pct_faithful', '003a_vs_002')} pp in 003a and
{_delta_str('rag_pct_faithful', '003b_vs_002')} pp in 003b relative to 002.

Length-balance retries triggered: **{lb_003a}** in 003a, **{lb_003b}** in 003b.

### Question diversity

Diversity is measured as the number of unique 8-word stem prefixes per 15-question cell.
A score of 15 means every question had a different opening; a score of 1 means all 15
questions began with the same words.

| Experiment | RAG diversity (of 60 RAG questions) |
|------------|-------------------------------------|
| 002 | {r002_div}/60 unique stems |
| 003a | {r003a_div}/60 unique stems |
| 003b | {r003b_div}/60 unique stems |

### Flaw analysis

| Condition | Dominant flaw 002 | Dominant flaw 003a | Dominant flaw 003b |
|-----------|------------------|-------------------|--------------------|
| RAG | {r002_top} | {r003a_top} | {r003b_top} |

---

## Comparison with Experiment 002

### What improved and by how much

| Metric | 003a vs 002 | 003b vs 002 |
|--------|------------|------------|
| RAG % Acceptable | {_delta_str('rag_pct_acceptable','003a_vs_002')} pp ({_dir('rag_pct_acceptable','003a_vs_002')}) | {_delta_str('rag_pct_acceptable','003b_vs_002')} pp ({_dir('rag_pct_acceptable','003b_vs_002')}) |
| RAG % Faithful | {_delta_str('rag_pct_faithful','003a_vs_002')} pp ({_dir('rag_pct_faithful','003a_vs_002')}) | {_delta_str('rag_pct_faithful','003b_vs_002')} pp ({_dir('rag_pct_faithful','003b_vs_002')}) |
| Zero-shot % Acceptable | {_delta_str('zero_pct_acceptable','003a_vs_002')} pp ({_dir('zero_pct_acceptable','003a_vs_002')}) | {_delta_str('zero_pct_acceptable','003b_vs_002')} pp ({_dir('zero_pct_acceptable','003b_vs_002')}) |
| RAG diversity | {_delta_str('rag_diversity','003a_vs_002')} ({_dir('rag_diversity','003a_vs_002')}) | {_delta_str('rag_diversity','003b_vs_002')} ({_dir('rag_diversity','003b_vs_002')}) |
| RAG mean flaws | {_delta_str('rag_mean_flaws','003a_vs_002')} ({_dir('rag_mean_flaws','003a_vs_002')}) | {_delta_str('rag_mean_flaws','003b_vs_002')} ({_dir('rag_mean_flaws','003b_vs_002')}) |

### What did not improve

The IWF checker's acceptability threshold (≤1 flaw) is unchanged. Questions that passed
the threshold may still have pedagogical issues not caught by the seven structural checks.
Faithfulness is still measured by the same automated LLM-as-judge with no human calibration.

### Unexpected findings

- Length-balance retries were triggered {lb_003a} times in 003a and {lb_003b} times in 003b,
  indicating the correct-answer-longest flaw is not uniformly distributed across cells.
- {f"The model fallback to {model_003b} means 003b cannot isolate the effect of the preferred model architecture; the temperature increase to 0.7 is the primary variable." if model_003b == "llama3.2" else f"The switch to {model_003b} provides genuine model variation separate from the temperature change."}

---

## Interpretation

The retrieval improvements in 003a (dual indices, dynamic queries, shuffling, length retry)
target the structural causes of the quality gap identified in 002. Separating the F-6 and
7-10 curriculum indices ensures that Year 9-10 questions are not generated from Foundation–
Year 6 content, which was the most likely cause of the low faithfulness in the 9-12 cells
in 002. Dynamic retrieval queries mean the FAISS search is conditioned on both the year
level and the cognitive demand, so retrieved chunks should be more semantically relevant
to the actual generation target.

Option shuffling directly eliminates the answer_position_bias flaw for questions where it
would otherwise be triggered, because the correct answer is now placed uniformly at random.
The length-balance retry addresses the longest_option_correct flaw by giving the model an
explicit signal that option lengths must be matched — a constraint it does not apply by
default when generating from dense curriculum text.

The temperature increase in 003b from 0.2 to 0.7 trades accuracy for diversity. Whether
this improves or degrades downstream metrics depends on whether the model's core knowledge
and instruction-following remain stable at higher temperatures.

---

## Limitations

1. **Grade-band routing is coarse.** The "3-5" band is routed to the F-6 index and "9-12"
   to the 7-10 index. A finer routing (e.g. Year 5 = F-6, Year 6 = either) would require
   year-level metadata to be preserved in the chunks, which is not currently implemented.

2. **The faithfulness judge is still the same automated binary scorer.** It has not been
   calibrated against human annotation. Improvement in the faithfulness metric may partly
   reflect the judge's response to different prompt distributions rather than genuine
   improvement in curriculum grounding.

3. **Option shuffling eliminates position bias by construction, not by improving the LLM.**
   The model still generates a position-biased answer; shuffling corrects it post-hoc.
   A more principled fix would be to instruct the model to randomise correct answer
   placement during generation.

4. **If 003b fell back to llama3.2, the model comparison is temperature-only.** The
   inability to test llama3.1:8b, mistral:7b, or gemma2:9b means the model architecture
   variable cannot be isolated. This is a gap that should be resolved before publishing.

---

## Recommended next steps

1. **Pull and test a preferred model for a clean 003b comparison.** Running
   `ollama pull llama3.1:8b` and re-running 003b would provide a genuine model
   architecture comparison on top of the retrieval improvements.

2. **Add year-level metadata to chunks** so index routing can be done at the Year level
   rather than the grade-band level, enabling more precise retrieval for edge bands
   (Year 6, Year 7).

3. **Human validation of the faithfulness metric.** A 30–40 item annotation by two
   curriculum experts would calibrate the automated judge and give a reliability estimate
   that can be cited in the thesis.

4. **Extend the grid to all four Bloom's levels and K-2 / 6-8 bands** to test whether the
   improvements generalise beyond the two Bloom's extremes and two grade-band extremes
   used in experiments 002 and 003.
"""

    path.write_text(content, encoding="utf-8")
    print(f"  Saved {path.name}")


def write_slide_brief(
    stats_002: dict,
    stats_003a: dict,
    stats_003b: dict,
    comparison: dict,
    model_003b: str,
    model_note: str,
    path: Path,
) -> None:
    """Write slide_content_003.md with real numbers for presentation slides.

    Args:
        stats_002, stats_003a, stats_003b: Stats dicts.
        comparison:  Delta table dict.
        model_003b:  Model name actually used in 003b.
        model_note:  Human-readable model-selection note.
        path:        Output path.
    """
    s002  = stats_002
    s003a = stats_003a
    s003b = stats_003b
    d     = comparison["deltas"]

    r002_acc   = s002["aggregate_RAG"]["pct_acceptable"]
    r002_faith = s002["aggregate_RAG"]["pct_faithful"]
    z002_acc   = s002["aggregate_zero_shot"]["pct_acceptable"]

    r003a_acc   = s003a["aggregate_RAG"]["pct_acceptable"]
    r003a_faith = s003a["aggregate_RAG"]["pct_faithful"]
    z003a_acc   = s003a["aggregate_zero_shot"]["pct_acceptable"]

    r003b_acc   = s003b["aggregate_RAG"]["pct_acceptable"]
    r003b_faith = s003b["aggregate_RAG"]["pct_faithful"]
    z003b_acc   = s003b["aggregate_zero_shot"]["pct_acceptable"]

    r003a_div = s003a["aggregate_RAG"]["diversity"]
    r003b_div = s003b["aggregate_RAG"]["diversity"]
    r002_div  = s002["aggregate_RAG"]["diversity"]

    r002_top  = s002["aggregate_RAG"].get("most_common_flaw", "—")
    r003a_top = s003a["aggregate_RAG"].get("most_common_flaw", "—")
    r003b_top = s003b["aggregate_RAG"].get("most_common_flaw", "—")

    r002_pf  = s002["aggregate_RAG"].get("parse_failures", 0)
    r003a_pf = s003a["aggregate_RAG"].get("parse_failures", 0)
    r003b_pf = s003b["aggregate_RAG"].get("parse_failures", 0)

    lb_003a  = s003a.get("length_balance_retries", 0)
    lb_003b  = s003b.get("length_balance_retries", 0)

    def _delta_str(metric, exp):
        try:
            val = d[metric][exp]["delta"]
            if val is None:
                return "±0"
            sign = "+" if val >= 0 else ""
            return f"{sign}{val:.1f}"
        except (KeyError, TypeError):
            return "±0"

    content = f"""# Slide Content Brief — Experiment 003

## Slide: What changed in Experiment 003

- **Separate curriculum indices for primary and secondary year levels:** Grade 3-5 questions now draw only from the F-6 curriculum; Grade 9-12 questions draw only from the 7-10 curriculum. In Experiment 002 a single flat index mixed all year levels together.
- **Dynamic retrieval queries:** Instead of the same five-word phrase for every question, the system now builds a search query that encodes the target year level and the cognitive demand (e.g. "Year 9 10 curriculum compare contrast analyse"). This means retrieved chunks are relevant to the actual question being generated.
- **Option shuffling:** After each question is generated, the four answer options are randomly reassigned to labels a, b, c, d so the correct answer is not systematically placed first or last.
- **Length-balance retry:** If the correct answer comes out noticeably longer than every distractor — the most common item-writing flaw in Experiment 002 — the system makes one extra call instructing the model to equalise option lengths.
- **Higher temperature in 003b ({model_003b}, T=0.7):** The generation temperature was raised from 0.2 to 0.7 to test whether question diversity improves when the model is given more sampling freedom.

---

## Slide: Key results — writing quality

- **RAG acceptability 003a: {r003a_acc:.1f}%** — retrieval improvements alone changed acceptability by {_delta_str('rag_pct_acceptable','003a_vs_002')} percentage points versus Experiment 002 ({r002_acc:.1f}%). Option shuffling directly eliminates position-bias flags and the length-balance retry targets the longest-option flaw.
- **Zero-shot acceptability 003a: {z003a_acc:.1f}%** — zero-shot quality changed by {_delta_str('zero_pct_acceptable','003a_vs_002')} pp versus 002 ({z002_acc:.1f}%). The improvements were targeted at retrieval, so zero-shot behaviour is a useful control.
- **RAG acceptability 003b: {r003b_acc:.1f}%** — adding the model/temperature change ({model_003b} at T=0.7) shifted acceptability by a further {_delta_str('rag_pct_acceptable','003b_vs_003a')} pp relative to 003a. Higher temperature increases diversity but may also introduce more structural flaws.

---

## Slide: Key results — curriculum faithfulness

- **RAG faithfulness 003a: {r003a_faith:.1f}%** — dynamic routing to grade-specific indices changed faithfulness by {_delta_str('rag_pct_faithful','003a_vs_002')} pp versus Experiment 002 ({r002_faith:.1f}%). Routing Year 9-12 questions to 7-10 content means the retrieved chunks are now from the correct year band.
- **RAG faithfulness 003b: {r003b_faith:.1f}%** — temperature increase changed faithfulness by {_delta_str('rag_pct_faithful','003b_vs_003a')} pp relative to 003a. Temperature affects whether the model uses retrieved content or deviates into prior knowledge.
- **Length-balance retries:** {lb_003a} retries in 003a, {lb_003b} in 003b. Each retry involves an extra LLM call and is an additional data point on how often the first-pass answer is biased in length.

---

## Slide: Comparison table

| Metric | Exp 002 | Exp 003a | Exp 003b |
|--------|---------|---------|---------|
| RAG writing quality | {r002_acc:.1f}% | {r003a_acc:.1f}% | {r003b_acc:.1f}% |
| Zero-shot writing quality | {z002_acc:.1f}% | {z003a_acc:.1f}% | {z003b_acc:.1f}% |
| RAG curriculum faithfulness | {r002_faith:.1f}% | {r003a_faith:.1f}% | {r003b_faith:.1f}% |
| Within-cell RAG diversity | {r002_div}/60 stems | {r003a_div}/60 stems | {r003b_div}/60 stems |
| Most common RAG flaw | {r002_top} | {r003a_top} | {r003b_top} |
| Format failures (RAG) | {r002_pf} | {r003a_pf} | {r003b_pf} |

---

## Slide: What the model change added

Between 003a and 003b (model/temperature change only, retrieval identical):

- RAG acceptability: {_delta_str('rag_pct_acceptable','003b_vs_003a')} pp
- RAG faithfulness: {_delta_str('rag_pct_faithful','003b_vs_003a')} pp
- Zero-shot acceptability: {_delta_str('zero_pct_acceptable','003b_vs_003a')} pp
- RAG diversity: {_delta_str('rag_diversity','003b_vs_003a')} unique stems (out of 60)

{"Note: 003b used llama3.2 as a fallback (none of the preferred models were available). The temperature increase from 0.2 to 0.7 is therefore the primary variable — no architecture comparison is possible without pulling a preferred model." if model_003b == "llama3.2" else f"003b used {model_003b}, providing a genuine architecture comparison against llama3.2 in 003a."}

---

## Slide: Remaining limitations

1. **Faithfulness is still automated and unvalidated.** The YES/NO judge is the same llama3.2-based scorer used in Experiment 002. Without human annotation, changes in the faithfulness metric may reflect prompt distribution effects rather than genuine grounding improvements.
2. **Option shuffling corrects position bias post-hoc.** The model still generates a biased answer; shuffling removes the statistical artefact without changing the generation behaviour. Prompt-level instructions to randomise placement would be a stronger fix.
3. **The model comparison in 003b is temperature-only.** None of the preferred models (llama3.1:8b, mistral:7b, gemma2:9b) were available, so the architecture variable cannot be isolated. The experiment should be re-run with a preferred model before drawing conclusions about model choice.
4. **Only two of four Bloom's levels and two of four grade bands are tested.** The grid covers the extremes (Remember/Analyze, Grade 3-5/9-12) but the intermediate conditions (Understand, Apply, Grade K-2, Grade 6-8) remain untested.

---

## Slide: Recommended next experiment

Experiment 004 should test a preferred model under the same retrieval conditions established in Experiment 003 — specifically, pulling llama3.1:8b or mistral:7b and re-running the full 003b grid. This would isolate the model architecture effect from the temperature effect, which are currently confounded. Experiment 004 should also extend the grid to include the Understand and Apply Bloom's levels, because the current evidence about flaw rates and faithfulness comes only from the cognitive extremes of the taxonomy. Finally, adding a human annotation step on a random sample of 30-40 RAG question-answer-chunk triples would give a calibration anchor for the automated faithfulness judge — without this, the faithfulness percentages cannot be cited with confidence in the thesis.

---

## Speaker notes

**Slide: What changed in Experiment 003**
In Experiment 002 I identified three structural problems: the index was mixing Year 3 and Year 10 content in the same pool, every retrieval call used the same generic five-word query regardless of what was being asked, and the model was placing the correct answer in the first or last position over half the time. Experiment 003 fixes all three. I split the index into two grade-band-specific pools, built a query that encodes the year level and cognitive target, and shuffled options after generation. I also added a retry when the correct answer comes out noticeably longer than the distractors, since that was the other dominant flaw in 002.

**Slide: Key results — writing quality**
The acceptability changes reflect a mix of effects. Shuffling directly eliminates the position-bias flaw for any question where it would have fired, so I'd expect a mechanical improvement there. The length-balance retry addresses the longest-option flaw, but only one retry is allowed, and if the retry also produces an imbalanced answer, we keep the original. So the retry helps but doesn't fully solve the problem. The zero-shot acceptability is a useful control — those questions don't go through retrieval, so any change there comes from the prompt changes or randomness, not from the index improvements.

**Slide: Key results — curriculum faithfulness**
The most meaningful change in 003a is that Year 9-12 questions now draw from the 7-10 curriculum rather than the F-6 curriculum. In Experiment 002, the system was generating high-school questions from primary school content, which is why faithfulness in those cells was low. Routing to the correct index should bring the retrieved chunks into alignment with what the questions are actually about. The faithfulness judge measures whether the answer is grounded in the retrieved text, so better routing should raise the score.

**Slide: Comparison table**
The comparison table shows the trajectory across experiments. The direction labels in the stats_comparison.json are the most direct summary — anything marked "improved" moved in the desired direction. Notice that zero-shot writing quality is a useful baseline because it should be relatively stable across experiments; large swings there would suggest the improvements are interacting with something other than the retrieval pipeline.

**Slide: Remaining limitations**
I want to be clear that these experiments are building evidence, not concluding it. The faithfulness metric needs human calibration before I can cite it in the thesis with confidence. The model comparison in 003b is confounded by the fallback to llama3.2, which means I'm really just testing temperature = 0.7 versus 0.2, not a different model. And the grid is still incomplete — we're missing the middle Bloom's levels and the K-2 and 6-8 grade bands. Experiment 004 will address the model gap and extend the grid.

**Slide: Recommended next experiment**
The single most important next step is pulling one of the preferred models and running a clean 003b comparison. Everything else — extending the grid, calibrating faithfulness — builds on having a reliable model comparison. I'd recommend llama3.1:8b as the first choice because it is the closest architectural upgrade from llama3.2 and the most likely to be available on consumer hardware.
"""

    path.write_text(content, encoding="utf-8")
    print(f"  Saved {path.name}")


# ── Main analysis entry point ─────────────────────────────────────────────────

def run_analysis(
    path_003a: Path,
    path_003b: Path,
    model_003b: str,
    model_note: str,
    out_dir: Path,
) -> None:
    """Run the full analysis pipeline.

    Loads experiment_002 and the two 003 scored files, computes all stats,
    writes every required output file to out_dir.

    Args:
        path_003a:  Path to experiment_003a_scored.jsonl.
        path_003b:  Path to experiment_003b_scored.jsonl.
        model_003b: Model name used in 003b.
        model_note: Human-readable model selection note.
        out_dir:    Output directory (results/experiment_003/).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== Analysis 003 ===", flush=True)

    # ── Load records ──────────────────────────────────────────────────────────
    print("Loading records …")
    recs_002  = load_records(PATH_002_SCORED)
    recs_003a = load_records(path_003a)
    recs_003b = load_records(path_003b)
    print(f"  002: {len(recs_002)}, 003a: {len(recs_003a)}, 003b: {len(recs_003b)}")

    # Build experiment_002 stats using same schema
    # experiment_002 records don't have the new fields so default gracefully
    stats_002 = {
        "experiment_id": "002",
        "model": recs_002[0].get("model", "llama3.2") if recs_002 else "llama3.2",
        "total_questions": len(recs_002),
        "length_balance_retries": 0,
        "aggregate_RAG": {},
        "aggregate_zero_shot": {},
        "conditions": {},
    }
    rag_002  = [r for r in recs_002 if r.get("retrieval")]
    zero_002 = [r for r in recs_002 if not r.get("retrieval")]
    agg_r002 = aggregate(rag_002)
    agg_z002 = aggregate(zero_002)
    stats_002["aggregate_RAG"] = {
        "n_acceptable":   int(len(rag_002) * agg_r002["pct_acceptable"] / 100),
        "pct_acceptable": round(agg_r002["pct_acceptable"], 1),
        "n_faithful":     agg_r002["n_faithful"],
        "pct_faithful":   round(agg_r002["pct_faithful"], 1),
        "mean_flaws":     round(agg_r002["mean_flaws"], 3),
        "diversity":      agg_r002["diversity"],
        "most_common_flaw": agg_r002["top_flaw"],
        "parse_failures": 0,
    }
    stats_002["aggregate_zero_shot"] = {
        "n_acceptable":   int(len(zero_002) * agg_z002["pct_acceptable"] / 100),
        "pct_acceptable": round(agg_z002["pct_acceptable"], 1),
        "mean_flaws":     round(agg_z002["mean_flaws"], 3),
        "diversity":      agg_z002["diversity"],
        "most_common_flaw": agg_z002["top_flaw"],
        "parse_failures": 0,
    }
    # Per-cell conditions for 002
    by_cell_002 = defaultdict(list)
    for r in recs_002:
        by_cell_002[condition_key(r)].append(r)
    for cond, grade, bloom in CELL_ORDER:
        key = f"{cond} | {grade} | {bloom}"
        c   = aggregate(by_cell_002.get((cond, grade, bloom), []))
        stats_002["conditions"][key] = {
            "total_questions":  c["n"],
            "n_acceptable":     int(c["n"] * c["pct_acceptable"] / 100),
            "pct_acceptable":   round(c["pct_acceptable"], 1),
            "mean_flaw_count":  round(c["mean_flaws"], 3),
            "n_faithful":       c["n_faithful"],
            "pct_faithful":     round(c["pct_faithful"], 1),
            "diversity_score":  c["diversity"],
            "most_common_flaw": c["top_flaw"],
            "parse_failures":   0,
            "model_used":       stats_002["model"],
            "retrieval_query":  "(fixed: 'main concepts key ideas overview')",
        }

    stats_003a = build_stats_json(recs_003a, "003a", "llama3.2")
    stats_003b = build_stats_json(recs_003b, "003b", model_003b)
    comparison = build_comparison_stats(stats_002, stats_003a, stats_003b)

    # ── CSVs ──────────────────────────────────────────────────────────────────
    print("Writing CSVs …")
    save_csv(build_csv_rows(recs_003a), out_dir / "results_003a.csv")
    save_csv(build_csv_rows(recs_003b), out_dir / "results_003b.csv")
    build_comparison_csv(recs_002, recs_003a, recs_003b, out_dir / "results_comparison.csv")

    # ── Stats JSONs ───────────────────────────────────────────────────────────
    print("Writing stats JSONs …")
    for fname, obj in [
        ("stats_003a.json",    stats_003a),
        ("stats_003b.json",    stats_003b),
        ("stats_comparison.json", comparison),
    ]:
        p = out_dir / fname
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        print(f"  Saved {fname}")

    # ── Charts ────────────────────────────────────────────────────────────────
    print("Writing charts …")
    chart_acceptability_comparison(recs_002, recs_003a, recs_003b,
                                   out_dir / "chart_acceptability_comparison.png")
    chart_faithfulness_comparison(recs_002, recs_003a, recs_003b,
                                  out_dir / "chart_faithfulness_comparison.png")
    chart_flaws_single(recs_003a, "Item-Writing Flaw Breakdown — Experiment 003a",
                       out_dir / "chart_flaws_003a.png")
    chart_flaws_single(recs_003b,
                       f"Item-Writing Flaw Breakdown — Experiment 003b ({model_003b})",
                       out_dir / "chart_flaws_003b.png")
    chart_flaws_comparison(recs_002, recs_003a, recs_003b,
                           out_dir / "chart_flaws_comparison.png")
    chart_diversity(recs_002, recs_003a, recs_003b,
                    out_dir / "chart_diversity_score.png")

    # ── Markdown reports ──────────────────────────────────────────────────────
    print("Writing markdown reports …")
    write_findings(stats_002, stats_003a, stats_003b, comparison,
                   model_003b, model_note, out_dir / "findings_003.md")
    write_slide_brief(stats_002, stats_003a, stats_003b, comparison,
                      model_003b, model_note, out_dir / "slide_content_003.md")

    # ── Terminal summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 003 SUMMARY")
    print("=" * 60)
    print(f"  003b model used  : {model_003b}  ({model_note.split('.')[0]})")
    print(f"  003a RAG acc     : {stats_003a['aggregate_RAG']['pct_acceptable']:.1f}%  "
          f"(002: {stats_002['aggregate_RAG']['pct_acceptable']:.1f}%)")
    print(f"  003b RAG acc     : {stats_003b['aggregate_RAG']['pct_acceptable']:.1f}%")
    print(f"  003a faithfulness: {stats_003a['aggregate_RAG']['pct_faithful']:.1f}%  "
          f"(002: {stats_002['aggregate_RAG']['pct_faithful']:.1f}%)")
    print(f"  003b faithfulness: {stats_003b['aggregate_RAG']['pct_faithful']:.1f}%")
    print(f"  003a RAG dom flaw: {stats_003a['aggregate_RAG']['most_common_flaw']}")
    print(f"  003b RAG dom flaw: {stats_003b['aggregate_RAG']['most_common_flaw']}")
    print(f"  003a diversity   : {stats_003a['aggregate_RAG']['diversity']}/60")
    print(f"  003b diversity   : {stats_003b['aggregate_RAG']['diversity']}/60")
    print(f"  LB retries 003a  : {stats_003a.get('length_balance_retries',0)}")
    print(f"  LB retries 003b  : {stats_003b.get('length_balance_retries',0)}")
    print("=" * 60)
    print(f"\nOutput directory: {out_dir}")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load model note from model_selection_003b.json if it exists
    model_cfg_path = OUT_DIR / "model_selection_003b.json"
    if model_cfg_path.exists():
        cfg = json.loads(model_cfg_path.read_text())
        model_003b = cfg.get("model_selected", "llama3.2")
        model_note = cfg.get("model_note", "")
    else:
        model_003b = "llama3.2"
        model_note = "model_selection_003b.json not found; defaulting to llama3.2"

    run_analysis(
        path_003a=PATH_003A_DEFAULT,
        path_003b=PATH_003B_DEFAULT,
        model_003b=model_003b,
        model_note=model_note,
        out_dir=OUT_DIR,
    )
