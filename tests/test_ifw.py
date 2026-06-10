"""
tests/test_ifw.py — unit tests for all 7 IWF checks in ifw_check.py.

Two tests per check:
  - one question that SHOULD trigger the flaw  (assert flaw IN result["flaws"])
  - one question that should NOT trigger it    (assert flaw NOT IN result["flaws"])

Run from rag-tutor/ with:
    python -m pytest tests/test_ifw.py -v
"""

import sys
from pathlib import Path

# Allow importing ifw_check from the parent rag-tutor/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from ifw_check import score_question


# ── Shared helper ─────────────────────────────────────────────────────────────

def make_q(stem: str, options: dict, answer: str) -> dict:
    """Build a minimal question dict matching the JSONL schema."""
    return {"question": stem, "options": options, "answer": answer}


# ═════════════════════════════════════════════════════════════════════════════
# 1. longest_option_correct
#    Flaw fires when the correct option is strictly longer than every distractor.
# ═════════════════════════════════════════════════════════════════════════════

def test_longest_option_correct_triggers():
    # 'b' is dramatically longer than the three one-word distractors.
    # A test-wise student can guess 'b' without knowing the content.
    q = make_q(
        "What is photosynthesis?",
        {
            "a": "Sleep",
            "b": "The process by which green plants use sunlight to convert carbon dioxide and water into glucose",
            "c": "Rain",
            "d": "Wind",
        },
        "b",
    )
    result = score_question(q)
    assert "longest_option_correct" in result["flaws"], (
        "Expected flaw when correct option is much longer than all distractors"
    )


def test_longest_option_correct_does_not_trigger():
    # All four options are short city names — no length cue available.
    q = make_q(
        "What is the capital of France?",
        {
            "a": "London",
            "b": "Berlin",
            "c": "Paris",
            "d": "Madrid",
        },
        "c",
    )
    result = score_question(q)
    assert "longest_option_correct" not in result["flaws"], (
        "Should not fire when options are similar in length"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2. answer_position_bias
#    Flaw fires when the correct answer is the first or last option key.
# ═════════════════════════════════════════════════════════════════════════════

def test_answer_position_bias_triggers():
    # Correct answer is 'd' — the last option in sorted order ["a","b","c","d"].
    # Test-makers sometimes unconsciously place the correct answer last.
    q = make_q(
        "Which planet is closest to the Sun?",
        {
            "a": "Earth",
            "b": "Venus",
            "c": "Jupiter",
            "d": "Mercury",
        },
        "d",
    )
    result = score_question(q)
    assert "answer_position_bias" in result["flaws"], (
        "Expected flaw when correct answer is in last position"
    )


def test_answer_position_bias_does_not_trigger():
    # Correct answer is 'b' — a middle position, not first or last.
    q = make_q(
        "Which planet is closest to the Sun?",
        {
            "a": "Earth",
            "b": "Mercury",
            "c": "Jupiter",
            "d": "Saturn",
        },
        "b",
    )
    result = score_question(q)
    assert "answer_position_bias" not in result["flaws"], (
        "Should not fire when correct answer is in a middle position"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3. all_none_of_above
#    Flaw fires when any option text contains "all of the above" or
#    "none of the above" (case-insensitive).
# ═════════════════════════════════════════════════════════════════════════════

def test_all_none_of_above_triggers():
    # Option 'd' is "All of the above" — students with partial knowledge can
    # exploit this without full understanding.
    q = make_q(
        "Which of the following are primary colours?",
        {
            "a": "Red",
            "b": "Blue",
            "c": "Yellow",
            "d": "All of the above",
        },
        "d",
    )
    result = score_question(q)
    assert "all_none_of_above" in result["flaws"], (
        "Expected flaw when an option contains 'all of the above'"
    )


def test_all_none_of_above_does_not_trigger():
    # All four options are specific, substantive answers — no catch-all phrase.
    q = make_q(
        "Which of the following are primary colours?",
        {
            "a": "Red",
            "b": "Blue",
            "c": "Yellow",
            "d": "Orange",
        },
        "a",
    )
    result = score_question(q)
    assert "all_none_of_above" not in result["flaws"], (
        "Should not fire when no option contains a catch-all phrase"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4. negated_stem
#    Flaw fires when the stem contains the all-caps word NOT or EXCEPT.
# ═════════════════════════════════════════════════════════════════════════════

def test_negated_stem_triggers():
    # "NOT" appears in full capitals — easy to miss under exam pressure.
    q = make_q(
        "Which of the following is NOT a mammal?",
        {
            "a": "Dog",
            "b": "Whale",
            "c": "Bat",
            "d": "Salmon",
        },
        "d",
    )
    result = score_question(q)
    assert "negated_stem" in result["flaws"], (
        "Expected flaw when stem contains 'NOT' in full capitals"
    )


def test_negated_stem_does_not_trigger():
    # "not" appears in lowercase — the check is case-sensitive for capitals only.
    # Lowercase negation is considered normal prose and does not trigger.
    q = make_q(
        "Which animal is not a reptile?",
        {
            "a": "Lizard",
            "b": "Snake",
            "c": "Turtle",
            "d": "Dog",
        },
        "d",
    )
    result = score_question(q)
    assert "negated_stem" not in result["flaws"], (
        "Should not fire when 'not' appears only in lowercase"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 5. duplicate_options
#    Flaw fires when two or more options are identical after lowercasing and
#    collapsing all whitespace.
# ═════════════════════════════════════════════════════════════════════════════

def test_duplicate_options_triggers():
    # 'b' and 'c' are "Four" / "four" — identical after normalisation.
    # The duplicated slot is wasted and may confuse test-takers.
    q = make_q(
        "What is 2 + 2?",
        {
            "a": "Three",
            "b": "Four",
            "c": "four",        # same as 'b' after lowercasing
            "d": "Five",
        },
        "b",
    )
    result = score_question(q)
    assert "duplicate_options" in result["flaws"], (
        "Expected flaw when two options are identical ignoring case"
    )


def test_duplicate_options_does_not_trigger():
    # All four options are distinct, even after normalisation.
    q = make_q(
        "What is 2 + 2?",
        {
            "a": "Three",
            "b": "Four",
            "c": "Five",
            "d": "Six",
        },
        "b",
    )
    result = score_question(q)
    assert "duplicate_options" not in result["flaws"], (
        "Should not fire when all options are distinct"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 6. stem_answer_overlap
#    Flaw fires when a 3-word sequence from the stem appears verbatim in the
#    correct answer but in none of the distractors.
# ═════════════════════════════════════════════════════════════════════════════

def test_stem_answer_overlap_triggers():
    # The 3-gram "water cycle describes" is in both the stem and option 'a'.
    # It does NOT appear in any distractor, so a student can match without
    # knowing what the water cycle actually is.
    q = make_q(
        "The water cycle describes the continuous movement of water on Earth.",
        {
            "a": "The water cycle describes how water moves between the surface and atmosphere",
            "b": "A classification system for ocean temperatures",
            "c": "A method used in agricultural irrigation",
            "d": "The study of underground water reservoirs",
        },
        "a",
    )
    result = score_question(q)
    assert "stem_answer_overlap" in result["flaws"], (
        "Expected flaw when a 3-gram from stem appears only in the correct answer"
    )


def test_stem_answer_overlap_does_not_trigger():
    # The same 3-gram "water cycle describes" now also appears in distractor 'b'.
    # Both correct and incorrect options share the phrase, so it provides no cue.
    q = make_q(
        "The water cycle describes the continuous movement of water on Earth.",
        {
            "a": "The water cycle describes how water moves between the surface and atmosphere",
            "b": "The water cycle describes only the process of evaporation",  # 3-gram also here
            "c": "A method used in agricultural irrigation",
            "d": "The study of underground water reservoirs",
        },
        "a",
    )
    result = score_question(q)
    assert "stem_answer_overlap" not in result["flaws"], (
        "Should not fire when the overlapping phrase also appears in a distractor"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 7. implausible_distractor
#    Flaw fires when any distractor is shorter than 40 % of the mean option
#    length across all options.
# ═════════════════════════════════════════════════════════════════════════════

def test_implausible_distractor_triggers():
    # Distractor 'd' is the single word "No" — far below 40 % of mean length.
    # Students can eliminate it immediately without subject-matter knowledge.
    q = make_q(
        "What is the primary function of mitochondria in a cell?",
        {
            "a": "Mitochondria controls the cell cycle and regulates division",
            "b": "Mitochondria synthesises proteins from amino acid sequences",
            "c": "Mitochondria produces ATP through cellular respiration",
            "d": "No",   # implausibly short distractor
        },
        "c",
    )
    result = score_question(q)
    assert "implausible_distractor" in result["flaws"], (
        "Expected flaw when a distractor is far shorter than the mean option length"
    )


def test_implausible_distractor_does_not_trigger():
    # All options are short but comparable in length — none falls below threshold.
    q = make_q(
        "What is the primary function of mitochondria?",
        {
            "a": "Protein synthesis",
            "b": "Cell division",
            "c": "Energy production",
            "d": "Waste removal",
        },
        "c",
    )
    result = score_question(q)
    assert "implausible_distractor" not in result["flaws"], (
        "Should not fire when all options are similar in length"
    )
