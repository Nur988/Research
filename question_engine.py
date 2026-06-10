"""
question_engine.py — standalone question generation for the K-12 RAG system.

No Streamlit dependency; safe to import in tests, notebooks, or CLI scripts.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
_RETRIEVAL_QUERY = "main concepts key ideas overview"
_RETRIEVAL_K = 8
_CHUNK_PREVIEW = 600

GRADE_DESCRIPTIONS: Dict[str, str] = {
    "K-2":  "kindergarten to grade 2 (ages 5–7). Use very simple words, short sentences, and concrete examples.",
    "3-5":  "grades 3 to 5 (ages 8–10). Use clear language and relatable everyday examples.",
    "6-8":  "grades 6 to 8 (ages 11–13). Use moderate vocabulary and expect some prior knowledge.",
    "9-12": "grades 9 to 12 (ages 14–18). Use academic vocabulary and expect abstract reasoning.",
}

BLOOM_DESCRIPTIONS: Dict[str, str] = {
    "Remember":   "recall facts and basic concepts (e.g., define, list, name, identify)",
    "Understand": "explain ideas in own words (e.g., describe, summarise, explain, classify)",
    "Apply":      "use information in new situations (e.g., solve, demonstrate, use, calculate)",
    "Analyze":    "draw connections and break down information (e.g., compare, contrast, differentiate)",
}

_MCQ_SCHEMA   = '{"question": "...", "options": {"a": "...", "b": "...", "c": "...", "d": "..."}, "correct": "a", "explanation": "..."}'
_SHORT_SCHEMA = '{"question": "...", "sample_answer": "...", "explanation": "..."}'
_TF_SCHEMA    = '{"question": "...", "correct": "True" or "False", "explanation": "..."}'

_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Your previous response could not be parsed as a JSON array. "
    "Return ONLY a raw JSON array — start with [ and end with ]. "
    "No markdown fences, no extra keys, no commentary outside the array."
)


def _build_prompt(
    context: Optional[str],
    n: int,
    q_type: str,
    grade_band: str,
    bloom_level: str,
) -> str:
    """Build the generation prompt.

    When context is None the context block is omitted entirely (zero-shot
    condition). The rest of the prompt — instructions, grade/Bloom's
    descriptions, JSON schema, rules — is identical in both cases.
    """
    grade_desc = GRADE_DESCRIPTIONS[grade_band]
    bloom_desc = BLOOM_DESCRIPTIONS[bloom_level]

    if q_type == "MCQ":
        schema = _MCQ_SCHEMA
        type_instruction = "multiple-choice questions with 4 options (a, b, c, d) and one correct answer"
    elif q_type == "Short Answer":
        schema = _SHORT_SCHEMA
        type_instruction = "short-answer questions with a sample answer (2–4 sentences)"
    else:  # True/False
        schema = _TF_SCHEMA
        type_instruction = "true/false questions"

    if context is not None:
        context_block = (
            "Use ONLY the information in the context below. Do not use external knowledge.\n\n"
            f"Context:\n{context}\n\n"
        )
    else:
        context_block = ""

    return (
        "You are a K-12 curriculum expert creating assessment questions.\n\n"
        f"Your task: Generate exactly {n} {type_instruction}.\n"
        f"Target audience: {grade_desc}\n"
        f"Bloom's Taxonomy level: {bloom_level} — {bloom_desc}\n\n"
        f"{context_block}"
        "Rules:\n"
        "- Questions must be appropriate for the grade band in vocabulary and complexity.\n"
        "- Each question must target the specified Bloom's level.\n"
        "- Do not repeat questions.\n"
        "- Return ONLY a valid JSON array. No markdown, no explanation outside the JSON.\n\n"
        f"JSON schema for each item:\n{schema}\n\n"
        f"Return a JSON array of exactly {n} items."
    )


def _parse_response(raw: str) -> List[dict]:
    """Parse LLM output into a list of question dicts.

    Strips markdown code fences defensively even when format='json' is set,
    then validates the result is a non-empty list of objects.

    Raises:
        json.JSONDecodeError: output is not valid JSON.
        ValueError: output is valid JSON but not a list of objects.
    """
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    data = json.loads(text)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}: {text[:120]}")
    if data and not isinstance(data[0], dict):
        raise ValueError(f"Array elements must be objects, got {type(data[0]).__name__}")

    return data


def generate_questions(
    grade_band: str,
    bloom_level: str,
    question_type: str,
    n: int,
    retrieval: bool = True,
    model: Optional[str] = None,
    ollama_url: Optional[str] = None,
    temperature: float = 0.2,
    vs=None,
) -> Dict[str, Any]:
    """Generate K-12 assessment questions, optionally grounded in FAISS retrieval.

    Args:
        grade_band:    One of "K-2", "3-5", "6-8", "9-12".
        bloom_level:   One of "Remember", "Understand", "Apply", "Analyze".
        question_type: One of "MCQ", "Short Answer", "True/False".
        n:             Number of questions to generate.
        retrieval:     True  → RAG condition: FAISS similarity search grounds the prompt.
                       False → zero-shot condition: same prompt, no context block.
        model:         Ollama chat model name. Defaults to DEFAULT_MODEL.
        ollama_url:    Ollama base URL. Defaults to DEFAULT_OLLAMA_URL.
        temperature:   LLM sampling temperature.
        vs:            FAISS vectorstore instance. Required when retrieval=True.

    Returns:
        {
            "questions":        list[dict],  # parsed question objects ([] on total failure)
            "retrieved_chunks": list[dict],  # chunks used as context ([] when retrieval=False)
            "raw_prompt":       str,         # prompt sent on the last attempt
            "parse_failed":     bool,        # True if both parse attempts failed
            "model":            str,
            "retrieval":        bool,
        }

    Raises:
        ValueError: if retrieval=True and vs is None.
    """
    model = model or DEFAULT_MODEL
    url = ollama_url or DEFAULT_OLLAMA_URL

    if retrieval and vs is None:
        raise ValueError("vs (vectorstore) must be provided when retrieval=True")

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieved_chunks: List[dict] = []
    context: Optional[str] = None

    if retrieval:
        sample_docs: List[Document] = vs.similarity_search(_RETRIEVAL_QUERY, k=_RETRIEVAL_K)
        for d in sample_docs:
            retrieved_chunks.append({
                "source":  d.metadata.get("source", ""),
                "chunk":   d.metadata.get("chunk", ""),
                "content": d.page_content[:_CHUNK_PREVIEW],
            })
        context = "\n\n".join(c["content"] for c in retrieved_chunks)

    # ── Build prompt and LLM ─────────────────────────────────────────────────
    prompt = _build_prompt(context, n, question_type, grade_band, bloom_level)
    llm = ChatOllama(model=model, temperature=temperature, base_url=url, format="json")

    # ── Attempt 1 ────────────────────────────────────────────────────────────
    response = llm.invoke(prompt)
    raw = response.content.strip()
    logger.debug("Attempt 1 raw output [model=%s retrieval=%s] (first 300 chars): %s",
                 model, retrieval, raw[:300])

    try:
        questions = _parse_response(raw)
        logger.info("Questions generated successfully on attempt 1 [model=%s retrieval=%s n=%d]",
                    model, retrieval, len(questions))
        return {
            "questions":        questions,
            "retrieved_chunks": retrieved_chunks,
            "raw_prompt":       prompt,
            "parse_failed":     False,
            "model":            model,
            "retrieval":        retrieval,
        }
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Parse attempt 1 failed [model=%s retrieval=%s]: %s | raw=%s",
            model, retrieval, exc, raw[:200],
        )

    # ── Attempt 2 — retry with clarification appended ────────────────────────
    retry_prompt = prompt + _RETRY_SUFFIX
    response2 = llm.invoke(retry_prompt)
    raw2 = response2.content.strip()
    logger.debug("Attempt 2 raw output [model=%s retrieval=%s] (first 300 chars): %s",
                 model, retrieval, raw2[:300])

    try:
        questions = _parse_response(raw2)
        logger.info("Questions generated successfully on attempt 2 (retry) [model=%s retrieval=%s n=%d]",
                    model, retrieval, len(questions))
        return {
            "questions":        questions,
            "retrieved_chunks": retrieved_chunks,
            "raw_prompt":       retry_prompt,
            "parse_failed":     False,
            "model":            model,
            "retrieval":        retrieval,
        }
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error(
            "Parse attempt 2 (retry) failed [model=%s retrieval=%s]: %s | raw=%s",
            model, retrieval, exc, raw2[:200],
        )
        return {
            "questions":        [],
            "retrieved_chunks": retrieved_chunks,
            "raw_prompt":       retry_prompt,
            "parse_failed":     True,
            "model":            model,
            "retrieval":        retrieval,
        }
