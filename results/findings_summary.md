# Experiment 002 — Findings Summary

**Dataset:** Australian Curriculum: Science F–6 v9 (182 chunks, ~1,000 tokens each)  
**Model:** llama3.2 · **Embeddings:** nomic-embed-text · **Questions:** 120 MCQ (8 cells × 15)

---

## Headline Numbers

| Condition | % IWF-Acceptable | % Faithful to Curriculum |
|-----------|-----------------|--------------------------|
| RAG | 82% | 55% |
| Zero-shot | 95% | N/A |

---

## Per-Condition Results

| Condition | N | Mean Flaws | % Acceptable | Top Flaw |
|-----------|---|-----------|-------------|----------|
| RAG \| Grade 3–5 \| Remember | 15 | 1.07 | 86.7% | answer\_position\_bias |
| RAG \| Grade 3–5 \| Analyze | 15 | 0.20 | 100.0% | duplicate\_options |
| RAG \| Grade 9–12 \| Remember | 15 | 1.60 | 46.7% | longest\_option\_correct |
| RAG \| Grade 9–12 \| Analyze | 15 | 0.93 | 93.3% | answer\_position\_bias |
| Zero-shot \| Grade 3–5 \| Remember | 15 | 0.87 | 100.0% | longest\_option\_correct |
| Zero-shot \| Grade 3–5 \| Analyze | 15 | 1.07 | 93.3% | longest\_option\_correct |
| Zero-shot \| Grade 9–12 \| Remember | 15 | 0.07 | 100.0% | longest\_option\_correct |
| Zero-shot \| Grade 9–12 \| Analyze | 15 | 0.93 | 86.7% | answer\_position\_bias |

---

## Key Findings

- **RAG degrades item quality at higher grade bands.** The worst-performing cell was RAG × Grade 9–12 × Remember, where only 46.7% of questions were acceptable. The model retrieved dense academic curriculum text and then produced options of very uneven length — the correct answer was consistently longer than all distractors — making the answer guessable without content knowledge.

- **Zero-shot produces cleaner items but cannot be curriculum-faithful.** Across all four zero-shot cells, acceptability averaged 95% with a mean flaw count of 0.73. However, those questions were drawn entirely from the model's pretrained knowledge (e.g. stop signs, photosynthesis, metaphors) rather than the target curriculum, meaning they cannot serve as valid assessments of what was actually taught.

- **RAG faithfulness is moderate, not high.** Even with a real 182-chunk curriculum index, 45% of RAG questions were judged unfaithful by the automated evaluator. Many retrieved chunks provided partial or contextual information rather than a direct supporting statement for the specific answer generated, suggesting that retrieval relevance and answer grounding are not equivalent.

---

## Interpreting the RAG vs Zero-Shot Tradeoff

RAG and zero-shot generation optimise for different goals that currently pull in opposite directions. Zero-shot produces items that score well on surface-level psychometric criteria — options are balanced in length, answers are spread across positions, distractors are plausible — because the model draws on a large and diverse training distribution. RAG forces the model to work within a narrow retrieved context, which introduces two compounding pressures: the correct answer tends to mirror the retrieved text closely (making it longer and more precise than the distractors), and the model must generate three plausible-but-wrong options for a topic it has just been shown once. The result is a meaningful quality penalty in exchange for curriculum alignment. For practical deployment, this suggests that RAG is necessary for assessment validity — items must be grounded in what was taught — but that prompt engineering, distractor-generation strategies, and post-hoc item review will be needed to close the quality gap before RAG-generated items are usable in classrooms.

---

## Limitations

1. **IWF checks are surface-level proxies, not expert judgement.** The seven item-writing flaw checks are deterministic rules derived from the psychometrics literature. They catch structural cues (option length, answer position, verbatim overlap) that correlate with poor item quality, but they do not assess content correctness, cognitive alignment with the stated Bloom's level, or pedagogical appropriateness. A question can pass all seven checks and still be factually wrong or educationally inappropriate.

2. **The faithfulness judge is automated and unvalidated against human raters.** The LLM-as-judge approach (asking llama3.2 to return YES/NO on whether an answer is grounded in the retrieved text) has not been calibrated against human expert annotation. The judge may be over-lenient (accepting paraphrases of curriculum ideas as faithful even when the answer introduces outside knowledge) or over-strict (rejecting correct answers because the retrieved chunk is a close but not exact match). Without an inter-rater reliability study, the 55% faithfulness figure should be treated as a directional estimate rather than a precise measurement.

---

## Next Steps

1. **Improve retrieval-aware prompting to reduce flaw rates.** The most common RAG flaws (longest option correct, answer position bias) are structural artifacts of how the model generates options from dense retrieved text. Explicitly instructing the model to match option lengths, randomise the correct answer's position across generations, and generate distractors before the correct answer are low-cost prompt changes that could close much of the 13-percentage-point acceptability gap without changing the retrieval setup.

2. **Validate the faithfulness metric with human expert annotation.** A small-scale human validation study (two curriculum experts rating 30–40 randomly sampled RAG question–answer–chunk triples) would establish whether the automated YES/NO judge is well-calibrated. If agreement is high, the metric can be used with confidence across larger experiments; if not, a more structured faithfulness rubric (e.g. fully supported / partially supported / unsupported) with explicit chunk-highlighting would produce more reliable scores.
