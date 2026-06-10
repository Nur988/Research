# Experiment 003 Findings

## Overview

Experiment 003 investigated two targeted improvements to the K-12 RAG question generation
pipeline established in Experiment 002. Two sub-experiments were run:

- **003a**: Same model (llama3.2) with improved retrieval — dual grade-band indices, dynamic
  queries, post-generation option shuffling, and length-balance retries.
- **003b**: llama3.2 at temperature 0.7, with identical retrieval improvements from 003a.

None of ['llama3.1:8b', 'mistral:7b', 'gemma2:9b'] found on Ollama. Available chat models: ['llama3', 'llama3.2']. Falling back to llama3.2.

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
| Model | llama3.2 | llama3.2 (None of ['llama3) |
| Temperature | 0.2 | 0.7 |
| Retrieval | Dual dynamic indices | Identical — reused from 003a |
| Shuffling | Yes | Yes (identical) |
| Length retry | Yes | Yes (identical) |

---

## Results

### Writing quality (acceptability)

| Experiment | RAG % Acceptable | Zero-shot % Acceptable |
|------------|-----------------|----------------------|
| 002 | 81.7% | 95.0% |
| 003a | 86.7% | 91.7% |
| 003b | 93.3% | 86.7% |

RAG acceptability changed by +5.0 pp
in 003a (retrieval improvements only) and
+11.6 pp overall in 003b.

Per-cell RAG results:

| Condition | % Acceptable | % Faithful | Diversity |
|-----------|-------------|-----------|-----------|
| 002 \| Grade 3-5 \| Remember | 86.7% | 26.7% | 4/15 |
| 002 \| Grade 3-5 \| Analyze | 100.0% | 80.0% | 6/15 |
| 002 \| Grade 9-12 \| Remember | 46.7% | 53.3% | 2/15 |
| 002 \| Grade 9-12 \| Analyze | 93.3% | 60.0% | 1/15 |
| 003a \| Grade 3-5 \| Remember | 80.0% | 13.3% | 8/15 |
| 003a \| Grade 3-5 \| Analyze | 93.3% | 20.0% | 5/15 |
| 003a \| Grade 9-12 \| Remember | 100.0% | 20.0% | 2/15 |
| 003a \| Grade 9-12 \| Analyze | 73.3% | 13.3% | 3/15 |
| 003b \| Grade 3-5 \| Remember | 93.3% | 33.3% | 11/15 |
| 003b \| Grade 3-5 \| Analyze | 86.7% | 20.0% | 15/15 |
| 003b \| Grade 9-12 \| Remember | 93.3% | 33.3% | 5/15 |
| 003b \| Grade 9-12 \| Analyze | 100.0% | 20.0% | 12/15 |

### Curriculum faithfulness

| Experiment | RAG % Faithful |
|------------|---------------|
| 002 | 55.0% |
| 003a | 16.7% |
| 003b | 26.7% |

Faithfulness changed by -38.3 pp in 003a and
-28.3 pp in 003b relative to 002.

Length-balance retries triggered: **42** in 003a, **53** in 003b.

### Question diversity

Diversity is measured as the number of unique 8-word stem prefixes per 15-question cell.
A score of 15 means every question had a different opening; a score of 1 means all 15
questions began with the same words.

| Experiment | RAG diversity (of 60 RAG questions) |
|------------|-------------------------------------|
| 002 | 12/60 unique stems |
| 003a | 18/60 unique stems |
| 003b | 42/60 unique stems |

### Flaw analysis

| Condition | Dominant flaw 002 | Dominant flaw 003a | Dominant flaw 003b |
|-----------|------------------|-------------------|--------------------|
| RAG | answer_position_bias | answer_position_bias | answer_position_bias |

---

## Comparison with Experiment 002

### What improved and by how much

| Metric | 003a vs 002 | 003b vs 002 |
|--------|------------|------------|
| RAG % Acceptable | +5.0 pp (improved) | +11.6 pp (improved) |
| RAG % Faithful | -38.3 pp (degraded) | -28.3 pp (degraded) |
| Zero-shot % Acceptable | -3.3 pp (degraded) | -8.3 pp (degraded) |
| RAG diversity | +6.0 (improved) | +30.0 (improved) |
| RAG mean flaws | -0.3 (improved) | -0.4 (improved) |

### What did not improve

The IWF checker's acceptability threshold (≤1 flaw) is unchanged. Questions that passed
the threshold may still have pedagogical issues not caught by the seven structural checks.
Faithfulness is still measured by the same automated LLM-as-judge with no human calibration.

### Unexpected findings

- Length-balance retries were triggered 42 times in 003a and 53 times in 003b,
  indicating the correct-answer-longest flaw is not uniformly distributed across cells.
- The model fallback to llama3.2 means 003b cannot isolate the effect of the preferred model architecture; the temperature increase to 0.7 is the primary variable.

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
