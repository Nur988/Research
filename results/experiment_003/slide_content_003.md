# Slide Content Brief — Experiment 003

## Slide: What changed in Experiment 003

- **Separate curriculum indices for primary and secondary year levels:** Grade 3-5 questions now draw only from the F-6 curriculum; Grade 9-12 questions draw only from the 7-10 curriculum. In Experiment 002 a single flat index mixed all year levels together.
- **Dynamic retrieval queries:** Instead of the same five-word phrase for every question, the system now builds a search query that encodes the target year level and the cognitive demand (e.g. "Year 9 10 curriculum compare contrast analyse"). This means retrieved chunks are relevant to the actual question being generated.
- **Option shuffling:** After each question is generated, the four answer options are randomly reassigned to labels a, b, c, d so the correct answer is not systematically placed first or last.
- **Length-balance retry:** If the correct answer comes out noticeably longer than every distractor — the most common item-writing flaw in Experiment 002 — the system makes one extra call instructing the model to equalise option lengths.
- **Higher temperature in 003b (llama3.2, T=0.7):** The generation temperature was raised from 0.2 to 0.7 to test whether question diversity improves when the model is given more sampling freedom.

---

## Slide: Key results — writing quality

- **RAG acceptability 003a: 86.7%** — retrieval improvements alone changed acceptability by +5.0 percentage points versus Experiment 002 (81.7%). Option shuffling directly eliminates position-bias flags and the length-balance retry targets the longest-option flaw.
- **Zero-shot acceptability 003a: 91.7%** — zero-shot quality changed by -3.3 pp versus 002 (95.0%). The improvements were targeted at retrieval, so zero-shot behaviour is a useful control.
- **RAG acceptability 003b: 93.3%** — adding the model/temperature change (llama3.2 at T=0.7) shifted acceptability by a further +6.6 pp relative to 003a. Higher temperature increases diversity but may also introduce more structural flaws.

---

## Slide: Key results — curriculum faithfulness

- **RAG faithfulness 003a: 16.7%** — dynamic routing to grade-specific indices changed faithfulness by -38.3 pp versus Experiment 002 (55.0%). Routing Year 9-12 questions to 7-10 content means the retrieved chunks are now from the correct year band.
- **RAG faithfulness 003b: 26.7%** — temperature increase changed faithfulness by +10.0 pp relative to 003a. Temperature affects whether the model uses retrieved content or deviates into prior knowledge.
- **Length-balance retries:** 42 retries in 003a, 53 in 003b. Each retry involves an extra LLM call and is an additional data point on how often the first-pass answer is biased in length.

---

## Slide: Comparison table

| Metric | Exp 002 | Exp 003a | Exp 003b |
|--------|---------|---------|---------|
| RAG writing quality | 81.7% | 86.7% | 93.3% |
| Zero-shot writing quality | 95.0% | 91.7% | 86.7% |
| RAG curriculum faithfulness | 55.0% | 16.7% | 26.7% |
| Within-cell RAG diversity | 12/60 stems | 18/60 stems | 42/60 stems |
| Most common RAG flaw | answer_position_bias | answer_position_bias | answer_position_bias |
| Format failures (RAG) | 0 | 0 | 0 |

---

## Slide: What the model change added

Between 003a and 003b (model/temperature change only, retrieval identical):

- RAG acceptability: +6.6 pp
- RAG faithfulness: +10.0 pp
- Zero-shot acceptability: -5.0 pp
- RAG diversity: +24.0 unique stems (out of 60)

Note: 003b used llama3.2 as a fallback (none of the preferred models were available). The temperature increase from 0.2 to 0.7 is therefore the primary variable — no architecture comparison is possible without pulling a preferred model.

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
