# Case study: a RAG over one blog

This is the study that rag-eval-harness was built from. The system under test is a small RAG over Hamel Husain's 20 blog posts (about 61,000 words), with BM25 retrieval and Claude Sonnet 5.5. The method is the one in the main README, applied step by step. `FINDINGS.md` has the detailed lab notes, with every number and decision.

## What we found, in order

### 1. Error analysis: no hallucinations, and two other failure modes

Fifty test questions were written with the dimension-tuple method: 20 tuples by hand and 30 by Claude, then one question for each tuple. The owner read all 50 traces on a local review page and labeled them. After a check against the full corpus, 45 passed and 5 failed.

| Failure mode | What happens | Count |
|---|---|---|
| Retrieval miss | The blog has the answer, but the chunk with it ranks below 5, so the model never sees it. | 3 |
| Over-decline | The passages support a partial answer, but the model replies that the blog does not answer. | 2 |

No answer stated a false fact. The first question was "how faithful is it?", but the data said that the problems were elsewhere.

### 2. Retrieval: the user's words matter, and chunk size hardly does

A reverse method built 109 question pairs. Claude copied one fact from a chunk and wrote a question that the fact answers. Half of the questions use any wording. The other half are adversarial: they use the words of other, similar chunks.

- BM25 finds 92–95% of the questions that reuse the blog's words, and only 40–42% of the adversarial ones.
- Across 12 chunk settings, recall at the same amount of context changes by less than the noise.
- A query rewrite fixed 17 adversarial questions and broke 3, but it cost more than the answer itself and nearly doubled the wait. It was not added.

A team that tests only with plain synthetic questions sees a retriever that looks finished.

### 3. A judge needs failures, so we made real ones

The normal prompt almost never adds facts, so it gave too few fails to measure a judge. A loose prompt ("use the passages where they help") answered the same 109 questions. Claude then mixed in facts from its memory of the blog. These are real failures with natural wording, not planted ones.

### 4. The faithfulness judge

The judge is Jev, a decision model that returns a probability for a typed yes-or-no question. Each sentence of an answer gets one question. The answer scores as its weakest sentence, and below 0.50 it fails.

- **Labels:** 191 answers, 28 fails. Opus subagents labeled them blind: they saw only the question, the passages and the answer, plus one written rule sheet. The owner checked all 28 fails and agreed with each one. The 163 passes were not checked by a person.
- **Design, chosen on train and dev only:** sending the question and the earlier sentences with each sentence raised balanced accuracy from 0.71 to 0.80.
- **Test split, run once (86 answers, 13 fails):** TPR 0.89 [0.82, 0.96], TNR 0.69 [0.46, 0.92].

| Run | Raw judge pass rate | Corrected rate | Labeled rate |
|---|---|---|---|
| normal prompt | 0.89 | 0.995 [0.83, 1.00] | 0.98 |
| loose prompt | 0.69 | 0.65 [0.32, 0.84] | 0.78 |

The raw rate is wrong in both directions. The correction moves both toward the labels.

### 5. Over-decline is tracked in code, not by a judge

Over-decline was 2 of 50 baseline answers. With so few cases, a test split would hold 2 or 3, and a judge's TNR could be anything. Code tracks the decline rate instead. An over-careful prompt raised it from 0.10 to 0.17.

### 6. A monitor and a gate

| Run | Corrected faithfulness | Declines | Gate |
|---|---|---|---|
| normal prompt | 0.995 | 0.10 | pass |
| loose prompt | 0.65 | 0.00 | fail |
| cautious prompt | 0.997 | 0.17 | fail |
| `ci` on GitHub Actions (2026-11-10) | 0.940 [0.75, 1.00] | 0.11 | pass |

It is a monitor and not a guardrail. With TPR 0.89, a guardrail would block about 1 good answer in 9.

## How this folder maps to the general tool

- `workspace/` is a normal rag-eval workspace. Its judge is the validated judge, word for word, and `calibrations/faithfulness.json` holds the frozen test scores from 2026-11-03.
- The calibration was not made again with the general split. The general tool puts each trace into train, dev or test by a hash, and that would move answers that chose the judge design (dev) into test.
- `hamel/contract.py` is the glue that a user writes for their own RAG: it turns the reference RAG's traces into the trace contract.
- `hamel/` holds the original study code: the crawler, the reference RAG, the tuple questions, the retrieval grid, and the slice 4 judge experiments.

## Run it

You need [uv](https://docs.astral.sh/uv/) and the two API keys from the main README. The blog posts are Hamel Husain's copyrighted writing, so `data/`, `traces/` and the workspace traces are never committed, and the MIT license of this code does not cover them.

```bash
cd examples/hamel-blog
```

```bash
uv run python -m hamel.corpus
```

| Step | Command (from `examples/hamel-blog`) | Cost |
|---|---|---|
| Ask one question | `uv run python -m hamel.rag "What is axial coding?"` | under $0.01 |
| Answer the 109 synthetic questions into the workspace | `uv run python -m hamel.contract answer --run mine` | about $0.70 |
| Judge them, then gate | `cd workspace && uv run rag-eval scan --run mine && uv run rag-eval gate --run mine` | about $0.03 |
| Retrieval grid | `uv run python -m hamel.retrieval grid` | free |
| Decline report | `uv run python -m hamel.judge declines` | free |
| Score the frozen test split again with the general engine | `uv run python -m hamel.contract parity` | about $0.01 |
| Offline checks | `for t in test_slice*.py; do uv run python "$t"; done` | free |

## Limits

1. **The passes were labeled by Claude and not checked by a person.** The owner checked all 28 fails.
2. **The owner is a careful reader of the blog, not its author.** The method asks for one domain expert. Here, that expert is a learner.
3. **TNR rests on 13 test fails,** so its interval is 46 points wide. The gate uses the point estimate for this reason.
4. **Most fails come from one prompt.** 24 of the 28 fails are from the loose prompt.
5. **The questions are synthetic.** They may not match how real users ask.
