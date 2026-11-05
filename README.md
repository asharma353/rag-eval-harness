# rag-eval-harness

How do you evaluate the faithfulness of a RAG system without an eval platform? This repository is one complete, small answer. Every number in it comes from a few lines of plain Python or from one validated yes-or-no question.

## The short answer

1. **Read the traces before you measure anything.** Label each answer pass or fail, and write down what went wrong. The failure modes come from the data, not from a list of generic metrics.
2. **Write one binary judge for each failure mode.** Here, the faithfulness judge asks one question for each sentence of the answer: "Do the passages state everything that this sentence claims?"
3. **Measure the judge against labels that it never saw.** Split the labels into train, dev and test. Choose the judge design and its threshold on train and dev only, then run it once on test. Report TPR (the judge passes a good answer) and TNR (the judge fails a bad answer).
4. **Correct the judge's pass rate with its known errors.** A judge with TPR 0.89 and TNR 0.69 gives a wrong raw rate. The formula `(p_obs + TNR − 1) / (TPR + TNR − 1)` and a bootstrap interval give the true rate with its uncertainty.
5. **Monitor, do not block.** The judge runs on the trace log after the user has the answer. A CI gate fails when the corrected rate drops.

With these steps, the corrected faithfulness rate was 0.995 for the normal prompt and 0.65 for a loose prompt. The hand labels gave 0.98 and 0.78, and both intervals hold the labeled rate.

## What is in the box

- **The system under test:** a small RAG over Hamel Husain's 20 blog posts (about 61,000 words). It uses BM25 retrieval and Claude Sonnet 5.5 (`claude-sonnet-5-5`), and it writes one JSON trace for each answer.
- **The harness:** about 1,460 lines of Python in `rag_eval/`. `pyproject.toml` has no dependencies, and no eval or RAG library is used. Every API call is plain HTTP in `rag_eval/llm.py`, so you can read each request field.
- **The method:** Hamel Husain's, from his evals FAQ and his posts on LLM judges. Error analysis comes first, the labels are binary, and every judge is validated before anyone trusts it.

## What we found, in order

### 1. Error analysis: no hallucinations, and two other failure modes

Fifty test questions were written with Hamel's dimension-tuple method: 20 tuples by hand and 30 by Claude, then one question for each tuple. The owner read all 50 traces on a local review page and labeled them. After a check against the full corpus, 45 passed and 5 failed.

| Failure mode | What happens | Count |
|---|---|---|
| Retrieval miss | The blog has the answer, but the chunk with it ranks below 5, so the model never sees it. | 3 |
| Over-decline | The passages support a partial answer, but the model replies that the blog does not answer. | 2 |

No answer stated a false fact. The first question was "how faithful is it?", but the data said that the problems were elsewhere.

### 2. Retrieval: the user's words matter, and chunk size hardly does

A reverse method built 109 question pairs. Claude copied one fact from a chunk and wrote a question that the fact answers. Half of the questions use any wording. The other half are adversarial: they use the words of other, similar chunks.

- BM25 finds 92–95% of the questions that reuse the blog's words, and only 40–42% of the adversarial ones.
- Across 12 chunk settings, recall at the same amount of context changes by less than the noise.
- A query rewrite fixed 17 adversarial questions and broke 3, but it cost more than the answer itself and nearly doubled the wait. It was not added. The test is in `FINDINGS.md`, ready to run again when real traffic shows a vocabulary gap.

A team that tests only with plain synthetic questions sees a retriever that looks finished.

### 3. A judge needs failures, so we made real ones

The normal prompt almost never adds facts, so it gave too few fails to measure a judge. A loose prompt ("use the passages where they help") answered the same 109 questions. Claude then mixed in facts from its memory of the blog. These are real failures with natural wording, not planted ones.

### 4. The faithfulness judge

The judge is [Jev](https://typesafe.ai), a decision model that returns a probability for a typed yes-or-no question. Each sentence of an answer gets one question. The answer scores as its weakest sentence, and below 0.50 it fails.

- **Labels:** 191 answers, 28 fails. Opus subagents labeled them blind: they saw only the question, the passages and the answer, plus one written rule sheet. The owner checked all 28 fails and agreed with each one. The 163 passes were not checked by a person.
- **Design, chosen on train and dev only:** sending the question and the earlier sentences with each sentence raised balanced accuracy from 0.71 to 0.80. Without them, Jev failed short sentences such as "No." because it could not tell what they claim.
- **Test split, run once (86 answers, 13 fails):**

| | Value | 95% interval |
|---|---|---|
| TPR: passes a faithful answer | 0.89 | 0.82 to 0.96 |
| TNR: fails an unfaithful answer | 0.69 | 0.46 to 0.92 |

| Run | Raw judge pass rate | Corrected rate | Labeled rate |
|---|---|---|---|
| normal prompt | 0.89 | 0.995 [0.83, 1.00] | 0.98 |
| loose prompt | 0.69 | 0.65 [0.32, 0.84] | 0.78 |

The raw rate is wrong in both directions. The correction moves both toward the labels. A full run of 109 answers costs about $0.03 in Jev calls.

### 5. Over-decline is tracked in code, not by a judge

Over-decline was 2 of 50 baseline answers. With so few cases, a test split would hold 2 or 3, and a judge's TNR could be anything. Code tracks the in-scope decline rate instead. Each synthetic question has a known evidence span, so code can also tell a retrieval miss from a decline with the evidence in view. An over-careful prompt raised the decline rate from 0.10 to 0.17. It declined 8 questions that the normal prompt answered, and the reverse never happened.

### 6. A monitor and a gate

`rag_eval/monitor.py` reads the trace log. `scan` judges each new trace and stores the score in it. `gate` exits with code 1 when the corrected faithfulness rate is below 0.90 or the in-scope decline rate is above 0.15.

| Run | Corrected faithfulness | In-scope declines | Gate |
|---|---|---|---|
| normal prompt | 0.995 | 0.10 | pass |
| loose prompt | 0.65 | 0.00 | fail |
| cautious prompt | 0.997 | 0.17 | fail |

It is a monitor and not a guardrail. With TPR 0.89, a guardrail would block about 1 good answer in 9.

## "Without a third-party tool"

The judge calls an outside API, and so does the RAG itself, so it is fair to ask what "no third-party tool" means here. We read the question as "no eval platform": nothing like Ragas, TruLens or a hosted eval service decides what faithfulness means or how to score it.

- The method does not depend on Jev. `judge_faith` in `rag_eval/judge.py` is the only function that calls it. Any judge that returns a score for each sentence can replace it, for example one Claude call with a yes-or-no output.
- Everything that makes a judge trustworthy is plain Python in this repository: the label split, TPR and TNR, the threshold sweep, the corrected rate and the bootstrap.
- A new judge must be validated again on the same test split. The frozen numbers in `evalset/faith_judge.json` are true only for `jev-1.14.0`, and the monitor stops if a trace was judged by another model.

## Limits

1. **The passes were labeled by Claude and not checked by a person.** The owner checked all 28 fails. An unfaithful answer that was labeled pass would change both TPR and TNR.
2. **The owner is a careful reader of the blog, not its author.** Hamel's method asks for one domain expert. Here, that expert is a learner.
3. **TNR rests on 13 test fails,** so its interval is 46 points wide. The lower end of the corrected rate is about 0.83 even for a perfect run. The gate uses the point estimate for this reason.
4. **Most fails come from one prompt.** 24 of the 28 fails are from the loose prompt. The normal prompt's rare errors, such as a wrong citation, are subtler.
5. **The questions are synthetic.** Claude wrote them from the post headings, so they may not match how real users ask.
6. **Jev is not fully deterministic.** Two runs of the same split can give slightly different scores.

## Run it

You need [uv](https://docs.astral.sh/uv/) and two API keys: Anthropic and TypeSafe. On macOS, put them in the Keychain as `anthropic-api-key` and `typesafe-api-key`. Elsewhere, set `ANTHROPIC_API_KEY` and `TYPESAFE_API_KEY`.

```bash
uv run python -m rag_eval.corpus
```

That downloads the blog posts into `data/`. The posts are Hamel Husain's copyrighted writing, so `data/` and `traces/` are never committed, and the MIT license of this code does not cover them.

| Step | Command | Cost |
|---|---|---|
| Ask one question | `uv run python -m rag_eval.rag "What is axial coding?"` | under $0.01 |
| Answer the 109 synthetic questions | `uv run python -m rag_eval.monitor ask --run mine` | about $0.70 |
| Judge them with Jev | `uv run python -m rag_eval.monitor scan --run mine` | about $0.03 |
| Gate on the stored scores | `uv run python -m rag_eval.monitor gate --run mine` | free |
| Label traces by hand | `uv run python -m rag_eval.review --run mine` | your time |
| Retrieval grid | `uv run python -m rag_eval.retrieval grid` | free |
| Decline report | `uv run python -m rag_eval.judge declines` | free |
| Offline tests | `for t in test_slice*.py; do uv run python "$t"; done` | free |

CI (`.github/workflows/eval.yml`) runs the offline tests on every push. The paid gate runs only when you start it by hand, because each run costs about $0.73 and downloads the blog again.

## Files

| Path | What it holds |
|---|---|
| `rag_eval/rag.py` | The system under test: retrieve, answer, write a trace. |
| `rag_eval/queries.py` | The 50 test questions from dimension tuples. |
| `rag_eval/review.py`, `review.html` | The local review page for labels, including the blind mode for judge labels. |
| `rag_eval/retrieval.py`, `bm25.py`, `chunk.py`, `metrics.py` | The retrieval eval and its parts. |
| `rag_eval/judge.py` | The faithfulness judge, its validation, and the decline report. |
| `rag_eval/monitor.py` | The monitor and the CI gate. |
| `evalset/` | Questions, evidence spans (word positions, not text), labels, and the frozen judge scores. |
| `FINDINGS.md` | The detailed lab notes for each step, with every number and decision. |
| `test_slice*.py` | One offline check file for each step. |
