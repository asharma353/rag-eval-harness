# Findings: baseline error analysis

Run `baseline`: 50 questions, BM25 retrieval, top 5 chunks of 200 words with 50 words of overlap, Claude Sonnet 5.5 at effort `low`. Reviewed on 2026-10-29.

## Result

The owner's first labels gave 44 pass and 6 fail. Claude then compared each failure note, and each declined answer, with the full corpus. That check proposed 5 label changes, and the owner accepted all five on 2026-10-30. The run now has 45 passes and 5 failures in two failure modes.

| Failure mode | What happens | Traces | Count |
|---|---|---|---|
| **Retrieval miss** | The blog answers the question, but the chunk with the answer ranks below 5, so the model never sees it. | q011 (rank 7), q041 (rank 14), q039 (rank 28) | 3 |
| **Over-decline** | The retrieved chunks support a partial answer, but the model replies "The blog posts I have do not answer this." | q029, q044 | 2 |
| **Total** | | | **5 of 50** |

Both modes end in the same answer to the user: a refusal or a thin answer. No answer in this run stated a false fact or cited a bad passage. The first upstream failure differs, so each mode needs a different fix.

## Label changes from the corpus check

The owner accepted all five changes. Each one is a new line in `evalset/labels.jsonl` with `labeled_by` set to the corpus check, so the old labels stay in the log.

| Trace | First label | Now | Evidence |
|---|---|---|---|
| q013 | fail | pass | The answer is correct and cites the two chunks that matter ([3] and [4], both from `eval-smell`). Chunks 1 and 2 are noise, but the user never sees them. This is a retrieval precision issue for slice 3, not a failed answer. |
| q024 | fail | pass | The corpus does not mention Weights & Biases or experiment tracking. No right chunk exists, so the decline is the correct answer. |
| q026 | fail | pass | The corpus does not mention LoRA or QLoRA. The decline is correct. |
| q011 | pass | fail | The answer is in the blog: `evals-faq#011` ("How much of my development budget should I allocate to evals?") says 60–80% of development time. It ranks 7th, so the model declined. |
| q044 | pass | fail | Chunk 1 (`evals-faq#064`) says that product managers and engineers can work together on error analysis. That supports a partial answer, but the model declined. |

The other four failures stay failures, but for three of them the evidence points at a different cause than the note:

- **q039** (the owner's note: the answer misuses good chunks). The blog defines a trace: "A trace is the complete record of all actions…". That text is in `evals-faq#008`, which ranks 28th. The chunk starts in the section before, so its heading is "Product evals" and not "Q: What is a trace?". This is the heading bug from slice 1. The answer itself is honest about what the chunks lack. The first failure is retrieval.
  *Correction from slice 3:* the heading bug is real, but it is not the main cause. With section-bounded chunks, the definition has the right heading and still ranks 17th. The main cause is the words. See the slice 3 section below.
- **q041** (the owner's note: the answer does not join the chunks well). The two chunks that rank first are a short recap of the steps. The detailed section, "Step 5: Build Your LLM as A Judge, Iteratively", ranks 14th. The answer can only list what the recap says. The first failure is retrieval.
- **q029** (the owner's note: retrieval). Chunks 2, 3 and 4 hold Hamel's view: teams spend too much time on choices such as which vector database to use, and the real goal is to give the model the context it needs. The right chunks were retrieved, and the model declined anyway. The first failure is the answer.

One decision belonged to the owner as the domain expert. Hamel never names a vector database (q029), and he only touches on product managers (q044). Is a decline acceptable when the blog has a related view but no direct answer? The owner's decision: it is not acceptable. A user who asks "what does Hamel recommend" gets more from "He does not name one. He says that this choice matters less than X" than from a refusal. This rule is the pass/fail line for the over-decline judge in slice 4.

## Other observations

These are not failures, but they matter for the next slices.

1. **Some scope labels are wrong.** Claude guessed the scope when it wrote each tuple. q003 is labeled "one post", but the Axolotl post covers debugger setup, not a training loss that does not go down, so its decline is correct. q029 is labeled "not in the blog", but the blog has a view on it. Treat the scope labels as hypotheses, not as ground truth.
2. **The out-of-scope questions are easy.** 14 of the 15 declined correctly, and the other one (q022) answered only the part that the blog covers. They test the decline path and little else. Slice 3 needs harder adversarial questions: false premises, and topics that the blog mentions only in passing.
3. **The retrieval misses are near misses.** q011 is at rank 7. If k goes from 5 to 10, it is found. q039 and q041 need better chunks, not a larger k.
4. **50 traces with 5 failures is too few to find all failure modes.** Hamel says to review until new traces stop showing new modes. Slice 4 needs 100 or more labels anyway, so that review continues there.

## Method lesson: the review page primed the notes

Five of the owner's six notes repeat the three causes that the notes placeholder listed ("retrieval (the right chunk is missing), the answer (it misuses good chunks), or the question itself"). Open coding needs the reviewer's own words about what they saw, for example "declined, but chunk 3 says 60–80%". A list of categories makes the reviewer pick a category before looking. The mistake was Claude's, who wrote the page. The placeholder now asks only what the reviewer saw, and it names no categories.

## What this means for the next slices

- **Slice 3 (retrieval):** Recall@5 measures the retrieval miss mode directly. Test chunks that start at section boundaries (for q039), and test k = 10 (for q011).
- **Slice 4 (judges):** The over-decline mode is a good first Jev judge. It asks one narrow, binary question: "The answer declines. Do the passages support at least a partial answer?" Validate it against the owner's labels before trusting it.

---

# Slice 3: retrieval eval (2026-10-30)

## Result

**Retrieval fails when the user's words differ from the blog's words. Chunking hardly matters.** BM25 finds 92–95% of the questions that reuse the blog's wording, and only 40–42% of the questions that do not (200-word chunks). Across the 12 chunk settings, recall at an equal amount of context changes by less than the noise in the measurement. Keep 200 words with 50 words of overlap.

## The question set

`evalset/retrieval_pairs.jsonl` has 112 pairs, made with the reverse method: Claude reads a chunk, copies one fact from it (the evidence), and writes a question that the fact answers. The file stores each evidence as a word span in its post, not as text, so no blog text goes into the repo, and the same span works for any chunk size. A chunk is relevant when it holds at least half of the evidence words.

| Style | Pairs | What it tests |
|---|---|---|
| plain | 59 | Any wording. On average, 45% of a question's search terms come from its evidence. |
| adversarial | 50 | The question uses everyday words or words of the two most similar other chunks. A gate drops any question that shares more than 30% of its terms with the evidence. On average, it shares 11%. |
| real | 3 | The baseline queries whose answer retrieval missed: q011, q039, q041. |

The first version had no gate. Its "adversarial" questions copied 43% of their evidence terms, which is the same as the plain ones, and BM25 found 100% of them. The prompt asked for confusing words, and Claude added them to the evidence words, not in place of them. Only a measured gate made the set hard. This repeats the slice 2 lesson: read the synthetic data before you trust it.

## The grid

R@5 is Recall@5. "R@~1000 words" uses the k that gives about 1,000 words of context for each chunk size, because a fixed k gives more text to bigger chunks. The real columns show the rank of the first relevant chunk.

| Size | Overlap | Sections | Chunks | R@5 | R@10 | R@~1000 words (k) | MRR | P@5 | nDCG@5 | R@5 plain | R@5 adversarial | Real ranks: q011 / q039 / q041 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 0 | no | 595 | 0.62 | 0.65 | 0.65 (10) | 0.54 | 0.13 | 0.55 | 0.92 | 0.28 | 9 / 31 / 5 |
| 100 | 0 | yes | 761 | 0.66 | 0.69 | 0.69 (10) | 0.53 | 0.13 | 0.55 | 0.93 | 0.34 | 17 / 25 / 24 |
| 100 | 50 | no | 1164 | 0.67 | 0.70 | 0.70 (10) | 0.55 | 0.23 | 0.52 | 0.92 | 0.38 | 23 / 42 / 7 |
| 100 | 50 | yes | 1062 | 0.67 | 0.70 | 0.70 (10) | 0.56 | 0.20 | 0.54 | 0.95 | 0.34 | 21 / 38 / 23 |
| 200 | 0 | no | 302 | 0.71 | 0.73 | 0.71 (5) | 0.59 | 0.14 | 0.61 | 0.95 | 0.42 | 27 / 19 / 11 |
| 200 | 0 | yes | 480 | 0.70 | 0.74 | 0.70 (5) | 0.60 | 0.14 | 0.61 | 0.93 | 0.42 | 36 / 18 / 15 |
| **200** | **50** | **no** | **395** | **0.69** | **0.75** | **0.69 (5)** | **0.60** | **0.17** | **0.59** | **0.92** | **0.42** | **7 / 28 / 7** |
| 200 | 50 | yes | 506 | 0.70 | 0.74 | 0.70 (5) | 0.61 | 0.17 | 0.61 | 0.95 | 0.40 | 40 / 17 / 13 |
| 400 | 0 | no | 156 | 0.76 | 0.80 | 0.63 (2) | 0.65 | 0.15 | 0.67 | 0.92 | 0.58 | 3 / 16 / 2 |
| 400 | 0 | yes | 375 | 0.72 | 0.76 | 0.64 (2) | 0.62 | 0.14 | 0.64 | 0.93 | 0.48 | 31 / 17 / 9 |
| 400 | 50 | no | 175 | 0.74 | 0.83 | 0.65 (2) | 0.66 | 0.17 | 0.65 | 0.90 | 0.56 | 58 / 14 / 2 |
| 400 | 50 | yes | 376 | 0.73 | 0.75 | 0.64 (2) | 0.63 | 0.15 | 0.64 | 0.93 | 0.50 | 31 / 16 / 10 |

The bold row is the current setting. Recall and MRR are means over the 109 synthetic pairs. Run it again with `uv run python -m rag_eval.retrieval grid`. It takes about one second.

## What the grid says

1. **Plain synthetic questions cannot tune retrieval.** They score 0.90–0.95 in every setting, because they reuse the blog's words. A team that tests only with such questions sees a retriever that looks finished.
2. **Vocabulary mismatch is the main retrieval failure.** Adversarial recall is 0.28–0.58. Two of the three real misses have the same cause. q039 asks about "a trace in an LLM app". "trace" appears in almost every evals post, "app" pulls the Dokku post ("Deploying Apps") to the top, and the definition does not use "LLM" or "app". The synthetic question for the same evidence ("a trace in the context of evaluating AI systems") misses too, at rank 45.
3. **Bigger chunks only look better.** At k = 5, 400-word chunks win (0.76 against 0.69). At an equal context of about 1,000 words, they lose (0.63–0.65 against 0.69–0.71). They win at a fixed k only because they give the model twice the text.
4. **No chunk setting is clearly best.** At equal context, all settings fall between 0.63 and 0.71. With 109 pairs, the standard error of a recall near 0.7 is about 0.04, so most of these differences are noise.
5. **Section-bounded chunks do not help on average.** They give q039 the right heading, but it still ranks 17th.
6. **k = 10 helps a little.** At 200/50, Recall goes from 0.69 to 0.75, and q011 (rank 7) is found. It doubles the context tokens, which adds about $0.003 per query.
7. **The real ranks jump around** (q011 ranks anywhere from 3 to 58). Three queries are too few to choose a setting with. The real queries show the size of the problem, not the best setting.

## Limits of this measurement

- **One evidence per question.** Hamel repeats some ideas in several posts, for example "build your own trace viewer" and "use binary judgments". If retrieval returns another post that also answers, the eval counts a miss. So adversarial recall is a floor, not an exact value.
- **Some adversarial questions are disguised more than a real user would disguise them.** For example, r035 says "the main container orchestration tool" and not "K8s".
- **Single run.** Claude wrote the questions once. Another run gives different questions and slightly different numbers.

## What this means for the next step

The biggest retrieval lever is the query, not the chunks. Two from-scratch options can close the vocabulary gap without an embedding library:

- **Query rewrite:** Claude rewrites the question in the blog's words before BM25.
- **Hypothetical answer (HyDE):** Claude writes a short imagined answer, and BM25 searches with that text.

Each one adds one Claude call per query, about $0.002 and 1–2 seconds. The adversarial set and the three real misses can measure each option against BM25 alone.

## Query rewrite and HyDE (2026-10-30)

**Both methods close most of the vocabulary gap when they are added to the user's question, and neither one helps when it replaces the question. On the three real misses, neither one wins, so the RAG stays unchanged.**

The test runs at the current setting (200-word chunks, 50 words of overlap) and uses the same 112 pairs. Run it again with `uv run python -m rag_eval.retrieval rewrite`. Claude's outputs are cached in `data/query_variants.jsonl`, outside git, so a rerun costs nothing.

- **Rewrite:** Claude reads the post titles and section headings (no post text) and writes a 5 to 15 word search query in the blog's terms.
- **HyDE:** Claude writes a 60 to 100 word imagined answer, and BM25 searches with that text.
- **Question + X:** BM25 searches with the user's question and X together.

| Query sent to BM25 | R@5 plain | R@5 adversarial | R@10 adversarial | MRR adversarial | Real ranks: q011 / q039 / q041 | Cost and time per query |
|---|---|---|---|---|---|---|
| question (baseline) | 0.92 | 0.42 | 0.52 | 0.32 | 7 / 28 / 7 | none |
| rewrite | 0.76 | 0.66 | 0.76 | 0.46 | 2 / 9 / 57 | $0.012, 2.5 s |
| **question + rewrite** | 0.85 | **0.70** | **0.84** | **0.55** | 1 / 8 / 29 | $0.012, 2.5 s |
| HyDE | 0.76 | 0.60 | 0.68 | 0.35 | 1 / 1 / 33 | $0.002, 3.6 s |
| question + HyDE | 0.86 | 0.66 | 0.70 | 0.45 | 1 / 2 / 21 | $0.002, 3.6 s |

For comparison, one answer from the RAG costs about $0.007 and takes about 3 seconds.

### What the test says

1. **Keep the user's question in the query.** Question + rewrite fixes 17 adversarial questions and breaks 3 (exact sign test p ≈ 0.003). Over all synthetic questions, the rewrite alone fixes 18 and breaks 15, so on balance it gains almost nothing. A rewrite loses words that the user typed and that the blog also uses.
2. **The cost is in the plain questions.** Question + rewrite breaks 4 plain questions and fixes none. Over all 109 synthetic questions, Recall@5 goes from 0.69 to 0.78 (17 fixed, 7 broken, p = 0.06). Whether that is a real gain depends on how real users word their questions. This harness does not know that yet, because the 50 baseline queries were also written by Claude from the post headings.
3. **The rewrite and HyDE are equal on recall.** Each one is better on about the same number of questions (10 against 9). HyDE costs one fifth as much.
4. **HyDE is partly memory.** For q011 its imagined answer says "60 to 80 percent", and for q039 it starts "A trace is the complete record…". Those are Hamel's own words, from Claude's training data. On a private corpus, which is the AWS client's case, HyDE would gain less. The rewrite sees only titles and headings, so its result transfers better.
5. **The rewrite works through the headings.** For q011 it reproduced the FAQ heading almost word for word. For q041 it added "critique shadowing", which pulls the short recap chunks to the top and not the Step 5 section, so the rank got worse (7 to 29). A rewrite can only aim at the post or section level that the headings describe.
6. **The real misses come out even.** With 10 chunks, BM25 alone finds 2 of the 3 (q011 and q041). Question + rewrite also finds 2 (q011 and q039). Different queries fail, but the count is the same.

### Decision

Do not add a rewrite step to the RAG now. It costs more than the answer itself ($0.012 against $0.007, because the outline and prompt are almost 6,000 tokens). It nearly doubles the wait for each answer. Its gain on realistic questions is not proven. Add it when real traffic shows that users word their questions differently from the blog. Measure it then with this same table. Prompt caching of the outline would cut most of the rewrite cost.

Spent: $1.60 for 112 rewrites and 112 HyDE passages. The estimate before the run was about $1, because it counted the outline at 3,200 tokens. The measured input is almost 6,000 tokens per rewrite.

# Slice 4: a faithfulness judge with Jev (2026-11-03)

## Result

A Jev judge catches 69% of unfaithful answers and passes 89% of faithful ones, on a held-out test set of 86 answers. A full run costs about $0.03 in Jev calls. It corrects the raw pass rate of a run to an estimate with a confidence interval. On the normal prompt, the estimate is 0.995 (95% interval 0.83 to 1.00). The labels give 48 of 49 for the same run.

| Test split (86 answers, 13 fails) | Value | 95% interval |
|---|---|---|
| TPR: the judge passes a faithful answer | 0.89 | 0.82 to 0.96 |
| TNR: the judge fails an unfaithful answer | 0.69 | 0.46 to 0.92 |

| Run | Answers | Raw judge pass rate | Corrected pass rate | Labeled pass rate |
|---|---|---|---|---|
| synthetic (normal prompt) | 98 | 0.89 | 0.995 [0.83, 1.00] | 48 of 49 (0.98) |
| loose prompt | 109 | 0.69 | 0.65 [0.32, 0.84] | 85 of 109 (0.78) |

The judge's raw rate is wrong in both directions: too low for the normal prompt and too high for the loose prompt. The correction moves both toward the labels, and both intervals hold the labeled rate.

## How the judge works

1. Code splits the answer into sentences and finds the citations in each one.
2. For each sentence, Jev gets one `noul` question: "Do the passages state everything that sentence_to_check claims?" The state holds the passages that the sentence cites (all five if it cites none), the user's question, and the answer's earlier sentences.
3. The answer's score is its lowest sentence score. Below 0.50 is a fail.
4. The corrected rate is `(p_obs + TNR − 1) / (TPR + TNR − 1)`. The interval comes from 2,000 bootstrap samples of the test labels and of the run's judgments.

Run it with `uv run python -m rag_eval.judge test`. The monitor (slice 5) gives the corrected rate of a run. Check: `test_slice4.py`.

## The labels

- 191 answers: 109 from the loose prompt, 49 synthetic and 33 baseline answers from the normal prompt. 28 fail.
- Claude labeled them at the owner's request: Opus subagents saw only the question, the passages and the answer, with one written rule sheet. Each record has `"labeler": "claude-opus-5-5"`. **On 2026-11-04 the owner checked all 28 fails and agreed with each one.** Each check is a new record with `"audited_by": "owner"`. The 163 passes were not checked by a person, so an unfaithful answer labeled pass is still possible.
- The labels split 15/40/45 into train, dev and test, stratified by verdict.

## How the design was chosen (train and dev only)

| State sent to Jev | Train + dev balanced accuracy |
|---|---|
| sentence + cited passages | 0.71 |
| **+ question and earlier sentences** | **0.80** |
| + all five passages | 0.78 |
| + a second question: "is this sentence only about the passages?" | 0.77 |

- Without context, Jev failed short sentences such as "No.", "Generally, yes." and lead-ins that end in a colon, because it could not tell what they claim.
- Scoring an answer by its second-lowest sentence looked better on dev (TPR 0.89, TNR 0.91), but caught only 1 of the 4 train fails. It also contradicts the rubric, where one unsupported sentence is a fail. The lowest sentence stayed.
- The threshold 0.50 is in the middle of a flat part of the dev curve (TPR 0.83, TNR 0.82 from 0.45 to 0.50). It was written into the code before the test split ran once.

## Limits

1. **The passes are Claude's labels.** This is the largest limit; see above.
2. **TNR rests on 13 test fails,** so its interval is 46 points wide. The corrected rate inherits that width, mostly at the low end.
3. **Two of the four missed fails are close calls.** q017 and q019 come from the baseline run. The owner passed them in slice 2, where the question was whether the answer is good. In the audit, the owner agreed that they fail the stricter faithfulness rubric. If they were passes, test TNR would be 9 of 11 (0.82).
4. **Most fails come from one prompt.** 24 of 28 fails are from the loose prompt, so the judge is tested mostly on its kind of error. The normal prompt's errors are rarer and subtler (a wrong citation, a link between two ideas).
5. **Jev is not fully deterministic.** The same train split gave a best threshold of 0.79 in one run and 0.72 in the next, with the first design.
6. **False fails are often true sentences that Jev doubts.** Examples: a list of cloud providers, or the remote-command claim about Dokku. The judge is stricter than the labels, which the correction absorbs.

Spent: about $0.25 on Jev for all design runs, the test run and two rate runs, plus $0.38 for the 55 extra loose answers.

## Over-decline: tracked in code, not by a judge (2026-11-03)

Over-decline is too rare to validate a judge: 2 of 50 baseline answers, and no certain case in 109 synthetic answers. With so few fails, a test split would hold 2 or 3, and TNR could be anything. Code tracks it instead, with `uv run python -m rag_eval.judge declines`.

| Run | Answers | Declined | In-scope decline rate | Over-decline (evidence in context) | Check by hand | Evidence not retrieved | Planned out of scope |
|---|---|---|---|---|---|---|---|
| baseline | 50 | 17 | 0.09 | 0 | 3 | 0 | 14 |
| synthetic (normal prompt) | 109 | 11 | 0.10 | 0 | 0 | 11 | 0 |
| cautious prompt | 109 | 19 | 0.17 | 0 | 0 | 19 | 0 |
| loose prompt | 109 | 0 | 0.00 | 0 | 0 | 0 | 0 |

How code sorts each decline:

- **The question has an evidence span** (all synthetic questions): if a retrieved chunk holds the evidence, the decline is a certain over-decline. If not, the retriever missed it.
- **No span, planned out of scope:** the decline is expected.
- **No span, planned in scope:** a person must read it. For the baseline, that is 3 answers. The owner's slice 2 labels say that q003 was a correct decline, q011 a retrieval miss, and q044 an over-decline.

What the numbers say:

1. **The in-scope decline rate catches an over-careful prompt.** The cautious prompt declines 8 questions that the normal prompt answers. The reverse never happens (sign test p = 0.008). This is the metric for the monitor in slice 5.
2. **No decline is certain from code alone.** The model never declined with the evidence in context. Over-declines come from related, partial passages, as with q044, and code cannot see those. They hide in "evidence not retrieved" and in "check by hand".
3. **The planned scope can be wrong.** q029 was planned as out of scope, but the owner labeled its decline a fail in slice 2. The "out of scope" group is only as good as the scope labels.

The decline label pool, its rubric and the `label-decline` page are removed. The cautious run stays as the test case for this report.

## Slice 5: a monitor and a CI gate (2026-11-04)

The judge runs as a monitor, not as a guardrail. With TPR 0.89, a guardrail would block about 1 good answer in 9. The monitor reads the trace log after the answers are sent, so the user waits for nothing.

| Run | Corrected faithfulness rate | In-scope decline rate | Gate |
|---|---|---|---|
| synthetic (normal prompt) | 0.995 [0.83, 1.00] | 0.10 | pass |
| loose prompt | 0.65 [0.33, 0.84] | 0.00 | fail: faithfulness below 0.90 |
| cautious prompt | 0.997 [0.84, 1.00] | 0.17 | fail: declines above 0.15 |
| ci (new run, clean copy, keys from environment variables) | 1.000 [0.86, 1.00] | 0.08 | pass |

How it works:

1. `uv run python -m rag_eval.monitor scan --run <name>` judges each trace that has no `monitor` field yet, and writes the score and the weakest sentence into the trace. A second scan makes no calls.
2. `report --run <name>` prints the two rates and the flagged answers. `gate --run <name>` prints the same and exits with code 1 if a check fails. Both read the stored scores and make no calls.
3. The correction uses the frozen test-split scores in `evalset/faith_judge.json`. The file holds numbers only. The Jev model is pinned to `jev-1.14.0`, because the TPR and TNR are true only for that model. The report stops if a trace was judged by another model.
4. CI (`.github/workflows/eval.yml`) runs the five offline test files on every push and pull request, for free. The paid gate runs only on a manual start: it downloads the blog, answers the 109 synthetic questions ($0.70), scans them ($0.02 to $0.03), and runs the gate. It needs the repository secrets `ANTHROPIC_API_KEY` and `TYPESAFE_API_KEY`.

Limits:

- The gate uses the point estimate. The lower end of the interval is about 0.83 even for a perfect run, because the test split has only 13 fails.
- The decline threshold has a small margin: expect a false alarm about 1 run in 20.
- The workflow has not run on GitHub yet. Each step ran locally, in a clean copy without `data/` or `traces/`, with the macOS Keychain command hidden.

Check: `test_slice5.py`.
