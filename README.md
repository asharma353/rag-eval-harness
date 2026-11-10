# rag-eval-harness

Measure how your RAG system fails, on your own documents, without an eval platform. You bring the documents and the system. rag-eval makes test questions with known evidence, checks your retriever, and gives you yes-or-no judges that are measured against a person's labels before you trust them. Then it reports corrected pass rates and gates your CI.

About 1,200 lines of plain Python, with no required dependencies. Every API call is plain HTTP in `rag_eval/llm.py`, so you can read each request field.

## The method, in five steps

1. **Read your traces before you measure anything.** Label each answer pass or fail, and write down what went wrong. The failure modes come from your data, not from a list of generic metrics.
2. **Write one yes-or-no judge for each failure mode.** For example, the starter faithfulness judge asks one question for each sentence of an answer: "Do the passages state everything that this sentence claims?"
3. **Measure each judge against labels that it never saw.** The labels go into train, dev and test splits. You choose the threshold on dev, then run the judge once on test. The result is TPR (the judge passes a good answer) and TNR (the judge fails a bad answer).
4. **Correct the judge's pass rate with its known errors.** A judge with TPR 0.89 and TNR 0.69 gives a wrong raw rate. The formula `(p_obs + TNR − 1) / (TPR + TNR − 1)` and a bootstrap interval give the true rate with its uncertainty.
5. **Monitor, do not block.** The judges run on your trace log after your users have their answers. A CI gate fails when a corrected rate drops below its limit.

## Quick start

You need [uv](https://docs.astral.sh/uv/). In this clone:

```bash
uv tool install --editable .
```

Then make a workspace, read your documents, and write test questions:

```bash
rag-eval init my-eval
```

```bash
cd my-eval
```

```bash
rag-eval add ~/docs/handbook.pdf ~/docs/faq/ urls.txt
```

```bash
rag-eval questions --n 100
```

Now your RAG answers `questions.jsonl` and writes `traces/<run>.jsonl`, one JSON object for each answer. [AGENTS.md](AGENTS.md) has the format and a glue script. After that:

| Step | Command | Cost |
|---|---|---|
| Is the evidence in your retrieved passages? | `rag-eval recall` | free |
| Read the answers and note what goes wrong | `rag-eval review --run <run>` | your time |
| Label pass or fail for one judge | `rag-eval review --judge faithfulness` | your time |
| Measure the judge on your labels | `rag-eval validate --judge faithfulness --part dev`, then `--part test` (once) | cents |
| Score a run, then read the report | `rag-eval scan`, then `rag-eval report` | about $0.03 for 100 answers |
| Gate in CI | `rag-eval gate` | free |

`rag-eval add` reads Markdown, text, HTML and PDF, from files, folders, URLs, or a text file of URLs. PDF needs the optional extra: `uv tool install --editable '.[pdf]'`.

API keys: `ANTHROPIC_API_KEY` (question generation and the `claude` judge backend) and `TYPESAFE_API_KEY` (the `jev` judge backend). On macOS, the Keychain items `anthropic-api-key` and `typesafe-api-key` also work.

## With a coding agent

[AGENTS.md](AGENTS.md) tells a coding agent (Claude Code, Codex, Cursor and others) how to run the workflow for you: the steps, what each one costs, how to write the glue for your RAG, and the rules. The most important rule is that the agent never labels: a person does. `CLAUDE.md` imports `AGENTS.md`.

## Your own failure modes

Each judge is one TOML file in `judges/`. The workspace starts with `faithfulness.toml`. When your reading finds another failure mode, add a file:

```toml
name = "competitor"
question = "Does the answer recommend a competitor's product?"
pass_when = "no"
unit = "answer"
backend = "jev"
min_rate = 0.95

[criteria]
pass = "The answer names only our products, or no products."
fail = "The answer suggests that the user buys or uses a competitor's product."
```

Then label it, validate it, and scan with it. Until it has a calibration, the report shows its raw rate marked "NOT CALIBRATED", and the gate skips it.

## What a workspace holds

| Path | What it holds | Commit it? |
|---|---|---|
| `rag-eval.toml` | Settings: decline phrases, citation pattern, the decline limit | yes |
| `judges/` | One judge file for each failure mode | yes |
| `labels/` | The person's verdicts, append-only | yes |
| `calibrations/` | Each judge's measured TPR, TNR and test scores, tied to the exact judge definition | yes |
| `corpus.jsonl`, `questions.jsonl` | Your documents' text, and questions with evidence copied from it | no |
| `traces/`, `scores/` | Your RAG's answers, and the judges' scores | no |

## "Without a third-party tool"

The judges call an outside API, and so does your RAG, so it is fair to ask what "no third-party tool" means here. Here it means no eval platform: nothing like Ragas, TruLens or a hosted eval service decides what a failure is or how to score it.

- The judge backend is a seam. `jev` ([TypeSafe's](https://typesafe.ai) decision model) returns a probability. `claude` asks Claude Haiku 5.5 for yes or no. Any model that answers yes or no can be a third backend.
- Everything that makes a judge trustworthy is plain Python in this repository: the label split, TPR and TNR, the threshold sweep, the corrected rate and the bootstrap.
- A calibration holds only for the exact judge that produced it. A change to the question, the criteria, the backend or the model makes a new judge, and the tool asks you to validate it again.

## Case study: a RAG over one blog

[`examples/hamel-blog/`](examples/hamel-blog/) is the study that this tool was built from: a RAG over 20 blog posts, 50 hand-labeled traces, a faithfulness judge validated on 191 labels, and a gate that runs on GitHub Actions. Its main results:

- Error analysis found no hallucinations. The failures were retrieval misses and over-declines.
- Keyword search found 92–95% of plain questions and only 40–42% of adversarial ones.
- The faithfulness judge has test TPR 0.89 and TNR 0.69. The corrected rate holds the labeled rate for a normal prompt (0.995 against 0.98) and for a loose prompt (0.65 against 0.78).

## Limits

1. **A judge is only as good as its labels.** One careful person who knows the domain is better than many people who do not.
2. **TNR needs fails.** With fewer than 10 fails in the test split, TNR is very uncertain, and so is the corrected rate. The tool warns.
3. **Generated questions are not real users.** Add your own with `--real`, and score production traces when you have them.
4. **A model judge is not fully deterministic.** Two scans of the same answers can give slightly different scores.

## Files

| Path | What it holds |
|---|---|
| `rag_eval/cli.py` | The `rag-eval` command. |
| `rag_eval/workspace.py` | The workspace: the only code that reads or writes files, and the trace contract checks. |
| `rag_eval/reader.py`, `chunk.py` | Reading documents; splitting them into passages. |
| `rag_eval/questions.py`, `bm25.py` | Test questions with evidence, by the reverse method. |
| `rag_eval/judge.py` | Judges, backends, the split, TPR and TNR, corrected rates, `scan`, `report`, `gate`. |
| `rag_eval/review.py`, `review.html` | The local label page. |
| `rag_eval/starter/` | The files that `init` copies. |
| `test_general.py` | The offline check of the whole flow. |
| `examples/hamel-blog/` | The case study. |

## License

MIT. See [LICENSE](LICENSE).
