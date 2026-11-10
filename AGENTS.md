# AGENTS.md

This file tells a coding agent how to use rag-eval-harness for a user. Read all of it before you run a step.

## What the tool does

rag-eval measures how the user's RAG system fails. It reads the user's documents, makes test questions with known evidence, and reads the answers that the user's RAG gives. A person labels some answers pass or fail. Then rag-eval measures each judge against those labels and reports corrected pass rates, with a gate for CI.

## Rules

1. **A person labels. You never label.** Labels are the ground truth that measures each judge. Do not write to `labels/`, do not mark traces on the review page, and do not suggest verdicts to the user while they label. You can start the review page and explain how it works.
2. **Ask before each paid step.** Tell the user the step and the cost from the table below, and wait for a yes. `questions`, `scan` and `validate` call outside APIs. All other steps are free.
3. **The user's documents leave the machine in paid steps.** `questions` sends passages to Anthropic. `scan` and `validate` send questions, passages and answers to the judge's backend (TypeSafe for `jev`, Anthropic for `claude`). If the documents are private, ask the user first.
4. **Never commit document text.** `corpus.jsonl`, `questions.jsonl`, `traces/` and `scores/` hold it. The workspace `.gitignore` keeps them out; do not change that.
5. **Run `validate --part test` once for each judge.** The tool stops a second run. Do not get around it: if you change the judge file to get better test numbers, the test split stops being a fair test. To try a new judge design, use `train` and `dev` only.
6. **Do not edit a judge to make the gate pass.** A changed judge has no calibration, and the gate skips it with a warning. Report the failure to the user.
7. **Use `uv`.** Never use `pip`.

## Install

In the clone of this repository:

```bash
uv tool install --editable .
```

To read PDF files, install the `pdf` extra:

```bash
uv tool install --editable '.[pdf]'
```

API keys: `ANTHROPIC_API_KEY` for `questions` and the `claude` backend, and `TYPESAFE_API_KEY` for the `jev` backend. On macOS, the tool also reads the Keychain items `anthropic-api-key` and `typesafe-api-key`. Never print a key.

## The workflow

Run each step inside the workspace folder. Every step prints what it did.

| Step | Command | Who | Cost |
|---|---|---|---|
| 1 | `rag-eval init my-eval`, then `cd my-eval` | agent | free |
| 2 | `rag-eval add <files, folders, URLs, or urls.txt>` | agent | free |
| 3 | `rag-eval questions --n 100 [--real my_questions.txt]` | agent, after a yes | about $1 for 100 passages |
| 4 | Glue: the user's RAG answers `questions.jsonl` and writes `traces/<run>.jsonl` | agent writes the glue | the user's RAG |
| 5 | `rag-eval recall --run <run>` | agent | free |
| 6 | `rag-eval review --run <run>`: read the answers and note what goes wrong | person | the person's time |
| 7 | Write one judge file for each failure mode that the person found (see below) | agent and person | free |
| 8 | `rag-eval review --judge <name>`: label pass or fail | person | the person's time |
| 9 | `rag-eval validate --judge <name> --part train`, then `dev`, then `test` once | agent, after a yes | about $0.01 to $0.05 a part with `jev` |
| 10 | `rag-eval scan --run <run>`, then `rag-eval report --run <run>` | agent, after a yes | about $0.03 for 100 answers with `jev` |
| 11 | `rag-eval gate --run <run>` in CI | CI | free (reads stored scores) |

Notes:

- Step 2 reads Markdown, text, HTML and PDF. A `.txt` file that holds only URLs, one for each line, is a URL list. A source that cannot be read is skipped with a message.
- Step 3 writes half plain and half adversarial questions. Adversarial questions use the words of other passages, so keyword search finds them less easily. `--real` adds the user's own questions, which have no evidence.
- Step 6 comes before any judge. Do not choose failure modes from a generic list; take them from what the person writes.
- A test split with fewer than 10 fails gives a TNR that is very uncertain. The tool warns. Ask the person to label more fails.

## The trace contract (step 4)

One line of `traces/<run>.jsonl` is one answer:

```json
{"id": "a1", "qid": "g001", "question": "How long do refunds take?", "passages": [{"text": "Refunds are paid within 30 days.", "source": "refunds.md"}], "answer": "Within 30 days [1].", "declined": false, "meta": {}}
```

- Required: `id` (unique in the run), `question`, `passages` (a list of `{text, source}`, in the order the retriever returned them), and `answer`.
- Optional: `qid` (the `qid` from `questions.jsonl`; `recall` needs it), `declined` (true when the system said it cannot answer; without it, the tool uses `decline_phrases` in `rag-eval.toml`), and `meta` (anything).
- Citations: a sentence judge checks a sentence against the passages that it cites. The default pattern is `[1]` for the first passage. If the user's RAG cites in a different way, set `citation_pattern` in `rag-eval.toml`.
- The tool never changes a trace file. A bad line stops the command with the file name and line number.

A glue script for the user's RAG has this shape. Replace `my_rag.answer` with the user's own call:

```python
import json
import sys

import my_rag  # the user's system

run = sys.argv[1]
with open("questions.jsonl") as f, open(f"traces/{run}.jsonl", "w") as out:
    for q in map(json.loads, f):
        result = my_rag.answer(q["question"])  # must return the passages it used and the answer
        out.write(json.dumps({
            "id": q["qid"], "qid": q["qid"], "question": q["question"],
            "passages": [{"text": p.text, "source": p.source} for p in result.passages],
            "answer": result.text,
        }) + "\n")
```

For production traffic, write traces from the logs in the same format, without `qid`.

## A judge file (step 7)

One file in `judges/` for each failure mode. Example:

```toml
name = "competitor"
question = "Does the answer recommend a competitor's product?"
pass_when = "no"       # the answer to `question` that means the trace passes
unit = "answer"        # "answer": one call for each answer; "sentence": each sentence, and the answer scores as its weakest one
backend = "jev"        # "jev" (a probability) or "claude" (yes or no, model claude-haiku-5-5 by default)
min_rate = 0.95        # the gate fails when the corrected pass rate is below this

[criteria]
pass = "The answer names only our products, or no products."
fail = "The answer suggests that the user buys or uses a competitor's product."
```

- The judge sees the question, the passages and the answer (`unit = "answer"`), or the question, the earlier sentences, the sentence and its cited passages (`unit = "sentence"`).
- Judges do not score declined answers. The decline rate is a separate check, with its limit in `rag-eval.toml`.
- A change to `question`, `pass_when`, `criteria`, `unit`, `backend` or `model` makes a new judge: it needs `dev` and `test` again. A change to `name` or `min_rate` does not.
- The starter `judges/faithfulness.toml` checks that the passages state every claim of each sentence.

## How the numbers work

- **Split:** each labeled trace goes to train (15%), dev (40%) or test (45%) by a hash of its run and id. A trace never moves to another split.
- **TPR:** the share of human passes that the judge passes. **TNR:** the share of human fails that the judge fails.
- **Corrected pass rate:** `(p_obs + TNR - 1) / (TPR + TNR - 1)`, where `p_obs` is the judge's raw pass rate on the run. The 95% interval comes from a bootstrap over the run's answers and the judge's test scores.
- **Gate:** it fails when a calibrated judge's corrected rate is below its `min_rate`, or when the decline rate is above `max_decline_rate`. The decline rate counts answers to generated questions only, when the run has them. A judge without a calibration does not fail the gate; the report marks it "NOT CALIBRATED".

## CI

Add the gate to the user's CI after their own job writes the traces of a run:

```yaml
- run: uv tool install --editable path/to/rag-eval-harness
- run: rag-eval scan --run ci && rag-eval gate --run ci
  working-directory: my-eval
  env:
    TYPESAFE_API_KEY: ${{ secrets.TYPESAFE_API_KEY }}
```

`scan` scores only traces that have no score from the judge's current definition, so a second run costs nothing.

## Checks

- `uv run python test_general.py` runs the offline check of the whole flow (no network, no keys).
- `examples/hamel-blog/` is a full case study. See its README.
