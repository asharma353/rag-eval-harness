# RAG eval harness

The harness measures how a RAG system fails. It validates one judge for each failure mode against labels from a person, and it reports corrected rates for each run of the system.

## Language

### Where the data lives

**Workspace**:
One folder that holds everything for the evaluation of one RAG system: its corpus, questions, runs, judges, labels and calibrations.
_Avoid_: project, eval folder, data dir

**Corpus**:
The documents that the RAG system answers from, as the harness read them.
_Avoid_: dataset, knowledge base, docs

**Run**:
One batch of traces from one version of the RAG system, for example one prompt or one CI build.
_Avoid_: experiment, batch, job

### What the RAG system produced

**Trace**:
The record of one answer: the question, the passages that the system retrieved, the answer, and whether the system declined.
_Avoid_: log, sample, response

**Passage**:
One piece of text that the user's retriever returned for a question, with its source.
_Avoid_: chunk (a chunk is only the reference RAG's own split of the corpus), context, document

**Decline**:
An answer in which the system says that it cannot answer from its passages.
_Avoid_: refusal, abstention, no-answer

**Evidence**:
The exact text in the corpus that answers a generated question.
_Avoid_: ground truth, gold passage

### How failures are measured

**Failure mode**:
One specific way in which an answer goes wrong, found by reading traces.
_Avoid_: metric, error category

**Judge**:
One yes-or-no question that detects one failure mode in a trace. A decline is not a judge, because code finds it without labels.
_Avoid_: evaluator, grader, metric

**Test split**:
The part of a judge's labels that is held out until the judge design is final, and is then used once to measure it. A trace stays in the same split for its whole life.
_Avoid_: holdout, eval set

**Label**:
A person's pass-or-fail verdict on one trace for one judge.
_Avoid_: annotation, rating, ground truth

**Calibration**:
The measured accuracy of one judge on held-out labels (its threshold, TPR and TNR), tied to the exact judge definition that produced it.
_Avoid_: frozen numbers, validation results

**Corrected rate**:
A run's pass rate for one judge, corrected for the judge's known errors.
_Avoid_: true rate, adjusted score
