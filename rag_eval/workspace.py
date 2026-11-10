"""The workspace: one folder that holds everything for the evaluation of one RAG system.

This is the only module that knows file names, reads or writes a file, or sees a raw trace line.

rag-eval.toml          settings
corpus.jsonl           the documents, as `add` read them               (not committed: their text)
questions.jsonl        test questions with their evidence               (not committed: corpus text)
traces/<run>.jsonl     the user's RAG output, one trace per line        (not committed; never rewritten)
scores/<run>.jsonl     judge scores, append-only, latest wins           (not committed)
judges/<name>.toml     one judge per failure mode
labels/<judge>.jsonl   a person's verdicts, append-only, latest wins
calibrations/<judge>.json  the judge's measured accuracy
"""
import json
import tomllib
from pathlib import Path

CONFIG = "rag-eval.toml"


class Workspace:
    def __init__(self, root="."):
        self.root = Path(root).resolve()
        if not (self.root / CONFIG).exists():
            raise SystemExit(f"no {CONFIG} in {self.root}: run `rag-eval init <folder>`, then work inside that folder")
        self.settings = tomllib.loads((self.root / CONFIG).read_text())

    # --- plain files ---

    def _rows(self, rel):
        """[(line number, record)] of a JSONL file; a bad line stops with the file and line number."""
        path = self.root / rel
        if not path.exists():
            return []
        out = []
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append((n, json.loads(line)))
            except json.JSONDecodeError:
                raise SystemExit(f"{rel}:{n}: not valid JSON") from None
        return out

    def _write(self, rel, rows):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def _append(self, rel, rows):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)

    def corpus(self):
        return [r for _, r in self._rows("corpus.jsonl")]

    def write_corpus(self, docs):
        self._write("corpus.jsonl", docs)

    def questions(self):
        return [r for _, r in self._rows("questions.jsonl")]

    def write_questions(self, questions):
        self._write("questions.jsonl", questions)

    # --- traces ---

    def runs(self):
        return sorted(p.stem for p in (self.root / "traces").glob("*.jsonl"))

    def run_name(self, run=None):
        """The run to use: the one named, or the only one there is."""
        runs = self.runs()
        if run is None:
            if len(runs) != 1:
                raise SystemExit(f"pass --run: the runs are {runs}" if runs else "no traces yet: write traces/<run>.jsonl")
            return runs[0]
        if run not in runs:
            raise SystemExit(f"no run '{run}': the runs are {runs}")
        return run

    def traces(self, run=None):
        """The checked traces of one run: {id, question, passages: [{text, source}], answer, declined, qid, meta}."""
        rel = f"traces/{self.run_name(run)}.jsonl"
        phrases = [p.casefold() for p in self.settings.get("traces", {}).get("decline_phrases", [])]
        out, seen = [], set()
        for n, t in self._rows(rel):
            bad = _contract_error(t)
            if bad:
                raise SystemExit(f"{rel}:{n}: {bad}")
            if t["id"] in seen:
                raise SystemExit(f"{rel}:{n}: duplicate trace id {t['id']}")
            seen.add(t["id"])
            declined = t["declined"] if "declined" in t else any(p in t["answer"].casefold() for p in phrases)
            out.append({"id": t["id"], "question": t["question"], "passages": [{"text": p["text"], "source": p["source"]} for p in t["passages"]],
                        "answer": t["answer"], "declined": declined, "qid": t.get("qid"), "meta": t.get("meta", {})})
        return out

    # --- judges, labels, scores, calibrations ---

    def judges(self):
        return [tomllib.loads(p.read_text()) for p in sorted((self.root / "judges").glob("*.toml"))]

    def labels(self, judge):
        """{(run, trace id): latest label}."""
        return {(r["run"], r["trace_id"]): r for _, r in self._rows(f"labels/{judge}.jsonl")}

    def add_label(self, judge, rec):
        self._append(f"labels/{judge}.jsonl", [rec])

    def scores(self, run):
        """{(trace id, judge): latest score record}."""
        return {(r["trace_id"], r["judge"]): r for _, r in self._rows(f"scores/{run}.jsonl")}

    def add_scores(self, run, recs):
        self._append(f"scores/{run}.jsonl", recs)

    def calibration(self, judge):
        path = self.root / "calibrations" / f"{judge}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def save_calibration(self, judge, rec):
        path = self.root / "calibrations" / f"{judge}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(rec, indent=1) + "\n")


def _contract_error(t):
    """What is wrong with one trace line, or None."""
    if not isinstance(t, dict):
        return "a trace must be a JSON object"
    for key in ("id", "question", "answer"):
        if not isinstance(t.get(key), str):
            return f"'{key}' must be a string"
    if not isinstance(t.get("passages"), list):
        return "'passages' must be a list of {text, source}"
    for i, p in enumerate(t["passages"], 1):
        if not (isinstance(p, dict) and isinstance(p.get("text"), str) and isinstance(p.get("source"), str)):
            return f"passage {i} needs 'text' and 'source' strings"
    if "declined" in t and not isinstance(t["declined"], bool):
        return "'declined' must be true or false"
    if "meta" in t and not isinstance(t["meta"], dict):
        return "'meta' must be an object"
    return None
