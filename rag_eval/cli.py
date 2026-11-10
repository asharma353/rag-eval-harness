"""rag-eval: measure how your RAG system fails. Run each step inside a workspace folder.

  init <folder>       make a workspace with a starter faithfulness judge
  add <paths or urls> read documents into corpus.jsonl (files, folders, URLs, or a .txt list of URLs)
  questions           write questions.jsonl: generated questions with evidence, plus your own (--real)
  (your RAG)          answer questions.jsonl and write traces/<run>.jsonl (see AGENTS.md for the format)
  recall              is the evidence in the passages that your retriever returned?
  review --judge X    label traces pass or fail for one judge, on a local page
  validate --judge X --part train|dev|test   measure the judge on your labels (test runs once)
  scan                score new traces with every judge
  report              corrected pass rates and the decline rate
  gate                report, then exit 1 when a check fails (for CI)
"""
import argparse
import shutil
from pathlib import Path

from . import judge as J
from . import questions as qgen
from . import reader
from .workspace import Workspace

STARTER = Path(__file__).with_name("starter")


def init(folder):
    dest = Path(folder)
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{folder} exists and is not empty")
    shutil.copytree(STARTER, dest, dirs_exist_ok=True)
    (dest / "gitignore").rename(dest / ".gitignore")
    print(f"workspace ready: cd {folder}, then `rag-eval add <files, folder or urls.txt>`")


def add(paths):
    ws = Workspace()
    found = reader.sources(paths)
    docs = {d["source"]: d for d in ws.corpus()}
    added, skipped = 0, []
    for source, kind in found:
        try:
            doc = reader.read(source, kind)
        except OSError as e:  # URLError and HTTPError are OSErrors
            skipped.append(f"could not fetch: {source} ({e})")
            continue
        if doc is None:
            skipped.append(f"no text: {source}")
            continue
        docs[source] = doc
        added += 1
    ws.write_corpus(list(docs.values()))
    words = sum(len(d["text"].split()) for d in docs.values())
    print(f"read {added} documents; the corpus now has {len(docs)} documents, {words} words")
    for line in skipped:
        print(f"  skipped, {line}")


def questions(n, real):
    ws = Workspace()
    if n and not ws.corpus():
        raise SystemExit("the corpus is empty: run `rag-eval add` first")
    out = qgen.generate(ws.corpus(), n) if n else []
    if real:
        lines = [x.strip() for x in open(real) if x.strip()]
        out += [{"qid": f"r{i:03d}", "style": "real", "question": q, "evidence": None} for i, q in enumerate(lines, 1)]
    ws.write_questions(out)
    print(f"{len(out)} questions -> questions.jsonl")


def recall(run):
    """Recall at 1, 3, 5 and all passages, by question style, over traces of questions that have evidence."""
    ws = Workspace()
    evidence = {q["qid"]: q for q in ws.questions() if q.get("evidence")}
    rows = []
    for t in ws.traces(run):
        q = evidence.get(t["qid"])
        if q:
            rank = next((i for i, p in enumerate(t["passages"], 1) if qgen.holds(q["evidence"], p["text"])), None)
            rows.append((q, rank))
    if not rows:
        raise SystemExit("no trace has the qid of a question with evidence: write the qid into each trace")
    out = {}
    for style in sorted({q["style"] for q, _ in rows}) + ["all"]:
        ranks = [r for q, r in rows if style in ("all", q["style"])]
        out[style] = {k: sum(r is not None and r <= k for r in ranks) / len(ranks) for k in (1, 3, 5)}
        out[style] |= {"all": sum(r is not None for r in ranks) / len(ranks), "n": len(ranks)}
    print(f"{'style':12s} {'n':>4s}  R@1   R@3   R@5   R@all")
    for style, v in out.items():
        print(f"{style:12s} {v['n']:4d}  {v[1]:.2f}  {v[3]:.2f}  {v[5]:.2f}  {v['all']:.2f}")
    for q, r in rows:
        if r is None:
            print(f"  missed {q['qid']} ({q['style']}): {q['question'][:100]}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rag-eval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = ap.add_subparsers(dest="step", required=True)
    steps.add_parser("init").add_argument("folder")
    steps.add_parser("add").add_argument("paths", nargs="+")
    sp = steps.add_parser("questions")
    sp.add_argument("--n", type=int, default=100, help="passages to write questions from (some are dropped); 0 for none")
    sp.add_argument("--real", help="a text file with your own questions, one per line")
    steps.add_parser("recall").add_argument("--run")
    sp = steps.add_parser("review")
    sp.add_argument("--judge", help="label traces for this judge")
    sp.add_argument("--run", help="read one run (without --judge: open reading, before any judge exists)")
    sp.add_argument("--port", type=int, default=8765)
    sp = steps.add_parser("validate")
    sp.add_argument("--judge", required=True)
    sp.add_argument("--part", required=True, choices=["train", "dev", "test"])
    sp = steps.add_parser("scan")
    sp.add_argument("--run")
    sp.add_argument("--judge", help="only this judge")
    for step in ("report", "gate"):
        steps.add_parser(step).add_argument("--run")
    args = ap.parse_args(argv)
    if args.step == "init":
        init(args.folder)
    elif args.step == "add":
        add(args.paths)
    elif args.step == "questions":
        questions(args.n, args.real)
    elif args.step == "recall":
        return recall(args.run)
    elif args.step == "review":
        from .review import serve
        serve(Workspace(), args.judge, args.run, args.port)
    elif args.step == "validate":
        return J.validate(Workspace(), args.judge, args.part)
    elif args.step == "scan":
        return J.scan(Workspace(), args.run, args.judge)
    elif args.step == "report":
        return J.report(Workspace(), args.run)
    elif args.step == "gate":
        return J.gate(Workspace(), args.run)


if __name__ == "__main__":
    main()
