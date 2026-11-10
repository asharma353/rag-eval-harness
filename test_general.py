"""Offline checks for the general flow. Run: uv run python test_general.py

Each section tests one agreed seam. No network and no API key: judges use the `stub` backend,
and question generation gets a stub LLM.
"""
import json
import os
import tempfile
from pathlib import Path

from rag_eval.cli import main
from rag_eval.workspace import Workspace


def stops(fn, *words):
    """fn() must stop with a message that holds every word."""
    try:
        fn()
    except SystemExit as e:
        msg = str(e.code)
        assert all(w in msg for w in words), msg
        return msg
    raise AssertionError(f"expected a stop with {words}")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def trace(i, answer="It is X [1].", **extra):
    return {"id": f"t{i}", "question": f"Q{i}?", "passages": [{"text": "X is so.", "source": "doc.md"}], "answer": answer, **extra}


home = Path(tempfile.mkdtemp())
os.chdir(home)

# --- Seam 1 (CLI): init makes a workspace; other steps stop outside one ---
stops(lambda: Workspace(), "rag-eval.toml", "init")
main(["init", "ws"])
ws_dir = home / "ws"
assert (ws_dir / "rag-eval.toml").exists() and (ws_dir / "judges" / "faithfulness.toml").exists()
assert "traces/" in (ws_dir / ".gitignore").read_text() and "corpus.jsonl" in (ws_dir / ".gitignore").read_text()
stops(lambda: main(["init", "ws"]), "ws", "exists")
os.chdir(ws_dir)
ws = Workspace()
assert [j["name"] for j in ws.judges()] == ["faithfulness"]

# --- Seam 2: Workspace.traces() checks the trace contract ---
assert ws.runs() == []
write_jsonl(ws_dir / "traces" / "a.jsonl", [trace(1), trace(2, "I cannot answer that from my documents."), trace(3, declined=True)])
got = ws.traces()  # one run: no name needed
assert [t["id"] for t in got] == ["t1", "t2", "t3"]
assert [t["declined"] for t in got] == [False, True, True], "declined from the field, else from [traces] decline_phrases"
assert got[0]["passages"] == [{"text": "X is so.", "source": "doc.md"}] and got[0]["meta"] == {} and got[0]["qid"] is None

write_jsonl(ws_dir / "traces" / "b.jsonl", [trace(1), trace(1)])
stops(lambda: ws.traces(), "--run", "a", "b")
stops(lambda: ws.traces("b"), "b.jsonl:2", "duplicate", "t1")
write_jsonl(ws_dir / "traces" / "b.jsonl", [trace(1), {"id": "t2", "question": "Q?", "answer": "A"}])
stops(lambda: ws.traces("b"), "b.jsonl:2", "passages")
(ws_dir / "traces" / "b.jsonl").write_text(json.dumps(trace(1)) + "\nnot json\n")
stops(lambda: ws.traces("b"), "b.jsonl:2")
write_jsonl(ws_dir / "traces" / "b.jsonl", [trace(1, passages=[{"text": "x"}])])
stops(lambda: ws.traces("b"), "b.jsonl:1", "source")
(ws_dir / "traces" / "b.jsonl").unlink()

# --- Seam 3: add reads Markdown, text, HTML (from a file or a URL list) into corpus.jsonl ---
docs = home / "docs"
(docs / "sub").mkdir(parents=True)
(docs / "guide.md").write_text("# Setup Guide\n\nInstall   the tool.\n\n## Run it\nType `go`.\n### Flags\n-v means verbose.\n")
(docs / "sub" / "notes.txt").write_text("Plain notes.\n\nSecond   line.\n")
(docs / "sub" / "logo.png").write_bytes(b"\x89PNG")
(docs / "page.html").write_text(
    "<html><head><title>Billing FAQ</title><style>p{}</style></head><body><nav>Home | About</nav>"
    "<main><h1>Billing</h1><p>Pay by <b>card</b>.</p><script>track()</script><h2>Refunds</h2><p>Within 30 days.</p></main>"
    "<footer>Copyright</footer></body></html>")
(docs / "empty.md").write_text("\n\n")
(home / "urls.txt").write_text(f"{(docs / 'page.html').as_uri()}\n\n")

main(["add", str(docs)])
corpus = {Path(d["source"]).name: d for d in ws.corpus()}
assert sorted(corpus) == ["guide.md", "notes.txt", "page.html"], sorted(corpus)  # the PNG and the empty file are skipped
assert corpus["guide.md"]["title"] == "Setup Guide"
assert corpus["guide.md"]["text"] == "Install the tool.\n## Run it\nType `go`.\n## Flags\n-v means verbose.", repr(corpus["guide.md"]["text"])
assert corpus["notes.txt"]["title"] == "notes" and corpus["notes.txt"]["text"] == "Plain notes.\nSecond line."
assert corpus["page.html"]["title"] == "Billing FAQ"
assert corpus["page.html"]["text"] == "## Billing\nPay by card.\n## Refunds\nWithin 30 days.", repr(corpus["page.html"]["text"])
assert len({d["id"] for d in ws.corpus()}) == 3

main(["add", str(docs / "guide.md"), str(home / "urls.txt")])  # the same file again replaces it; a URL list adds the page
corpus = ws.corpus()
assert len(corpus) == 4 and sum(d["source"].endswith("guide.md") for d in corpus) == 1
by_url = [d for d in corpus if d["source"].startswith("file:")]
assert by_url and by_url[0]["text"] == "## Billing\nPay by card.\n## Refunds\nWithin 30 days."

main(["add", (docs / "gone.html").as_uri()])  # a URL that cannot be fetched is skipped, and the corpus stays as it was
assert len(ws.corpus()) == 4

# --- Seam 6: generated questions keep only evidence that is really in the corpus ---
from rag_eval import questions as qgen  # noqa: E402


def stub_llm(system, user, schema=None):
    """Copies 10 target words as evidence, except for the zebra document, where it invents the evidence."""
    target = user.split("<target>\n", 1)[1].split("\n</target>", 1)[0]
    evidence = "Zebras are known to fly south every winter" if "zebra" in target else " ".join(target.split()[:10])
    question = "How would someone handle this kind of thing?" if "<other>" in user else f"What is true about {target.split()[0]}?"
    return {"text": json.dumps({"evidence": evidence, "question": question}), "cost_usd": 0.001}


topics = ["billing", "refunds", "shipping", "returns", "zebra"]
qdocs = [{"id": t, "source": f"{t}.md", "title": t.title(),
          "text": " ".join(f"{t} fact{i} is about {t} rule{i}." for i in range(12))} for t in topics]
qs = qgen.generate(qdocs, 8, llm=stub_llm)
assert qs and len({q["qid"] for q in qs}) == len(qs)
assert all(q["doc"] != "zebra" for q in qs), "invented evidence is dropped"
texts = {d["id"]: d["text"] for d in qdocs}
assert all(qgen.holds(q["evidence"], texts[q["doc"]]) for q in qs)
assert {q["style"] for q in qs} == {"plain", "adversarial"}
assert qgen.holds("Pay by card", "## Billing\nYou can pay, by CARD, today.") and not qgen.holds("pay by cheque only from today", "pay by card")
assert qgen.holds("a b c d", "x a b y") and not qgen.holds("a b c d", "a x b y"), "at least half of the words, in one run"

try:
    import pypdf  # noqa: F401
except ImportError:
    (docs / "scan.pdf").write_bytes(b"%PDF-1.4")
    stops(lambda: main(["add", str(docs / "scan.pdf")]), "pdf")

# --- Seam 1 (CLI): questions --real adds the user's own questions; recall checks the user's retriever ---
(home / "real.txt").write_text("How do I get a refund?\n\nCan I pay in cash?\n")
main(["questions", "--n", "0", "--real", str(home / "real.txt")])  # --n 0: no LLM call
assert [(q["qid"], q["style"], q["evidence"]) for q in ws.questions()] == [("r001", "real", None), ("r002", "real", None)]

ws.write_questions([
    {"qid": "g001", "style": "plain", "question": "How do I pay?", "evidence": "Pay by card."},
    {"qid": "g002", "style": "adversarial", "question": "Money back?", "evidence": "Refunds are paid within 30 days of the return."},
    {"qid": "r001", "style": "real", "question": "Cash?", "evidence": None},
])
p = lambda text: {"text": text, "source": "page.html"}
write_jsonl(ws_dir / "traces" / "r.jsonl", [
    trace(1, qid="g001", passages=[p("Pay by card.")]),                                         # rank 1
    trace(2, qid="g002", passages=[p("Shipping is free."), p("Returns: refunds are paid within 30")]),  # rank 2
    trace(3, qid="r001", passages=[p("Cash is fine.")]),                                        # no evidence: not counted
    trace(4, passages=[p("x")]),                                                                # no qid: not counted
])
r = main(["recall", "--run", "r"])
assert r == {"plain": {1: 1.0, 3: 1.0, 5: 1.0, "all": 1.0, "n": 1}, "adversarial": {1: 0.0, 3: 1.0, 5: 1.0, "all": 1.0, "n": 1},
             "all": {1: 0.5, 3: 1.0, 5: 1.0, "all": 1.0, "n": 2}}, r

# --- Seam 4: the judge engine, with the stub backend ---
# The stub answers the judge's question "yes" (0.9) unless the text it judges holds WRONG (0.1).
from rag_eval import judge as J  # noqa: E402


def gate_code(argv):
    try:
        main(argv)
    except SystemExit as e:
        return e.code
    return 0


def edit(path, old, new):
    text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new))


os.chdir(home)
main(["init", "jw"])
jw = home / "jw"
os.chdir(jw)
edit(jw / "judges" / "faithfulness.toml", 'backend = "jev"', 'backend = "stub"')
edit(jw / "judges" / "faithfulness.toml", 'model = "jev-1.14.0"', 'model = "stub-1"')

# The split is a fixed function of (run, trace id): about 15/40/45, the same on every call.
parts = [J.split_of("lab", f"t{i}") for i in range(3000)]
assert all(abs(parts.count(p) / 3000 - share) < 0.03 for p, share in (("train", 0.15), ("dev", 0.40), ("test", 0.45)))
assert [J.split_of("lab", f"t{i}") for i in range(50)] == parts[:50]

# Labeled run: in test, 10 human passes (the judge fails 2) and 10 human fails (the judge passes 3): TPR 0.8, TNR 0.7.
# In train and dev, the judge agrees with every label.
want = {("test", "pass", "ok"): 8, ("test", "pass", "bad"): 2, ("test", "fail", "bad"): 7, ("test", "fail", "ok"): 3,
        ("dev", "pass", "ok"): 6, ("dev", "fail", "bad"): 4, ("train", "pass", "ok"): 2, ("train", "fail", "bad"): 1}
lab, i = [], 0
while any(want.values()):
    part = J.split_of("lab", f"t{i}")
    for key in want:
        if key[0] == part and want[key]:
            want[key] -= 1
            lab.append((f"t{i}", key[1], key[2]))
            break
    i += 1
answer_of = {"ok": "Good first. Fine second [1].", "bad": "Good first. WRONG second [1]."}
write_jsonl(jw / "traces" / "lab.jsonl", [trace(t, answer_of[kind]) | {"id": t} for t, _, kind in lab])
jws = Workspace()
for t, verdict, _ in lab:
    jws.add_label("faithfulness", {"run": "lab", "trace_id": t, "verdict": verdict, "notes": ""})

stops(lambda: main(["validate", "--judge", "faithfulness", "--part", "test"]), "--part dev")
dev = main(["validate", "--judge", "faithfulness", "--part", "dev"])
assert (dev["tpr"], dev["tnr"]) == (1.0, 1.0) and dev["n_pass"] == 6 and dev["n_fail"] == 4
test = main(["validate", "--judge", "faithfulness", "--part", "test"])
assert (round(test["tpr"], 6), round(test["tnr"], 6)) == (0.8, 0.7) and (test["n_pass"], test["n_fail"]) == (10, 10)
stops(lambda: main(["validate", "--judge", "faithfulness", "--part", "test"]), "already")
stops(lambda: main(["validate", "--judge", "faithfulness", "--part", "dev"]), "already")
cal = jws.calibration("faithfulness")
assert cal["tpr"] == test["tpr"] and len(cal["passes"]) == 10 and len(cal["fails"]) == 10 and cal["tested_at"]

# What the backend sees: a sentence judge gets the question, the earlier sentences, and only the passages its
# sentence cites (all of them when it cites none).
J.STUB_SEEN.clear()
faith = next(j for j in jws.judges() if j["name"] == "faithfulness")
three = trace(9, "First claim [2]. Second claim.", passages=[{"text": f"P{n}", "source": "d"} for n in (1, 2, 3)])
J.score_traces(faith, [three], jws.settings)
assert [s["sentence_to_check"] for s in J.STUB_SEEN] == ["First claim [2].", "Second claim."]
assert [[p["text"] for p in s["passages"]] for s in J.STUB_SEEN] == [["P2"], ["P1", "P2", "P3"]]
assert J.STUB_SEEN[1]["answer_so_far"] == "First claim [2]." and J.STUB_SEEN[0]["question"] == "Q9?"

# --- Seam 1 (CLI): scan, report, gate on a production run ---
# 10 answers (the judge passes 7) and 2 declines, no qids: declines 2/12 = 0.17;
# corrected faithfulness (0.7 + 0.7 - 1) / (0.8 + 0.7 - 1) = 0.8.
prod = [trace(n, answer_of["ok" if n < 7 else "bad"]) for n in range(10)] + [trace(10, "I don't know."), trace(11, declined=True)]
write_jsonl(jw / "traces" / "prod.jsonl", prod)
stops(lambda: main(["report", "--run", "prod"]), "scan")
assert main(["scan", "--run", "prod"]) == 10, "declined traces are not judged"
assert main(["scan", "--run", "prod"]) == 0, "a second scan finds nothing new"
failed = main(["report", "--run", "prod"])
assert len(failed) == 2 and "faithfulness" in failed[0] and "0.800" in failed[0] and "decline" in failed[1] and "0.167" in failed[1], failed
assert gate_code(["gate", "--run", "prod"]) == 1

# The gate limit is not part of the judge, so a new limit keeps the calibration.
edit(jw / "judges" / "faithfulness.toml", "min_rate = 0.90", "min_rate = 0.75")
edit(jw / "rag-eval.toml", "max_decline_rate = 0.15", "max_decline_rate = 0.20")
assert gate_code(["gate", "--run", "prod"]) == 0

# A new judge without labels: scan scores it, report shows its raw rate, and the gate skips it.
(jw / "judges" / "upsell.toml").write_text(
    'name = "upsell"\nquestion = "Does the answer push the user to buy something?"\npass_when = "no"\nunit = "answer"\n'
    'backend = "stub"\nmin_rate = 0.5\n[criteria]\npass = "No sales push."\nfail = "A sales push."\n')
stops(lambda: main(["report", "--run", "prod"]), "scan", "upsell")
assert main(["scan", "--run", "prod"]) == 10
assert main(["report", "--run", "prod"]) == [], "raw rate 0.3 is below 0.5, but an uncalibrated judge does not fail the gate"
assert main(["scan", "--run", "prod", "--judge", "upsell"]) == 0

# A changed question is a new judge: old scores and the calibration no longer count.
edit(jw / "judges" / "faithfulness.toml", "check only sentence_to_check.", "check only sentence_to_check, nothing else.")
stops(lambda: main(["report", "--run", "prod"]), "scan", "faithfulness")
main(["scan", "--run", "prod"])
assert main(["report", "--run", "prod"]) == []
stops(lambda: main(["validate", "--judge", "faithfulness", "--part", "test"]), "--part dev")

# Two runs can use the same trace id (the AGENTS.md glue uses the qid). Each labeled trace keeps its own score.
(jw / "judges" / "dup.toml").write_text(
    'name = "dup"\nquestion = "Is the answer right?"\npass_when = "yes"\nunit = "answer"\nbackend = "stub"\n'
    '[criteria]\npass = "Right."\nfail = "Wrong."\n')
dev_runs = [r for r in (f"x{n}" for n in range(200)) if J.split_of(r, "same") == "dev"][:2]
for run, answer, verdict in zip(dev_runs, ["Fine.", "WRONG."], ["pass", "fail"]):
    write_jsonl(jw / "traces" / f"{run}.jsonl", [trace(0, answer) | {"id": "same"}])
    jws.add_label("dup", {"run": run, "trace_id": "same", "verdict": verdict, "notes": ""})
dup = main(["validate", "--judge", "dup", "--part", "dev"])
assert (dup["tpr"], dup["tnr"]) == (1.0, 1.0), dup

# --- Seam 5: Hamel parity. The general code, given the frozen test scores, gives the published TPR and TNR. ---
frozen = json.loads((Path(__file__).resolve().parent / "examples/hamel-blog/evalset/faith_judge.json").read_text())
m = J.measure(frozen["passes"], frozen["fails"], frozen["threshold"])
assert (round(m["tpr"], 2), round(m["tnr"], 2), len(frozen["passes"]), len(frozen["fails"])) == (0.89, 0.69, 73, 13), m

print("general checks pass")
