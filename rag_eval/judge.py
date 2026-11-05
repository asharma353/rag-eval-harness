"""Slice 4: Jev judges, validated against hand labels.

runs   Answer the synthetic retrieval questions with the normal prompt (run `synthetic`), half of them
       with the loose prompt (run `loose`), and all of them with the cautious prompt (run `cautious`).
       Loose answers mix in facts from Claude's memory: real failures for the faithfulness judge to catch.
       Cautious answers test whether the decline report notices an over-careful prompt.
Label pools are blind: the review page gets no run name, prompt, or cost, and the order is shuffled.
train  Score the train split of the faithfulness labels with Jev, and show the best threshold and the misses.
dev    The same for the dev split, where the threshold is chosen.
test   Run the frozen judge once on the test split: TPR and TNR with bootstrap intervals. The scores go to
       evalset/faith_judge.json, which the monitor (slice 5) uses to correct a run's pass rate.
declines  Over-decline is tracked in code, not by a judge: it is too rare (2 of 50 baseline answers) to
       validate one. Each decline is sorted by what code can know; only "check by hand" needs a person.

uv run python -m rag_eval.judge runs
uv run python -m rag_eval.review --set faithfulness
uv run python -m rag_eval.judge test
uv run python -m rag_eval.judge declines
"""
import argparse
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .chunk import load_chunks
from .llm import jev
from .rag import CAUTIOUS, LOOSE, TRACES, run_all
from .retrieval import PAIRS, relevant

EVALSET = PAIRS.parent
FROZEN = EVALSET / "faith_judge.json"

RUBRICS = {
    "faithfulness": {
        "text": "Pass: the passages state every claim in the answer. A claim with a citation must be in the passage it cites; "
                "a claim without one must be in one of the passages. Fail: any claim that the passages do not state, "
                "even if it is true. Wording may differ; meaning may not. The prompt tells the model the author is "
                "Hamel Husain, so naming him is not a claim.",
        "placeholder": "Which sentence is not supported? What do its passages say instead?",
    },
}


def loose_qids(pairs):
    qids = sorted(p["pid"] for p in pairs)
    return set(random.Random(0).sample(qids, len(qids) // 2))


def synthetic_items():
    pairs = [p for p in map(json.loads, open(PAIRS)) if p["style"] != "real"]
    return [{"qid": p["pid"], "query": p["question"], "style": p["style"]} for p in pairs]


def runs():
    pairs = [p for p in map(json.loads, open(PAIRS)) if p["style"] != "real"]
    items = synthetic_items()
    run_all("synthetic", items)
    loose = loose_qids(pairs)
    run_all("loose", [it for it in items if it["qid"] in loose], system=LOOSE)
    run_all("cautious", items, system=CAUTIOUS)
    # The first half gave 14 fails, too few to measure a judge, so the loose prompt also answers the other half.
    run_all("loose", items, system=LOOSE)


def pool(name):
    """The traces to label for one judge, in a fixed shuffled order.

    faithfulness: answers from baseline, loose, and synthetic. A synthetic answer is left out where its
    question is in the first loose half, so that first pool of 136 had no question twice. Its labels
    were done before the second loose half ran, so its synthetic answers stay in.
    """
    traces = [json.loads(p.read_text()) for p in TRACES.glob("*.json")]
    traces = [t for t in traces if t.get("run") in ("baseline", "synthetic", "loose", "cautious")]
    loose = loose_qids([p for p in map(json.loads, open(PAIRS)) if p["style"] != "real"])
    if name == "faithfulness":
        traces = [t for t in traces if t["run"] != "cautious" and not t["declined"] and not (t["run"] == "synthetic" and t["qid"] in loose)]
    else:
        raise ValueError(f"unknown pool {name}")
    traces.sort(key=lambda t: t["id"])
    random.Random(0).shuffle(traces)
    return traces


# --- Faithfulness judge: one Jev noul per answer sentence; the answer scores as its weakest sentence. ---

FAITH_Q = {
    "type": "noul",
    "instructions": "Do the passages state everything that sentence_to_check claims? Use question and "
                    "answer_so_far only to understand what the sentence refers to; check only sentence_to_check.",
    "criteria": {
        "true": "Every fact, name, number, example, reason, recommendation, and qualifier in the sentence is stated "
                "in the passages, possibly in other words. Also yes when the sentence makes no claim about the topic, "
                "such as saying what the passages do or do not cover. The blog's author is Hamel Husain, so naming "
                "him needs no support.",
        "false": "The sentence adds something that the passages do not state: a new fact, example, reason, or "
                 "recommendation, a link between two ideas that the passages do not make, or a claim stronger or "
                 "broader than the passages make (such as 'always' for 'often', or advice where the passage only "
                 "reports). A true fact that is not in the passages is still no.",
    },
}
# From the dev split (TPR 0.83, TNR 0.82 from 0.45 to 0.50), frozen before the test split ran. Dev also chose the
# state: with the question and the earlier sentences, balanced accuracy on train + dev rose from 0.71 to 0.80.
THRESHOLD = 0.50
JEV_MODEL = "jev-1.14.0"  # pinned: the test split's TPR and TNR hold only for the model that produced them
SPLIT = {"train": 0.15, "dev": 0.40}  # test gets the rest


def split_sentences(answer):
    """Answer text -> [(sentence, cited passage numbers)]. List markers are dropped; lines without letters too."""
    out = []
    for line in answer.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(*])", line):
            if re.search(r"[A-Za-z]", part):
                out.append((part, [int(n) for n in re.findall(r"\[(\d+)\]", part)]))
    return out


def sentence_state(trace, sents, i):
    """What Jev sees for sentence i: the passages it cites (all of them if it cites none), plus the question and the
    earlier sentences, so that "No." or "He also says…" can be understood."""
    sentence, cites = sents[i]
    passages = [r for r in trace["retrieved"] if r["rank"] in cites] or trace["retrieved"]
    return {"question": trace["query"], "answer_so_far": " ".join(s for s, _ in sents[:i]), "sentence_to_check": sentence,
            "passages": [{"n": r["rank"], "title": r["title"], "heading": r["heading"], "text": r["text"]} for r in passages]}


def judge_faith(traces, workers=8):
    """{trace id: [(sentence, P(supported))]} plus total cost. One Jev request per sentence, in parallel."""
    jobs = [(t, sents, i) for t in traces for sents in [split_sentences(t["answer"])] for i in range(len(sents))]
    with ThreadPoolExecutor(workers) as pool_:
        resps = list(pool_.map(lambda j: jev(sentence_state(*j), {"supported": FAITH_Q}, model=JEV_MODEL), jobs))
    out = {t["id"]: [] for t in traces}
    for (t, sents, i), r in zip(jobs, resps):
        out[t["id"]].append((sents[i][0], r["answers"]["supported"]["noul"]))
    return out, sum(r["cost_usd"] for r in resps)


def answer_score(sentence_scores):
    return min((p for _, p in sentence_scores), default=1.0)


def split_labels(verdicts, seed=0):
    """{id: 'pass'|'fail'} -> {id: 'train'|'dev'|'test'}, stratified by verdict, same result for the same labels."""
    rng, out = random.Random(seed), {}
    for v in ("pass", "fail"):
        ids = sorted(i for i, x in verdicts.items() if x == v)
        rng.shuffle(ids)
        a, b = round(SPLIT["train"] * len(ids)), round((SPLIT["train"] + SPLIT["dev"]) * len(ids))
        out.update({i: "train" for i in ids[:a]} | {i: "dev" for i in ids[a:b]} | {i: "test" for i in ids[b:]})
    return out


def tpr_tnr(rows, threshold):
    """rows: [(human passed, judge score)]. TPR = judge passes a human pass; TNR = judge fails a human fail."""
    passes = [s >= threshold for h, s in rows if h]
    fails = [s < threshold for h, s in rows if not h]
    return sum(passes) / len(passes), sum(fails) / len(fails)


def sweep(rows):
    """The threshold with the best balanced accuracy (TPR + TNR) / 2; ties go to the lowest threshold."""
    cuts = sorted({s for _, s in rows} | {1.01})
    return max(cuts, key=lambda c: (sum(tpr_tnr(rows, c)), -c))


def corrected_rate(p_obs, tpr, tnr):
    """The true pass rate implied by the judge's observed pass rate and its known errors, clipped to [0, 1]."""
    if tpr + tnr <= 1:
        return float("nan")  # the judge is no better than chance, so its rate says nothing
    return min(1.0, max(0.0, (p_obs + tnr - 1) / (tpr + tnr - 1)))


def bootstrap(stat, samples, n=2000, seed=0):
    """95% percentile interval of stat(*samples), each sample resampled with replacement on its own."""
    rng, vals = random.Random(seed), []
    for _ in range(n):
        v = stat(*[rng.choices(s, k=len(s)) for s in samples])
        if v == v:  # drop nan
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def faith_rows(part):
    """Score one split with Jev. Returns [(trace, human passed, score, sentence scores)] and the cost."""
    from .review import load_labels  # review imports this module
    traces = {t["id"]: t for t in pool("faithfulness")}
    labels = {k: v["verdict"] for k, v in load_labels(EVALSET / "labels_faithfulness.jsonl").items() if k in traces and v["verdict"]}
    ids = [i for i, p in split_labels(labels).items() if p == part]
    scores, cost = judge_faith([traces[i] for i in ids])
    return [(traces[i], labels[i] == "pass", answer_score(scores[i]), scores[i]) for i in ids], cost


def show_misses(rows, threshold):
    for t, human, score, sents in sorted(rows, key=lambda r: r[2]):
        if human != (score >= threshold):
            worst = min(sents, key=lambda x: x[1])
            print(f"  {'missed fail' if not human else 'false fail '} {t['qid']} {score:.2f}  {worst[0][:150]}")


def faith(part):
    rows, cost = faith_rows(part)
    pairs = [(h, s) for _, h, s, _ in rows]
    n_fail = sum(not h for h, _ in pairs)
    print(f"{part}: {len(rows)} answers, {n_fail} human fails, Jev cost ${cost:.4f}")
    if part in ("train", "dev"):
        t = THRESHOLD if part == "train" and THRESHOLD else sweep(pairs)
        tpr, tnr = tpr_tnr(pairs, t)
        print(f"best threshold {t:.2f}: TPR {tpr:.2f}  TNR {tnr:.2f}")
        show_misses(rows, t)
        return
    assert THRESHOLD is not None, "set THRESHOLD from the dev split first"
    tpr, tnr = tpr_tnr(pairs, THRESHOLD)
    passes, fails = [s for h, s in pairs if h], [s for h, s in pairs if not h]
    lo_hi = lambda f: bootstrap(lambda p, q: f(p, q), [passes, fails])
    tpr_ci = lo_hi(lambda p, q: sum(s >= THRESHOLD for s in p) / len(p))
    tnr_ci = lo_hi(lambda p, q: sum(s < THRESHOLD for s in q) / len(q))
    print(f"threshold {THRESHOLD}: TPR {tpr:.2f} [{tpr_ci[0]:.2f}, {tpr_ci[1]:.2f}]  TNR {tnr:.2f} [{tnr_ci[0]:.2f}, {tnr_ci[1]:.2f}]")
    show_misses(rows, THRESHOLD)
    source = "judge.FAITH_Q answer scores on the test split of evalset/labels_faithfulness.jsonl"
    FROZEN.write_text(json.dumps({"judge_model": JEV_MODEL, "threshold": THRESHOLD, "source": source,
                                  "passes": passes, "fails": fails}) + "\n")


# --- Over-decline: tracked in code. ---

DECLINE_KINDS = ("over-decline", "check by hand", "evidence not retrieved", "out of scope")


def decline_kind(trace, pair, chunks):
    """Why one declined answer declined, as far as code can tell.
    pair: the question's evidence span, if it has one. chunks: chunk id -> chunk with word span."""
    if pair:
        held = any(relevant(chunks[r["id"]], pair) for r in trace["retrieved"])
        return "over-decline" if held else "evidence not retrieved"
    if trace.get("tuple", {}).get("scope") == "out_of_scope":
        return "out of scope"
    return "check by hand"


def declines():
    """Decline rate per run, each decline sorted by kind, and the ones that need a person."""
    pairs = {p["pid"]: p for p in map(json.loads, open(PAIRS))}
    chunks = {c["id"]: c for c in load_chunks()}
    traces = [json.loads(p.read_text()) for p in TRACES.glob("*.json")]
    traces = [t for t in traces if t.get("run")]
    print(f"{'run':10s} {'answers':>7s} {'declined':>8s} {'in-scope rate':>13s}  " + "  ".join(DECLINE_KINDS))
    review = []
    for run in sorted({t["run"] for t in traces}):
        ts = [t for t in traces if t["run"] == run]
        kinds = Counter(decline_kind(t, pairs.get(t["qid"]), chunks) for t in ts if t["declined"])
        in_scope = [t for t in ts if t.get("tuple", {}).get("scope") != "out_of_scope"]
        rate_ = (sum(kinds.values()) - kinds["out of scope"]) / len(in_scope)
        print(f"{run:10s} {len(ts):7d} {sum(kinds.values()):8d} {rate_:13.2f}  " + "  ".join(f"{kinds[k]:{len(k)}d}" for k in DECLINE_KINDS))
        review += [(run, t) for t in ts if t["declined"] and decline_kind(t, pairs.get(t["qid"]), chunks) in DECLINE_KINDS[:2]]
    for run, t in sorted(review, key=lambda x: (x[0], x[1]["qid"])):
        print(f"  {decline_kind(t, pairs.get(t['qid']), chunks):13s} {run:9s} {t['qid']}  {t['query'][:90]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["runs", "train", "dev", "test", "declines"])
    step = ap.parse_args().step
    if step == "runs":
        runs()
    elif step == "declines":
        declines()
    else:
        faith(step)


if __name__ == "__main__":
    main()
