"""Judges: one yes-or-no question per failure mode, validated on a person's labels before anyone trusts it.

A judge is a TOML file in judges/. A backend scores it: `jev` (a probability), `claude` (yes or no),
or `stub` (offline tests). The answer scores as its weakest sentence for unit = "sentence".

validate  train and dev: choose the threshold (dev saves it); test: run once, measure TPR and TNR, save
          the calibration. A changed judge (new hash) starts again.
scan      score every new trace of a run with every judge, into scores/<run>.jsonl.
report    for each calibrated judge, the corrected pass rate with a 95% interval; raw rates for the rest;
          the decline rate. gate = report, then exit 1 when a check fails.
"""
import hashlib
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

SPLIT = {"train": 0.15, "dev": 0.40}  # test gets the rest
DEFAULT_MODEL = {"jev": "jev-1.14.0", "claude": "claude-haiku-5-5", "stub": "stub"}
RAW_THRESHOLD = 0.5  # for judges that have no calibration yet
STUB_SEEN = []  # the states the stub backend received, for the offline tests


# --- The judge file ---

def check(j):
    """Stop with a clear message when a judge file is incomplete."""
    name = j.get("name", "?")
    need = {"name": str, "question": str, "pass_when": str, "unit": str, "backend": str}
    for key, kind in need.items():
        if not isinstance(j.get(key), kind):
            raise SystemExit(f"judge {name}: '{key}' is missing")
    if j["pass_when"] not in ("yes", "no"):
        raise SystemExit(f"judge {name}: pass_when must be \"yes\" or \"no\"")
    if j["unit"] not in ("answer", "sentence"):
        raise SystemExit(f"judge {name}: unit must be \"answer\" or \"sentence\"")
    if j["backend"] not in BACKENDS:
        raise SystemExit(f"judge {name}: backend must be one of {sorted(BACKENDS)}")
    if not (isinstance(j.get("criteria", {}).get("pass"), str) and isinstance(j["criteria"].get("fail"), str)):
        raise SystemExit(f"judge {name}: [criteria] needs 'pass' and 'fail'")
    if not 0 <= j.get("min_rate", 0) <= 1:
        raise SystemExit(f"judge {name}: min_rate must be from 0 to 1")
    return j


def model_of(j):
    return j.get("model") or DEFAULT_MODEL[j["backend"]]


def citation_pattern(settings):
    return settings.get("traces", {}).get("citation_pattern", r"\[(\d+)\]")


def judge_hash(j, settings):
    """Everything that changes a judge's scores. Not the name, the gate limit, or the threshold."""
    parts = {k: j[k] for k in ("question", "pass_when", "criteria", "unit", "backend")} | {"model": model_of(j)}
    if j["unit"] == "sentence":
        parts["citation_pattern"] = citation_pattern(settings)
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


# --- What the backend sees ---

def split_sentences(answer, pattern=r"\[(\d+)\]"):
    """Answer text -> [(sentence, cited passage numbers)]. List markers are dropped; lines without letters too."""
    out = []
    for line in answer.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(*])", line):
            if re.search(r"[A-Za-z]", part):
                out.append((part, [int(n) for n in re.findall(pattern, part)]))
    return out


def _passages(ps, numbers=None):
    return [{"n": n, "source": p["source"], "text": p["text"]} for n, p in enumerate(ps, 1) if numbers is None or n in numbers]


def states(j, trace, settings):
    """One state per call. A sentence is checked against the passages it cites (all of them if it cites none);
    the question and the earlier sentences come along, so that "No." or "It also says…" can be understood."""
    if j["unit"] == "answer":
        return [{"question": trace["question"], "passages": _passages(trace["passages"]), "answer": trace["answer"]}]
    sents = split_sentences(trace["answer"], citation_pattern(settings))
    out = []
    for i, (sentence, cites) in enumerate(sents):
        cited = [n for n in cites if 1 <= n <= len(trace["passages"])]
        out.append({"question": trace["question"], "answer_so_far": " ".join(s for s, _ in sents[:i]),
                    "sentence_to_check": sentence, "passages": _passages(trace["passages"], cited or None)})
    return out


# --- Backends: score(judge, states) -> ([P(pass)], cost in $) ---

def _p_pass(j, p_yes):
    return p_yes if j["pass_when"] == "yes" else 1 - p_yes


def score_stub(j, sts):
    STUB_SEEN.extend(sts)
    return [_p_pass(j, 0.1 if "WRONG" in s.get("sentence_to_check", s.get("answer", "")) else 0.9) for s in sts], 0.0


def score_jev(j, sts, workers=8):
    from .llm import jev
    yes, no = (j["criteria"]["pass"], j["criteria"]["fail"]) if j["pass_when"] == "yes" else (j["criteria"]["fail"], j["criteria"]["pass"])
    question = {j["name"]: {"type": "noul", "instructions": j["question"], "criteria": {"true": yes, "false": no}}}
    with ThreadPoolExecutor(workers) as pool:
        resps = list(pool.map(lambda s: jev(s, question, model=model_of(j)), sts))
    return [_p_pass(j, r["answers"][j["name"]]["noul"]) for r in resps], sum(r["cost_usd"] for r in resps)


def score_claude(j, sts, workers=8):
    """Claude answers yes or no, so the score is 1.0 or 0.0 and the threshold is always 0.5 (no sweep)."""
    from .llm import claude
    yes, no = (j["criteria"]["pass"], j["criteria"]["fail"]) if j["pass_when"] == "yes" else (j["criteria"]["fail"], j["criteria"]["pass"])
    system = "You check one output of a question-answering system. Answer the question about the state with yes or no, by the criteria."
    schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}}, "required": ["answer"], "additionalProperties": False}

    def one(s):
        user = f"<state>\n{json.dumps(s, indent=1)}\n</state>\n\nQuestion: {j['question']}\nAnswer yes when: {yes}\nAnswer no when: {no}"
        out = claude(system, user, model=model_of(j), effort="low", max_tokens=4000, schema=schema, fallbacks=False)
        return _p_pass(j, 1.0 if json.loads(out["text"])["answer"] == "yes" else 0.0), out["cost_usd"]

    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(one, sts))
    return [p for p, _ in res], sum(c for _, c in res)


BACKENDS = {"jev": score_jev, "claude": score_claude, "stub": score_stub}


def score_traces(j, traces, settings):
    """({trace id: (score, weakest sentence or None)}, cost in $). The answer scores as its weakest sentence."""
    jobs = [(t["id"], s) for t in traces for s in states(j, t, settings)]
    scores, cost = BACKENDS[j["backend"]](j, [s for _, s in jobs])
    out = {}
    for (tid, s), p in zip(jobs, scores):
        if tid not in out or p < out[tid][0]:
            out[tid] = (p, s.get("sentence_to_check"))
    return {t["id"]: out.get(t["id"], (1.0, None)) for t in traces}, cost  # no sentence: no claim


# --- Statistics ---

def split_of(run, trace_id):
    """train, dev or test: a fixed function of the trace, so new labels never move old ones."""
    x = int(hashlib.sha256(f"{run}/{trace_id}".encode()).hexdigest(), 16) % 1000 / 1000
    return "train" if x < SPLIT["train"] else "dev" if x < SPLIT["train"] + SPLIT["dev"] else "test"


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


def measure(passes, fails, threshold):
    """TPR and TNR with 95% intervals, from the judge's scores on human passes and human fails."""
    tpr = lambda p: sum(s >= threshold for s in p) / len(p)
    tnr = lambda q: sum(s < threshold for s in q) / len(q)
    return {"tpr": tpr(passes), "tnr": tnr(fails), "tpr_ci": bootstrap(lambda p, q: tpr(p), [passes, fails]),
            "tnr_ci": bootstrap(lambda p, q: tnr(q), [passes, fails])}


def corrected(cal, judged):
    """(corrected pass rate, 95% interval) of a run. judged: [judge passed] for each answered trace."""
    t = cal["threshold"]

    def est(p, q, j):
        return corrected_rate(sum(j) / len(j), sum(s >= t for s in p) / len(p), sum(s < t for s in q) / len(q))

    return est(cal["passes"], cal["fails"], judged), bootstrap(est, [cal["passes"], cal["fails"], judged])


# --- Steps ---

def get_judge(ws, name):
    for j in ws.judges():
        if j.get("name") == name:
            return check(j)
    raise SystemExit(f"no judge '{name}' in judges/: the judges are {[j.get('name') for j in ws.judges()]}")


def validate(ws, name, part):
    j = get_judge(ws, name)
    h = judge_hash(j, ws.settings)
    cal = ws.calibration(name)
    if cal and cal["hash"] == h and cal.get("tested_at"):
        raise SystemExit(f"judge {name} is already tested (TPR {cal['tpr']:.2f}, TNR {cal['tnr']:.2f}). "
                         "The test split runs once per judge; to try a new design, change the judge file")
    if part == "test" and not (cal and cal["hash"] == h):
        raise SystemExit(f"judge {name} has no threshold for its current definition: run `rag-eval validate --judge {name} --part dev` first")

    labels = {k: v["verdict"] == "pass" for k, v in ws.labels(name).items() if v.get("verdict") in ("pass", "fail")}
    if not labels:
        raise SystemExit(f"no labels for {name}: label traces with `rag-eval review --judge {name}`")
    traces = {}
    for run in sorted({r for r, _ in labels}):
        if run in ws.runs():
            traces |= {(run, t["id"]): t for t in ws.traces(run) if not t["declined"]}
    keys = [k for k in labels if k in traces]
    counts = {p: [labels[k] for k in keys if split_of(*k) == p] for p in ("train", "dev", "test")}
    print("labels: " + ", ".join(f"{p} {len(v)} ({v.count(False)} fails)" for p, v in counts.items()))
    if len(counts["test"]) - sum(counts["test"]) < 10:
        print("warning: fewer than 10 fails in the test split, so TNR will be very uncertain. Label more fails first.")

    mine = [k for k in keys if split_of(*k) == part]
    if not mine or all(labels[k] for k in mine) or not any(labels[k] for k in mine):
        raise SystemExit(f"the {part} split needs at least one pass and one fail: label more traces")
    # Two runs can share a trace id, so each trace is scored under its "run/id" key.
    scores, cost = score_traces(j, [dict(traces[k], id="/".join(k)) for k in mine], ws.settings)
    scores = {k: scores["/".join(k)] for k in mine}
    rows = [(labels[k], scores[k][0]) for k in mine]
    threshold = cal["threshold"] if part == "test" else RAW_THRESHOLD if j["backend"] == "claude" else sweep(rows)
    m = measure([s for h_, s in rows if h_], [s for h_, s in rows if not h_], threshold)
    print(f"{part}: {len(rows)} answers, cost ${cost:.4f}. Threshold {threshold:.2f}: "
          f"TPR {m['tpr']:.2f} [{m['tpr_ci'][0]:.2f}, {m['tpr_ci'][1]:.2f}]  TNR {m['tnr']:.2f} [{m['tnr_ci'][0]:.2f}, {m['tnr_ci'][1]:.2f}]")
    for k, (human, s) in zip(mine, rows):
        if human != (s >= threshold):
            print(f"  {'missed fail' if not human else 'false fail '} {k[0]}/{k[1]} {s:.2f}  {(scores[k][1] or traces[k]['answer'])[:150]}")

    out = {"judge": name, "hash": h, "backend": j["backend"], "model": model_of(j), "threshold": threshold,
           "tpr": m["tpr"], "tnr": m["tnr"], "n_pass": sum(r[0] for r in rows), "n_fail": sum(not r[0] for r in rows)}
    if part == "dev":
        ws.save_calibration(name, out | {"chosen_on": "dev"})
    elif part == "test":
        ws.save_calibration(name, out | {"tpr_ci": m["tpr_ci"], "tnr_ci": m["tnr_ci"], "passes": [s for h_, s in rows if h_],
                                         "fails": [s for h_, s in rows if not h_], "tested_at": _now()})
    return out


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def judges_to_run(ws, only=None):
    js = [check(j) for j in ws.judges()]
    return [get_judge(ws, only)] if only else js


def scan(ws, run, only=None):
    """Score every answered trace that has no score from the judge's current definition. Returns the count."""
    run = ws.run_name(run)
    traces = [t for t in ws.traces(run) if not t["declined"]]
    have = ws.scores(run)
    total = 0
    for j in judges_to_run(ws, only):
        h = judge_hash(j, ws.settings)
        todo = [t for t in traces if have.get((t["id"], j["name"]), {}).get("hash") != h]
        if not todo:
            continue
        scores, cost = score_traces(j, todo, ws.settings)
        now = _now()
        ws.add_scores(run, [{"trace_id": tid, "judge": j["name"], "hash": h, "score": s, "weakest": w, "scanned_at": now}
                            for tid, (s, w) in scores.items()])
        print(f"{j['name']}: scored {len(todo)} traces, cost ${cost:.4f}")
        total += len(todo)
    return total


def report(ws, run):
    """Print each judge's rate and the decline rate; return the failed checks."""
    run = ws.run_name(run)
    traces = ws.traces(run)
    answered = [t for t in traces if not t["declined"]]
    have = ws.scores(run)
    if not answered:
        raise SystemExit(f"run {run} declined every question, so there is nothing to judge")
    failed = []
    print(f"run {run}: {len(traces)} traces, {len(traces) - len(answered)} declined")
    for j in judges_to_run(ws):
        h = judge_hash(j, ws.settings)
        rows = [have.get((t["id"], j["name"])) for t in answered]
        if any(r is None or r["hash"] != h for r in rows):
            raise SystemExit(f"judge {j['name']} has traces without a score from its current definition: run `rag-eval scan --run {run}`")
        cal = ws.calibration(j["name"])
        min_rate = j.get("min_rate", 0)
        if cal and cal["hash"] == h and cal.get("tested_at"):
            rate, (lo, hi) = corrected(cal, [r["score"] >= cal["threshold"] for r in rows])
            print(f"{j['name']}: corrected pass rate {rate:.3f} [{lo:.3f}, {hi:.3f}], gate minimum {min_rate}")
            threshold = cal["threshold"]
            if not rate >= min_rate:  # nan (a chance-level judge) fails too
                failed.append(f"{j['name']}: corrected pass rate {rate:.3f} is below {min_rate}")
        else:
            threshold = RAW_THRESHOLD
            raw = sum(r["score"] >= threshold for r in rows) / len(rows)
            why = "its definition changed since validation" if cal else "it has no calibration"
            print(f"{j['name']}: raw pass rate {raw:.3f}, NOT CALIBRATED ({why}); the gate skips it")
        for t, r in sorted(zip(answered, rows), key=lambda x: x[1]["score"]):
            if r["score"] < threshold:
                print(f"  flagged {t['id']} {r['score']:.2f}  {(r['weakest'] or t['answer'])[:150]}")

    evidence = {q["qid"] for q in ws.questions() if q.get("evidence")}
    answerable = [t for t in traces if t["qid"] in evidence] or traces
    decline = sum(t["declined"] for t in answerable) / len(answerable)
    max_decline = ws.settings.get("gate", {}).get("max_decline_rate", 0.15)
    print(f"decline rate {decline:.3f} over {len(answerable)} {'answerable questions' if answerable is not traces else 'traces'}, gate maximum {max_decline}")
    if decline > max_decline:
        failed.append(f"decline rate {decline:.3f} is above {max_decline}")
    return failed


def gate(ws, run):
    failed = report(ws, run)
    print("\n".join(f"FAIL {f}" for f in failed) or "PASS")
    if failed:
        sys.exit(1)
    return failed
