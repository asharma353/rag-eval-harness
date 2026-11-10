"""Slice 5: a monitor that reads the trace log. It reports and blocks nothing.

The judge passes 89% of good answers (test TPR), so as a guardrail it would stop about 1 good answer in 9.
As a monitor it runs after the answers are sent: the user waits for nothing, and after a crash you run it again.

ask     Answer the 109 synthetic questions with the normal prompt into one run (what CI does).
scan    Judge every trace of a run that has no `monitor` field yet, and write the result into the trace.
report  The corrected faithfulness rate with its interval, the in-scope decline rate, and the flagged answers.
gate    The report, then exit 1 if a check fails. It reads the stored scores, so it makes no calls.

uv run python -m rag_eval.monitor scan --run synthetic
uv run python -m rag_eval.monitor gate --run synthetic
"""
import argparse
import json
import sys
from datetime import datetime, timezone

from .judge import FROZEN, JEV_MODEL, answer_score, bootstrap, corrected_rate, judge_faith, synthetic_items
from .rag import TRACES, run_all

# Normal prompt 0.995, loose 0.65. The point estimate, not the interval's lower end: with 13 test-split
# fails that end is about 0.83 even for a perfect run, so a gate on it would always fail.
MIN_FAITH = 0.90
# Normal prompt 0.10, cautious 0.17. With 109 questions 0.15 is about 1.7 standard deviations above 0.10,
# so expect a false alarm about 1 run in 20.
MAX_DECLINE = 0.15


def traces_of(run):
    traces = [json.loads(p.read_text()) for p in TRACES.glob("*.json")]
    return [t for t in traces if t.get("run") == run]


def scan(run):
    todo = [t for t in traces_of(run) if "monitor" not in t]
    scores, cost = judge_faith([t for t in todo if not t["declined"]])
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for t in todo:
        s = scores.get(t["id"])
        t["monitor"] = {"declined": t["declined"], "score": None if s is None else answer_score(s),
                        "weakest": min(s, key=lambda x: x[1])[0] if s else None,
                        "judge_model": JEV_MODEL, "scanned_at": now}
        (TRACES / f"{t['id']}.json").write_text(json.dumps(t, indent=2))
    print(f"{run}: scanned {len(todo)} new traces, {len(scores)} judged, Jev cost ${cost:.4f}")


def summary(traces, frozen):
    """(corrected faithfulness rate, its 95% interval, in-scope decline rate) from the scores that scan stored."""
    t, passes, fails = frozen["threshold"], frozen["passes"], frozen["fails"]
    judged = [x["monitor"]["score"] >= t for x in traces if not x["declined"]]

    def est(p, q, j):
        return corrected_rate(sum(j) / len(j), sum(s >= t for s in p) / len(p), sum(s < t for s in q) / len(q))

    in_scope = [x for x in traces if x.get("tuple", {}).get("scope") != "out_of_scope"]
    decline = sum(x["declined"] for x in in_scope) / len(in_scope)
    return est(passes, fails, judged), bootstrap(est, [passes, fails, judged]), decline


def failed_checks(faith, decline):
    out = []
    if not faith >= MIN_FAITH:  # nan (a chance-level judge) fails too
        out.append(f"corrected faithfulness rate {faith:.3f} is below {MIN_FAITH}")
    if decline > MAX_DECLINE:
        out.append(f"in-scope decline rate {decline:.3f} is above {MAX_DECLINE}")
    return out


def report(run):
    """Print the run's numbers and flagged answers; return the failed checks."""
    frozen, traces = json.loads(FROZEN.read_text()), traces_of(run)
    if not traces or any("monitor" not in t for t in traces):
        sys.exit(f"run '{run}' has no traces or unscanned ones: run `monitor scan --run {run}` first")
    if models := {t["monitor"]["judge_model"] for t in traces} - {frozen["judge_model"]}:
        sys.exit(f"traces were judged by {models}, but the frozen numbers are for {frozen['judge_model']}")
    if all(t["declined"] for t in traces):
        sys.exit(f"run '{run}' declined every question, so there is nothing to judge")
    faith, (lo, hi), decline = summary(traces, frozen)
    flagged = sorted((t for t in traces if not t["declined"] and t["monitor"]["score"] < frozen["threshold"]),
                     key=lambda t: t["monitor"]["score"])
    print(f"{run}: {len(traces)} traces, {sum(t['declined'] for t in traces)} declined, {len(flagged)} flagged by the judge")
    print(f"corrected faithfulness rate {faith:.3f} [{lo:.3f}, {hi:.3f}]   in-scope decline rate {decline:.3f}")
    for t in flagged:
        print(f"  {t.get('qid')} {t['monitor']['score']:.2f}  {t['monitor']['weakest'][:150]}")
    return failed_checks(faith, decline)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["ask", "scan", "report", "gate"])
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    if args.step == "ask":
        run_all(args.run, synthetic_items())
    elif args.step == "scan":
        scan(args.run)
    else:
        failed = report(args.run)
        if args.step == "gate":
            print("\n".join(f"FAIL {f}" for f in failed) or f"PASS faithfulness >= {MIN_FAITH}, declines <= {MAX_DECLINE}")
            sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
