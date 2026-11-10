"""The reference RAG as a "user's RAG": its traces in the trace contract, for the workspace in workspace/.

answer --run R   Answer the 109 synthetic questions with the reference RAG and write workspace/traces/R.jsonl.
                 This is the glue a user writes for their own RAG (AGENTS.md): read questions, write traces.
calibrate        Write workspace/calibrations/faithfulness.json from the frozen test scores in
                 evalset/faith_judge.json. The example keeps its frozen 2026-11-03 split: a split by trace hash
                 would move answers that chose the judge design (dev) into test.
parity           Score the frozen test split again through the general judge engine (Jev, about $0.01) and
                 compare TPR and TNR with the frozen numbers. The general engine sends each passage's source
                 as "title > heading" in place of separate fields, so this checks that the calibration still holds.

cd examples/hamel-blog
uv run python -m hamel.contract calibrate
uv run python -m hamel.contract answer --run ci
cd workspace && uv run rag-eval scan --run ci && uv run rag-eval gate --run ci
"""
import argparse
import json
from pathlib import Path

from rag_eval import judge as J
from rag_eval.workspace import Workspace

from .judge import FROZEN, pool, split_labels, synthetic_items
from .rag import run_all

WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"


def to_trace(t):
    """One reference-RAG trace in the contract format."""
    return {"id": t["id"], "qid": t.get("qid"), "question": t["query"], "answer": t["answer"], "declined": t["declined"],
            "passages": [{"text": r["text"], "source": f"{r['title']} > {r['heading'] or 'Introduction'}"} for r in t["retrieved"]],
            "meta": {"run": t.get("run"), "prompt": t["config"]["prompt"], "model": t["model"], "cost_usd": t["cost_usd"]}}


def answer(run):
    traces = run_all(run, synthetic_items())  # new traces only; a rerun answers what is missing
    done = WORKSPACE / "traces" / f"{run}.jsonl"
    done.parent.mkdir(exist_ok=True)
    with open(done, "a") as f:
        f.writelines(json.dumps(to_trace(t)) + "\n" for t in traces)
    print(f"{len(traces)} traces -> {done.relative_to(WORKSPACE.parent)}")


def calibrate():
    ws = Workspace(WORKSPACE)
    judge = J.get_judge(ws, "faithfulness")
    frozen = json.loads(FROZEN.read_text())
    assert frozen["judge_model"] == J.model_of(judge), "the frozen scores are for another model"
    m = J.measure(frozen["passes"], frozen["fails"], frozen["threshold"])
    ws.save_calibration("faithfulness", {
        "judge": "faithfulness", "hash": J.judge_hash(judge, ws.settings), "backend": judge["backend"], "model": J.model_of(judge),
        "threshold": frozen["threshold"], "tpr": m["tpr"], "tnr": m["tnr"], "tpr_ci": m["tpr_ci"], "tnr_ci": m["tnr_ci"],
        "n_pass": len(frozen["passes"]), "n_fail": len(frozen["fails"]), "passes": frozen["passes"], "fails": frozen["fails"],
        "tested_at": "2026-11-03", "source": "evalset/faith_judge.json: the frozen test split of evalset/labels_faithfulness.jsonl"})
    print(f"calibration written: TPR {m['tpr']:.2f}, TNR {m['tnr']:.2f}, threshold {frozen['threshold']}")


def parity():
    from .review import load_labels
    ws = Workspace(WORKSPACE)
    judge = J.get_judge(ws, "faithfulness")
    traces = {t["id"]: t for t in pool("faithfulness")}
    labels = {k: v["verdict"] for k, v in load_labels(FROZEN.parent / "labels_faithfulness.jsonl").items() if k in traces and v["verdict"]}
    ids = [i for i, part in split_labels(labels).items() if part == "test"]
    scores, cost = J.score_traces(judge, [to_trace(traces[i]) for i in ids], ws.settings)
    frozen = json.loads(FROZEN.read_text())
    passes = [scores[i][0] for i in ids if labels[i] == "pass"]
    fails = [scores[i][0] for i in ids if labels[i] == "fail"]
    now, then = J.measure(passes, fails, frozen["threshold"]), J.measure(frozen["passes"], frozen["fails"], frozen["threshold"])
    print(f"test split: {len(passes)} passes, {len(fails)} fails, Jev cost ${cost:.4f}")
    print(f"frozen engine:  TPR {then['tpr']:.2f}  TNR {then['tnr']:.2f}")
    print(f"general engine: TPR {now['tpr']:.2f}  TNR {now['tnr']:.2f}")
    # Same labels, same split function, same order as the frozen run, so the lists line up answer by answer.
    t = frozen["threshold"]
    flips = sum((a >= t) != (b >= t) for a, b in zip(passes + fails, frozen["passes"] + frozen["fails"]))
    print(f"answers whose verdict changed: {flips} of {len(ids)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["answer", "calibrate", "parity"])
    ap.add_argument("--run")
    args = ap.parse_args()
    if args.step == "answer":
        if not args.run:
            ap.error("answer needs --run")
        answer(args.run)
    else:
        {"calibrate": calibrate, "parity": parity}[args.step]()


if __name__ == "__main__":
    main()
