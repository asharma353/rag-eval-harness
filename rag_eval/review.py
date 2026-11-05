"""Review page for error analysis: read each trace, mark pass or fail, write a note.

Run: uv run python -m rag_eval.review [--run baseline | --set faithfulness] [--port 8765]
Then open http://127.0.0.1:8765

--run  reviews one run, for open coding. Labels go to evalset/labels.jsonl.
--set  labels a blind pool for one judge (slice 4): the page shows the judge's rubric and gets no run
       name, prompt, tuple, or cost. Labels go to evalset/labels_<set>.jsonl.
Each label file is an append-only log; the last line for a trace wins.
Nothing is ever overwritten, so a mistake in the page cannot erase an earlier label.
"""
import argparse
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .judge import RUBRICS, pool
from .queries import EVALSET
from .rag import TRACES

LABELS = EVALSET / "labels.jsonl"
PAGE = Path(__file__).with_name("review.html")
VERDICTS = {"pass", "fail", None}
BLIND_KEYS = ("id", "query", "answer", "declined", "retrieved", "cited_chunk_ids", "bad_citations")


def load_traces(run):
    traces = [json.loads(p.read_text()) for p in TRACES.glob("*.json")]
    traces = [t for t in traces if t.get("run") == run]
    return sorted(traces, key=lambda t: (t.get("qid", ""), t["created_at"]))


def load_labels(path=LABELS):
    """Latest label per trace id."""
    latest = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                latest[rec["trace_id"]] = rec
    return latest


def append_label(rec, path=LABELS):
    path.parent.mkdir(exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def make_handler(load, labels_path, title, rubric=None, blind=False):
    """load() returns the traces to review; it runs on every request, so new traces show up on reload."""
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/traces":
                traces = load()
                if blind:
                    traces = [{k: t[k] for k in BLIND_KEYS} for t in traces]
                self._send(200, {"run": title, "rubric": rubric, "traces": traces, "labels": load_labels(labels_path)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/label":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                traces = {t["id"]: t for t in load()}
                trace = traces[body["trace_id"]]
                if body.get("verdict") not in VERDICTS:
                    raise ValueError("verdict must be pass, fail, or null")
                notes = str(body.get("notes", ""))
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": f"bad label: {e}"})
            rec = {
                "trace_id": trace["id"],
                "qid": trace.get("qid"),
                "run": trace.get("run"),
                "set": title if blind else None,
                "verdict": body.get("verdict"),
                "notes": notes,
                "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            append_label(rec, labels_path)
            self._send(200, rec)

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="baseline")
    ap.add_argument("--set", choices=sorted(RUBRICS), help="label a blind pool for one judge instead of a run")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.set:
        handler = make_handler(lambda: pool(args.set), EVALSET / f"labels_{args.set}.jsonl", args.set, RUBRICS[args.set], blind=True)
        what = f"label set '{args.set}' ({len(pool(args.set))} traces, blind)"
    else:
        handler = make_handler(lambda: load_traces(args.run), LABELS, args.run)
        what = f"run '{args.run}' ({len(load_traces(args.run))} traces)"
    # Bound to localhost only: the page can write labels, so nothing outside this machine may reach it.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Reviewing {what} at http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
