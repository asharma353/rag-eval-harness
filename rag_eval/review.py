"""The review page: a person reads each trace and labels it pass or fail, with a note.

review --judge X   labels for one judge. The page shows the judge's own question and criteria, and it is
                   blind: no run name, and a fixed shuffled order. Declined traces are left out, because
                   judges do not score them. Labels go to labels/X.jsonl.
review --run R     open reading of one run, before any judge exists: what goes wrong, in your own words.
                   Labels go to labels/open-coding.jsonl.
Label files are append-only, so a mistake on the page cannot erase an earlier label.
The server listens on 127.0.0.1 only, because the page can write files.
"""
import hashlib
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .judge import citation_pattern, get_judge, judge_hash

PAGE = Path(__file__).with_name("review.html")
OPEN_CODING = "open-coding"


def _key(run, trace_id):
    return hashlib.sha1(f"{run}/{trace_id}".encode()).hexdigest()[:12]


def pool(ws, judge=None, run=None):
    """{key: (run, trace)} for the page, in a fixed order that hides the run."""
    runs = [ws.run_name(run)] if run or not judge else ws.runs()
    out = {_key(r, t["id"]): (r, t) for r in runs for t in ws.traces(r) if not (judge and t["declined"])}
    return dict(sorted(out.items()))  # sorted by a hash: a shuffle that stays the same


def make_handler(ws, judge=None, run=None):
    name = judge["name"] if judge else OPEN_CODING
    jhash = judge_hash(judge, ws.settings) if judge else None
    pattern = citation_pattern(ws.settings)

    def labels():
        return {_key(r, tid): rec for (r, tid), rec in ws.labels(name).items()}

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
                traces = [{"key": k, **{f: t[f] for f in ("question", "passages", "answer", "declined")}} for k, (_, t) in pool(ws, judge, run).items()]
                self._send(200, {"title": name if judge else f"run {run}", "judge": judge, "citation_pattern": pattern,
                                 "traces": traces, "labels": labels()})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/label":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                r, t = pool(ws, judge, run)[body["key"]]
                if body.get("verdict") not in ("pass", "fail", None):
                    raise ValueError("verdict must be pass, fail, or null")
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": f"bad label: {e}"})
            rec = {"run": r, "trace_id": t["id"], "verdict": body.get("verdict"), "notes": str(body.get("notes", "")),
                   "judge_hash": jhash, "labeled_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            ws.add_label(name, rec)
            self._send(200, rec)

        def log_message(self, *args):
            pass

    return Handler


def serve(ws, judge_name=None, run=None, port=8765):
    if not judge_name and not run and len(ws.runs()) != 1:
        raise SystemExit("pass --judge X to label for a judge, or --run R to read one run")
    judge = get_judge(ws, judge_name) if judge_name else None
    run = run or (None if judge else ws.run_name())
    n = len(pool(ws, judge, run))
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(ws, judge, run))
    what = f"labels for judge {judge_name}" if judge else f"open reading of run {run}"
    print(f"{what}: {n} traces at http://127.0.0.1:{port}  (stop with Ctrl-C)")
    server.serve_forever()
