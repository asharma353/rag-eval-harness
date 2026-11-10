"""The system under test: retrieve with BM25, answer with Claude, save a trace.

Run: uv run python -m rag_eval.rag "What is axial coding?" [--k 5]
Each run writes traces/<id>.json with everything needed to review it later.
"""
import argparse
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from rag_eval.bm25 import BM25
from .chunk import index_text, load_chunks
from rag_eval.llm import SONNET, claude

TRACES = Path(__file__).resolve().parent.parent / "traces"
NO_ANSWER = "The blog posts I have do not answer this."

SYSTEM = f"""You answer questions about Hamel Husain's blog posts. Use only the numbered passages in the user message.

Rules:
- Use only facts that the passages state. Do not add facts from your own knowledge, even true ones.
- After each sentence that uses a passage, cite it by number, like [2]. A sentence can cite several: [1][3].
- If the passages do not answer the question, reply exactly: "{NO_ANSWER}"
- Keep the answer short: at most one paragraph or a short list."""

# A realistic, sloppy RAG prompt, used only to produce real faithfulness failures for judge validation
# (slice 4). Without the "only the passages" rule, Claude mixes in what it remembers about the blog.
LOOSE = """You answer questions about Hamel Husain's blog posts. Numbered passages from the blog are in the user message; use them where they help.
After each sentence that uses a passage, cite it by number, like [2]. Keep the answer short: at most one paragraph or a short list."""

# The over-careful mirror of LOOSE: it declines unless the passages answer in full. Slice 4 uses it to
# check whether the decline report notices a prompt that declines too often.
CAUTIOUS = SYSTEM.replace("If the passages do not answer the question", "If the passages do not fully and directly answer the question")
PROMPT_NAMES = {SYSTEM: "strict", LOOSE: "loose", CAUTIOUS: "cautious"}


@lru_cache(maxsize=None)
def _index(size, overlap):
    chunks = load_chunks(size, overlap)
    return chunks, BM25([index_text(c) for c in chunks])


def retrieve(query, k=5, size=200, overlap=50):
    chunks, bm25 = _index(size, overlap)
    return [dict(chunks[i], score=round(s, 3)) for i, s in bm25.top(query, k)]


def build_prompt(query, passages):
    blocks = "\n\n".join(f"[{n}] ({p['title']} > {p['heading'] or 'Introduction'})\n{p['text']}" for n, p in enumerate(passages, 1))
    return f"<passages>\n{blocks}\n</passages>\n\nQuestion: {query}"


def parse_citations(answer, n_passages):
    """(cited passage numbers in first-use order, numbers that point at no passage)."""
    nums = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]
    seen = list(dict.fromkeys(nums))
    return [x for x in seen if 1 <= x <= n_passages], [x for x in seen if not 1 <= x <= n_passages]


def answer(query, k=5, size=200, overlap=50, model=SONNET, effort="low", meta=None, system=SYSTEM):
    """meta: extra fields for the trace, such as qid, run name, and the question's tuple."""
    passages = retrieve(query, k, size, overlap)
    user = build_prompt(query, passages)
    out = claude(system, user, model=model, effort=effort)
    cited, bad = parse_citations(out["text"], len(passages))
    trace = {
        "id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(meta or {}),
        "query": query,
        "config": {"k": k, "chunk_size": size, "overlap": overlap, "retriever": "bm25", "model": model, "effort": effort,
                   "prompt": PROMPT_NAMES.get(system, "custom")},
        "retrieved": [{"rank": r, **p} for r, p in enumerate(passages, 1)],
        "prompt": {"system": system, "user": user},
        "answer": out["text"],
        "declined": out["text"].strip() == NO_ANSWER,
        "cited_chunk_ids": [passages[n - 1]["id"] for n in cited],
        "bad_citations": bad,
        "model": out["model"],
        "stop_reason": out["stop_reason"],
        "usage": out["usage"],
        "latency_s": out["latency_s"],
        "cost_usd": out["cost_usd"],
    }
    TRACES.mkdir(exist_ok=True)
    (TRACES / f"{trace['id']}.json").write_text(json.dumps(trace, indent=2))
    return trace


def run_all(run_name, items, system=SYSTEM, workers=4):
    """Answer each item ({qid, query, ...meta}) once per run; qids that already have a trace in this run are skipped."""
    done = {t.get("qid") for t in map(lambda p: json.loads(p.read_text()), TRACES.glob("*.json")) if t.get("run") == run_name}
    todo = [it for it in items if it["qid"] not in done]
    with ThreadPoolExecutor(workers) as pool:
        traces = list(pool.map(lambda it: answer(it["query"], meta={**{k: v for k, v in it.items() if k != "query"}, "run": run_name}, system=system), todo))
    cost = sum(t["cost_usd"] for t in traces)
    declined = sum(t["declined"] for t in traces)
    slowest = max((t["latency_s"] for t in traces), default=0)
    print(f"run '{run_name}': {len(traces)} new traces ({len(done)} already done), {declined} declined, slowest {slowest}s, cost ${cost:.4f}")
    return traces


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=50)
    args = ap.parse_args()
    t = answer(args.query, args.k, args.size, args.overlap)
    for p in t["retrieved"]:
        print(f"  {p['rank']}. {p['score']:>6}  {p['id']:<26} {p['heading'][:60]}")
    print(f"\n{t['answer']}\n")
    print(f"cited: {t['cited_chunk_ids']}  bad citations: {t['bad_citations']}")
    print(f"model {t['model']}  {t['latency_s']}s  ${t['cost_usd']:.4f}  trace: traces/{t['id']}.json")


if __name__ == "__main__":
    main()
