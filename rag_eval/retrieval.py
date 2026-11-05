"""Retrieval eval: reverse-method question pairs, scored over a grid of chunk settings.

gen   Sample chunks. For each one, Claude copies one fact (the evidence) and writes a question
      that the fact answers. Two styles alternate:
      - plain: any wording.
      - adversarial: written in the words of the two most similar other chunks, so keyword
        search is pulled toward the wrong chunks.
      Plus the real baseline queries whose answer retrieval missed (FINDINGS.md).
grid  Re-chunk and re-index the corpus for each setting, and score every pair.
rewrite  Compare BM25 on the user's question with BM25 on a Claude query rewrite or a HyDE passage.

The evidence is stored as a word span in its post (slug, start, end), not as text and not as a
chunk id. The span is valid under any chunking, and no blog text goes into the repo.
A chunk is relevant when it holds at least half of the evidence words.

uv run python -m rag_eval.retrieval gen   -> evalset/retrieval_pairs.jsonl
uv run python -m rag_eval.retrieval grid
uv run python -m rag_eval.retrieval rewrite   (cache: data/query_variants.jsonl)
"""
import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from statistics import mean

from .bm25 import BM25, tokenize
from .chunk import doc_words, index_text, load_chunks, load_docs
from .corpus import DATA
from .llm import claude
from .metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from .queries import EVALSET, QUERIES, post_outline

PAIRS = EVALSET / "retrieval_pairs.jsonl"
VARIANTS = DATA / "query_variants.jsonl"
N_CHUNKS = 120  # half plain, half adversarial; the leak gate drops some adversarial ones
GRID = [(size, overlap, by_section) for size in (100, 200, 400) for overlap in (0, 50) for by_section in (False, True)]
# Baseline retrieval misses: (qid, post, a short phrase that starts the evidence).
REAL = [
    ("q011", "evals-faq", "60-80% of our development time"),
    ("q039", "evals-faq", "A trace is the complete record of all actions"),
    ("q041", "llm-judge", "You cannot write a good judge prompt until"),
]

SCHEMA = {
    "type": "object",
    "properties": {"evidence": {"type": "string"}, "question": {"type": "string"}},
    "required": ["evidence", "question"],
    "additionalProperties": False,
}
RULES = """Rules:
- The fact must be something a reader of the blog wants to know: an idea, advice, a reason, or a result. Not a detail inside an example's data or code, and not an acknowledgment.
- evidence: copy, word for word, the one sentence or clause of the target passage that states the fact. 8 to 40 words.
- question: one question that the evidence answers. Ask about one thing only. It must make sense to someone who has not read the passage, so do not say "the passage", "the post" or "the author". Write it the way a real user types a question."""
MAX_LEAK = 0.3  # adversarial questions may share at most this share of their words with the evidence


def slug_of(chunk):
    return chunk["id"].rsplit("#", 1)[0]


def _norm(word):
    return re.sub(r"[^a-z0-9]", "", word.lower())


def locate(words, quote, lo=0, hi=None):
    """Word span [start, end) of quote in words[lo:hi], ignoring case and punctuation. None if absent."""
    pos = [i for i in range(lo, len(words) if hi is None else hi) if _norm(words[i])]
    toks = [_norm(words[i]) for i in pos]
    q = [t for t in map(_norm, quote.split()) if t]
    for j in range(len(toks) - len(q) + 1 if q else 0):
        if toks[j : j + len(q)] == q:
            return pos[j], pos[j + len(q) - 1] + 1
    return None


def fingerprint(words, start, end):
    """Detects a changed post: if Hamel edits it, the stored span no longer points at the evidence."""
    return hashlib.sha1(" ".join(map(_norm, words[start:end])).encode()).hexdigest()[:12]


def leak(question, evidence):
    """Share of the question's search terms that also appear in the evidence."""
    q = set(tokenize(question))
    return len(q & set(tokenize(evidence))) / len(q) if q else 0.0


def relevant(chunk, pair):
    held = min(chunk["end"], pair["end"]) - max(chunk["start"], pair["start"])
    return slug_of(chunk) == pair["slug"] and 2 * held >= pair["end"] - pair["start"]


def sample_chunks(chunks, n, seed=0):
    """Round-robin over posts, so every post gets questions and not only the long ones."""
    rng = random.Random(seed)
    by_post = {}
    for c in chunks:
        if c["end"] - c["start"] >= 100:  # skip short tail chunks
            by_post.setdefault(slug_of(c), []).append(c)
    for cs in by_post.values():
        rng.shuffle(cs)
    picked = []
    while len(picked) < n and any(by_post.values()):
        for cs in by_post.values():
            if cs and len(picked) < n:
                picked.append(cs.pop())
    return picked


def confusers(chunk, chunks, bm25, n=2):
    """The n chunks most similar to this one, other than the chunks that overlap it."""
    out = []
    for i, _ in bm25.top(chunk["text"], 10):
        c = chunks[i]
        if not (slug_of(c) == slug_of(chunk) and c["start"] < chunk["end"] and chunk["start"] < c["end"]):
            out.append(c)
    return out[:n]


def write_pair(target, others):
    system = "You write test questions for a search engine over Hamel Husain's blog."
    user = f"<target>\n{target['text']}\n</target>\n\n"
    if others:
        user += "\n\n".join(f"<other>\n{c['text']}\n</other>" for c in others)
        user += ("\n\nPick one fact that the target passage states and the other passages do not. Write the question the way "
                 "a user who has never read the target would ask it: use everyday words or words from the other passages, "
                 "and do not reuse the target's distinctive words. A keyword search must find it hard. Only the target's "
                 "fact may answer it.")
    else:
        user += "Pick one specific fact that the target passage states."
    out = claude(system, user + "\n\n" + RULES, schema=SCHEMA)
    return json.loads(out["text"]), out["cost_usd"]


def generate():
    words = {d["slug"]: doc_words(d)[0] for d in load_docs()}
    chunks = load_chunks()
    bm25 = BM25([index_text(c) for c in chunks])
    jobs = [(c, confusers(c, chunks, bm25) if i % 2 else []) for i, c in enumerate(sample_chunks(chunks, N_CHUNKS))]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda job: write_pair(*job), jobs))

    pairs, dropped, leaky, cost = [], [], [], 0.0
    for (c, others), (out, usd) in zip(jobs, results):
        cost += usd
        span = locate(words[slug_of(c)], out["evidence"], c["start"], c["end"])
        # Drop evidence that Claude did not copy exactly, and evidence already used: overlapping chunks share sentences.
        if not span or any(p["slug"] == slug_of(c) and p["start"] < span[1] and span[0] < p["end"] for p in pairs):
            dropped.append(c["id"])
            continue
        q_leak = leak(out["question"], " ".join(words[slug_of(c)][span[0] : span[1]]))
        if others and q_leak > MAX_LEAK:  # the adversarial question copied the evidence's words
            leaky.append(c["id"])
            continue
        pairs.append({"style": "adversarial" if others else "plain", "question": out["question"], "slug": slug_of(c),
                      "start": span[0], "end": span[1], "leak": round(q_leak, 2), "source_chunk": c["id"],
                      "confusers": [o["id"] for o in others]})
    queries = {q["qid"]: q["query"] for q in map(json.loads, open(QUERIES))}
    for qid, slug, phrase in REAL:
        span = locate(words[slug], phrase)
        assert span, f"{qid}: evidence phrase not found in {slug}"
        pairs.append({"style": "real", "qid": qid, "question": queries[qid], "slug": slug, "start": span[0], "end": span[1]})

    with open(PAIRS, "w") as f:
        for i, p in enumerate(pairs, 1):
            p["fp"] = fingerprint(words[p["slug"]], p["start"], p["end"])
            f.write(json.dumps({"pid": f"r{i:03d}", **p}) + "\n")
    print(f"{len(pairs)} pairs -> {PAIRS}   cost ${cost:.4f}\n  dropped (evidence not exact or repeated): {len(dropped)} {dropped}"
          f"\n  dropped (adversarial question leaked over {MAX_LEAK:.0%}): {len(leaky)} {leaky}")


def score(pairs, size, overlap, by_section):
    chunks = load_chunks(size, overlap, by_section)
    bm25 = BM25([index_text(c) for c in chunks])
    budget_k = max(1, round(1000 / size))  # about 1,000 words of context, the same for every chunk size
    rows = []
    for p in pairs:
        s = bm25.score(p["question"])
        rels = [int(relevant(chunks[i], p)) for i in sorted(range(len(chunks)), key=lambda i: -s[i])]
        assert any(rels), f"{p['pid']}: no chunk holds half of the evidence"
        rows.append({"pid": p["pid"], "style": p["style"], "qid": p.get("qid"), "rank": rels.index(1) + 1,
                     "r5": recall_at_k(rels, 5), "r10": recall_at_k(rels, 10), "rb": recall_at_k(rels, budget_k),
                     "p5": precision_at_k(rels, 5), "mrr": reciprocal_rank(rels), "ndcg5": ndcg_at_k(rels, 5)})
    return len(chunks), budget_k, rows


def grid():
    pairs = [json.loads(line) for line in open(PAIRS)]
    words = {d["slug"]: doc_words(d)[0] for d in load_docs()}
    changed = [p["pid"] for p in pairs if fingerprint(words[p["slug"]], p["start"], p["end"]) != p["fp"]]
    assert not changed, f"the corpus changed under these pairs, run gen again: {changed}"

    syn = sum(p["style"] != "real" for p in pairs)
    print(f"{syn} synthetic pairs ({sum(p['style'] == 'plain' for p in pairs)} plain, "
          f"{sum(p['style'] == 'adversarial' for p in pairs)} adversarial); real ranks are the rank of the first relevant chunk")
    for style in ("plain", "adversarial"):
        print(f"  {style}: mean share of question terms copied from the evidence {mean(p['leak'] for p in pairs if p['style'] == style):.2f}")
    print()
    print("| size | overlap | sections | chunks | R@5 | R@10 | R@~1000 words (k) | MRR | P@5 | nDCG@5 | R@5 plain | R@5 adversarial | real: " +
          " / ".join(q for q, _, _ in REAL) + " |")
    print("|---" * 13 + "|")
    for size, overlap, by_section in GRID:
        n, budget_k, rows = score(pairs, size, overlap, by_section)
        syn_rows = [r for r in rows if r["style"] != "real"]

        def avg(key, style=None):
            return f"{mean(r[key] for r in syn_rows if style in (None, r['style'])):.2f}"

        real = " / ".join(str(r["rank"]) for r in rows if r["style"] == "real")
        print(f"| {size} | {overlap} | {'yes' if by_section else 'no'} | {n} | {avg('r5')} | {avg('r10')} | {avg('rb')} ({budget_k}) "
              f"| {avg('mrr')} | {avg('p5')} | {avg('ndcg5')} | {avg('r5', 'plain')} | {avg('r5', 'adversarial')} | {real} |")


def rewrite(question):
    """Query rewrite: put the question in the blog's own terms. Claude sees only titles and headings, no post text."""
    system = "You turn a user's question into a query for a keyword search engine over Hamel Husain's blog."
    user = (f"The blog's posts and their section headings:\n\n{post_outline()}\n\nQuestion: {question}\n\n"
            "Write one search query of 5 to 15 words. Use the terms these posts would use for this idea: "
            "name the technique, tool or concept. Reply with the query only.")
    return claude(system, user)


def hyde(question):
    """HyDE (hypothetical document embeddings, here with BM25): search with an imagined answer, not the question."""
    system = "You write a short passage that answers a question, in the style of a technical blog about LLM evals and engineering."
    user = f"Question: {question}\n\nWrite a 60 to 100 word passage that answers it, as a blog post that answers it would. Reply with the passage only."
    return claude(system, user)


def variants(pairs):
    """Rewrite and HyDE text for each question, cached in data/ (outside git) so a rerun costs nothing."""
    cache = {v["question"]: v for v in map(json.loads, open(VARIANTS))} if VARIANTS.exists() else {}
    todo = [p["question"] for p in pairs if p["question"] not in cache]

    def one(q):
        r, h = rewrite(q), hyde(q)
        return {"question": q, "rewrite": r["text"].strip(), "hyde": h["text"].strip(),
                "cost": {"rewrite": r["cost_usd"], "hyde": h["cost_usd"]}, "latency": {"rewrite": r["latency_s"], "hyde": h["latency_s"]}}

    with ThreadPoolExecutor(8) as pool, open(VARIANTS, "a") as f:
        for v in pool.map(one, todo):
            cache[v["question"]] = v
            f.write(json.dumps(v) + "\n")
    return cache


def compare(size=200, overlap=50):
    """BM25 with the user's question against BM25 with Claude's rewrite or imagined answer, at the current chunk setting."""
    pairs = [json.loads(line) for line in open(PAIRS)]
    v = variants(pairs)
    methods = {
        "question (baseline)": lambda q: q,
        "rewrite": lambda q: v[q]["rewrite"],
        "question + rewrite": lambda q: f"{q} {v[q]['rewrite']}",
        "HyDE": lambda q: v[q]["hyde"],
        "question + HyDE": lambda q: f"{q} {v[q]['hyde']}",
    }
    print(f"chunks {size}/{overlap}; real ranks are the rank of the first relevant chunk\n")
    print("| Query sent to BM25 | R@5 plain | R@5 adversarial | R@10 adversarial | MRR adversarial | real: " +
          " / ".join(q for q, _, _ in REAL) + " |")
    print("|---" * 6 + "|")
    for name, make in methods.items():
        _, _, rows = score([dict(p, question=make(p["question"])) for p in pairs], size, overlap, False)

        def avg(key, style):
            return f"{mean(r[key] for r in rows if r['style'] == style):.2f}"

        real = " / ".join(str(r["rank"]) for r in rows if r["style"] == "real")
        print(f"| {name} | {avg('r5', 'plain')} | {avg('r5', 'adversarial')} | {avg('r10', 'adversarial')} | {avg('mrr', 'adversarial')} | {real} |")
    for m in ("rewrite", "hyde"):
        print(f"\n{m}: mean ${mean(x['cost'][m] for x in v.values()):.4f} and {mean(x['latency'][m] for x in v.values()):.1f}s per query")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["gen", "grid", "rewrite"])
    {"gen": generate, "grid": grid, "rewrite": compare}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
