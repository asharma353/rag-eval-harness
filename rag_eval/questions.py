"""Test questions with known evidence, made by the reverse method.

Sample passages of the corpus. For each one, an LLM copies one fact (the evidence) and writes a
question that the fact answers. Two styles alternate:
- plain: any wording.
- adversarial: in the words of the two most similar other passages, so keyword search is pulled
  toward the wrong passages. A team that tests only with plain questions sees a retriever that
  looks finished.
Evidence that the LLM did not copy word for word is dropped, so every kept question has evidence
that really is in the corpus.
"""
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from .bm25 import BM25, tokenize
from .chunk import chunk_doc, doc_words, index_text
from .llm import claude

SCHEMA = {
    "type": "object",
    "properties": {"evidence": {"type": "string"}, "question": {"type": "string"}},
    "required": ["evidence", "question"],
    "additionalProperties": False,
}
SYSTEM = "You write test questions for a search engine over a collection of documents."
RULES = """Rules:
- The fact must be something a user of these documents wants to know: an idea, an instruction, a reason, a number, or a result. Not a detail inside an example's data or code, and not an acknowledgment.
- evidence: copy, word for word, the one sentence or clause of the target passage that states the fact. 8 to 40 words.
- question: one question that the evidence answers. Ask about one thing only. It must make sense to someone who has not read the passage, so do not say "the passage", "the document" or "the author". Write it the way a real user types a question."""
ADVERSARIAL = ("Pick one fact that the target passage states and the other passages do not. Write the question the way "
               "a user who has never read the target would ask it: use everyday words or words from the other passages, "
               "and do not reuse the target's distinctive words. A keyword search must find it hard. Only the target's "
               "fact may answer it.")
MAX_LEAK = 0.3  # adversarial questions may share at most this share of their search terms with the evidence


def _norm(word):
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _tokens(text):
    return [t for t in map(_norm, text.split()) if t]


def locate(words, quote, lo=0, hi=None):
    """Word span [start, end) of quote in words[lo:hi], ignoring case and punctuation. None if absent."""
    pos = [i for i in range(lo, len(words) if hi is None else hi) if _norm(words[i])]
    toks = [_norm(words[i]) for i in pos]
    q = _tokens(quote)
    for j in range(len(toks) - len(q) + 1 if q else 0):
        if toks[j : j + len(q)] == q:
            return pos[j], pos[j + len(q) - 1] + 1
    return None


def holds(evidence, passage):
    """A passage holds the evidence when it has at least half of the evidence's words in one unbroken run
    (case and punctuation ignored). Half, because a passage boundary can cut the evidence in two."""
    e, p = _tokens(evidence), _tokens(passage)
    best, prev = 0, [0] * (len(p) + 1)
    for a in e:  # longest common run of words, by dynamic programming
        cur = [0] * (len(p) + 1)
        for j, b in enumerate(p, 1):
            if a == b:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return bool(e) and 2 * best >= len(e)


def leak(question, evidence):
    """Share of the question's search terms that also appear in the evidence."""
    q = set(tokenize(question))
    return len(q & set(tokenize(evidence))) / len(q) if q else 0.0


def doc_of(chunk):
    return chunk["id"].rsplit("#", 1)[0]


def sample_chunks(chunks, n, seed=0, min_words=30):
    """Round-robin over documents, so every document gets questions and not only the long ones."""
    rng = random.Random(seed)
    by_doc = {}
    for c in chunks:
        if c["end"] - c["start"] >= min_words:  # skip short tail chunks
            by_doc.setdefault(doc_of(c), []).append(c)
    for cs in by_doc.values():
        rng.shuffle(cs)
    picked = []
    while len(picked) < n and any(by_doc.values()):
        for cs in by_doc.values():
            if cs and len(picked) < n:
                picked.append(cs.pop())
    return picked


def confusers(chunk, chunks, bm25, n=2):
    """The n chunks most similar to this one, other than the chunks that overlap it."""
    out = []
    for i, _ in bm25.top(chunk["text"], 10):
        c = chunks[i]
        if not (doc_of(c) == doc_of(chunk) and c["start"] < chunk["end"] and chunk["start"] < c["end"]):
            out.append(c)
    return out[:n]


def write_pair(llm, target, others):
    user = f"<target>\n{target['text']}\n</target>\n\n"
    if others:
        user += "\n\n".join(f"<other>\n{c['text']}\n</other>" for c in others) + "\n\n" + ADVERSARIAL
    else:
        user += "Pick one specific fact that the target passage states."
    out = llm(SYSTEM, user + "\n\n" + RULES, schema=SCHEMA)
    return json.loads(out["text"]), out["cost_usd"]


def generate(docs, n, llm=claude, workers=8):
    """About n questions, half plain and half adversarial: [{qid, style, question, doc, source, evidence, leak}]."""
    words = {d["id"]: doc_words(d)[0] for d in docs}
    chunks = [c for d in docs for c in chunk_doc(d)]
    bm25 = BM25([index_text(c) for c in chunks])
    jobs = [(c, confusers(c, chunks, bm25) if i % 2 else []) for i, c in enumerate(sample_chunks(chunks, n))]
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda job: write_pair(llm, *job), jobs))

    out, spans, not_copied, leaky, cost = [], [], 0, 0, 0.0
    for (c, others), (pair, usd) in zip(jobs, results):
        cost += usd
        doc = doc_of(c)
        span = locate(words[doc], pair["evidence"], c["start"], c["end"])
        # Drop evidence that was not copied exactly, and evidence already used: overlapping chunks share sentences.
        if not span or any(d == doc and s < span[1] and span[0] < e for d, s, e in spans):
            not_copied += 1
            continue
        evidence = " ".join(words[doc][span[0] : span[1]])
        q_leak = leak(pair["question"], evidence)
        if others and q_leak > MAX_LEAK:  # the adversarial question copied the evidence's words
            leaky += 1
            continue
        spans.append((doc, *span))
        out.append({"qid": f"g{len(out) + 1:03d}", "style": "adversarial" if others else "plain", "question": pair["question"],
                    "doc": doc, "source": c["source"], "evidence": evidence, "leak": round(q_leak, 2)})
    print(f"{len(out)} questions from {len(jobs)} passages, cost ${cost:.4f}; dropped: {not_copied} evidence not copied "
          f"exactly or repeated, {leaky} adversarial questions that reused the evidence's words")
    return out
