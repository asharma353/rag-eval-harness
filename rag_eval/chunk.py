"""Split posts into fixed-size word windows that overlap.

Chunk size and overlap are hyperparameters: slice 3 measures Recall@k across a grid of them.
Each chunk remembers its post title and the nearest heading above its first word, and that
context goes in front of the chunk text when it is indexed (content-aware chunking).
"""
import json
from functools import lru_cache

from .corpus import DATA


def doc_words(doc):
    """The post's words without the heading lines, plus the heading and section number of each word."""
    words, heads, secs, heading, sec = [], [], [], "", 0
    for line in doc["text"].split("\n"):
        if line.startswith("## "):
            heading, sec = line[3:], sec + 1
            continue
        for w in line.split():
            words.append(w)
            heads.append(heading)
            secs.append(sec)
    return words, heads, secs


def chunk_doc(doc, size=200, overlap=50, by_section=False):
    """Word windows over one post. With by_section, no window crosses a heading.

    Each chunk keeps its word span [start, end) in the post, so an evidence span can be
    matched to chunks of any size.
    """
    assert 0 <= overlap < size, "overlap must be smaller than size"
    words, heads, secs = doc_words(doc)
    cuts = [i for i in range(1, len(words)) if by_section and secs[i] != secs[i - 1]]
    spans = list(zip([0] + cuts, cuts + [len(words)])) if words else []
    step = size - overlap
    windows = []
    for a, b in spans:
        # The last window would hold only overlap words already in the one before it, so stop early.
        windows += [(s, min(s + size, b)) for s in range(a, max(b - overlap, a + 1), step)]
    return [
        {
            "id": f"{doc['slug']}#{i:03d}",
            "url": doc["url"],
            "title": doc["title"],
            "heading": heads[s],
            "start": s,
            "end": e,
            "text": " ".join(words[s:e]),
        }
        for i, (s, e) in enumerate(windows)
    ]


def index_text(chunk):
    """What BM25 sees: title and heading in front of the text."""
    return f"{chunk['title']} | {chunk['heading']}\n{chunk['text']}"


def load_docs():
    with open(DATA / "corpus.jsonl") as f:
        return [json.loads(line) for line in f]


@lru_cache(maxsize=None)
def load_chunks(size=200, overlap=50, by_section=False):
    return tuple(c for d in load_docs() for c in chunk_doc(d, size, overlap, by_section))
