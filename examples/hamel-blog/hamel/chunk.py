"""Split posts into fixed-size word windows that overlap.

Chunk size and overlap are hyperparameters: slice 3 measures Recall@k across a grid of them.
Each chunk remembers its post title and the nearest heading above its first word, and that
context goes in front of the chunk text when it is indexed (content-aware chunking).
"""
import json
from functools import lru_cache

from rag_eval.chunk import chunk_doc as _chunk_doc
from rag_eval.chunk import doc_words, index_text  # noqa: F401  (used by the other example modules)

from .corpus import DATA


def chunk_doc(doc, size=200, overlap=50, by_section=False):
    """The general chunker, on a post that has `slug` and `url` in place of `id` and `source`."""
    return [dict(c, url=c["source"]) for c in _chunk_doc({**doc, "id": doc["slug"], "source": doc["url"]}, size, overlap, by_section)]


def load_docs():
    with open(DATA / "corpus.jsonl") as f:
        return [json.loads(line) for line in f]


@lru_cache(maxsize=None)
def load_chunks(size=200, overlap=50, by_section=False):
    return tuple(c for d in load_docs() for c in chunk_doc(d, size, overlap, by_section))
