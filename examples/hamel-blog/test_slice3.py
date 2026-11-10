"""Offline checks for slice 3. Run: uv run python test_slice3.py"""
import math

from hamel.chunk import chunk_doc, doc_words
from rag_eval.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from hamel.retrieval import leak, locate, relevant

# metrics on a ranking whose relevant chunks are at ranks 2 and 4
rels = [0, 1, 0, 1, 0, 0]
assert recall_at_k(rels, 1) == 0 and recall_at_k(rels, 2) == 1
assert precision_at_k(rels, 5) == 0.4
assert reciprocal_rank(rels) == 0.5
best = 1 + 1 / math.log2(3)
assert math.isclose(ndcg_at_k(rels, 5), (1 / math.log2(3) + 1 / math.log2(5)) / best)
assert ndcg_at_k([1, 1, 0], 5) == 1 and ndcg_at_k([0, 0, 0], 5) == 0 and reciprocal_rank([0, 0]) == 0

# locate ignores case, punctuation and curly quotes, and respects the search window
words = "Intro. We’ve spent 60–80% of our time on Error Analysis, mostly.".split()
assert locate(words, "we've spent 60-80% of our time") == (1, 7)
assert locate(words, "error analysis mostly") == (8, 11)
assert locate(words, "error analysis", lo=0, hi=9) is None
assert locate(words, "not here") is None and locate(words, "...") is None

# leak: share of the question's search terms (stopwords removed) that the evidence also has
assert leak("What is axial coding?", "Axial coding groups the notes.") == 1.0
assert leak("How do I group my notes?", "Axial coding groups the notes.") == 0.5  # terms: group, notes
assert leak("the of a", "anything") == 0.0

# a chunk is relevant when it holds at least half of the evidence span [10, 20)
pair = {"slug": "p", "start": 10, "end": 20}
assert relevant({"id": "p#000", "start": 0, "end": 15}, pair)
assert not relevant({"id": "p#000", "start": 0, "end": 14}, pair)
assert not relevant({"id": "other#000", "start": 0, "end": 30}, pair)

# by_section: no chunk crosses a heading; start and end match the text
doc = {"slug": "d", "url": "u", "title": "T", "text": "a b c\n## H1\n" + " ".join(f"w{i}" for i in range(25)) + "\n## H2\nx y z"}
words, heads, secs = doc_words(doc)
for by_section in (False, True):
    for c in chunk_doc(doc, size=10, overlap=3, by_section=by_section):
        assert c["text"] == " ".join(words[c["start"] : c["end"]])
        if by_section:
            assert len(set(secs[c["start"] : c["end"]])) == 1, c
            assert c["heading"] == heads[c["end"] - 1]
assert {c["text"] for c in chunk_doc(doc, 10, 3, True)} >= {"a b c", "x y z"}, "short sections become their own chunks"
covered = {i for c in chunk_doc(doc, 10, 3, True) for i in range(c["start"], c["end"])}
assert covered == set(range(len(words))), "no words dropped"

print("slice 3 checks passed")
