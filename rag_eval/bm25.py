"""BM25 keyword retrieval, from scratch.

score(query, doc) = sum over query terms t of
    idf(t) * tf(t, doc) * (k1 + 1) / (tf(t, doc) + k1 * (1 - b + b * len(doc) / avg_len))

- idf rewards rare terms: log(1 + (N - df + 0.5) / (df + 0.5)).
- k1 caps how much repeating a term helps (term-frequency saturation).
- b sets how much long documents are penalized (length normalization).
"""
import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    "a an and are as at be but by can do does for from has have how i if in into is it its "
    "my of on or so that the their them then there these they this to was we what when where "
    "which who why will with would you your".split()
)


def tokenize(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lens) / len(self.lens)
        n = len(docs)
        df = Counter(t for tf in self.tfs for t in tf)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query):
        terms = tokenize(query)
        scores = []
        for tf, length in zip(self.tfs, self.lens):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            scores.append(sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf))
        return scores

    def top(self, query, k=5):
        """[(doc_index, score)] for the k best documents, best first."""
        scores = self.score(query)
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, scores[i]) for i in best]
