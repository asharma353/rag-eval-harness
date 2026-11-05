"""Retrieval metrics, from scratch.

`rels` is one query's ranked list of 0/1 relevance over the whole index: rels[0] is the top chunk.
"""
import math


def recall_at_k(rels, k):
    """1 if a relevant chunk is in the top k, else 0.

    Each question in this harness has one relevant item: its evidence span. Any chunk that holds
    the evidence delivers it, so this is recall over evidence, not over chunks. Counting chunks
    would punish overlap: two overlapping chunks can both hold the same sentence.
    """
    return float(any(rels[:k]))


def precision_at_k(rels, k):
    """Share of the top k chunks that are relevant."""
    return sum(rels[:k]) / k


def reciprocal_rank(rels):
    """1 / rank of the first relevant chunk; 0 if there is none. MRR is the mean of this."""
    return next((1 / rank for rank, rel in enumerate(rels, 1) if rel), 0.0)


def ndcg_at_k(rels, k):
    """DCG of the top k, divided by the DCG of the best possible order. Rank r counts 1 / log2(r + 1)."""

    def dcg(rs):
        return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(rs[:k], 1))

    best = dcg(sorted(rels, reverse=True))
    return dcg(rels) / best if best else 0.0
