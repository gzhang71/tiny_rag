"""Retrieval ranking metrics — pure functions, no dependencies.

Ids are opaque hashable values; the evaluator matches chunks to relevance
judgements by `source` or `"source:chunk_index"`. `relevant` is a set of ids
(binary relevance) or a mapping id -> gain (graded relevance for NDCG).
"""
import math
from collections.abc import Mapping, Sequence
from typing import Hashable


def recall_at_k(
    retrieved: Sequence[Hashable], relevant: Mapping | set | frozenset, k: int
) -> float:
    """Fraction of the relevant ids that appear in the top-k retrieved."""
    if not relevant:
        return 0.0
    hits = set(retrieved[:k]) & set(relevant)
    return len(hits) / len(relevant)


def ndcg_at_k(
    retrieved: Sequence[Hashable], relevant: Mapping | set | frozenset, k: int
) -> float:
    """Normalised Discounted Cumulative Gain: rewards ranking relevant ids
    early. A set gives every relevant id gain 1; a mapping gives graded gains.
    Duplicate retrieved ids only earn their gain on first appearance."""
    if not relevant:
        return 0.0
    gains = (
        dict(relevant)
        if isinstance(relevant, Mapping)
        else {rid: 1.0 for rid in relevant}
    )
    dcg, seen = 0.0, set()
    for rank, rid in enumerate(retrieved[:k]):
        if rid in gains and rid not in seen:
            seen.add(rid)
            dcg += gains[rid] / math.log2(rank + 2)
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
