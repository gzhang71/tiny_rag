"""Maximal Marginal Relevance — diversity-aware selection after fusion.

Similarity ranking alone tends to fill the top-k with near-paraphrases of the
same passage (overlapping chunks make this worse). MMR greedily picks the
candidate maximising

    lambda * relevance(query, c)  -  (1 - lambda) * max_similarity(c, selected)

so each pick must add relevant information the already-selected chunks don't
carry. lambda=1 is pure relevance; lower values trade relevance for coverage.
"""
import numpy as np

from rag.ingest.embedder import Embedder
from rag.reranking.base import RerankStage
from rag.store.document import Chunk


class MMRReranker(RerankStage):
    """MMR rerank stage: embeds the query and candidates, then hands off to
    `mmr_select`. Selects the final `top_k`, so a later stage (e.g. the
    cross-encoder) only re-orders the survivors."""

    def __init__(self, embedder: Embedder, lambda_: float = 0.7):
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError("lambda_ must be in [0, 1]")
        self.embedder = embedder
        self.lambda_ = lambda_

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        chunks = [chunk for chunk, _ in candidates]
        return mmr_select(
            self.embedder.embed_one(query),
            self.embedder.embed([c.text for c in chunks]),
            chunks,
            top_k=top_k,
            lambda_=self.lambda_,
        )


def mmr_select(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidates: list[Chunk],
    top_k: int,
    lambda_: float = 0.7,
) -> list[tuple[Chunk, float]]:
    """Embeddings must be L2-normalised (dot product == cosine similarity).
    Returned scores are the MMR objective at selection time, so they decrease
    monotonically and reflect both relevance and novelty."""
    if not candidates:
        return []
    relevance = candidate_embeddings @ query_embedding
    selected: list[tuple[int, float]] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
            score = float(relevance[best])
        else:
            chosen = candidate_embeddings[[i for i, _ in selected]]
            redundancy = candidate_embeddings @ chosen.T  # sim of every candidate to each pick
            def mmr(i: int) -> float:
                return lambda_ * relevance[i] - (1.0 - lambda_) * float(redundancy[i].max())
            best = max(remaining, key=mmr)
            score = mmr(best)
        selected.append((best, score))
        remaining.remove(best)

    return [(candidates[i], score) for i, score in selected]
