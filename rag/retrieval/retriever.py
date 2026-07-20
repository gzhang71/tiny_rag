from enum import Enum

from rag.ingest.embedder import Embedder
from rag.query_processing import QueryExpander, QueryStage
from rag.reranking import MMRReranker, RerankStage
from rag.retrieval.base import RetrieveTunnel
from rag.retrieval.bm25 import BM25Tunnel
from rag.retrieval.dense import DenseTunnel
from rag.retrieval.lexical import LexicalTunnel
from rag.retrieval.ner import EntityTunnel
from rag.store.base import BaseVectorStore
from rag.store.document import Chunk


class Channel(str, Enum):
    DENSE = "dense"      # embedding similarity (semantic)
    BM25 = "bm25"        # keyword relevance (sparse lexical)
    LEXICAL = "lexical"  # exact phrase/span matching
    ENTITY = "entity"    # named-entity overlap (NER)


# every tunnel except DENSE is a corpus index built from the store's chunks
_CORPUS_TUNNELS = {
    Channel.BM25: BM25Tunnel,
    Channel.LEXICAL: LexicalTunnel,
    Channel.ENTITY: EntityTunnel,
}

DEFAULT_CHANNELS = (Channel.DENSE, Channel.BM25, Channel.LEXICAL, Channel.ENTITY)


def _rrf_fuse(
    result_lists: list[list[tuple[Chunk, float]]], k: int = 60
) -> list[tuple[Chunk, float]]:
    """Reciprocal Rank Fusion: score(c) = Σ 1/(k + rank). Rank-based, so the
    tunnels' incomparable score scales (cosine, BM25, span length, entity IDF)
    don't need calibrating. Tunnels that return nothing contribute nothing."""
    scores: dict[tuple[str, int], float] = {}
    chunks: dict[tuple[str, int], Chunk] = {}
    for results in result_lists:
        for rank, (chunk, _) in enumerate(results):
            key = (chunk.source, chunk.chunk_index)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            chunks.setdefault(key, chunk)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(chunks[key], score) for key, score in ranked]


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: BaseVectorStore,
        channels: tuple[Channel, ...] = DEFAULT_CHANNELS,
        rrf_k: int = 60,
        mmr_lambda: float | None = None,  # None = off; 0..1 trades relevance for diversity
        reranker: RerankStage | None = None,
        query_stages: tuple[QueryStage, ...] = (),  # applied to the query before the tunnels
        query_expanders: tuple[QueryExpander, ...] = (),  # fan the query out into several, RRF-fused
        candidate_multiplier: int = 4,  # pool size per tunnel = top_k * this, before fusion/rerank
    ):
        if not channels:
            raise ValueError("at least one retrieval channel is required")
        self.embedder = embedder
        self.store = store
        self.channels = tuple(channels)
        self.rrf_k = rrf_k
        self.query_stages = tuple(query_stages)
        self.query_expanders = tuple(query_expanders)
        self.candidate_multiplier = candidate_multiplier
        # rerank stages run in order over the fused pool; MMR first (selects
        # the final top_k), then any supplied stage re-orders the survivors
        self.stages: list[RerankStage] = []
        if mmr_lambda is not None:
            self.stages.append(MMRReranker(embedder, lambda_=mmr_lambda))
        if reranker is not None:
            self.stages.append(reranker)
        # corpus tunnels are rebuilt lazily whenever the store contents change,
        # so they work with persisted stores that already hold chunks at startup
        self._tunnels: dict[Channel, RetrieveTunnel] = {
            Channel.DENSE: DenseTunnel(embedder, store)
        }

    def _tunnel(self, channel: Channel) -> RetrieveTunnel:
        tunnel = self._tunnels.get(channel)
        if channel != Channel.DENSE and (tunnel is None or len(tunnel) != len(self.store)):
            tunnel = _CORPUS_TUNNELS[channel](self.store.chunks())
            self._tunnels[channel] = tunnel
        assert tunnel is not None
        return tunnel

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        for stage in self.query_stages:
            query = stage.process(query)

        # expanders fan the query out into several (sub-questions, HyDE
        # passages); each is run through every tunnel and all lists are fused
        queries = [query]
        for expander in self.query_expanders:
            seen = {q.lower() for q in queries}
            for expanded in (e for q in queries for e in expander.expand(q)):
                if expanded.lower() not in seen:
                    seen.add(expanded.lower())
                    queries.append(expanded)

        # over-fetch when a later stage (fusion or rerank) will re-order and cut
        pool = top_k * self.candidate_multiplier if (
            len(self.channels) > 1 or len(queries) > 1 or self.stages
        ) else top_k

        result_lists = [
            self._tunnel(ch).search(q, top_k=pool) for q in queries for ch in self.channels
        ]
        if len(result_lists) == 1:
            candidates = result_lists[0]
        else:
            candidates = _rrf_fuse(result_lists, k=self.rrf_k)

        # rerank stages see the user's (processed) query, not the expansions
        for stage in self.stages:
            candidates = stage.rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]
