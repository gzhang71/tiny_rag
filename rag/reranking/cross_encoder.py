from sentence_transformers import CrossEncoder

from rag.reranking.base import RerankStage
from rag.store.document import Chunk

_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(RerankStage):
    """Cross-encoder rerank stage: scores each (query, chunk) pair jointly,
    which is far more accurate than embedding similarity but too slow to run
    over the whole corpus — so it only re-orders the retrieved candidate pool.
    Runs locally, no API key needed.
    """

    def __init__(self, model: str = _MODEL):
        self.model = CrossEncoder(model)

    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], top_k: int = 5
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        scores = self.model.predict([(query, chunk.text) for chunk, _ in candidates])
        reranked = sorted(
            ((chunk, float(score)) for (chunk, _), score in zip(candidates, scores)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return reranked[:top_k]
