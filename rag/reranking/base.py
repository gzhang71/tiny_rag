from abc import ABC, abstractmethod

from rag.store.document import Chunk


class RerankStage(ABC):
    """One post-fusion rerank stage in the retrieval pipeline.

    Every stage maps a ranked candidate pool to a re-ordered (and usually
    truncated) list of (chunk, score) pairs. Stages are applied in sequence,
    each receiving the previous stage's output, so a stage that cuts to
    `top_k` leaves later stages only re-ordering the survivors. Score
    semantics are stage-specific (MMR objective, cross-encoder logit, ...);
    callers should treat them as opaque ranking keys.
    """

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[tuple[Chunk, float]], top_k: int = 5
    ) -> list[tuple[Chunk, float]]: ...
