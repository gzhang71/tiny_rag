from abc import ABC, abstractmethod

from rag.store.document import Chunk


class ContextStage(ABC):
    """One context-engineering stage, run between retrieval and generation.

    Every stage maps the ranked (chunk, score) list to a transformed one —
    compressed, deduplicated, packed to a budget. Stages are applied in
    sequence, each receiving the previous stage's output. Stages must not
    mutate the incoming chunks (they are the store's live objects); a stage
    that rewrites text returns copies.
    """

    @abstractmethod
    def apply(
        self, query: str, chunks: list[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]: ...
