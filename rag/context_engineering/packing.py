"""Context packing — fit the retrieved chunks into a coherent context window.

Retrieval ranks chunks by relevance, but the generator's context has a token
budget and reads top-to-bottom. The packer:

1. drops query-time near-duplicates (token-set Jaccard) that ingest-time
   dedupe can't catch — e.g. the same passage retrieved from two sources, or
   overlapping neighbour chunks
2. greedily keeps the highest-scored chunks that fit the character budget
   (a smaller lower-ranked chunk may fill space a bigger one couldn't)
3. re-orders the survivors into (source, chunk_index) document order, so
   adjacent chunks read as continuous text instead of jumping around
"""
from rag.context_engineering.base import ContextStage
from rag.retrieval.text import tokenize
from rag.store.document import Chunk


class ContextPacker(ContextStage):
    def __init__(
        self,
        max_chars: int | None = None,  # None = no budget, still dedupes + orders
        dedupe_threshold: float = 0.85,  # Jaccard similarity above which a chunk is dropped
    ):
        if not 0.0 < dedupe_threshold <= 1.0:
            raise ValueError("dedupe_threshold must be in (0, 1]")
        self.max_chars = max_chars
        self.dedupe_threshold = dedupe_threshold

    def apply(
        self, query: str, chunks: list[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        selected: list[tuple[Chunk, float]] = []
        selected_tokens: list[set[str]] = []
        used = 0
        for chunk, score in chunks:  # already ranked best-first
            if self.max_chars is not None and used + len(chunk.text) > self.max_chars:
                continue  # doesn't fit — a smaller lower-ranked chunk still might
            tokens = set(tokenize(chunk.text))
            if tokens and any(
                len(tokens & prev) / len(tokens | prev) >= self.dedupe_threshold
                for prev in selected_tokens
            ):
                continue  # near-duplicate of an already-packed chunk
            selected.append((chunk, score))
            selected_tokens.append(tokens)
            used += len(chunk.text)
        # document order: neighbouring chunks concatenate into readable text
        selected.sort(key=lambda pair: (pair[0].source, pair[0].chunk_index))
        return selected
