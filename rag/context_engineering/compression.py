"""Contextual compression — keep only the sentences that serve the query.

A retrieved chunk is usually only partly relevant; the rest dilutes the
generator's attention and spends tokens. The compressor splits each chunk
into sentences, scores every sentence against the query with the local
embedder (one batched encode, no API call), and keeps the top `keep` fraction
per chunk in their original order. Short chunks pass through untouched, and
compressed chunks are copies — the store's objects are never mutated.
"""
import math
from dataclasses import replace

from rag.context_engineering.base import ContextStage
from rag.ingest.chunker import _split_sentences
from rag.ingest.embedder import Embedder
from rag.store.document import Chunk

_MIN_SENTENCES = 3  # below this, compression can't drop anything meaningful


class ContextCompressor(ContextStage):
    def __init__(self, embedder: Embedder, keep: float = 0.6):
        if not 0.0 < keep <= 1.0:
            raise ValueError("keep must be in (0, 1]")
        self.embedder = embedder
        self.keep = keep

    def apply(
        self, query: str, chunks: list[tuple[Chunk, float]]
    ) -> list[tuple[Chunk, float]]:
        if not chunks:
            return []
        per_chunk = [_split_sentences(chunk.text) for chunk, _ in chunks]
        to_score = [s for sentences in per_chunk for s in sentences]
        if not to_score:
            return chunks
        query_embedding = self.embedder.embed_one(query)
        relevance = self.embedder.embed(to_score) @ query_embedding

        result: list[tuple[Chunk, float]] = []
        offset = 0
        for (chunk, score), sentences in zip(chunks, per_chunk):
            scores = relevance[offset : offset + len(sentences)]
            offset += len(sentences)
            if len(sentences) < _MIN_SENTENCES:
                result.append((chunk, score))
                continue
            keep_n = max(1, math.ceil(self.keep * len(sentences)))
            keep_idx = sorted(
                sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:keep_n]
            )  # rank by relevance, then restore original sentence order
            if len(keep_idx) == len(sentences):
                result.append((chunk, score))
                continue
            compressed = " ".join(sentences[i] for i in keep_idx)
            result.append(
                (
                    replace(
                        chunk,
                        text=compressed,
                        metadata={**chunk.metadata, "compressed_from_chars": len(chunk.text)},
                    ),
                    score,
                )
            )
        return result
