"""Okapi BM25 — the sparse/lexical retrieval tunnel.

Complements dense retrieval: exact keyword matches (names, error codes, IDs)
that embedding models blur together score highly here.
"""
import math
from collections import Counter

from rag.retrieval.base import RetrieveTunnel
from rag.retrieval.text import tokenize as _tokenize
from rag.store.document import Chunk


class BM25Tunnel(RetrieveTunnel):
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks = chunks
        self._term_freqs = [Counter(_tokenize(c.text)) for c in chunks]
        self._doc_lens = [sum(tf.values()) for tf in self._term_freqs]
        self._avg_len = (sum(self._doc_lens) / len(chunks)) if chunks else 0.0

        doc_freq: Counter[str] = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        n = len(chunks)
        self._idf = {
            term: math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in doc_freq.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        query_terms = _tokenize(query)
        scored = []
        for i, tf in enumerate(self._term_freqs):
            score = 0.0
            for term in query_terms:
                freq = tf.get(term)
                if not freq:
                    continue
                norm = 1.0 - self.b + self.b * self._doc_lens[i] / self._avg_len
                score += self._idf[term] * freq * (self.k1 + 1.0) / (freq + self.k1 * norm)
            if score > 0.0:
                scored.append((self._chunks[i], score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)
