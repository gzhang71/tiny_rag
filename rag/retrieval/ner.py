"""Named-entity retrieval tunnel.

Extracts entities from query and chunks, then scores chunks by IDF-weighted
entity overlap — a rare entity shared with the query ("ZX-9981") is worth far
more than a common one. Extraction is rule-based (regex heuristics) so it runs
with zero extra dependencies; swap `extract_entities` for a spaCy or
transformer NER model for higher recall on natural-language entity mentions.
"""
import math
import re
from collections import Counter

from rag.retrieval.base import RetrieveTunnel
from rag.retrieval.text import STOPWORDS
from rag.store.document import Chunk

_PATTERNS = [
    re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),   # emails
    re.compile(r"\bhttps?://\S+\b"),                                     # URLs
    re.compile(r"\b[A-Z]{1,6}-\d{2,}\b"),                                # ticket/CVE-style ids: ZX-9981
    re.compile(r"\b[A-Z]\d{3,}\b"),                                      # short codes: E4032
    re.compile(r"\bv?\d+\.\d+(?:\.\d+)+\b"),                             # versions: 2.14.1
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                                # ISO dates
    re.compile(r"\b(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)+\b"),               # multi-word proper nouns: New York
    re.compile(r"\b[A-Z][a-z]{2,}\b"),                                   # single capitalised words
]


def extract_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for pattern in _PATTERNS:
        for match in pattern.findall(text):
            lowered = match.lower()
            # capitalised sentence-openers like "The" or "When" are not entities
            if lowered not in STOPWORDS:
                entities.add(lowered)
    return entities


class EntityTunnel(RetrieveTunnel):
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._entities = [extract_entities(c.text) for c in chunks]

        doc_freq: Counter[str] = Counter()
        for ents in self._entities:
            doc_freq.update(ents)
        n = len(chunks)
        self._idf = {
            ent: math.log((n + 1) / (df + 0.5))
            for ent, df in doc_freq.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        query_entities = extract_entities(query)
        if not query_entities:
            return []
        scored = []
        for chunk, ents in zip(self._chunks, self._entities):
            score = sum(self._idf[e] for e in query_entities & ents)
            if score > 0.0:
                scored.append((chunk, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)
