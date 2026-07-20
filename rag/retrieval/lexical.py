"""Exact phrase matching — the strict lexical retrieval tunnel.

Where BM25 scores bag-of-words term overlap, this tunnel rewards chunks that
contain contiguous spans of the query verbatim (the equivalent of a
`match_phrase` clause in Elasticsearch). A chunk containing a longer exact
span of the query outranks one with scattered term hits.
"""
from rag.retrieval.base import RetrieveTunnel
from rag.retrieval.text import STOPWORDS, tokenize
from rag.store.document import Chunk


def _contains(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


class LexicalTunnel(RetrieveTunnel):
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._token_seqs = [tokenize(c.text) for c in chunks]

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        scored = []
        for chunk, tokens in zip(self._chunks, self._token_seqs):
            # longest contiguous query span present verbatim in the chunk;
            # spans of pure stopwords ("what was the") don't count as matches
            best = 0
            for n in range(len(terms), 0, -1):
                spans = (terms[i : i + n] for i in range(len(terms) - n + 1))
                if any(
                    _contains(tokens, span) for span in spans
                    if not all(t in STOPWORDS for t in span)
                ):
                    best = n
                    break
            if best:
                scored.append((chunk, best / len(terms)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._chunks)
