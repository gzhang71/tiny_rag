"""Corpus-driven query spell correction.

A query word that doesn't appear in the ingested corpus vocabulary is replaced
by its closest vocabulary word within a small edit distance (1 for short
words, 2 for longer), preferring more frequent words on ties. Guardrails keep
novel terms from being "corrected" away: only pure-alphabetic words of 4+
characters are candidates, and anything already in the vocabulary is left
alone. Needs no external dependency or API call.
"""
import re
from collections import Counter

from rag.retrieve.query.base import QueryStage
from rag.store.base import BaseVectorStore

_WORD = re.compile(r"[A-Za-z]+")
_MIN_LEN = 4  # shorter words are too ambiguous to correct safely


def _edit_distance(a: str, b: str, cap: int) -> int | None:
    """Levenshtein distance, or None once it provably exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return None
        prev = cur
    return prev[-1] if prev[-1] <= cap else None


class SpellCorrector(QueryStage):
    """The vocabulary is built lazily from `store.chunks()` and rebuilt when
    the store size changes, mirroring how the corpus tunnels stay fresh."""

    def __init__(self, store: BaseVectorStore):
        self.store = store
        self._vocab: Counter[str] = Counter()
        self._indexed = -1

    def _ensure_vocab(self) -> None:
        if self._indexed == len(self.store):
            return
        self._vocab = Counter(
            word.lower()
            for chunk in self.store.chunks()
            for word in _WORD.findall(chunk.text)
        )
        self._indexed = len(self.store)

    def _correct(self, word: str) -> str:
        lower = word.lower()
        if len(lower) < _MIN_LEN or lower in self._vocab:
            return word
        cap = 1 if len(lower) <= 6 else 2
        best: str | None = None
        best_key: tuple[int, int] | None = None
        for candidate, freq in self._vocab.items():
            distance = _edit_distance(lower, candidate, cap)
            if distance is None:
                continue
            key = (distance, -freq)
            if best_key is None or key < best_key:
                best, best_key = candidate, key
        return best if best is not None else word

    def process(self, query: str) -> str:
        self._ensure_vocab()
        if not self._vocab:
            return query
        return _WORD.sub(lambda m: self._correct(m.group(0)), query)
