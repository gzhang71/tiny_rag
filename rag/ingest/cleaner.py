"""Document cleaning, quality filtering, and duplicate-chunk removal."""
import hashlib
import unicodedata

from rag.store.document import Chunk


def clean_text(text: str) -> str:
    """Normalise a raw document before chunking: NFKC unicode normalisation,
    unified newlines, control characters stripped, runs of spaces/tabs
    collapsed, at most one blank line in a row."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )
    lines = [" ".join(line.split()) for line in text.split("\n")]
    cleaned: list[str] = []
    for line in lines:
        if line or (cleaned and cleaned[-1]):  # collapse blank-line runs to one
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def content_hash(text: str) -> str:
    """Stable fingerprint of a whole (cleaned) document — used to skip
    re-ingesting sources that haven't changed."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_low_quality(text: str) -> bool:
    """Boilerplate/noise heuristic: near-empty fragments and chunks that are
    mostly non-letters (page numbers, separator lines, headers/footers).
    Thresholds are deliberately loose so short legitimate sentences survive;
    CJK characters carry roughly 2x the information of latin ones, so they
    count double toward the length check."""
    stripped = text.strip()
    weighted_len = sum(
        2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in stripped
    )
    if weighted_len < 15:
        return True
    letters = sum(ch.isalpha() for ch in stripped)
    return letters / len(stripped) < 0.25


def _signature(text: str) -> str:
    """Order-insensitive exact fingerprint: hash of the sorted token set, so
    reformatted or reflowed copies of the same passage collide."""
    tokens = sorted(set(text.lower().split()))
    return hashlib.sha1(" ".join(tokens).encode("utf-8")).hexdigest()


def _simhash(tokens: list[str]) -> int:
    """64-bit SimHash: near-identical token multisets land within a few bits
    of each other, so hamming distance measures content similarity."""
    bits = [0] * 64
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:16], 16)
        for i in range(64):
            bits[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i, b in enumerate(bits) if b > 0)


class ChunkDeduper:
    """Exact + near-duplicate chunk filter, stateful across ingest calls.

    Exact duplicates are caught by the token-set signature; near-duplicates
    (a few words changed — different versions of the same passage) by SimHash
    within `hamming_threshold` bits. Measured on chunk-sized texts, versions
    of the same passage differ by ~3-5 bits while unrelated chunks sit around
    32 (never observed below 20), so 8 separates them with wide margin.
    `observe` records a text without filtering, used to seed state from an
    already-populated persisted store.
    """

    def __init__(self, hamming_threshold: int = 8):
        self.hamming_threshold = hamming_threshold
        self._exact: set[str] = set()
        self._simhashes: list[int] = []

    def observe(self, text: str) -> None:
        self._exact.add(_signature(text))
        self._simhashes.append(_simhash(text.lower().split()))

    def _is_duplicate(self, text: str) -> bool:
        if _signature(text) in self._exact:
            return True
        simhash = _simhash(text.lower().split())
        return any(
            (simhash ^ seen).bit_count() <= self.hamming_threshold
            for seen in self._simhashes
        )

    def dedupe(self, chunks: list[Chunk]) -> list[Chunk]:
        kept: list[Chunk] = []
        for chunk in chunks:
            if self._is_duplicate(chunk.text):
                continue
            self.observe(chunk.text)
            kept.append(chunk)
        return kept
