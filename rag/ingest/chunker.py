import re
from collections.abc import Iterator

from rag.ingest.cleaner import is_low_quality
from rag.store.document import Chunk

# sentence boundary: end punctuation + whitespace, CJK end punctuation
# (no trailing space in CJK text), or a paragraph break
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])|\n{2,}")

# markdown-style heading line; loaders synthesise these for HTML/docx too
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split on markdown headings into (breadcrumb, body) pairs. The
    breadcrumb is the heading path from the document root, e.g.
    'Configuration > Timeouts', so a chunk keeps its place in the document."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    if text[: matches[0].start()].strip():
        sections.append(("", text[: matches[0].start()]))
    stack: list[tuple[int, str]] = []  # (level, title) path to the current heading
    for match, following in zip(matches, [*matches[1:], None]):
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2).strip()))
        body = text[match.end() : following.start() if following else len(text)]
        sections.append((" > ".join(title for _, title in stack), body))
    return sections


def _pack_sentences(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    """Pack whole sentences into strings of at most `chunk_size` characters,
    carrying the trailing sentences (up to ~`overlap` chars) of each chunk
    into the next. Sentences longer than `chunk_size` are hard-split."""
    pieces: list[str] = []
    for sentence in _split_sentences(text):
        if len(sentence) <= chunk_size:
            pieces.append(sentence)
        else:
            pieces.extend(
                sentence[i : i + chunk_size] for i in range(0, len(sentence), chunk_size)
            )

    current: list[str] = []
    length = 0
    fresh = 0  # pieces in `current` that aren't carried-over overlap

    for piece in pieces:
        if fresh and length + 1 + len(piece) > chunk_size:
            yield " ".join(current)
            carried: list[str] = []
            carried_len = 0
            for sentence in reversed(current):
                if carried_len + len(sentence) + 1 > overlap:
                    break
                carried.insert(0, sentence)
                carried_len += len(sentence) + 1
            current, length, fresh = carried, carried_len, 0
        current.append(piece)
        length += len(piece) + (1 if len(current) > 1 else 0)
        fresh += 1

    if fresh:  # skip a final chunk that would be pure overlap of the previous one
        yield " ".join(current)


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """Structure- and sentence-aware chunking. The document is split into
    sections along markdown-style headings; each section's sentences are
    packed into overlapping chunks. A chunk's text is prefixed with its
    heading breadcrumb (also kept in `metadata["section"]`) so every
    retrieval channel and the generator see where it belongs. Low-quality
    chunks (near-empty, mostly non-letters) are dropped before indexing.
    """
    chunks: list[Chunk] = []
    for breadcrumb, body in _split_sections(text):
        for packed in _pack_sentences(body, chunk_size, overlap):
            if is_low_quality(packed):
                continue
            chunks.append(
                Chunk(
                    text=f"[{breadcrumb}] {packed}" if breadcrumb else packed,
                    source=source,
                    chunk_index=len(chunks),
                    metadata={"section": breadcrumb} if breadcrumb else {},
                )
            )
    return chunks
