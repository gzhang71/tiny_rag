"""File loading with per-format text extraction.

Every loader returns plain text in which section headings survive as
markdown-style ``# `` lines (native in Markdown, synthesised for HTML and
docx) — the chunker uses them for structure-aware chunking. PDF and docx
support needs the optional `pypdf` / `python-docx` packages, imported lazily
so plain-text ingestion works without them.
"""
from html.parser import HTMLParser
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".text"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".html", ".htm", ".docx"}


def load_file(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in (".html", ".htm"):
        return _html_to_text(path.read_text(encoding="utf-8"))
    if suffix == ".docx":
        return _load_docx(path)
    return path.read_text(encoding="utf-8")


def load_directory(directory: str | Path, glob: str | None = None) -> dict[str, str]:
    """Load every supported file under `directory`. Pass an explicit `glob`
    (e.g. "**/*.txt") to restrict; the default picks up all supported types."""
    root = Path(directory)
    if glob is not None:
        paths = root.glob(glob)
    else:
        paths = (p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    return {str(p): load_file(p) for p in sorted(paths) if p.is_file()}


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PDF ingestion requires pypdf (pip install pypdf)") from exc
    return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _load_docx(path: Path) -> str:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("docx ingestion requires python-docx (pip install python-docx)") from exc
    lines: list[str] = []
    for para in docx.Document(str(path)).paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style is not None else ""
        if style.startswith("Heading"):
            try:
                level = min(int(style.rsplit(" ", 1)[-1]), 6)
            except ValueError:
                level = 1
            lines.append(f"{'#' * level} {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines)


class _HTMLTextExtractor(HTMLParser):
    """Extracts readable text: skips script/style, turns <h1>-<h6> into
    markdown headings, and inserts paragraph breaks at block elements."""

    _SKIP = {"script", "style", "head", "noscript", "template"}
    _HEADINGS = {f"h{n}": n for n in range(1, 7)}
    _BLOCKS = {"p", "div", "li", "ul", "ol", "table", "tr", "br", "section",
               "article", "header", "footer", "blockquote", "pre", "nav"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._HEADINGS:
            self._parts.append(f"\n\n{'#' * self._HEADINGS[tag]} ")
        elif tag in self._BLOCKS:
            self._parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._HEADINGS or tag in self._BLOCKS:
            self._parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        collapsed = " ".join(data.split())
        if collapsed:
            self._parts.append(collapsed + " ")

    def text(self) -> str:
        return "".join(self._parts)


def _html_to_text(html: str) -> str:
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return extractor.text()
