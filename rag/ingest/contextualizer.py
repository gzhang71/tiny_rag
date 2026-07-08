"""Contextual retrieval: situate each chunk within its document.

An isolated chunk often loses the referents that make it findable ("It rose
by 3%" — what did?). For each chunk, one short Claude call writes a 1-2
sentence context describing where the chunk sits in the overall document;
the context is prepended to the chunk text, so the embedder and every
retrieval tunnel index it, and recorded in `metadata["context"]`.

The full document is sent with a prompt-cache breakpoint, so per-chunk calls
after the first read the document at ~10% of input price. Opt-in
(`--contextualize`): ingestion costs one API call per chunk.
"""
import anthropic

from rag.store.document import Chunk

_MODEL = "claude-opus-4-8"
_MAX_DOC_CHARS = 400_000  # keep the cached document comfortably inside the context window

_PROMPT = (
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Give a short succinct context (1-2 sentences) situating this chunk within "
    "the overall document for the purposes of improving search retrieval of "
    "the chunk. Answer only with the succinct context and nothing else."
)


class Contextualizer:
    def __init__(self, model: str = _MODEL):
        self.client = anthropic.Anthropic()
        self.model = model

    def contextualize(self, document: str, chunks: list[Chunk]) -> list[Chunk]:
        """Mutates and returns `chunks`: prepends the generated context to each
        chunk's text and records it in `metadata["context"]`."""
        document = document[:_MAX_DOC_CHARS]
        for chunk in chunks:
            context = self._context_for(document, chunk.text)
            if context:
                chunk.metadata["context"] = context
                chunk.text = f"{context}\n{chunk.text}"
        return chunks

    def _context_for(self, document: str, chunk_text: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            output_config={"effort": "low"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<document>\n{document}\n</document>",
                            # cached across the per-chunk calls for this document
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": _PROMPT.format(chunk=chunk_text)},
                    ],
                }
            ],
        )
        if response.stop_reason == "refusal":
            return ""
        return next(
            (block.text for block in response.content if block.type == "text"), ""
        ).strip()
