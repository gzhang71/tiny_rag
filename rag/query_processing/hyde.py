"""HyDE — Hypothetical Document Embeddings.

Questions and the passages that answer them live in different regions of
embedding space ("Who loves cats?" vs "Alice loves cats."). HyDE bridges the
gap: one short Claude call writes a hypothetical passage that *would* answer
the query, and that passage is retrieved with alongside the original query —
its embedding lands near real answer passages, and its vocabulary feeds the
sparse tunnels. The two ranked lists are RRF-fused, so a hallucinated detail
in the hypothetical can add noise to one list but can't fabricate a chunk.
"""
import anthropic

from rag.query_processing.base import QueryExpander

_MODEL = "claude-opus-4-8"

_SYSTEM = (
    "You support a document retrieval system (HyDE). Write a short passage "
    "(2-4 sentences) that plausibly answers the user's question, phrased the "
    "way an actual document would state it — declarative, concrete, no "
    "hedging and no meta-commentary. Invented specifics are acceptable; the "
    "passage is used only as a search probe, never shown to the user. "
    "Preserve identifiers (codes, versions, ticket ids, names) verbatim. "
    "Reply with the passage only."
)


class HyDEExpander(QueryExpander):
    def __init__(self, model: str = _MODEL):
        self.client = anthropic.Anthropic()
        self.model = model

    def expand(self, query: str) -> list[str]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            output_config={"effort": "low"},
            system=_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        if response.stop_reason == "refusal":
            return [query]
        passage = next(
            (block.text for block in response.content if block.type == "text"), ""
        ).strip()
        return [query, passage] if passage else [query]
