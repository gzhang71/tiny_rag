"""LLM query decomposition.

A complex or multi-hop question ("How does ingest dedupe interact with the
Chroma backend?") often retrieves poorly as a single string because its parts
compete for the embedding. One short Claude call splits it into self-contained
sub-questions; each is retrieved independently and the rankings are RRF-fused,
so evidence for every part of the question reaches the candidate pool. Falls
back to the original query alone when the model declines or the question is
already atomic.
"""
import anthropic

from rag.query_processing.base import QueryExpander

_MODEL = "claude-opus-4-8"
_MAX_SUBQUERIES = 4

_SYSTEM = (
    "You decompose search queries for a document retrieval system. "
    "If the user's question contains multiple distinct information needs, "
    "split it into at most {max_subqueries} self-contained sub-questions, one per line, "
    "each answerable from a single passage; resolve pronouns and shared "
    "references so every sub-question stands alone. Preserve identifiers "
    "(codes, versions, ticket ids, names) verbatim and do not answer the "
    "question. If the question is already a single atomic information need, "
    "reply with the single word ATOMIC. Reply with the sub-questions (or "
    "ATOMIC) only."
)


class QueryDecomposer(QueryExpander):
    def __init__(self, model: str = _MODEL, max_subqueries: int = _MAX_SUBQUERIES):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_subqueries = max_subqueries

    def expand(self, query: str) -> list[str]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            output_config={"effort": "low"},
            system=_SYSTEM.format(max_subqueries=self.max_subqueries),
            messages=[{"role": "user", "content": query}],
        )
        if response.stop_reason == "refusal":
            return [query]
        text = next(
            (block.text for block in response.content if block.type == "text"), ""
        ).strip()
        if not text or text.upper() == "ATOMIC":
            return [query]
        subqueries = [line.strip("-* \t") for line in text.splitlines() if line.strip()]
        subqueries = [s for s in subqueries if s][: self.max_subqueries]
        # keep the original query in the pool — fusion rewards chunks that
        # answer the whole question, not just one part
        return [query] + [s for s in subqueries if s.lower() != query.lower()]
