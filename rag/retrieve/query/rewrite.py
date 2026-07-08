"""LLM query rewriting (query understanding).

Reformulates the raw query into one that retrieves better — expands acronyms
and ambiguous references, adds close synonyms of key terms, fixes grammar —
via one short Claude call per query. Falls back to the original query if the
model returns nothing usable.
"""
import anthropic

from rag.retrieve.query.base import QueryStage

_MODEL = "claude-opus-4-8"

_SYSTEM = (
    "You rewrite search queries for a document retrieval system. "
    "Rewrite the user's query so it retrieves the most relevant passages: "
    "fix typos and grammar, expand acronyms and ambiguous references, and add "
    "close synonyms of key terms where helpful. Keep it a single short query, "
    "preserve identifiers (codes, versions, ticket ids, names) verbatim, and "
    "do not answer the question. Reply with the rewritten query only."
)


class QueryRewriter(QueryStage):
    def __init__(self, model: str = _MODEL):
        self.client = anthropic.Anthropic()
        self.model = model

    def process(self, query: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            output_config={"effort": "low"},  # rewriting is simple; keep it fast
            system=_SYSTEM,
            messages=[{"role": "user", "content": query}],
        )
        if response.stop_reason == "refusal":
            return query
        rewritten = next(
            (block.text for block in response.content if block.type == "text"), ""
        ).strip()
        return rewritten or query
