from abc import ABC, abstractmethod


class QueryStage(ABC):
    """One query-preprocessing stage, run before the retrieval tunnels.

    Every stage maps a raw query string to an improved one (typo-corrected,
    expanded, rewritten). Stages are applied in sequence, each receiving the
    previous stage's output; a stage that can't improve the query should
    return it unchanged.
    """

    @abstractmethod
    def process(self, query: str) -> str: ...


class QueryExpander(ABC):
    """One query-expansion stage, run after the `QueryStage`s.

    Where a `QueryStage` rewrites the query in place, an expander turns it
    into several queries that are each retrieved independently and RRF-fused
    (decomposition into sub-questions, HyDE hypothetical documents, ...).
    The original query should be included in the returned list unless the
    expander deliberately replaces it; an expander that can't improve the
    query should return `[query]`.
    """

    @abstractmethod
    def expand(self, query: str) -> list[str]: ...
