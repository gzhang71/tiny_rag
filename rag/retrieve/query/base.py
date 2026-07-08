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
