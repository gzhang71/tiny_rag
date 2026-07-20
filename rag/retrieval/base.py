from abc import ABC, abstractmethod

from rag.store.document import Chunk


class RetrieveTunnel(ABC):
    """One parallel retrieval channel a query runs through.

    Every tunnel maps a raw query string to a ranked list of (chunk, score)
    pairs. Scores only need to be internally consistent per tunnel — fusion
    across tunnels is rank-based (RRF), never score-based.
    """

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of chunks indexed — used to detect a stale index."""
