"""Dense retrieval tunnel — semantic similarity via the vector store."""
from rag.ingest.embedder import Embedder
from rag.retrieval.base import RetrieveTunnel
from rag.store.base import BaseVectorStore
from rag.store.document import Chunk


class DenseTunnel(RetrieveTunnel):
    def __init__(self, embedder: Embedder, store: BaseVectorStore):
        self.embedder = embedder
        self.store = store

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        return self.store.search(self.embedder.embed_one(query), top_k=top_k)

    def __len__(self) -> int:
        return len(self.store)
