import chromadb
import numpy as np

from rag.store.base import BaseVectorStore
from rag.store.document import Chunk


def _to_chunk(doc: str, meta: dict) -> Chunk:
    return Chunk(
        text=doc,
        source=meta["source"],
        chunk_index=meta["chunk_index"],
        metadata={k: v for k, v in meta.items() if k not in ("source", "chunk_index")},
    )


class ChromaStore(BaseVectorStore):
    """Chroma-backed store. Three modes, picked by constructor args:

    - persist_dir set (default): embedded DB persisted to that directory
    - host set: client for a standalone server (`chroma run --path <dir>`)
    - neither: ephemeral in-memory DB
    """

    def __init__(
        self,
        persist_dir: str | None = "./chroma_db",
        collection: str = "tiny_rag",
        host: str | None = None,
        port: int = 8000,
    ):
        if host is not None:
            self.client = chromadb.HttpClient(host=host, port=port)
        elif persist_dir is not None:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.EphemeralClient()

        # cosine space matches the FAISS backend: embeddings are L2-normalised,
        # so scores are comparable across backends
        self.collection = self.client.get_or_create_collection(
            name=collection,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        # ids are source:chunk_index so re-ingesting a document updates in place
        # instead of duplicating chunks across runs; chunk metadata rides along
        # (chroma only takes scalar values)
        self.collection.upsert(
            ids=[f"{c.source}:{c.chunk_index}" for c in chunks],
            embeddings=embeddings.astype(np.float32),
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    **{
                        k: v
                        for k, v in c.metadata.items()
                        if isinstance(v, (str, int, float, bool))
                    },
                }
                for c in chunks
            ],
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        count = len(self)
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=query_embedding.astype(np.float32).reshape(1, -1),
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        return [
            (
                _to_chunk(doc, meta),
                1.0 - dist,  # chroma returns cosine distance; 1 - distance == cosine similarity
            )
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def chunks(self) -> list[Chunk]:
        result = self.collection.get(include=["documents", "metadatas"])
        return [
            _to_chunk(doc, meta)
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]

    def __len__(self) -> int:
        return self.collection.count()
