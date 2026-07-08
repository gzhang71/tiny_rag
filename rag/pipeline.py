import warnings

from rag.generate.generator import Generator
from rag.ingest.chunker import chunk_text
from rag.ingest.cleaner import ChunkDeduper, clean_text, content_hash
from rag.ingest.embedder import Embedder
from rag.ingest.loader import load_file, load_directory
from rag.retrieve.query import QueryStage, SpellCorrector
from rag.retrieve.rerank import CrossEncoderReranker
from rag.retrieve.retriever import Channel, DEFAULT_CHANNELS, Retriever
from rag.store.base import BaseVectorStore, StoreBackend
from rag.store.vector_store import IndexType, VectorStore


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        top_k: int = 5,
        backend: StoreBackend = StoreBackend.FAISS,
        # retrieval channels
        channels: tuple[Channel, ...] = DEFAULT_CHANNELS,
        mmr_lambda: float | None = None,
        rerank: bool = False,
        rrf_k: int = 60,
        # query preprocessing
        spell_correct: bool = True,   # corpus-vocab typo correction, local
        query_rewrite: bool = False,  # LLM query understanding, one Claude call per query
        # ingest enrichment
        contextualize: bool = False,  # contextual retrieval, one Claude call per chunk
        # FAISS-specific
        index_type: IndexType = IndexType.FLAT,
        # Chroma-specific
        persist_dir: str | None = "./chroma_db",
        collection: str = "tiny_rag",
        chroma_host: str | None = None,
        chroma_port: int = 8000,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k

        self.embedder = Embedder()
        if chunk_size > self.embedder.max_chars:
            warnings.warn(
                f"chunk_size={chunk_size} chars likely exceeds the embedding model's "
                f"{self.embedder.max_seq_length}-token window (~{self.embedder.max_chars} "
                "chars); dense retrieval will silently ignore the tail of each chunk. "
                f"Consider chunk_size <= {self.embedder.max_chars}.",
                stacklevel=2,
            )
        self.store: BaseVectorStore
        if backend == StoreBackend.CHROMA:
            # imported lazily so the faiss backend works without chromadb installed
            from rag.store.chroma_store import ChromaStore

            self.store = ChromaStore(
                persist_dir=persist_dir,
                collection=collection,
                host=chroma_host,
                port=chroma_port,
            )
        else:
            self.store = VectorStore(dim=self.embedder.dim, index_type=index_type)

        query_stages: list[QueryStage] = []
        if spell_correct:
            query_stages.append(SpellCorrector(self.store))
        if query_rewrite:
            # imported lazily so query rewriting stays an opt-in Claude dependency
            from rag.retrieve.query.rewrite import QueryRewriter

            query_stages.append(QueryRewriter())

        self.retriever = Retriever(
            self.embedder,
            self.store,
            channels=channels,
            rrf_k=rrf_k,
            mmr_lambda=mmr_lambda,
            reranker=CrossEncoderReranker() if rerank else None,
            query_stages=tuple(query_stages),
        )
        self.generator = Generator()

        self.contextualizer = None
        if contextualize:
            # imported lazily so contextual retrieval stays an opt-in Claude dependency
            from rag.ingest.contextualizer import Contextualizer

            self.contextualizer = Contextualizer()

        # dedupe + per-source content hashes; seeded lazily from a persisted
        # store so skip/dedupe decisions survive across processes
        self._deduper = ChunkDeduper()
        self._known_sources: dict[str, str] = {}
        self._seeded = False

    def _seed_from_store(self) -> None:
        if self._seeded:
            return
        for chunk in self.store.chunks():
            self._deduper.observe(chunk.text)
            doc_hash = chunk.metadata.get("doc_hash")
            if doc_hash:
                self._known_sources[chunk.source] = doc_hash
        self._seeded = True

    def ingest_text(self, text: str, source: str = "inline") -> int:
        self._seed_from_store()
        cleaned = clean_text(text)
        doc_hash = content_hash(cleaned)
        if self._known_sources.get(source) == doc_hash:
            return 0  # source unchanged since its last ingest — skip re-embedding

        chunks = chunk_text(cleaned, source=source, chunk_size=self.chunk_size, overlap=self.overlap)
        chunks = self._deduper.dedupe(chunks)
        if not chunks:
            self._known_sources[source] = doc_hash
            return 0
        if self.contextualizer is not None:
            chunks = self.contextualizer.contextualize(cleaned, chunks)
        for chunk in chunks:
            chunk.metadata["doc_hash"] = doc_hash

        embeddings = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, embeddings)
        self._known_sources[source] = doc_hash
        return len(chunks)

    def ingest_file(self, path: str) -> int:
        return self.ingest_text(load_file(path), source=path)

    def ingest_directory(self, directory: str, glob: str | None = None) -> int:
        total = 0
        for source, text in load_directory(directory, glob=glob).items():
            total += self.ingest_text(text, source=source)
        return total

    def query(self, question: str) -> str:
        if len(self.store) == 0:
            raise RuntimeError("No documents ingested. Call ingest_text/ingest_file first.")
        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        return self.generator.generate(question, chunks)
