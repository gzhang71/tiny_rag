# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip3 install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

## Running

```bash
python3 main.py --text "Alice loves cats." "Who loves cats?"
python3 main.py --file path/to/doc.txt "What is the main topic?"
python3 main.py --dir path/to/docs/ "Summarize the key points"

# persistent Chroma DB: ingest once, then query without a source
python3 main.py --store chroma --dir path/to/docs/ "Summarize the key points"
python3 main.py --store chroma "Follow-up question"

# retrieval tunnels (default: all four, RRF-fused), MMR diversity, cross-encoder rerank
python3 main.py --channels dense,bm25 --mmr 0.7 --rerank --file doc.txt "Question"

# query preprocessing: spell correction is on by default (--no-spell disables);
# --rewrite adds an LLM query-understanding pass (one Claude call per query)
python3 main.py --rewrite --file doc.txt "hw does the pipline work?"

# .pdf / .html / .docx / .md are supported; --contextualize adds contextual
# retrieval (a Claude-written situating context per chunk, one call per chunk)
python3 main.py --contextualize --file report.pdf "What were the Q3 numbers?"
```

## Architecture

`RAGPipeline` (in `rag/pipeline.py`) is the single public entry point. It wires together four independent stages:

1. **Ingest** (`rag/ingest/`) —
   - `loader.py` reads files/directories with per-format extraction: `.txt`/`.md` natively, `.pdf` via `pypdf`, `.docx` via `python-docx`, `.html` via a stdlib parser — HTML/docx headings are synthesised as markdown `#` lines so all formats feed the same structure-aware chunker. `pypdf`/`python-docx` are imported lazily. `load_directory` picks up all supported extensions by default (pass `glob` to restrict).
   - `cleaner.py` normalises documents (NFKC unicode, newlines, control chars, whitespace); `is_low_quality` drops near-empty/mostly-non-letter chunks (CJK chars count double toward the length threshold); `ChunkDeduper` drops exact duplicates (token-set signature) and near-duplicates (64-bit SimHash, hamming ≤ 8) across ingest calls; `content_hash` fingerprints whole documents for incremental ingest.
   - `chunker.py` is structure- and sentence-aware: text splits into sections along markdown headings, each chunk's text is prefixed with its heading breadcrumb (`[Configuration > Timeouts] …`, also in `metadata["section"]`), sentences (incl. CJK `。！？` boundaries) are packed whole up to `chunk_size` chars with trailing sentences up to `overlap` chars carried into the next chunk; oversized sentences hard-split (`Chunk` dataclass from `rag/store/document.py`).
   - `contextualizer.py` (opt-in via `contextualize=True` / `--contextualize`) — contextual retrieval: one Claude call per chunk (`claude-opus-4-8`, document prompt-cached across calls) writes a 1-2 sentence situating context, prepended to the chunk text so every tunnel indexes it (also in `metadata["context"]`). Imported lazily.
   - `embedder.py` wraps `sentence-transformers` (`all-MiniLM-L6-v2`) to produce L2-normalised float32 vectors; exposes `max_seq_length`/`max_chars` — `RAGPipeline` warns when `chunk_size` exceeds what the encoder can embed (the tail would be silently ignored by dense retrieval only).

   **Incremental ingest**: `RAGPipeline` stores a `doc_hash` in every chunk's metadata and skips re-embedding a source whose cleaned content hash is unchanged; dedupe state and hashes are seeded lazily from the store, so this works across processes with a persisted Chroma DB (whose `add`/`chunks` round-trip chunk metadata).

2. **Store** (`rag/store/`) — two backends implementing `BaseVectorStore` (`base.py`), selected via the `StoreBackend` enum:
   - `VectorStore` (`vector_store.py`, `StoreBackend.FAISS`, default) wraps an in-memory FAISS index. Two index types via `IndexType`: `FLAT` (exact cosine, `IndexFlatIP`) and `HNSW` (approximate, `IndexHNSWFlat` with inner-product metric). Switch to `HNSW` at ~10k+ chunks.
   - `ChromaStore` (`chroma_store.py`, `StoreBackend.CHROMA`) wraps a Chroma collection (cosine space). Embedded + persisted to `persist_dir` (default `./chroma_db`) by default; pass `host`/`port` to connect to a `chroma run` server; both `None` gives an ephemeral in-memory DB. Ids are `source:chunk_index`, so re-ingesting a document upserts instead of duplicating. `chromadb` is imported lazily in `pipeline.py` — the FAISS path works without it installed.

   Both backends rely on L2-normalised vectors and return cosine-similarity scores.

3. **Retrieve** (`rag/retrieve/`) — before the tunnels run, optional query-preprocessing stages (`rag/retrieve/query/`, all implementing the `QueryStage` ABC in `query/base.py`: `process(query) -> str`, applied in order by `Retriever.query_stages`) transform the query:
   - `SpellCorrector` (`query/spell.py`, on by default) — corpus-driven typo correction: query words absent from the store's vocabulary (rebuilt lazily on store-size change, like the corpus tunnels) are replaced by the closest vocab word within edit distance 1–2, preferring frequent words; only pure-alphabetic words of 4+ chars are touched, so ids/codes/novel terms survive.
   - `QueryRewriter` (`query/rewrite.py`, opt-in via `query_rewrite=True`) — LLM query understanding: one short Claude call (`claude-opus-4-8`, `effort: low`) rewrites the query for retrieval (typos, acronyms, synonyms), falling back to the original on refusal/empty output. Imported lazily in `pipeline.py`.

   Then multi-tunnel retrieval: each enabled `Channel` produces a ranked list and the lists are fused with Reciprocal Rank Fusion (`_rrf_fuse` in `retriever.py`, rank-based so per-tunnel score scales never need calibrating). Tunnels live in `rag/retrieve/retrieve_tunnel/` and all inherit the `RetrieveTunnel` ABC (`base.py`: `search(query, top_k) -> list[(Chunk, score)]` + `__len__`):
   - `DenseTunnel` (`dense.py`) — embeds the query, searches the vector store (semantic).
   - `BM25Tunnel` (`bm25.py`) — self-contained Okapi BM25, sparse keyword relevance.
   - `LexicalTunnel` (`lexical.py`) — exact phrase matching; longest contiguous query span found verbatim in a chunk (pure-stopword spans don't count).
   - `EntityTunnel` (`ner.py`) — rule-based NER (regex: ticket ids, codes, emails, URLs, versions, ISO dates, proper-noun spans) scored by IDF-weighted entity overlap; `extract_entities` is the swap point for a spaCy/transformer model.

   All tunnels except dense are corpus indexes built lazily from `store.chunks()` and rebuilt when the store size changes (works with pre-populated persisted stores); shared tokenizer/stopwords in `retrieve_tunnel/text.py`.

   Post-fusion rerank stages (`rag/retrieve/rerank/`) all inherit the `RerankStage` ABC (`rerank/base.py`: `rerank(query, candidates, top_k) -> list[(Chunk, score)]`); `Retriever` applies its enabled stages in order (`Retriever.stages`), each receiving the previous stage's output:
   - **MMR** (`rerank/mmr.py`, `MMRReranker`) — optional diversity-aware selection: greedily maximises `λ·relevance − (1−λ)·max-sim-to-selected` over fresh chunk embeddings; picks the final `top_k` from the fused pool. Enabled by `mmr_lambda` (None = off).
   - **Cross-encoder** (`rerank/cross_encoder.py`, `CrossEncoderReranker`) — optional local cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores candidates (after MMR it only re-orders the survivors).

   Tunnels over-fetch `top_k * candidate_multiplier` when any later stage will cut. Returns `list[tuple[Chunk, float]]`; score semantics depend on the last stage (cosine / RRF / MMR objective / cross-encoder logit).

4. **Generate** (`rag/generate/generator.py`) — passes retrieved chunks as context to Claude (`claude-opus-4-8`) via the Anthropic SDK with adaptive thinking and streaming.

All imports are absolute from the `rag` package root (e.g. `from rag.store.document import Chunk`). The project root must be on `sys.path` (running `python3 main.py` from `tiny_rag/` handles this automatically).

## Key knobs

| Parameter | Where | Default |
|---|---|---|
| `channels` | `RAGPipeline(channels=(Channel.DENSE, Channel.BM25))` | all four |
| `spell_correct` | `RAGPipeline(spell_correct=False)` — corpus-vocab query typo correction | `True` |
| `query_rewrite` | `RAGPipeline(query_rewrite=True)` — LLM query rewrite before retrieval | `False` |
| `contextualize` | `RAGPipeline(contextualize=True)` — contextual retrieval at ingest | `False` |
| `mmr_lambda` | `RAGPipeline(mmr_lambda=0.7)` — MMR diversity stage, None = off | `None` |
| `rerank` | `RAGPipeline(rerank=True)` — cross-encoder rerank stage | `False` |
| `rrf_k` | `RAGPipeline(rrf_k=60)` — RRF fusion constant | 60 |
| `candidate_multiplier` | `Retriever(candidate_multiplier=4)` — per-channel over-fetch | 4 |
| `backend` | `RAGPipeline(backend=StoreBackend.CHROMA)` | `FAISS` |
| `persist_dir` | `RAGPipeline(persist_dir="./chroma_db")` (Chroma only) | `./chroma_db` |
| `chroma_host` / `chroma_port` | `RAGPipeline(chroma_host="localhost")` — use a `chroma run` server | `None` / 8000 |
| `collection` | `RAGPipeline(collection="tiny_rag")` (Chroma only) | `tiny_rag` |
| `index_type` | `RAGPipeline(index_type=IndexType.HNSW)` (FAISS only) | `FLAT` |
| `hnsw_m` | `VectorStore(hnsw_m=32)` | 32 |
| `hnsw_ef_construction` | `VectorStore(hnsw_ef_construction=200)` | 200 |
| `hnsw_ef_search` | `VectorStore(hnsw_ef_search=64)` or `store.set_ef_search(n)` | 64 |
| `chunk_size` / `overlap` | `RAGPipeline(chunk_size=512, overlap=64)` | 512 / 64 |
| `top_k` | `RAGPipeline(top_k=5)` | 5 |
