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

# query processing: spell correction is on by default (--no-spell disables);
# --rewrite adds an LLM query-understanding pass, --decompose splits multi-hop
# questions into sub-questions, --hyde retrieves with an LLM-written
# hypothetical answer passage (each: one Claude call per query)
python3 main.py --rewrite --file doc.txt "hw does the pipline work?"
python3 main.py --decompose --hyde --file doc.txt "How do dedupe and Chroma interact?"

# context engineering (local, no API call): --compress cuts chunks to their
# query-relevant sentences, --context-budget packs into a char budget
python3 main.py --compress --context-budget 4000 --dir docs/ "Summarize the risks"

# .pdf / .html / .docx / .md are supported; --contextualize adds contextual
# retrieval (a Claude-written situating context per chunk, one call per chunk)
python3 main.py --contextualize --file report.pdf "What were the Q3 numbers?"
```

## Architecture

`RAGPipeline` (in `rag/pipeline.py`) is the single public entry point. It wires together the pipeline stages; `rag/evaluation/` sits on top and drives the pipeline for measurement:

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

3. **Query processing** (`rag/query_processing/`) — transforms the query before the tunnels run. Two ABCs in `base.py`:

   `QueryStage` (`process(query) -> str`, applied in order by `Retriever.query_stages`):
   - `SpellCorrector` (`spell.py`, on by default) — corpus-driven typo correction: query words absent from the store's vocabulary (rebuilt lazily on store-size change, like the corpus tunnels) are replaced by the closest vocab word within edit distance 1–2, preferring frequent words; only pure-alphabetic words of 4+ chars are touched, so ids/codes/novel terms survive.
   - `QueryRewriter` (`rewrite.py`, opt-in via `query_rewrite=True`) — LLM query understanding: one short Claude call (`claude-opus-4-8`, `effort: low`) rewrites the query for retrieval (typos, acronyms, synonyms), falling back to the original on refusal/empty output. Imported lazily in `pipeline.py`.

   `QueryExpander` (`expand(query) -> list[str]`, applied by `Retriever.query_expanders`; every resulting query runs through every tunnel and all ranked lists are RRF-fused together):
   - `QueryDecomposer` (`decompose.py`, opt-in via `decompose=True`) — one Claude call splits a multi-hop question into ≤4 self-contained sub-questions (replies `ATOMIC` for single-need questions → no expansion); the original query stays in the pool. Imported lazily.
   - `HyDEExpander` (`hyde.py`, opt-in via `hyde=True`) — HyDE: one Claude call writes a short hypothetical answer passage, retrieved alongside the original query (its embedding lands near real answer passages; its vocabulary feeds the sparse tunnels). Imported lazily.

4. **Retrieval** (`rag/retrieval/`) — multi-tunnel hybrid search: each enabled `Channel` produces a ranked list per query and the lists are fused with Reciprocal Rank Fusion (`_rrf_fuse` in `retriever.py`, rank-based so per-tunnel score scales never need calibrating). All tunnels inherit the `RetrieveTunnel` ABC (`base.py`: `search(query, top_k) -> list[(Chunk, score)]` + `__len__`):
   - `DenseTunnel` (`dense.py`) — embeds the query, searches the vector store (semantic).
   - `BM25Tunnel` (`bm25.py`) — self-contained Okapi BM25, sparse keyword relevance.
   - `LexicalTunnel` (`lexical.py`) — exact phrase matching; longest contiguous query span found verbatim in a chunk (pure-stopword spans don't count).
   - `EntityTunnel` (`ner.py`) — rule-based NER (regex: ticket ids, codes, emails, URLs, versions, ISO dates, proper-noun spans) scored by IDF-weighted entity overlap; `extract_entities` is the swap point for a spaCy/transformer model.

   All tunnels except dense are corpus indexes built lazily from `store.chunks()` and rebuilt when the store size changes (works with pre-populated persisted stores); shared tokenizer/stopwords in `rag/retrieval/text.py`. `Retriever` (`retriever.py`) orchestrates stages → expanders → tunnels → fusion → rerank; tunnels over-fetch `top_k * candidate_multiplier` when fusion, multiple queries, or a later stage will cut. Returns `list[tuple[Chunk, float]]`; score semantics depend on the last stage (cosine / RRF / MMR objective / cross-encoder logit).

5. **Reranking** (`rag/reranking/`) — post-fusion stages, all inheriting the `RerankStage` ABC (`base.py`: `rerank(query, candidates, top_k) -> list[(Chunk, score)]`); `Retriever` applies its enabled stages in order (`Retriever.stages`), each receiving the previous stage's output, with the user's (processed, un-expanded) query:
   - **MMR** (`mmr.py`, `MMRReranker`) — optional diversity-aware selection: greedily maximises `λ·relevance − (1−λ)·max-sim-to-selected` over fresh chunk embeddings; picks the final `top_k` from the fused pool. Enabled by `mmr_lambda` (None = off).
   - **Cross-encoder** (`cross_encoder.py`, `CrossEncoderReranker`) — optional local cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores candidates (after MMR it only re-orders the survivors).

6. **Context engineering** (`rag/context_engineering/`) — shapes the retrieved chunks between retrieval and generation. Stages inherit the `ContextStage` ABC (`base.py`: `apply(query, chunks) -> chunks`), applied in order by `RAGPipeline.prepare_context` (`pipeline.context_stages`); stages return copies and never mutate the store's chunk objects:
   - `ContextCompressor` (`compression.py`, opt-in via `compress=True`) — extractive compression: splits each chunk into sentences, scores them against the query with the local embedder (one batched encode, no API call), keeps the top `keep` fraction (default 0.6) in original order; chunks under 3 sentences pass through; original length recorded in `metadata["compressed_from_chars"]`.
   - `ContextPacker` (`packing.py`, opt-in via `context_budget=N`) — drops query-time near-duplicates (token-set Jaccard ≥ 0.85), greedily fits the best-scoring survivors into the char budget, then re-orders into `(source, chunk_index)` document order.

7. **Generate** (`rag/generate/generator.py`) — passes the prepared chunks as context to Claude (`claude-opus-4-8`) via the Anthropic SDK with adaptive thinking and streaming.

**Evaluation** (`rag/evaluation/`) — not a pipeline stage; drives the pipeline over a labelled dataset:
   - `metrics.py` — `recall_at_k` / `ndcg_at_k`, pure functions over id sequences (ids are `source` or `"source:chunk_index"`; NDCG accepts a set for binary or a mapping for graded relevance).
   - `judge.py` (`LLMJudge`) — one structured-output Claude call per (question, context, answer): decomposes the answer into atomic claims labelled supported / not_in_context / contradicted plus a holistic 0-1 groundedness score. Faithfulness = supported/total (RAGAS definition); hallucination rate = 1 − faithfulness. Returns `None` on refusal.
   - `evaluator.py` (`RAGEvaluator(pipeline, judge=None).evaluate(dataset)`) — takes `EvalExample(question, relevant=[ids])`; always computes Recall@K/NDCG@K from `pipeline.retriever.retrieve`; with a judge it also runs `prepare_context` + the generator and judges the answer. Aggregates into `EvalReport` (means + `summary()`; per-example detail in `.results`).

All imports are absolute from the `rag` package root (e.g. `from rag.store.document import Chunk`). The project root must be on `sys.path` (running `python3 main.py` from `tiny_rag/` handles this automatically).

## Key knobs

| Parameter | Where | Default |
|---|---|---|
| `channels` | `RAGPipeline(channels=(Channel.DENSE, Channel.BM25))` | all four |
| `spell_correct` | `RAGPipeline(spell_correct=False)` — corpus-vocab query typo correction | `True` |
| `query_rewrite` | `RAGPipeline(query_rewrite=True)` — LLM query rewrite before retrieval | `False` |
| `decompose` | `RAGPipeline(decompose=True)` — LLM multi-hop decomposition into sub-questions | `False` |
| `hyde` | `RAGPipeline(hyde=True)` — HyDE hypothetical answer passage | `False` |
| `compress` | `RAGPipeline(compress=True)` — extractive sentence compression (local) | `False` |
| `context_budget` | `RAGPipeline(context_budget=4000)` — char budget for packed context, None = off | `None` |
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
