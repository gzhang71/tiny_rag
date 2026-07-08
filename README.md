# tiny_rag

A minimal RAG (Retrieval-Augmented Generation) system built with FAISS/Chroma, sentence-transformers, and Claude.

## Installation

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

## Usage

```bash
# Ask a question over inline text
python main.py --text "Alice loves cats. Bob loves dogs." "Who loves cats?"

# Ask a question over a single file (.txt, .md, .pdf, .html, .docx)
python main.py --file path/to/report.pdf "What is the main topic?"

# Ask a question over a directory (all supported file types are picked up)
python main.py --dir path/to/docs/ "Summarize the key points"
```

Documents are cleaned (unicode/whitespace normalisation), split into sentence-aware chunks along section headings (each chunk keeps its heading breadcrumb, e.g. `[Configuration > Timeouts]`), deduplicated — exact and near-duplicate (SimHash) — and skipped entirely if a source's content hasn't changed since its last ingest. With `--contextualize`, Claude additionally writes a 1-2 sentence situating context for every chunk at ingest ([contextual retrieval](https://www.anthropic.com/news/contextual-retrieval)) — the strongest retrieval-accuracy lever, at the cost of one API call per chunk (the document is prompt-cached across calls).

### Persistent vector DB (Chroma)

With `--store chroma`, embeddings are stored in a local Chroma database (default `./chroma_db`), so you ingest once and query as many times as you like:

```bash
# ingest once (re-running upserts — no duplicates)
python main.py --store chroma --dir path/to/docs/ "Summarize the key points"

# later: query without re-ingesting
python main.py --store chroma "What are the key risks mentioned?"
```

To run Chroma as a standalone server on your Mac instead of embedded mode:

```bash
chroma run --path ./chroma_data          # in one terminal
python main.py --store chroma --chroma-host localhost "Your question"
```

### Retrieval channels

Retrieval runs through several parallel channels whose ranked lists are merged with Reciprocal Rank Fusion (RRF):

| Channel | Kind | What it catches |
|---|---|---|
| `dense` | semantic | paraphrases, related concepts (vector similarity) |
| `bm25` | sparse lexical | keyword relevance (Okapi BM25) |
| `lexical` | exact match | verbatim phrases from the query |
| `entity` | NER | shared named entities: ticket ids, codes, dates, names, URLs |

All four are on by default; pick a subset with `--channels`. Two optional post-fusion stages:

- `--mmr [LAMBDA]` — MMR diversity selection: avoids filling the top-k with near-duplicate chunks; lambda in [0,1] trades relevance (1.0) against diversity (default 0.7).
- `--rerank` — re-score candidates with a local cross-encoder (most accurate, a bit slower).

And two query-preprocessing stages that run before the tunnels:

- **Spell correction** (on by default, `--no-spell` disables) — query typos are fixed against the ingested corpus vocabulary ("embeding" → "embedding"); identifiers and novel terms are left alone. Local, no API call.
- `--rewrite` — query understanding: Claude rewrites the query for retrieval (expands acronyms, adds synonyms) before the tunnels run. One extra API call per query.

```bash
python main.py --channels dense,bm25 --mmr --rerank --file doc.txt "What changed in v2.3?"
python main.py --rewrite --file doc.txt "hw does the ingest pipline work?"
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--top-k` | 5 | Number of chunks to retrieve |
| `--chunk-size` | 512 | Characters per chunk |
| `--overlap` | 64 | Overlap between chunks |
| `--store` | `faiss` | Vector store: `faiss` (in-memory) or `chroma` (persistent DB) |
| `--persist-dir` | `./chroma_db` | Chroma DB directory (embedded mode) |
| `--chroma-host` / `--chroma-port` | — / 8000 | Connect to a running `chroma run` server |
| `--channels` | all four | Comma-separated retrieval channels to fuse |
| `--mmr [LAMBDA]` | off | MMR diversity selection after fusion (lambda default 0.7) |
| `--rerank` | off | Cross-encoder rerank stage after fusion/MMR |
| `--no-spell` | spell on | Disable corpus-vocabulary query spell correction |
| `--rewrite` | off | LLM query rewrite (query understanding) before retrieval |
| `--contextualize` | off | Contextual retrieval: LLM-written chunk context at ingest |

## Python API

```python
from rag import RAGPipeline, IndexType

# Default: exact flat index, good for small corpora
pipeline = RAGPipeline()
pipeline.ingest_file("doc.txt")
print(pipeline.query("What is this about?"))

# HNSW: approximate index, sub-linear search for large corpora (10k+ chunks)
pipeline = RAGPipeline(index_type=IndexType.HNSW)
pipeline.ingest_directory("./docs/")
print(pipeline.query("What are the key themes?"))

# Chroma: persistent vector DB — survives restarts, ingest once / query forever
from rag import StoreBackend
pipeline = RAGPipeline(backend=StoreBackend.CHROMA, persist_dir="./chroma_db")
pipeline.ingest_directory("./docs/")   # only needed the first time
print(pipeline.query("What are the key themes?"))

# choose retrieval channels and add cross-encoder reranking
from rag import Channel
pipeline = RAGPipeline(channels=(Channel.DENSE, Channel.BM25), rerank=True)
```

## Workflow

Two phases: ingestion builds the indexes, querying runs a question through the retrieval tunnels and hands the winners to Claude.

```
INGEST                                QUERY
──────                                ─────
 --file / --dir / --text               question
        │                                  │
        ▼                                  ▼
   loader extracts text               spell correction (default; --no-spell)
   (txt/md/pdf/html/docx,             corpus-vocab typo fixes
    headings preserved)                    │
        │                             LLM query rewrite (--rewrite, optional)
        ▼                             acronyms, synonyms, grammar
   cleaner normalises text                 │
   (unicode, whitespace,                   │
    control chars)                         │
        │                                  ├─────────────┬─────────────┬─────────────┐
        ▼                                  ▼             ▼             ▼             ▼
   chunker packs whole                DenseTunnel    BM25Tunnel   LexicalTunnel  EntityTunnel
   sentences into section-            (embed query,  (keyword      (exact phrase (named-entity
   aware overlapping chunks;           vector store    relevance)    spans)        overlap)
   dupes + noise dropped,              search)            │             │             │
   unchanged sources skipped               │              │             │             │
        │                                  └──────┬───────┴─────┬───────┴─────────────┘
        ▼                                         ▼             each tunnel over-fetches
   contextualizer prepends               RRF fusion            top_k × candidate_multiplier
   LLM chunk context                     (rank-based merge + dedup)
   (--contextualize, optional)                    │
        │                                MMR selection (--mmr, optional)
        ▼                                diversity-aware top-k pick
   embedder encodes chunks                        │
   (MiniLM, L2-normalised)               cross-encoder rerank (--rerank, optional)
        │                                joint (query, chunk) scoring
        ▼                                         │
   vector store add/upsert                        ▼
   (FAISS in-memory, or                  top-k chunks + question → Claude → streamed answer
    Chroma persisted to disk)
```

Step by step for a query:

0. **The query is preprocessed.** Spell correction (on by default) fixes typos against the corpus vocabulary without touching identifiers or novel terms; with `--rewrite`, Claude then rewrites the query for retrieval — expanding acronyms and adding synonyms of key terms.
1. **Every enabled tunnel searches independently.** Dense embeds the query and searches the vector store; BM25, lexical, and entity search corpus indexes built lazily from the store's chunks (so they also work against a pre-populated Chroma DB). Each returns a ranked `(chunk, score)` list, over-fetching 4× `top_k` when a later stage will cut.
2. **Reciprocal Rank Fusion merges the lists.** RRF scores by rank position only (`Σ 1/(k + rank)`), so the tunnels' incomparable score scales — cosine, BM25, span length, entity IDF — never need calibrating. Duplicates found by several tunnels rise to the top.
3. **MMR (optional)** greedily picks a top-k that trades relevance against redundancy, so the context window isn't spent on near-duplicate chunks.
4. **Cross-encoder rerank (optional)** re-scores each surviving (query, chunk) pair jointly for the final ordering — the most accurate stage, kept cheap by running only on the candidate pool.
5. **Generation**: the top-k chunks are formatted with source/score attribution into the context of a Claude request, and the answer streams back.

## Repository architecture

```
tiny_rag/
├── main.py                        # CLI entry point (argparse → RAGPipeline)
├── requirements.txt
└── rag/
    ├── pipeline.py                # RAGPipeline — the single public entry point, wires all stages
    ├── ingest/
    │   ├── loader.py              # per-format extraction: txt/md/pdf/html/docx, headings preserved
    │   ├── cleaner.py             # normalisation, quality filter, exact + SimHash dedup, doc hashing
    │   ├── chunker.py             # section- and sentence-aware overlapping chunks
    │   ├── contextualizer.py      # contextual retrieval: LLM-written chunk context (opt-in)
    │   └── embedder.py            # sentence-transformers (all-MiniLM-L6-v2), L2-normalised vectors
    ├── store/
    │   ├── base.py                # BaseVectorStore ABC + StoreBackend enum
    │   ├── document.py            # Chunk dataclass (text, source, chunk_index, metadata)
    │   ├── vector_store.py        # FAISS backend (FLAT exact / HNSW approximate), in-memory
    │   └── chroma_store.py        # Chroma backend: embedded-persistent, server, or ephemeral
    ├── retrieve/
    │   ├── retriever.py           # orchestrator: query stages, tunnels, RRF fusion, rerank stages
    │   ├── query/                 # query preprocessing, applied before the tunnels
    │   │   ├── base.py            # QueryStage ABC — all stages inherit this
    │   │   ├── spell.py           # corpus-vocabulary spell correction (local)
    │   │   └── rewrite.py         # LLM query rewriting via Claude (opt-in)
    │   ├── retrieve_tunnel/       # the parallel retrieval tunnels
    │   │   ├── base.py            # RetrieveTunnel ABC — all tunnels inherit this
    │   │   ├── dense.py           # DenseTunnel: query embedding → vector store search
    │   │   ├── bm25.py            # BM25Tunnel: Okapi BM25 keyword relevance
    │   │   ├── lexical.py         # LexicalTunnel: exact phrase/span matching
    │   │   ├── ner.py             # EntityTunnel: rule-based NER + IDF-weighted entity overlap
    │   │   └── text.py            # shared tokenizer + stopwords
    │   └── rerank/                # post-fusion stages, applied in order
    │       ├── base.py            # RerankStage ABC — all stages inherit this
    │       ├── mmr.py             # MMR diversity selection
    │       └── cross_encoder.py   # local cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
    └── generate/
        └── generator.py           # Claude (claude-opus-4-8) with streaming, cited context
```

The dependency direction is one-way: `pipeline` → (`ingest`, `retrieve`, `generate`) → `store`. Tunnels never import each other; the retriever composes them. Swapping a piece (another embedding model, a new tunnel, a different vector DB) means implementing one small ABC — `RetrieveTunnel` for channels, `BaseVectorStore` for stores.

## Architecture notes

**Vector store** — two backends behind a common interface (`BaseVectorStore`):
- `faiss` (default) — in-process, in-memory. `FLAT` gives exact cosine search; switch to `HNSW` for large corpora to get sub-linear query time at the cost of approximate results.
- `chroma` — a real vector DB that runs entirely on your machine: embedded and persisted to a local directory by default, or as a standalone server via `chroma run`. Uses cosine space and stable `source:chunk_index` ids, so re-ingesting a document updates it in place.

Both backends consume the same L2-normalised embeddings and return cosine-similarity scores, so results are comparable across backends.

**Ingest** — `.txt`, `.md`, `.pdf`, `.html`, and `.docx` are extracted to text with headings preserved as markdown `#` lines. Documents are normalised (unicode, whitespace, control characters), split along section headings and sentence boundaries (chunks never cut mid-sentence and carry their heading breadcrumb, e.g. `[Configuration > Timeouts]`), quality-filtered (page numbers, separator lines), and deduplicated — exact copies by token-set signature, near-duplicates by SimHash. A source whose content hash is unchanged since its last ingest is skipped entirely (hashes and dedup state persist with the Chroma store). Chunks are capped by a warning when `chunk_size` exceeds the embedding model's token window. Optional contextual retrieval (`--contextualize`) prepends a Claude-written 1-2 sentence situating context to every chunk — the document is prompt-cached, so per-chunk calls read it at ~10% input price.

**Query preprocessing** — spell correction (on by default) fixes query typos against the corpus's own vocabulary — local, dependency-free, and safe for identifiers; optional LLM query rewriting (`--rewrite`) has Claude reformulate the query (acronyms, synonyms, grammar) before retrieval.

**Retrieval** — multi-tunnel: dense vectors for meaning, BM25 for keywords, exact phrase match for verbatim quotes, and entity overlap (rule-based NER) for ids/names/dates. Ranked lists are merged with Reciprocal Rank Fusion, which is rank-based so the tunnels' different score scales never need calibrating. Optional Maximal Marginal Relevance selection then picks a top-k that covers different aspects instead of near-duplicates, and an optional cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`, local) re-scores for the final ordering.

**Embeddings** — `all-MiniLM-L6-v2` via sentence-transformers (runs locally, no API key needed).

**Generation** — `claude-opus-4-8` with adaptive thinking via the Anthropic SDK.
