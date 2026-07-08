# tiny_rag

A minimal RAG (Retrieval-Augmented Generation) system built with FAISS/Chroma, sentence-transformers, and Claude.

## Installation

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

Most of the system runs locally: embeddings (`all-MiniLM-L6-v2`), all four retrieval channels, spell correction, MMR, and the cross-encoder reranker need no network access. The API key is used for **answer generation** (every query) and the two opt-in Claude stages, `--rewrite` and `--contextualize`. The first run downloads the embedding model (~90 MB) from Hugging Face; `--rerank` additionally downloads the cross-encoder (~90 MB) on first use.

## Usage

```bash
# Ask a question over inline text
python main.py --text "Alice loves cats. Bob loves dogs." "Who loves cats?"

# Ask a question over a single file (.txt, .md, .pdf, .html, .docx)
python main.py --file path/to/report.pdf "What is the main topic?"

# Ask a question over a directory (all supported file types are picked up)
python main.py --dir path/to/docs/ "Summarize the key points"
```

### Supported formats

| Extension | Extractor | Structure |
|---|---|---|
| `.txt`, `.text` | plain read | none |
| `.md`, `.markdown`, `.rst` | plain read | native `#` headings |
| `.html`, `.htm` | stdlib parser (script/style stripped) | `<h1>`–`<h6>` → `#` headings |
| `.docx` | `python-docx` | `Heading N` styles → `#` headings |
| `.pdf` | `pypdf` | flat text, pages joined |

All extractors emit markdown-style `#` heading lines, which the chunker uses for structure-aware chunking — so an HTML page and the equivalent Markdown file chunk identically.

### What happens at ingest

1. **Clean** — unicode NFKC normalisation, unified newlines, control characters stripped, whitespace collapsed.
2. **Skip if unchanged** — the cleaned document is content-hashed; if that source was already ingested with the same hash, ingestion returns immediately (`Ingested 0 chunks`). With a persistent Chroma store this works across runs, so re-running `--dir` on a mostly-unchanged corpus only re-embeds the files that actually changed.
3. **Chunk** — the document is split into sections along headings; within each section, whole sentences (including CJK `。！？` boundaries) are packed into chunks of at most `--chunk-size` characters, with the trailing sentences (~`--overlap` chars) carried into the next chunk. Chunks are prefixed with their heading breadcrumb, e.g. `[Configuration > Timeouts] Retries default to three.`
4. **Filter noise** — chunks that are near-empty or mostly non-letters (page numbers, `--- 42 ---` separators, header/footer debris) are dropped.
5. **Deduplicate** — exact copies (order-insensitive token-set hash) and near-duplicates (64-bit SimHash within 8 bits — a passage with a few words changed) are dropped, including against chunks already in a persisted store.
6. **Contextualize (optional, `--contextualize`)** — Claude writes a 1-2 sentence context situating each chunk in its document ([contextual retrieval](https://www.anthropic.com/news/contextual-retrieval)), prepended to the chunk text so *every* retrieval channel benefits, not just the dense one. A chunk like "It rose by 3%." becomes findable for "ACME Q3 revenue". Costs one API call per chunk; the document is sent with a prompt-cache breakpoint, so calls after the first read it at ~10% of input price.
7. **Embed and store** — chunks are encoded (L2-normalised MiniLM vectors) and upserted into the vector store.

A warning is printed if `--chunk-size` exceeds what the embedding model can encode (~1024 characters for MiniLM's 256-token window): oversized chunks still ingest, but dense retrieval only "sees" each chunk's head, while BM25/lexical/entity still see all of it.

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

- **Spell correction** (on by default, `--no-spell` disables) — query typos are fixed against the ingested corpus vocabulary ("embeding" → "embedding"), so it never needs a dictionary and adapts to domain jargon automatically. Guardrails keep it conservative: only pure-alphabetic words of 4+ characters are candidates, words already in the corpus are never touched, and corrections stay within edit distance 1 (short words) or 2 (longer) — so ticket ids, version strings, and genuinely novel terms pass through untouched. Local, no API call.
- `--rewrite` — query understanding: Claude rewrites the query for retrieval (expands acronyms and ambiguous references, adds synonyms of key terms, fixes grammar) before the tunnels run, preserving identifiers verbatim. Falls back to the original query if the model declines. One extra API call per query.

```bash
python main.py --channels dense,bm25 --mmr --rerank --file doc.txt "What changed in v2.3?"
python main.py --rewrite --file doc.txt "hw does the ingest pipline work?"
```

### Which flags should I turn on?

| Situation | Reach for |
|---|---|
| Default: small corpus, quick answers | nothing — the four fused channels are already strong |
| Users type fast and make typos | nothing — spell correction is already on |
| Chunks in results feel repetitive / near-identical | `--mmr` (lower lambda → more diversity) |
| Top results are close but ordered badly | `--rerank` (local, adds ~a second) |
| Queries are terse, jargon-heavy, or full of acronyms | `--rewrite` (1 API call/query) |
| Chunks lose meaning out of context ("It rose 3%") | `--contextualize` at ingest (1 API call/chunk, biggest accuracy lever) |
| Corpus grows past ~10k chunks | FAISS `IndexType.HNSW` (Python API) or `--store chroma` |
| Same corpus queried repeatedly | `--store chroma` — ingest once, unchanged files skipped on re-ingest |

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

# everything on: diverse results, reranked, LLM query rewrite + contextual retrieval
pipeline = RAGPipeline(mmr_lambda=0.7, rerank=True, query_rewrite=True, contextualize=True)
pipeline.ingest_file("report.pdf")
print(pipeline.query("What drove the Q3 revenue change?"))

# retrieval without generation — returns [(Chunk, score), ...]
for chunk, score in pipeline.retriever.retrieve("timeout defaults", top_k=3):
    print(f"{score:.3f}  [{chunk.source}:{chunk.chunk_index}]  {chunk.text[:80]}")
```

### `RAGPipeline` parameters

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` / `overlap` | 512 / 64 | Chunk size and carried overlap, in characters |
| `top_k` | 5 | Chunks handed to the generator |
| `backend` | `StoreBackend.FAISS` | `FAISS` (in-memory) or `CHROMA` (persistent) |
| `channels` | all four | Retrieval channels to run and fuse |
| `rrf_k` | 60 | RRF fusion constant (higher = flatter rank weighting) |
| `mmr_lambda` | `None` (off) | MMR diversity stage; 0..1, higher = more relevance |
| `rerank` | `False` | Cross-encoder rerank stage (local model) |
| `spell_correct` | `True` | Corpus-vocabulary query spell correction |
| `query_rewrite` | `False` | LLM query rewrite before retrieval (1 call/query) |
| `contextualize` | `False` | Contextual retrieval at ingest (1 call/chunk) |
| `index_type` | `IndexType.FLAT` | FAISS only: `FLAT` exact or `HNSW` approximate |
| `persist_dir` | `./chroma_db` | Chroma only: embedded DB directory (`None` = in-memory) |
| `collection` | `tiny_rag` | Chroma only: collection name |
| `chroma_host` / `chroma_port` | `None` / 8000 | Chroma only: connect to a `chroma run` server |

`ingest_text(text, source)`, `ingest_file(path)`, and `ingest_directory(dir, glob=None)` all return the number of chunks actually stored — `0` means the content was unchanged or fully deduplicated, not that ingestion failed.

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

The dependency direction is one-way: `pipeline` → (`ingest`, `retrieve`, `generate`) → `store`. Tunnels never import each other; the retriever composes them. Swapping a piece (another embedding model, a new tunnel, a different vector DB) means implementing one small ABC — `RetrieveTunnel` for channels, `BaseVectorStore` for stores, `QueryStage` for query preprocessing, `RerankStage` for post-fusion stages.

## Extending

Each seam in the pipeline is a small ABC; add a class, wire it in, done.

**A new file format** — add a branch in `loader.py::load_file` and the extension to `SUPPORTED_SUFFIXES`. Emit markdown `#` heading lines wherever the format has sections and the structure-aware chunker picks them up for free.

**A new query-preprocessing stage** (e.g. acronym expansion from a project glossary):

```python
from rag.retrieve.query.base import QueryStage

class GlossaryExpander(QueryStage):
    def __init__(self, glossary: dict[str, str]):
        self.glossary = glossary

    def process(self, query: str) -> str:
        for term, expansion in self.glossary.items():
            query = query.replace(term, f"{term} ({expansion})")
        return query

# pass to the retriever (order matters — stages run left to right)
pipeline.retriever.query_stages = (*pipeline.retriever.query_stages, GlossaryExpander({...}))
```

**A new retrieval tunnel** — subclass `RetrieveTunnel` (`search(query, top_k) -> list[(Chunk, score)]` + `__len__`). Scores only need to be self-consistent: fusion is rank-based, so they're never compared across tunnels.

**A new rerank stage** — subclass `RerankStage` (`rerank(query, candidates, top_k)`) and append it to `pipeline.retriever.stages`; each stage receives the previous one's output.

**A better NER model** — `rag/retrieve/retrieve_tunnel/ner.py::extract_entities` is the single swap point; replace the regex rules with a spaCy or transformer NER and the entity tunnel picks it up unchanged.

**A different embedding model** — `Embedder(model_name="...")` accepts any sentence-transformers model; the stores read the dimension from it. For multilingual corpora, e.g. `paraphrase-multilingual-MiniLM-L12-v2`.

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
