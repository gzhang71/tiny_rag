"""
tiny_rag CLI

Usage:
    python main.py --file path/to/doc.txt "What is the main topic?"
    python main.py --dir path/to/docs/ "Summarize the key points"
    python main.py --text "Alice loves cats. Bob loves dogs." "Who loves cats?"

    # persistent vector DB (Chroma): ingest once, then query without a source
    python main.py --store chroma --file path/to/doc.txt "What is the main topic?"
    python main.py --store chroma "What else does it say?"
"""
import argparse
import sys

from rag import Channel, DEFAULT_CHANNELS, RAGPipeline, StoreBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny RAG — ask questions over your documents")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", metavar="PATH",
                        help="Ingest a single file (.txt/.md/.pdf/.html/.docx)")
    source.add_argument("--dir", metavar="DIR",
                        help="Ingest all supported files in a directory")
    source.add_argument("--text", metavar="TEXT", help="Ingest raw text inline")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument(
        "--store", choices=[b.value for b in StoreBackend], default=StoreBackend.FAISS.value,
        help="Vector store backend: faiss (in-memory) or chroma (persistent DB) (default: faiss)",
    )
    parser.add_argument(
        "--persist-dir", default="./chroma_db",
        help="Directory for the Chroma DB when --store chroma (default: ./chroma_db)",
    )
    parser.add_argument(
        "--chroma-host", default=None,
        help="Connect to a running Chroma server (`chroma run`) instead of the embedded DB",
    )
    parser.add_argument("--chroma-port", type=int, default=8000)
    parser.add_argument(
        "--channels", default=",".join(c.value for c in DEFAULT_CHANNELS),
        help="Comma-separated retrieval channels, RRF-fused: "
             f"{', '.join(c.value for c in Channel)} (default: all)",
    )
    parser.add_argument(
        "--mmr", nargs="?", type=float, const=0.7, default=None, metavar="LAMBDA",
        help="Diversify results with MMR; optional lambda in [0,1], higher = more "
             "relevance, lower = more diversity (default when flag given: 0.7)",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Rerank candidates with a local cross-encoder before generation",
    )
    parser.add_argument(
        "--no-spell", dest="spell", action="store_false",
        help="Disable query spell correction against the corpus vocabulary (on by default)",
    )
    parser.add_argument(
        "--rewrite", action="store_true",
        help="Rewrite the query with Claude before retrieval (query understanding; "
             "one extra API call per query)",
    )
    parser.add_argument(
        "--decompose", action="store_true",
        help="Decompose multi-hop questions into sub-questions with Claude; each is "
             "retrieved separately and RRF-fused (one extra API call per query)",
    )
    parser.add_argument(
        "--hyde", action="store_true",
        help="HyDE: retrieve with a Claude-written hypothetical answer passage "
             "alongside the query (one extra API call per query)",
    )
    parser.add_argument(
        "--compress", action="store_true",
        help="Compress retrieved chunks to their query-relevant sentences before "
             "generation (local embeddings, no API call)",
    )
    parser.add_argument(
        "--context-budget", type=int, default=None, metavar="CHARS",
        help="Pack the generator context to at most CHARS characters, dropping "
             "near-duplicate chunks and re-ordering into document order",
    )
    parser.add_argument(
        "--contextualize", action="store_true",
        help="Contextual retrieval: prepend a Claude-written 1-2 sentence document "
             "context to each chunk at ingest (one API call per chunk)",
    )
    args = parser.parse_args()

    try:
        channels = tuple(Channel(name.strip()) for name in args.channels.split(",") if name.strip())
    except ValueError:
        parser.error(f"invalid --channels value {args.channels!r}; "
                     f"choose from: {', '.join(c.value for c in Channel)}")
    if not channels:
        parser.error("--channels must name at least one channel")

    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
        backend=StoreBackend(args.store),
        channels=channels,
        mmr_lambda=args.mmr,
        rerank=args.rerank,
        spell_correct=args.spell,
        query_rewrite=args.rewrite,
        decompose=args.decompose,
        hyde=args.hyde,
        compress=args.compress,
        context_budget=args.context_budget,
        contextualize=args.contextualize,
        persist_dir=args.persist_dir,
        chroma_host=args.chroma_host,
        chroma_port=args.chroma_port,
    )

    if args.file:
        n = pipeline.ingest_file(args.file)
        print(f"Ingested {n} chunks from {args.file}", file=sys.stderr)
    elif args.dir:
        n = pipeline.ingest_directory(args.dir)
        print(f"Ingested {n} chunks from {args.dir}", file=sys.stderr)
    elif args.text:
        n = pipeline.ingest_text(args.text, source="cli-inline")
        print(f"Ingested {n} chunks from inline text", file=sys.stderr)
    elif len(pipeline.store) == 0:
        parser.error("the store is empty — provide a source (--file, --dir, or --text)")
    else:
        print(f"Querying existing store ({len(pipeline.store)} chunks)", file=sys.stderr)

    answer = pipeline.query(args.question)
    print(answer)


if __name__ == "__main__":
    main()
