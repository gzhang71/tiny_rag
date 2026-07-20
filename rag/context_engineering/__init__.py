"""Context engineering — shapes retrieved chunks into the generator's context.

Stages implement the `ContextStage` ABC (`base.py`: `apply(query, chunks) ->
chunks`) and run between retrieval and generation, each receiving the
previous stage's output:

1. Compression (`compression.py`, `ContextCompressor`) — extractive: drops
   the sentences of each chunk least relevant to the query (local embeddings,
   no API call)
2. Packing (`packing.py`, `ContextPacker`) — drops query-time near-duplicate
   chunks, fits the survivors into a character budget by score, and re-orders
   them into document order for coherent reading

Both stages return copies; the store's chunk objects are never mutated.
"""
from rag.context_engineering.base import ContextStage
from rag.context_engineering.compression import ContextCompressor
from rag.context_engineering.packing import ContextPacker

__all__ = ["ContextStage", "ContextCompressor", "ContextPacker"]
