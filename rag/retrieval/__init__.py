"""Retrieval — the parallel tunnels a query runs through, and the Retriever
that fuses them.

Every tunnel inherits `RetrieveTunnel` (`base.py`): `search(query, top_k)`
returning ranked (chunk, score) pairs, and `__len__` reporting how many chunks
it has indexed. `DenseTunnel` searches the vector store; the others are
corpus indexes built from the store's chunks. `Retriever` (`retriever.py`)
runs the enabled tunnels (hybrid search) and fuses their rankings with RRF.
"""
from rag.retrieval.base import RetrieveTunnel
from rag.retrieval.bm25 import BM25Tunnel
from rag.retrieval.dense import DenseTunnel
from rag.retrieval.lexical import LexicalTunnel
from rag.retrieval.ner import EntityTunnel
from rag.retrieval.retriever import Channel, DEFAULT_CHANNELS, Retriever

__all__ = [
    "RetrieveTunnel",
    "DenseTunnel",
    "BM25Tunnel",
    "LexicalTunnel",
    "EntityTunnel",
    "Retriever",
    "Channel",
    "DEFAULT_CHANNELS",
]
