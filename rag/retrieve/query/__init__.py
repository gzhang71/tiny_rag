"""Query-preprocessing stages. All stages implement the `QueryStage` ABC
(`base.py`) and are applied in this order when enabled:

1. Spell correction (`spell.py`, `SpellCorrector`) — corpus-vocabulary typo fixes
2. LLM rewrite (`rewrite.py`, `QueryRewriter`) — query understanding via Claude

`rewrite.py` is imported lazily by the pipeline so the rest of the package
works without the `anthropic` SDK on the import path at query-stage-build time.
"""
from rag.retrieve.query.base import QueryStage
from rag.retrieve.query.spell import SpellCorrector

__all__ = ["QueryStage", "SpellCorrector"]
