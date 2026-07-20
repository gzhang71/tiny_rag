"""Query processing — transforms the query before the retrieval tunnels run.

Two kinds of stage, both defined in `base.py`:

`QueryStage` (`process(query) -> str`, applied in order when enabled):
1. Spell correction (`spell.py`, `SpellCorrector`) — corpus-vocabulary typo fixes
2. LLM rewrite (`rewrite.py`, `QueryRewriter`) — query understanding via Claude

`QueryExpander` (`expand(query) -> list[str]`, each result retrieved
independently and RRF-fused):
3. Decomposition (`decompose.py`, `QueryDecomposer`) — split multi-hop
   questions into self-contained sub-questions via Claude
4. HyDE (`hyde.py`, `HyDEExpander`) — retrieve with a Claude-written
   hypothetical answer passage alongside the query

`rewrite.py`, `decompose.py`, and `hyde.py` are imported lazily by the
pipeline so the rest of the package works without the `anthropic` SDK on the
import path at stage-build time.
"""
from rag.query_processing.base import QueryExpander, QueryStage
from rag.query_processing.spell import SpellCorrector

__all__ = ["QueryStage", "QueryExpander", "SpellCorrector"]
