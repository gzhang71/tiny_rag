"""Evaluation — measure retrieval and generation quality.

- `metrics.py` — Recall@K and NDCG@K over retrieved chunk/document ids
  (pure functions, no dependencies)
- `judge.py` (`LLMJudge`) — claim-level faithfulness, holistic groundedness,
  and hallucination rate via one structured-output Claude call per answer
- `evaluator.py` (`RAGEvaluator`) — runs a labelled `EvalExample` dataset
  through a `RAGPipeline` and aggregates everything into an `EvalReport`
  (retrieval metrics always; judge metrics only when a judge is passed)
"""
from rag.evaluation.evaluator import EvalExample, EvalReport, ExampleResult, RAGEvaluator
from rag.evaluation.metrics import ndcg_at_k, recall_at_k

__all__ = [
    "EvalExample",
    "EvalReport",
    "ExampleResult",
    "RAGEvaluator",
    "ndcg_at_k",
    "recall_at_k",
]
