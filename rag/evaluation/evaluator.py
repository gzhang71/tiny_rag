"""Evaluation harness — run a labelled dataset through the pipeline.

Each example carries a question and the ids of the chunks/documents that
should be retrieved (`source`, or `"source:chunk_index"` for chunk-level
judgements). Retrieval is scored with Recall@K and NDCG@K; when a judge is
supplied, the full pipeline answer is scored for faithfulness, groundedness,
and hallucination rate (one Claude call per example).

    evaluator = RAGEvaluator(pipeline, judge=LLMJudge())
    report = evaluator.evaluate([
        EvalExample("Who loves cats?", relevant=["pets.txt"]),
    ])
    print(report.summary())
"""
from dataclasses import dataclass, field

from rag.evaluation.judge import JudgeVerdict, LLMJudge
from rag.evaluation.metrics import ndcg_at_k, recall_at_k
from rag.store.document import Chunk


@dataclass
class EvalExample:
    question: str
    relevant: list[str]  # source names, or "source:chunk_index" for chunk-level


@dataclass
class ExampleResult:
    question: str
    recall: float
    ndcg: float
    retrieved: list[str]  # "source:chunk_index" of the top-k, in rank order
    answer: str | None = None
    verdict: JudgeVerdict | None = None


@dataclass
class EvalReport:
    top_k: int
    results: list[ExampleResult] = field(default_factory=list)

    def _mean(self, values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @property
    def recall_at_k(self) -> float | None:
        return self._mean([r.recall for r in self.results])

    @property
    def ndcg_at_k(self) -> float | None:
        return self._mean([r.ndcg for r in self.results])

    @property
    def faithfulness(self) -> float | None:
        return self._mean([r.verdict.faithfulness for r in self.results if r.verdict])

    @property
    def groundedness(self) -> float | None:
        return self._mean([r.verdict.groundedness for r in self.results if r.verdict])

    @property
    def hallucination_rate(self) -> float | None:
        return self._mean(
            [r.verdict.hallucination_rate for r in self.results if r.verdict]
        )

    def summary(self) -> str:
        lines = [f"Examples: {len(self.results)} (top_k={self.top_k})"]
        for name in (
            "recall_at_k", "ndcg_at_k", "faithfulness", "groundedness",
            "hallucination_rate",
        ):
            value = getattr(self, name)
            if value is not None:
                lines.append(f"{name}: {value:.3f}")
        return "\n".join(lines)


def _chunk_ids(chunk: Chunk) -> tuple[str, str]:
    return chunk.source, f"{chunk.source}:{chunk.chunk_index}"


class RAGEvaluator:
    def __init__(self, pipeline, judge: LLMJudge | None = None):
        self.pipeline = pipeline
        self.judge = judge

    def evaluate(self, dataset: list[EvalExample], top_k: int | None = None) -> EvalReport:
        top_k = top_k or self.pipeline.top_k
        report = EvalReport(top_k=top_k)
        for example in dataset:
            retrieved = self.pipeline.retriever.retrieve(example.question, top_k=top_k)
            relevant = set(example.relevant)
            # a retrieved chunk counts under whichever id form the label uses
            ranked_ids = [
                doc_id if doc_id in relevant else chunk_id
                for doc_id, chunk_id in (_chunk_ids(c) for c, _ in retrieved)
            ]
            result = ExampleResult(
                question=example.question,
                recall=recall_at_k(ranked_ids, relevant, k=top_k),
                ndcg=ndcg_at_k(ranked_ids, relevant, k=top_k),
                retrieved=[chunk_id for _, chunk_id in (_chunk_ids(c) for c, _ in retrieved)],
            )
            if self.judge is not None:
                chunks = self.pipeline.prepare_context(example.question, retrieved)
                result.answer = self.pipeline.generator.generate(example.question, chunks)
                context = "\n\n---\n\n".join(chunk.text for chunk, _ in chunks)
                result.verdict = self.judge.judge(
                    example.question, context, result.answer
                )
            report.results.append(result)
        return report
