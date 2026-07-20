"""LLM judge for generation quality — faithfulness, groundedness, hallucination.

One Claude call per (question, context, answer) triple, with a structured
output schema so the verdict always parses. The judge decomposes the answer
into atomic factual claims and labels each against the context:

- `supported`       — the context states or directly entails the claim
- `not_in_context`  — the context neither supports nor contradicts it
- `contradicted`    — the context says otherwise

From those labels: faithfulness = supported / total claims (the RAGAS
definition), hallucination rate = 1 - faithfulness. Groundedness is the
judge's holistic 0-1 score of how well the answer as a whole stays within
the context, including whether it admits when the context is insufficient.
"""
from dataclasses import dataclass

import anthropic

_MODEL = "claude-opus-4-8"

_SYSTEM = (
    "You are an evaluation judge for a retrieval-augmented generation system. "
    "You receive a question, the retrieved context passages, and the answer a "
    "model generated from them. Decompose the answer into its atomic factual "
    "claims (skip hedges, meta-commentary, and restatements of the question) "
    "and label each claim strictly against the provided context — not against "
    "your own knowledge: 'supported' if the context states or directly entails "
    "it, 'contradicted' if the context says otherwise, 'not_in_context' if the "
    "context is silent on it. Also give a holistic groundedness score from 0.0 "
    "to 1.0 for how well the answer stays within the context, rewarding "
    "answers that admit when the context is insufficient."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "not_in_context", "contradicted"],
                    },
                },
                "required": ["claim", "verdict"],
                "additionalProperties": False,
            },
        },
        "groundedness": {
            "type": "number",
            "description": "0.0 (entirely ungrounded) to 1.0 (fully grounded)",
        },
    },
    "required": ["claims", "groundedness"],
    "additionalProperties": False,
}


@dataclass
class JudgeVerdict:
    faithfulness: float       # supported claims / total claims
    groundedness: float       # judge's holistic 0-1 score
    hallucination_rate: float  # 1 - faithfulness
    claims: list[tuple[str, str]]  # (claim, verdict)


class LLMJudge:
    def __init__(self, model: str = _MODEL):
        self.client = anthropic.Anthropic()
        self.model = model

    def judge(self, question: str, context: str, answer: str) -> JudgeVerdict | None:
        """Returns None when the judge call is refused."""
        import json

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
            system=_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"<question>\n{question}\n</question>\n\n"
                        f"<context>\n{context}\n</context>\n\n"
                        f"<answer>\n{answer}\n</answer>"
                    ),
                }
            ],
        )
        if response.stop_reason == "refusal":
            return None
        text = next(block.text for block in response.content if block.type == "text")
        data = json.loads(text)
        claims = [(c["claim"], c["verdict"]) for c in data["claims"]]
        supported = sum(1 for _, verdict in claims if verdict == "supported")
        faithfulness = supported / len(claims) if claims else 1.0
        groundedness = min(1.0, max(0.0, float(data["groundedness"])))
        return JudgeVerdict(
            faithfulness=faithfulness,
            groundedness=groundedness,
            hallucination_rate=1.0 - faithfulness,
            claims=claims,
        )
