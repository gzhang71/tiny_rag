import numpy as np
from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    @property
    def max_seq_length(self) -> int:
        """Token window of the encoder — input beyond it is silently truncated."""
        return int(self.model.max_seq_length)

    @property
    def max_chars(self) -> int:
        """Rough character capacity of the encoder window (~4 chars/token)."""
        return self.max_seq_length * 4
