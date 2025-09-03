"""
Tiny, dependency-free hashed embeddings (fixed dim).
"""
from typing import List
import numpy as np
import hashlib

class HashedEmbeddings:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in text.lower().split() if t]

    def _bucket(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(h, 16) % self.dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            tokens = self._tokenize(t)
            for tok in tokens:
                idx = self._bucket(tok)
                vecs[i, idx] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] = vecs[i] / norm
        return vecs
