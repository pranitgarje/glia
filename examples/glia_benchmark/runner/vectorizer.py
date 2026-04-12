"""
runner/vectorizer.py
────────────────────
Local embedding provider for the benchmark.

Uses sentence-transformers all-MiniLM-L6-v2 (384-dim) for fast, free,
offline embeddings — no OpenAI key required.  Falls back to a deterministic
random projection if the model can't be loaded (e.g. in a constrained CI env).
"""
from __future__ import annotations

import hashlib
import struct
from typing import List

DIMS = 384


class BenchmarkVectorizer:
    """
    sentence-transformers wrapper that satisfies GliaManager's vectorizer
    contract: embed(text) -> List[float]  and  embed_many(texts) -> List[List[float]].
    """

    def __init__(self) -> None:
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer   # type: ignore
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[Vectorizer] Loaded all-MiniLM-L6-v2 (384-dim).")
        except Exception as exc:
            print(f"[Vectorizer] sentence-transformers unavailable ({exc}); "
                  "falling back to deterministic hash projection.")
            self._model = None

    def embed(self, text: str) -> List[float]:
        if self._model is not None:
            return self._model.encode(text, normalize_embeddings=True).tolist()
        return self._hash_project(text)

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        if self._model is not None:
            return self._model.encode(
                texts, normalize_embeddings=True, batch_size=64
            ).tolist()
        return [self._hash_project(t) for t in texts]

    # ── Deterministic fallback ────────────────────────────────────────────────
    @staticmethod
    def _hash_project(text: str) -> List[float]:
        """
        Produces a stable 384-dim float vector from a SHA-512 hash of the text.
        Not semantically meaningful, but reproducible across runs — sufficient
        for benchmarking the caching machinery itself.
        """
        digest = hashlib.sha512(text.encode()).digest()   # 64 bytes
        # Repeat digest 6× → 384 bytes → 96 floats; tile to 384 floats
        raw = (digest * 6)[:384]
        # Unpack as 96 signed ints, scale to [-1, 1]
        ints = struct.unpack("96b", raw[:96])
        floats = [i / 128.0 for i in ints]
        # Tile 4× to reach 384 dims
        full = (floats * 4)[:384]
        # L2-normalise
        norm = sum(x * x for x in full) ** 0.5 or 1.0
        return [x / norm for x in full]
