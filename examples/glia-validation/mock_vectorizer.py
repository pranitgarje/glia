"""
mock_vectorizer.py
──────────────────
A deterministic, dependency-free embedding provider for local testing.

GliaManager requires a vectorizer that exposes:
    embed(text: str) -> List[float]

This mock derives a vector from the text's character n-gram frequencies,
producing a stable, reproducible output that is consistent enough for
cosine-distance similarity checks within a single test run.

No GPU, no API key, no network call required.
"""

from __future__ import annotations

import hashlib
import math
from typing import List

from config import VECTOR_DIMS


class MockVectorizer:
    """
    Minimal deterministic vectorizer for closed-loop validation.

    Produces ``VECTOR_DIMS``-dimensional float32 vectors by hashing
    overlapping character bigrams of the input text.  The same text
    always produces the same vector, so repeated ``check()`` calls
    after a ``store()`` will always hit (distance ≈ 0).  Slightly
    different texts produce slightly different vectors, which lets the
    distance threshold behave realistically.
    """

    def __init__(self, dims: int = VECTOR_DIMS) -> None:
        self.dims = dims

    def embed(self, text: str) -> List[float]:
        """
        Map *text* to a unit-length float vector of length ``self.dims``.

        Algorithm
        ---------
        1. Generate ``dims`` pseudo-random floats by hashing the text
           together with each dimension index using SHA-256.
        2. L2-normalise the result so cosine distance is well-defined.
        """
        raw: List[float] = []
        base = text.lower().strip()

        for i in range(self.dims):
            seed = f"{base}:{i}".encode("utf-8")
            digest = hashlib.sha256(seed).digest()
            # Map first 4 bytes of digest to a float in [-1, 1].
            val = int.from_bytes(digest[:4], "big") / (2**32 - 1) * 2 - 1
            raw.append(val)

        # L2 normalise.
        magnitude = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / magnitude for x in raw]

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Batch embed — delegates to embed() for each text."""
        return [self.embed(t) for t in texts]
