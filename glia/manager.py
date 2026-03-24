"""
glia/manager.py
───────────────
Layer 3 — Cache Core

Defines GliaManager, the primary entry point for developers interacting
with Glia. Owns the Redis connection, the RediSearch index lifecycle,
and the store() / check() public methods.

Architectural constraints:
- Imports SchemaBuilder from glia.schema (Layer 1).
- MUST NOT instantiate or import any embedding provider internally;
  the vectorizer is always injected by the caller.
- MUST NOT modify the developer's existing document parsing, LLM
  interaction, or vectorizer initialization.
- This file MUST NOT import from Layer 4 (adapters/), Layer 5
  (runners.py, watcher.py), or Layer 6 (__init__.py).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from glia.schema import SchemaBuilder


class GliaManager:
    """
    Primary cache manager for Glia.

    Wraps a RediSearch-backed semantic cache and exposes a minimal
    surface — ``store()`` and ``check()`` — that slots into an existing
    RAG workflow without requiring any changes to surrounding code.

    The developer instantiates ``GliaManager`` once, passing in the
    embedding provider they have already initialised.  The manager
    never creates or swaps the vectorizer; it is the sole embedding
    source for the lifetime of this instance.

    Parameters
    ----------
    vectorizer : Any
        Developer-instantiated embedding provider.  Must expose:
        - ``embed(text: str) -> List[float]`` for single-query lookup.
        - ``embed_many(texts: List[str]) -> List[List[float]]`` for
          batch ingestion.
    redis_url : str
        Connection string for the Redis Stack instance,
        e.g. ``"redis://localhost:6379"``.
    index_name : str, optional
        Name of the RediSearch index.  Defaults to ``"llmcache"``.
    distance_threshold : float, optional
        Cosine-distance ceiling for a cache hit.  Entries with a
        distance above this value are treated as misses.
        Defaults to ``0.2``.
    vector_dims : int, optional
        Dimensionality of the embedding vectors produced by
        ``vectorizer``.  Must match the provider's output size.
        Defaults to ``768`` (textembedding-gecko@003).
    custom_schema : List[Dict[str, Any]], optional
        Additional field definitions to extend the default schema.
        Each entry must be a dict accepted by ``SchemaBuilder``
        (tag or numeric fields only).  Defaults to ``None``.
    ttl_seconds : int, optional
        If set, cache entries expire after this many seconds.
        Prevents orphaned entries if the watcher fails.
        Defaults to ``None`` (no expiry).
    """

    def __init__(
        self,
        vectorizer: Any,
        redis_url: str,
        index_name: str = "llmcache",
        distance_threshold: float = 0.2,
        vector_dims: int = 768,
        custom_schema: Optional[List[Dict[str, Any]]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self.vectorizer: Any = vectorizer
        self.redis_url: str = redis_url
        self.index_name: str = index_name
        self.distance_threshold: float = distance_threshold
        self.vector_dims: int = vector_dims
        self.custom_schema: Optional[List[Dict[str, Any]]] = custom_schema
        self.ttl_seconds: Optional[int] = ttl_seconds

        # SchemaBuilder is initialised here so the index schema is
        # ready before any store() or check() call is made.
        self._schema_builder: SchemaBuilder = SchemaBuilder(
            vector_dims=self.vector_dims,
            custom_fields=self.custom_schema or [],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        prompt: str,
        response: str,
        source_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Embed ``prompt`` and persist a tagged cache entry to Redis.

        On each call this method must:
        1. Embed ``prompt`` via ``self.vectorizer.embed()``.
        2. Generate a deterministic cache key.
        3. Serialise the payload as a JSON document and write it to
           Redis under the active RediSearch index.
        4. Store ``source_id`` as a TAG field so entries can be
           targeted during invalidation.

        Parameters
        ----------
        prompt : str
            The original user query or question to cache.
        response : str
            The LLM-generated answer to associate with ``prompt``.
        source_id : str
            Identifier of the source document that produced this
            entry (e.g. a LlamaIndex ``node.id_``).  Used by
            ``CacheInvalidator.delete_by_tag()`` to remove all entries
            derived from a given source when that source changes.
        metadata : Dict[str, Any], optional
            Arbitrary extra fields to store alongside the entry.
            Keys must correspond to fields declared in the schema
            (default or custom).  Defaults to ``None``.

        Returns
        -------
        None
        """
        ...

    def check(
        self,
        prompt: str,
        filter: Optional[str] = None,
    ) -> Optional[str]:
        """
        Look up a cached response for ``prompt`` via vector similarity.

        The method embeds ``prompt``, performs a KNN search against the
        ``prompt_vector`` field, and returns the stored response string
        if the nearest neighbour is within ``self.distance_threshold``.

        Parameters
        ----------
        prompt : str
            The user query to look up.
        filter : str, optional
            A RediSearch filter expression applied *before* similarity
            scoring, allowing pre-filtering by any TAG field (e.g.
            ``"@source_id:{doc_42}"``).  Defaults to ``None``
            (no pre-filter).

        Returns
        -------
        str
            The cached response string on a hit.
        None
            If no entry is within ``self.distance_threshold``.
        """
        ...