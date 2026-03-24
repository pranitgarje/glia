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

import hashlib
import json
import struct
from typing import Any, Dict, List, Optional

import redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from glia.schema import SchemaBuilder
from glia.events import EventEmitter, WatcherEvent

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vector_to_bytes(vector: List[float]) -> bytes:
    """Pack a list of floats into a little-endian binary blob for Redis."""
    return struct.pack(f"<{len(vector)}f", *vector)


def _make_cache_key(index_name: str, prompt: str) -> str:
    """
    Generate a deterministic Redis key from the index name and the prompt.

    Using SHA-256 keeps keys fixed-length and avoids special-character issues
    while still being collision-resistant enough for a semantic cache.
    """
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{index_name}:entry:{digest}"


# ---------------------------------------------------------------------------
# GliaManager
# ---------------------------------------------------------------------------

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
    emitter : EventEmitter, optional
        Shared event emitter for cache_hit / cache_miss events.
        If ``None``, a private emitter is created automatically.
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
        emitter: Optional[EventEmitter] = None,
    ) -> None:
        # ── Injected dependencies (never created here) ──────────────────────
        self.vectorizer: Any = vectorizer

        # ── Configuration ────────────────────────────────────────────────────
        self.redis_url: str = redis_url
        self.index_name: str = index_name
        self.distance_threshold: float = distance_threshold
        self.vector_dims: int = vector_dims
        self.custom_schema: Optional[List[Dict[str, Any]]] = custom_schema
        self.ttl_seconds: Optional[int] = ttl_seconds

        # ── Observability ────────────────────────────────────────────────────
        self._emitter: EventEmitter = emitter or EventEmitter()

        # ── Redis connection ─────────────────────────────────────────────────
        self._redis: redis.Redis = redis.from_url(
            self.redis_url,
            decode_responses=False,   # we store raw bytes for the vector field
        )

        # ── Schema / index ───────────────────────────────────────────────────
        self._schema_builder: SchemaBuilder = SchemaBuilder(
            vector_dims=self.vector_dims,
            custom_fields=self.custom_schema or [],
        )
        self._ensure_index()

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """
        Create the RediSearch index if it does not already exist.

        The index is built over JSON documents stored under the key
        prefix ``<index_name>:entry:``.  The schema always includes:

        * ``$.prompt``        – stored text (not indexed)
        * ``$.response``      – stored text (not indexed)
        * ``$.source_id``     – TAG field, used for invalidation
        * ``$.prompt_vector`` – VECTOR field (cosine, FLAT algorithm)

        Any ``custom_schema`` fields declared at construction time are
        appended after the defaults.
        """
        try:
            self._redis.ft(self.index_name).info()
            # Index already exists — nothing to do.
            return
        except Exception:
            pass  # Index does not exist yet; create it below.

        schema = self._schema_builder.build()

        definition = IndexDefinition(
            prefix=[f"{self.index_name}:entry:"],
            index_type=IndexType.JSON,
        )

        self._redis.ft(self.index_name).create_index(
            fields=schema,
            definition=definition,
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

        Steps
        -----
        1. Embed ``prompt`` via ``self.vectorizer.embed()``.
        2. Generate a deterministic cache key (SHA-256 of the prompt).
        3. Serialise the payload as a JSON document and write it to
           Redis under the active RediSearch index.
        4. Store ``source_id`` as a TAG field so entries can be
           targeted during invalidation.
        5. Apply ``ttl_seconds`` expiry if configured.

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
        # 1. Embed the prompt using the injected vectorizer.
        vector: List[float] = self.vectorizer.embed(prompt)

        # 2. Deterministic cache key.
        key = _make_cache_key(self.index_name, prompt)

        # 3. Build the JSON payload.
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            # Store source_id as a plain string; RediSearch indexes it as TAG.
            "source_id": source_id,
            # Store the vector as a list of floats — redis-py's JSON client
            # will serialise this correctly; RediSearch reads it via $.prompt_vector.
            "prompt_vector": vector,
        }
        if metadata:
            payload.update(metadata)

        # Write to Redis as a JSON document.
        self._redis.json().set(key, "$", payload)

        # 4. Apply optional TTL.
        if self.ttl_seconds is not None:
            self._redis.expire(key, self.ttl_seconds)

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
        # Embed the incoming prompt.
        query_vector: List[float] = self.vectorizer.embed(prompt)
        query_bytes: bytes = _vector_to_bytes(query_vector)

        # Build the KNN query string.
        #
        # RediSearch KNN syntax:
        #   (prefilter_expression)=>[KNN k @field $param AS score_alias]
        #
        # We always request k=1 — we only care about the single nearest entry.
        prefilter = f"({filter})" if filter else "*"
        query_str = f"{prefilter}=>[KNN 1 @prompt_vector $vec AS vector_score]"

        query = (
            Query(query_str)
            .sort_by("vector_score", asc=True)
            .return_fields("response", "vector_score", "source_id")
            .dialect(2)           # required for KNN / vector search
        )

        try:
            results = self._redis.ft(self.index_name).search(
                query,
                query_params={"vec": query_bytes},
            )
        except Exception:
            # Index not found or Redis unavailable — treat as a miss.
            self._emitter.emit(
                "cache_miss",
                WatcherEvent(
                    event_type="cache_miss",
                    payload={"prompt": prompt, "reason": "search_error"}
                )
            )
            return None

        if not results.docs:
            self._emitter.emit(
                "cache_miss",
                WatcherEvent(
                    event_type="cache_miss",
                    payload={"prompt": prompt}
                )
            )
            return None

        top_doc = results.docs[0]

        # vector_score is a cosine *distance* in [0, 2]; lower is better.
        distance = float(getattr(top_doc, "vector_score", 2.0))

        if distance > self.distance_threshold:
            self._emitter.emit(
                "cache_miss",
                WatcherEvent(
                    event_type="cache_miss",
                    payload= {
                        "prompt": prompt,
                        "distance": distance,
                        "threshold": self.distance_threshold,
                    }
                ),
            )
            return None

        response: str = top_doc.response

        self._emitter.emit(
            "cache_hit",
            WatcherEvent(
                    event_type="cache_hit",
                    source_id = getattr(top_doc, "source_id", None),
                    payload= {
                        "prompt": prompt,
                        "distance": distance,
                    },
                ),
        )
        return response

    # ------------------------------------------------------------------
    # Index management (utility, referenced in class diagram)
    # ------------------------------------------------------------------

    def delete_index(self, drop_documents: bool = False) -> None:
        """
        Drop the RediSearch index.

        Parameters
        ----------
        drop_documents : bool
            If ``True``, also delete all underlying JSON documents.
            Defaults to ``False`` (index metadata only).
        """
        try:
            self._redis.ft(self.index_name).dropindex(delete_documents=drop_documents)
        except Exception:
            pass  # Index may not exist; silently ignore.

    # ------------------------------------------------------------------
    # EventEmitter pass-through (convenience)
    # ------------------------------------------------------------------

    def on(self, event_name: str, callback: Any) -> None:
        """
        Register an event listener on the internal emitter.

        Example
        -------
        >>> manager.on("cache_hit", lambda p: print("HIT", p))
        """
        self._emitter.on(event_name, callback)