"""
glia/invalidator.py
───────────────────
Layer 3 — Cache Core

Defines CacheInvalidator, responsible for targeted removal of stale
cache entries by source_id tag.

Architectural constraints:
- Receives a GliaManager instance at init; shares its Redis connection
  — no second connection is opened.
- Single responsibility: given a source_id, find all matching cache
  keys via RediSearch and delete them atomically.
- MUST NOT import from Layer 4 (adapters/), Layer 5 (runners.py,
  watcher.py), or Layer 6 (__init__.py).
- Can be invoked directly by the developer or called internally by
  CacheWatcher when a source document changes.
"""

from __future__ import annotations

from glia.manager import GliaManager


class CacheInvalidator:
    """
    Targeted cache invalidation by ``source_id`` tag.

    ``CacheInvalidator`` wraps a ``GliaManager`` instance and exposes a
    single method — ``delete_by_tag()`` — that atomically removes every
    cache entry associated with a given source document.

    Because it receives the manager at construction time, it reuses the
    manager's existing Redis connection rather than opening a second one.
    It can be used standalone by the developer or wired into
    ``CacheWatcher`` for automatic invalidation on source changes.

    Parameters
    ----------
    cache_manager : GliaManager
        The already-initialised cache manager whose Redis connection
        and RediSearch index this invalidator will operate against.
    """

    def __init__(self, cache_manager: GliaManager) -> None:
        self.cache_manager: GliaManager = cache_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def delete_by_tag(self, source_id: str) -> int:
        """
        Atomically delete all cache entries tagged with ``source_id``.

        This method must:
        1. Query the RediSearch index for every entry whose
           ``source_id`` TAG field matches the provided value.
        2. Delete all matched entries in a single batched operation,
           removing both the JSON document and its index reference so
           no orphaned index entries remain.
        3. Return the count of deleted entries.
        4. Be idempotent: if no entries are found, return ``0`` without
           raising an error.

        Parameters
        ----------
        source_id : str
            The tag value to match against the ``source_id`` field,
            e.g. a LlamaIndex ``node.id_`` or any string identifier
            used when the entry was stored.

        Returns
        -------
        int
            Number of cache entries deleted.  Returns ``0`` if no
            entries matched ``source_id``.
        """
        ...