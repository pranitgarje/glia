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

from redis.commands.search.query import Query
import redis

from glia.manager import GliaManager
from glia.exceptions import InvalidationError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of keys to fetch per RediSearch page.
# RediSearch defaults to returning 10 results; we use a large page size so
# a single round-trip handles most real-world invalidation calls.
# For sources with thousands of cached entries the scroll loop below handles
# overflow without any caller change.
_PAGE_SIZE = 1_000


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
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _redis(self):
        """Convenience shortcut to the shared Redis client."""
        return self.cache_manager._redis

    @property
    def _index_name(self) -> str:
        return self.cache_manager.index_name

    def _search_keys_for_tag(self, source_id: str) -> list[str]:
        """
        Return every Redis key in the index whose ``source_id`` TAG
        matches ``source_id``.

        RediSearch TAG values that contain special characters (hyphens,
        dots, slashes, …) must be escaped with a backslash before being
        embedded in a query string.  We escape here so callers never
        have to think about it.

        The method pages through results with LIMIT / OFFSET so it is
        safe for sources with arbitrarily many cached entries.
        """
        # Escape characters that RediSearch treats as TAG delimiters or
        # query operators: , . < > { } [ ] " ' : ; ! @ # $ % ^ & * ( )
        # - + = ~ |  and the backslash itself.
        special = r'\,.<>{}[]"\':;!@#$%^&*()-+=~|'
        escaped = "".join(f"\\{c}" if c in special else c for c in source_id)

        ft = self._redis.ft(self._index_name)
        keys: list[str] = []
        offset = 0

        while True:
            # We only need the document IDs (keys), not any field values,
            # so we request 0 return fields — this minimises network traffic.
            query = (
                Query(f"@source_id:{{{escaped}}}")
                .no_content()           # return keys only, skip field data
                .paging(offset, _PAGE_SIZE)
                .dialect(2)
            )

            try:
                results = ft.search(query)
            except redis.exceptions.ResponseError:
                # Index does not exist yet; nothing to delete.
                break
            except Exception as exc:
                # Catch actual connection drops/timeouts and wrap them
                raise InvalidationError(
                    f"Failed to query RediSearch for tag '{source_id}'"
                ) from exc

            if not results.docs:
                break

            keys.extend(doc.id for doc in results.docs)

            # If we got fewer results than the page size we've seen everything.
            if len(results.docs) < _PAGE_SIZE:
                break

            offset += _PAGE_SIZE

        return keys

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def delete_by_tag(self, source_id: str) -> int:
        """
        Atomically delete all cache entries tagged with ``source_id``.

        Steps
        -----
        1. Query the RediSearch index for every key whose ``source_id``
           TAG field matches the provided value (paged to handle large
           result sets without memory pressure).
        2. Delete all matched keys in a single Redis pipeline so the
           entire operation lands in one network round-trip.  Deleting
           a JSON document under a key that the index tracks also
           removes its index references automatically — no orphaned
           index entries are left behind.
        3. Return the count of deleted entries.
        4. Idempotent: returns ``0`` without error when no entries match.

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
        keys = self._search_keys_for_tag(source_id)

        if not keys:
            return 0

        # Issue all DEL commands inside a single pipeline so they arrive
        # at Redis atomically in one batch — no partial deletes if the
        # connection drops mid-way.
        pipe = self._redis.pipeline(transaction=True)
        for key in keys:
            pipe.delete(key)
        results = pipe.execute()

        # Each pipe.delete() returns 1 if the key existed, 0 if it was
        # already gone (race condition).  Sum gives the true deleted count.
        deleted = sum(r for r in results if isinstance(r, int))
        return deleted