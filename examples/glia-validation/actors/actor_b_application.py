"""
actors/actor_b_application.py
──────────────────────────────
Actor B — The Glia Application ("The User")

Simulates a RAG application that:
  1. Uses GliaManager to store and check cached LLM responses.
  2. Runs CacheWatcher in a background thread to monitor PostgreSQL via
     RelationalDBAdapter (changelog strategy, polling mode).
  3. Subscribes to all internal Glia events for full observability.

Public interface (used by actor_c_verifier.py and run_validation.py):
  - GliaApplication.start()           → connect everything, start watcher
  - GliaApplication.stop()            → stop watcher, disconnect
  - GliaApplication.ask(prompt)       → cache check → returns (result, hit)
  - GliaApplication.store(prompt, response, source_id)
  - GliaApplication.wait_for_invalidation(source_id, timeout) → bool
  - GliaApplication.event_log         → list of all observed WatcherEvent dicts
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras

# Allow imports from parent directory when run standalone.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CACHE_INDEX_NAME,
    DISTANCE_THRESHOLD,
    INVALIDATION_TIMEOUT_SECONDS,
    PG_DSN,
    POLL_INTERVAL_SECONDS,
    REDIS_URL,
    VECTOR_DIMS,
)
from mock_vectorizer import MockVectorizer

from glia.manager import GliaManager
from glia.invalidator import CacheInvalidator
from glia.watcher import CacheWatcher
from glia.adapters.relational import RelationalDBAdapter
from glia.events import EventEmitter, WatcherEvent

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [ACTOR-B / APPLICATION]  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("actor_b")


class GliaApplication:
    """
    Self-contained Glia-powered RAG application for closed-loop validation.

    Wires together GliaManager + CacheWatcher and exposes a clean test API.
    """

    def __init__(self) -> None:
        self._vectorizer = MockVectorizer(dims=VECTOR_DIMS)
        self._emitter = EventEmitter()

        # Register event listeners before creating any components so that
        # no events are missed even during initialisation.
        self._emitter.on("cache_hit",             self._on_cache_hit)
        self._emitter.on("cache_miss",            self._on_cache_miss)
        self._emitter.on("invalidation_complete", self._on_invalidation_complete)

        # Thread-safe event log — both actor B's thread and the runner
        # background thread write to this list.
        self.event_log: list[dict] = []
        self._event_lock = threading.Lock()

        # Invalidation events keyed by source_id so wait_for_invalidation()
        # can block until a specific ID is invalidated.
        self._invalidation_events: dict[str, threading.Event] = {}
        self._inv_event_lock = threading.Lock()

        # ── Glia components ──────────────────────────────────────────────
        self.manager: Optional[GliaManager] = None
        self.invalidator: Optional[CacheInvalidator] = None
        self.watcher: Optional[CacheWatcher] = None
        self._pg_conn = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to Redis + PostgreSQL, build the watcher, start polling."""
        log.info("🚀 GliaApplication starting…")

        # 1. Build GliaManager (opens Redis, creates RediSearch index).
        self.manager = GliaManager(
            vectorizer=self._vectorizer,
            redis_url=REDIS_URL,
            index_name=CACHE_INDEX_NAME,
            distance_threshold=DISTANCE_THRESHOLD,
            vector_dims=VECTOR_DIMS,
            emitter=self._emitter,
        )
        log.info("✅ GliaManager connected to Redis (index=%r).", CACHE_INDEX_NAME)

        # 2. Build the invalidator (shares the manager's Redis connection).
        self.invalidator = CacheInvalidator(cache_manager=self.manager)

        # 3. Connect to PostgreSQL and create the RelationalDBAdapter.
        self._pg_conn = self._connect_postgres()
        adapter = RelationalDBAdapter(
            connection=self._pg_conn,
            table="documents",
            updated_at_col="updated_at",
            mode="polling",
            source_id_field="id",
            poll_interval=POLL_INTERVAL_SECONDS,
            last_cursor=None,           # bootstrap: read current state first
            changelog_table="document_changelog",
        )
        log.info(
            "✅ RelationalDBAdapter ready (table=documents, poll_interval=%ds).",
            POLL_INTERVAL_SECONDS,
        )

        # 4. Build and start the CacheWatcher.
        self.watcher = CacheWatcher(
            invalidator=self.invalidator,
            adapters=[adapter],
            on_invalidation=self._on_invalidation_callback,
            emitter=self._emitter,
        )
        self.watcher.start()
        log.info("✅ CacheWatcher started — monitoring PostgreSQL changelog.")
        log.info("🟢 GliaApplication fully operational.")

    def stop(self) -> None:
        """Stop the watcher and close all connections."""
        log.info("🛑 GliaApplication stopping…")
        if self.watcher:
            self.watcher.stop()
            log.info("   CacheWatcher stopped.")
        if self._pg_conn:
            try:
                self._pg_conn.close()
            except Exception:
                pass
            log.info("   PostgreSQL connection closed.")
        log.info("🔴 GliaApplication stopped.")

    # ------------------------------------------------------------------
    # Cache operations (public API used by actor_c)
    # ------------------------------------------------------------------

    def ask(self, prompt: str, source_id: str | None = None) -> tuple[str | None, bool]:
        """
        Check the cache for *prompt*.

        Returns
        -------
        (response, is_hit) :
            response  — cached string on hit, None on miss.
            is_hit    — True on cache hit, False on miss.
        """
        filter_expr = f"@source_id:{{{source_id}}}" if source_id else None
        result = self.manager.check(prompt, filter=filter_expr)
        hit = result is not None
        status = "HIT ✅" if hit else "MISS ❌"
        log.info("🔍 Cache %s for prompt=%r", status, prompt[:80])
        return result, hit

    def store(self, prompt: str, response: str, source_id: str) -> None:
        """Store a (prompt, response) pair tagged with *source_id*."""
        self.manager.store(prompt=prompt, response=response, source_id=source_id)
        log.info(
            "💾 Stored cache entry — source_id=%r, prompt=%r",
            source_id, prompt[:60],
        )

    def wait_for_invalidation(
        self,
        source_id: str,
        timeout: float = INVALIDATION_TIMEOUT_SECONDS,
    ) -> bool:
        """
        Block until an ``invalidation_complete`` event is observed for
        *source_id*, or until *timeout* seconds have elapsed.

        Returns True if the invalidation was observed, False on timeout.
        """
        with self._inv_event_lock:
            if source_id not in self._invalidation_events:
                self._invalidation_events[source_id] = threading.Event()
            evt = self._invalidation_events[source_id]

        log.info(
            "⏳ Waiting up to %.0fs for invalidation of source_id=%r…",
            timeout, source_id,
        )
        observed = evt.wait(timeout=timeout)
        if observed:
            log.info("🔔 Invalidation confirmed for source_id=%r.", source_id)
        else:
            log.warning(
                "⚠️  Timeout waiting for invalidation of source_id=%r.", source_id
            )
        return observed

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    def _on_cache_hit(self, event: WatcherEvent) -> None:
        self._log_event(event)
        log.info(
            "   📦 cache_hit  source_id=%r  distance=%.4f",
            event.source_id,
            event.payload.get("distance", 0),
        )

    def _on_cache_miss(self, event: WatcherEvent) -> None:
        self._log_event(event)
        log.info(
            "   🕳️  cache_miss  reason=%r",
            event.payload.get("reason", "threshold_exceeded"),
        )

    def _on_invalidation_complete(self, event: WatcherEvent) -> None:
        self._log_event(event)
        log.info(
            "   🗑️  invalidation_complete  source_id=%r  deleted=%d  mode=%s",
            event.source_id,
            event.deleted_count,
            event.detection_mode,
        )
        # Signal any thread blocked in wait_for_invalidation().
        with self._inv_event_lock:
            if event.source_id in self._invalidation_events:
                self._invalidation_events[event.source_id].set()

    def _on_invalidation_callback(self, source_id: str, deleted_count: int) -> None:
        """Developer-style on_invalidation callback (simpler signature)."""
        log.info(
            "   🔁 on_invalidation callback: source_id=%r, deleted=%d",
            source_id, deleted_count,
        )

    def _log_event(self, event: WatcherEvent) -> None:
        with self._event_lock:
            self.event_log.append(event.to_dict())

    # ------------------------------------------------------------------
    # PostgreSQL connection helper
    # ------------------------------------------------------------------

    def _connect_postgres(self):
        import time
        for attempt in range(1, 11):
            try:
                conn = psycopg2.connect(PG_DSN)
                conn.autocommit = True
                log.info("   PostgreSQL connected (attempt %d).", attempt)
                return conn
            except psycopg2.OperationalError as exc:
                log.warning("   PG not ready (attempt %d): %s", attempt, exc)
                time.sleep(3)
        raise RuntimeError("Could not connect to PostgreSQL after 10 attempts.")


# ── Standalone entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    app = GliaApplication()
    app.start()
    log.info("Running — press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()
