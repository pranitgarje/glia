"""
glia/events.py
--------------
Observability primitives for the Glia library.

ISOLATION GUARANTEE: This file has *zero* imports from any other glia module.
It can be instantiated, tested, and reused in complete isolation.

Any component that needs to fire an event receives an EventEmitter via
constructor injection — it never imports from manager.py or watcher.py.

Public surface
--------------
- WatcherEvent  : structured event payload (dataclass)
- EventEmitter  : lightweight pub/sub dispatcher
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Signature expected of every registered callback:
#:   callback(event: WatcherEvent) -> None
EventCallback = Callable[["WatcherEvent"], None]


# ---------------------------------------------------------------------------
# WatcherEvent
# ---------------------------------------------------------------------------

@dataclass
class WatcherEvent:
    """
    Immutable-ish structured payload emitted by Glia components.

    Every event produced by the library — cache hits, misses,
    invalidations, and file-system watcher notifications — is
    represented as a ``WatcherEvent`` instance.

    The dictionary representation (``to_dict()``) is the form passed to
    developer-supplied ``on_invalidation`` callbacks and to standard
    Python logging handlers.

    Attributes
    ----------
    event_type:
        Discriminator string.  One of:
        ``"cache_hit"``, ``"cache_miss"``,
        ``"invalidation_complete"``, ``"watcher_event"``.
    source_id:
        The cache tag / document identifier relevant to this event.
        ``None`` for ``"cache_miss"`` events where no entry was matched.
    adapter_type:
        Identifier for the adapter that surfaced the event
        (e.g. ``"filesystem"``, ``"vector"``, ``"graph"``, ``"relational"``).
        ``None`` when the event originates from the cache core rather than
        an adapter.
    detection_mode:
        How the change was detected: ``"polling"``, ``"cdc"``, or
        ``"direct"`` (developer-invoked invalidation).
        ``None`` when not applicable (e.g. cache_hit / cache_miss).
    deleted_count:
        Number of cache entries removed.  ``0`` for non-invalidation events.
    timestamp:
        UTC datetime when the event was created.  Populated automatically
        via ``field(default_factory=…)`` — do not set manually.
    payload:
        Arbitrary extra data attached by the emitting component.
        For ``"cache_hit"``: ``{"similarity_score": float}``.
        For ``"cache_miss"``: ``{"prompt_excerpt": str}`` (≤ 200 chars).
        For ``"watcher_event"``: ``{"file_path": str, "fs_event_type": str}``.
    """

    event_type: str
    source_id: Optional[str] = None
    adapter_type: Optional[str] = None
    detection_mode: Optional[str] = None
    deleted_count: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    payload: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """
        Return a plain-dictionary representation suitable for passing to
        developer callbacks or ``logging.Logger`` extra fields.

        The ``timestamp`` is serialised as an ISO-8601 string so the dict
        is JSON-serialisable without a custom encoder.

        The returned structure is always flat and complete — every key is
        present even when its value is ``None`` or ``0``, so callback
        authors can rely on a stable schema without defensive ``get()``
        calls::

            {
                "event_type":     "invalidation_complete",
                "source_id":      "node-abc123",
                "adapter_type":   "filesystem",
                "detection_mode": "polling",
                "deleted_count":  4,
                "timestamp":      "2024-01-15T09:23:11.045821",
                "payload": {
                    "file_path":    "data/report.md",
                    "fs_event_type": "modified"
                }
            }
        """
        return {
            "event_type":     self.event_type,
            "source_id":      self.source_id,
            "adapter_type":   self.adapter_type,
            "detection_mode": self.detection_mode,
            "deleted_count":  self.deleted_count,
            "timestamp":      self.timestamp.isoformat(),
            "payload":        self.payload,
        }


# ---------------------------------------------------------------------------
# EventEmitter
# ---------------------------------------------------------------------------

class EventEmitter:
    """
    Lightweight, synchronous pub/sub event dispatcher.

    Components that need to fire events receive an ``EventEmitter``
    instance via constructor injection.  They call ``emit()``; registered
    listeners receive the ``WatcherEvent`` payload.

    Design constraints
    ------------------
    - Zero imports from the rest of ``glia/``.
    - Falls back to ``logging.getLogger("glia.events")`` when no listeners
      are registered for an event type, so events are never silently lost.
    - Thread-safe for concurrent ``emit()`` calls from background runners
      (uses a simple ``list`` copy-on-read pattern rather than a lock,
      acceptable for the low-frequency events this library produces).

    Usage::

        emitter = EventEmitter()
        emitter.on("cache_hit", lambda e: print(e.to_dict()))

        event = WatcherEvent(event_type="cache_hit", source_id="doc-42",
                             payload={"similarity_score": 0.05})
        emitter.emit("cache_hit", event)
    """

    def __init__(self) -> None:
        """
        Initialise the emitter with an empty callback registry and a
        module-level logger as the default sink.
        """
        self._callbacks: Dict[str, List[EventCallback]] = {}
        self._logger: logging.Logger = logging.getLogger("glia.events")

    def on(self, event_name: str, callback: EventCallback) -> None:
        """
        Register *callback* to be invoked whenever *event_name* is emitted.

        Multiple callbacks may be registered for the same event name;
        they are called in registration order.

        Parameters
        ----------
        event_name:
            One of ``"cache_hit"``, ``"cache_miss"``,
            ``"invalidation_complete"``, ``"watcher_event"``.
        callback:
            Callable that accepts a single ``WatcherEvent`` argument and
            returns ``None``.
        """
        if event_name not in self._callbacks:
            self._callbacks[event_name] = []
        self._callbacks[event_name].append(callback)

    def emit(self, event_name: str, event: WatcherEvent) -> None:
        """
        Dispatch *event* to all callbacks registered under *event_name*.

        If no callbacks are registered, the event is forwarded to the
        module logger at ``DEBUG`` level so it is never silently dropped.

        Callbacks are invoked against a snapshot of the registered list
        taken at the moment ``emit()`` is called.  This means a callback
        that calls ``on()`` to add further listeners during dispatch will
        not affect the current emission — safe for background-thread callers.

        Any exception raised inside a callback is caught, logged at
        ``ERROR`` level with the offending callback's name, and then
        re-raised so the caller knows dispatch did not fully complete.

        Parameters
        ----------
        event_name:
            The event type string (must match the ``event.event_type`` field
            by convention, but is not enforced here).
        event:
            The fully populated ``WatcherEvent`` instance to dispatch.
        """
        # Snapshot the list to protect against mid-iteration mutation.
        listeners: List[EventCallback] = list(
            self._callbacks.get(event_name, [])
        )

        if not listeners:
            # Nothing registered — fall back to the logger so the event
            # is never silently swallowed.
            self._logger.debug(
                "glia.events: no listeners for '%s' — event dropped to log: %s",
                event_name,
                event.to_dict(),
            )
            return

        for callback in listeners:
            try:
                callback(event)
            except Exception:
                self._logger.error(
                    "glia.events: callback '%s' raised an exception "
                    "while handling event '%s'.",
                    getattr(callback, "__name__", repr(callback)),
                    event_name,
                    exc_info=True,
                )
                raise