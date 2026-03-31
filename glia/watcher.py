"""
glia/watcher.py
───────────────
Layer 5 — Watcher Engine

Defines CacheWatcher, the orchestration hub of the watcher engine.

CacheWatcher inspects each adapter's ``mode`` at ``start()`` time, spawns
the appropriate runner (PollingRunner or CDCRunner), and holds references to
all running threads.  It is the sole owner of the invalidation pathway:
runners surface ``source_id`` values back through ``_dispatch()``, which
then calls ``CacheInvalidator.delete_by_tag()`` and fires the
``invalidation_complete`` event via the shared ``EventEmitter``.

Architectural constraints:
- Imports CacheInvalidator from glia.invalidator (Layer 3).
- Imports PollingRunner and CDCRunner from glia.runners (Layer 5 peer).
- Imports EventEmitter and WatcherEvent from glia.events (Layer 1).
- MUST NOT import from Layer 6 (__init__.py).
- No polling or CDC stream logic lives here; that belongs in runners.py.

Threading model
---------------
CacheWatcher itself runs on the caller's thread.  It delegates all
background work to runner-owned threads.  ``start()`` and ``stop()`` are
blocking calls: ``start()`` returns only after all runner threads have been
launched; ``stop()`` returns only after all runner threads have been joined.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from glia.invalidator import CacheInvalidator
from glia.runners import CDCRunner, PollingRunner
from glia.events import EventEmitter, WatcherEvent
from glia.adapters.base import DatabaseAdapter

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("glia.watcher")

# ---------------------------------------------------------------------------
# Type alias for the optional developer callback
# ---------------------------------------------------------------------------

#: Signature of the optional ``on_invalidation`` callback the developer may
#: supply.  Receives the resolved ``source_id`` and the number of cache
#: entries deleted.
#:   on_invalidation(source_id: str, deleted_count: int) -> None
InvalidationCallback = Callable[[str, int], None]


# ---------------------------------------------------------------------------
# CacheWatcher
# ---------------------------------------------------------------------------

class CacheWatcher:
    """
    Orchestration hub for background cache invalidation.

    ``CacheWatcher`` is the top-level watcher object that most developers
    interact with.  It accepts a list of adapters at construction time,
    inspects each adapter's ``mode`` when ``start()`` is called, and spawns
    either a :class:`~glia.runners.PollingRunner` or a
    :class:`~glia.runners.CDCRunner` for each one.  All runners share a
    single reference back to :meth:`_dispatch`, which is the single choke
    point through which every ``source_id`` flows before reaching
    :class:`~glia.invalidator.CacheInvalidator`.

    Parameters
    ----------
    invalidator : CacheInvalidator
        The already-initialised invalidator whose
        :meth:`~glia.invalidator.CacheInvalidator.delete_by_tag` method
        will be called each time a changed ``source_id`` is surfaced by a
        runner.  The invalidator (and therefore its ``GliaManager``) must
        remain alive for the full lifetime of the watcher.
    adapters : List[DatabaseAdapter]
        One or more concrete adapter instances (VectorDBAdapter,
        GraphDBAdapter, RelationalDBAdapter, etc.) to be monitored.
        Each adapter's ``mode`` attribute (``"polling"`` or ``"cdc"``)
        determines which runner type is created for it.
    on_invalidation : InvalidationCallback, optional
        Developer-supplied callback invoked *after* each successful
        ``delete_by_tag()`` call.  Receives ``(source_id, deleted_count)``
        as positional arguments.  Defaults to ``None`` (no callback).
    emitter : EventEmitter, optional
        Shared event emitter.  If ``None``, a private emitter is created.
        Pass the same emitter instance as ``GliaManager`` to consolidate
        all ``cache_hit``, ``cache_miss``, and ``invalidation_complete``
        events into a single subscriber pipeline.

    Attributes
    ----------
    invalidator : CacheInvalidator
        The invalidator used to delete stale cache entries.
    adapters : List[DatabaseAdapter]
        The adapter instances registered with this watcher.
    _on_invalidation : InvalidationCallback or None
        The optional developer callback.
    _emitter : EventEmitter
        The event emitter used to fire ``invalidation_complete`` and
        ``watcher_event`` events.
    _runners : List[PollingRunner | CDCRunner]
        Runner instances created by ``start()``.  Empty until ``start()``
        is called and cleared after ``stop()`` completes.
    """

    def __init__(
        self,
        invalidator: CacheInvalidator,
        adapters: List[DatabaseAdapter],
        on_invalidation: Optional[InvalidationCallback] = None,
        emitter: Optional[EventEmitter] = None,
    ) -> None:
        """
        Initialise the watcher without starting any background threads.

        No adapter connections are opened here; that happens in
        :meth:`start`.

        Parameters
        ----------
        invalidator : CacheInvalidator
            The invalidator to call on each change event.
        adapters : List[DatabaseAdapter]
            Adapters to monitor.  Must contain at least one entry.
        on_invalidation : InvalidationCallback, optional
            Developer callback invoked after each successful invalidation.
        emitter : EventEmitter, optional
            Shared event emitter.  A private one is created if omitted.

        Raises
        ------
        ValueError
            If ``adapters`` is empty.
        """
        if not adapters:
            raise ValueError(
                "CacheWatcher requires at least one adapter; "
                "received an empty list."
            )

        self.invalidator: CacheInvalidator = invalidator
        self.adapters: List[DatabaseAdapter] = list(adapters)
        self._on_invalidation: Optional[InvalidationCallback] = on_invalidation
        self._emitter: EventEmitter = emitter or EventEmitter()

        # Populated by start(); cleared by stop().
        self._runners: List[PollingRunner | CDCRunner] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Connect all adapters and launch their background runner threads.

        For each adapter in :attr:`adapters`:

        1. Call ``adapter.connect()`` to open the upstream data-source
           connection.
        2. Inspect ``adapter.mode``:
           - ``"polling"`` → create a :class:`~glia.runners.PollingRunner`.
           - ``"cdc"``     → create a :class:`~glia.runners.CDCRunner`.
        3. Call ``runner.start()`` to launch the background thread.
        4. Append the runner to :attr:`_runners`.

        Returns only after *all* runner threads have been started.
        If any ``adapter.connect()`` call raises, the exception propagates
        to the caller and no runner is created for that adapter (already-
        started runners for earlier adapters continue running).

        Raises
        ------
        AdapterConnectionError
            If any adapter fails to connect.
        RuntimeError
            If ``start()`` is called while runners are already active.
        """
        ...

    def stop(self) -> None:
        """
        Stop all runner threads and disconnect all adapters.

        For each runner in :attr:`_runners`:

        1. Call ``runner.stop()`` (blocks until that runner's thread exits).

        Then for each adapter in :attr:`adapters`:

        2. Call ``adapter.disconnect()`` to cleanly close the upstream
           connection.

        Clears :attr:`_runners` after all threads have been joined.
        Idempotent: safe to call when already stopped.
        """
        ...

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, source_id: str, event: WatcherEvent) -> None:
        """
        Central handler called by runners whenever a changed record is detected.

        This is the single choke point through which every ``source_id``
        flows before reaching :class:`~glia.invalidator.CacheInvalidator`.

        Steps
        -----
        1. Call :meth:`~glia.invalidator.CacheInvalidator.delete_by_tag`
           with ``source_id`` and capture ``deleted_count``.
        2. Emit an ``"invalidation_complete"`` :class:`~glia.events.WatcherEvent`
           via :attr:`_emitter`, carrying ``source_id``, ``deleted_count``,
           and the ``detection_mode`` from the incoming ``event``.
        3. If :attr:`_on_invalidation` is set, invoke it with
           ``(source_id, deleted_count)``.
        4. Log the outcome at ``DEBUG`` level.

        All exceptions from ``delete_by_tag()`` are caught, logged at
        ``ERROR`` level, and re-raised so the runner can handle retries.
        Exceptions from the developer-supplied ``_on_invalidation`` callback
        are caught and logged but *not* re-raised, so a buggy callback can
        never crash the watcher.

        Parameters
        ----------
        source_id : str
            The resolved cache tag to invalidate.  Passed directly to
            :meth:`~glia.invalidator.CacheInvalidator.delete_by_tag`.
        event : WatcherEvent
            The raw event as surfaced by the runner.  Used to carry
            ``detection_mode`` and ``adapter_type`` into the
            ``"invalidation_complete"`` event payload.
        """
        ...