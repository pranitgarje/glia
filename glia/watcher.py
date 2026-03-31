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

Jupyter / nest_asyncio compatibility
-------------------------------------
Runner threads are plain ``threading.Thread`` daemons — they never create
or interact with an event loop, so they are compatible with Jupyter's
already-running asyncio loop without requiring ``nest_asyncio``.  If the
caller is inside a Jupyter cell and needs to await the watcher, the
recommended pattern is::

    import asyncio, concurrent.futures
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    # Run blocking start() off the event-loop thread so the notebook cell
    # returns immediately while the watcher runs in the background.
    loop.run_in_executor(executor, watcher.start)

No changes to this file are required for that usage — runner threads are
already daemon threads and will be cleaned up when the interpreter exits.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, List, Optional, Union

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

# Internal type alias for the union of runner types.
_AnyRunner = Union[PollingRunner, CDCRunner]


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
    _lock : threading.Lock
        Guards ``_runners`` mutations so concurrent ``start()`` / ``stop()``
        calls from different threads (e.g. a Jupyter interrupt) are safe.
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
        self._runners: List[_AnyRunner] = []

        # Protects _runners against concurrent start()/stop() races.
        # Important in Jupyter where keyboard-interrupt handlers may call
        # stop() on a different OS thread from the cell that called start().
        self._lock: threading.Lock = threading.Lock()

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
        started runners for earlier adapters continue running — the caller
        is responsible for calling ``stop()`` to clean up).

        Raises
        ------
        AdapterConnectionError
            If any adapter fails to connect.
        RuntimeError
            If ``start()`` is called while runners are already active.
        ValueError
            If an adapter exposes an unrecognised ``mode`` value.
        """
        with self._lock:
            if self._runners:
                raise RuntimeError(
                    "CacheWatcher is already running. "
                    "Call stop() before calling start() again."
                )

        # Connect adapters and create runners outside the lock so a slow
        # adapter.connect() call does not hold _lock for a long time.
        # The runners list is only written back under the lock once all
        # runners have been successfully started.
        new_runners: List[_AnyRunner] = []

        for adapter in self.adapters:
            adapter_name = type(adapter).__name__

            logger.debug("CacheWatcher: connecting adapter %s …", adapter_name)
            # AdapterConnectionError propagates here — callers must handle it.
            adapter.connect()
            logger.debug("CacheWatcher: adapter %s connected.", adapter_name)

            mode = adapter.mode
            if mode == "polling":
                runner: _AnyRunner = PollingRunner(
                    adapter=adapter,       # type: ignore[arg-type]
                    dispatch=self._dispatch,
                )
            elif mode == "cdc":
                runner = CDCRunner(
                    adapter=adapter,       # type: ignore[arg-type]
                    dispatch=self._dispatch,
                )
            else:
                raise ValueError(
                    f"Adapter {adapter_name} has unrecognised mode={mode!r}. "
                    "Expected 'polling' or 'cdc'."
                )

            runner.start()
            new_runners.append(runner)
            logger.info(
                "CacheWatcher: %s started for adapter %s (mode=%s).",
                type(runner).__name__,
                adapter_name,
                mode,
            )

        with self._lock:
            self._runners = new_runners

        logger.info(
            "CacheWatcher: all %d runner(s) active.", len(self._runners)
        )

    def stop(self) -> None:
        """
        Stop all runner threads and disconnect all adapters.

        For each runner in :attr:`_runners`:

        1. Call ``runner.stop()`` — blocks until that runner's thread exits.

        Then for each adapter in :attr:`adapters`:

        2. Call ``adapter.disconnect()`` to cleanly close the upstream
           connection.

        Clears :attr:`_runners` after all threads have been joined.
        Idempotent: safe to call when already stopped.
        """
        with self._lock:
            runners_snapshot = list(self._runners)

        if not runners_snapshot:
            logger.debug("CacheWatcher.stop(): no active runners — nothing to do.")
            return

        for runner in runners_snapshot:
            runner_name = type(runner).__name__
            adapter_name = type(runner.adapter).__name__
            logger.debug(
                "CacheWatcher: stopping %s for adapter %s …",
                runner_name,
                adapter_name,
            )
            runner.stop()
            logger.info(
                "CacheWatcher: %s for adapter %s stopped.",
                runner_name,
                adapter_name,
            )

        for adapter in self.adapters:
            adapter_name = type(adapter).__name__
            try:
                adapter.disconnect()
                logger.debug("CacheWatcher: adapter %s disconnected.", adapter_name)
            except Exception:
                # A disconnect failure must not prevent the remaining adapters
                # from being disconnected or the runners list from being cleared.
                logger.warning(
                    "CacheWatcher: adapter %s raised during disconnect — "
                    "continuing cleanup.",
                    adapter_name,
                    exc_info=True,
                )

        with self._lock:
            self._runners.clear()

        logger.info("CacheWatcher: all runners stopped and adapters disconnected.")

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
           and the ``detection_mode`` / ``adapter_type`` from the incoming
           ``event``.
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
            The resolved cache tag to invalidate.
        event : WatcherEvent
            The raw event as surfaced by the runner.  Supplies
            ``detection_mode`` and ``adapter_type`` for the outbound
            ``"invalidation_complete"`` event.
        """
        # ------------------------------------------------------------------
        # Step 1: Invalidate — errors are logged and re-raised so the
        #         calling runner can decide whether to retry.
        # ------------------------------------------------------------------
        try:
            deleted_count: int = self.invalidator.delete_by_tag(source_id)
        except Exception:
            logger.error(
                "CacheWatcher._dispatch: delete_by_tag() failed for "
                "source_id=%r — re-raising for runner retry logic.",
                source_id,
                exc_info=True,
            )
            raise  # Let the runner's error-handling loop decide what to do.

        # ------------------------------------------------------------------
        # Step 2: Emit invalidation_complete event.
        # ------------------------------------------------------------------
        completion_event = WatcherEvent(
            event_type="invalidation_complete",
            source_id=source_id,
            adapter_type=event.adapter_type,
            detection_mode=event.detection_mode,
            deleted_count=deleted_count,
        )
        try:
            self._emitter.emit("invalidation_complete", completion_event)
        except Exception:
            # An emitter callback failure must not block the invalidation
            # pathway — log it but continue.
            logger.error(
                "CacheWatcher._dispatch: emitter raised while firing "
                "'invalidation_complete' for source_id=%r.",
                source_id,
                exc_info=True,
            )

        # ------------------------------------------------------------------
        # Step 3: Invoke the optional developer callback.
        # Exceptions are swallowed so a buggy callback never crashes
        # the runner thread.
        # ------------------------------------------------------------------
        if self._on_invalidation is not None:
            try:
                self._on_invalidation(source_id, deleted_count)
            except Exception:
                logger.error(
                    "CacheWatcher._dispatch: on_invalidation callback raised "
                    "for source_id=%r — ignoring to protect runner thread.",
                    source_id,
                    exc_info=True,
                )

        # ------------------------------------------------------------------
        # Step 4: Debug log.
        # ------------------------------------------------------------------
        logger.debug(
            "CacheWatcher._dispatch: invalidation complete — "
            "source_id=%r, deleted=%d, mode=%s, adapter=%s.",
            source_id,
            deleted_count,
            event.detection_mode,
            event.adapter_type,
        )