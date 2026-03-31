"""
glia/runners.py
───────────────
Layer 5 — Watcher Engine

Defines the two execution engines that CacheWatcher spawns per adapter:

  - PollingRunner  — drives the timed poll loop for PollingAdapter instances.
  - CDCRunner      — drives the blocking stream listener for CDCAdapter instances.

Architectural constraints:
- Imports ONLY from the abstract adapter contracts (PollingAdapter, CDCAdapter)
  and from the events / exceptions layers.  Never imports a concrete adapter class.
- All invalidation logic is delegated back to CacheWatcher via the ``dispatch``
  callback supplied at construction.  Runners never call CacheInvalidator directly.
- MUST NOT import from Layer 6 (__init__.py).

Threading model
---------------
Each runner owns exactly one ``threading.Thread``.  The thread is created in
``__init__()`` but not started until ``start()`` is called.  ``stop()`` signals
the thread to exit and joins it, blocking the caller until the thread terminates.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from glia.adapters.polling import PollingAdapter
from glia.adapters.cdc import CDCAdapter
from glia.events import WatcherEvent

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("glia.runners")

# ---------------------------------------------------------------------------
# Type alias for the dispatch callback
# ---------------------------------------------------------------------------

#: Signature of the callback runners hand events back to CacheWatcher with.
#:   dispatch(source_id: str, event: WatcherEvent) -> None
DispatchCallback = Callable[[str, WatcherEvent], None]


# ---------------------------------------------------------------------------
# PollingRunner
# ---------------------------------------------------------------------------

class PollingRunner:
    """
    Drives the timed poll loop for a single :class:`PollingAdapter`.

    ``CacheWatcher`` creates one ``PollingRunner`` per adapter whose
    ``mode`` is ``"polling"``.  The runner owns a background thread that
    calls :meth:`~glia.adapters.polling.PollingAdapter.poll` on the adapter
    at regular intervals, maps each yielded record to a ``source_id`` via
    :meth:`~glia.adapters.base.DatabaseAdapter.map_to_source_id`, and
    forwards the result to ``CacheWatcher`` through the ``dispatch`` callback.

    Parameters
    ----------
    adapter : PollingAdapter
        The concrete polling adapter to drive.  Must already be connected
        (``CacheWatcher.start()`` is responsible for calling
        ``adapter.connect()`` before creating the runner).
    dispatch : DispatchCallback
        Callable provided by ``CacheWatcher``.  Receives the resolved
        ``source_id`` string and the associated ``WatcherEvent`` each time
        a changed record is surfaced by ``poll()``.  The runner never
        calls ``CacheInvalidator`` directly — all invalidation is
        delegated through this callback.

    Attributes
    ----------
    adapter : PollingAdapter
        The adapter instance this runner drives.
    _dispatch : DispatchCallback
        The watcher callback; called once per changed record.
    _thread : threading.Thread
        The background thread managed by this runner.  Assigned in
        ``__init__()`` and started in ``start()``.
    _stop_event : threading.Event
        Internal signal used to request a clean shutdown.  Set by
        ``stop()``; checked at each iteration boundary inside ``_run()``.
    """

    def __init__(
        self,
        adapter: PollingAdapter,
        dispatch: DispatchCallback,
    ) -> None:
        self.adapter: PollingAdapter = adapter
        self._dispatch: DispatchCallback = dispatch

        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread = threading.Thread(
            target=self._run,
            name=f"glia.PollingRunner[{type(adapter).__name__}]",
            daemon=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the background polling thread.

        Raises
        ------
        RuntimeError
            If ``start()`` is called more than once on the same instance.
        """
        if self._thread.is_alive():
            raise RuntimeError(
                f"PollingRunner for {type(self.adapter).__name__} is already running. "
                "Call stop() before calling start() again."
            )
        # Clear any residual stop signal from a previous run so the loop
        # doesn't exit immediately on a restarted runner.
        self._stop_event.clear()
        self._thread.start()
        logger.debug(
            "PollingRunner[%s]: background thread started (interval=%.1fs).",
            type(self.adapter).__name__,
            self.adapter.poll_interval,
        )

    def stop(self) -> None:
        """
        Signal the polling loop to exit and block until the thread terminates.

        Idempotent: safe to call on an already-stopped runner.
        """
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join()
            logger.debug(
                "PollingRunner[%s]: background thread stopped.",
                type(self.adapter).__name__,
            )

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """
        Body of the background polling thread.

        Loops until ``_stop_event`` is set.  Each iteration:

        1. Calls ``adapter.poll()`` and iterates the returned records.
        2. For each record, resolves ``source_id`` via
           ``adapter.map_to_source_id(record)``.
        3. If ``source_id`` is not ``None``, builds a ``WatcherEvent`` and
           forwards it to ``self._dispatch(source_id, event)``.
        4. After exhausting all records, advances the cursor via
           ``adapter.advance_cursor(adapter.get_cursor())``.
        5. Sleeps for ``adapter.poll_interval`` seconds, interruptible by
           ``_stop_event`` so ``stop()`` is responsive even during long
           sleep intervals.
        6. Catches and logs all unexpected exceptions at ERROR level then
           continues — transient failures must not crash the thread.
        """
        adapter_name = type(self.adapter).__name__

        while not self._stop_event.is_set():
            try:
                records_seen = 0
                for record in self.adapter.poll():
                    if self._stop_event.is_set():
                        # Honour a stop request mid-batch without waiting for
                        # the full iteration to finish.
                        break

                    source_id: Optional[str] = self.adapter.map_to_source_id(record)
                    if source_id is None:
                        # Adapter signalled this record is not relevant —
                        # skip without raising.
                        continue

                    event = WatcherEvent(
                        event_type="watcher_event",
                        source_id=source_id,
                        adapter_type=adapter_name,
                        detection_mode="polling",
                    )

                    try:
                        self._dispatch(source_id, event)
                    except Exception:
                        logger.error(
                            "PollingRunner[%s]: dispatch raised for source_id=%r.",
                            adapter_name,
                            source_id,
                            exc_info=True,
                        )

                    records_seen += 1

                # Advance cursor only if we weren't asked to stop mid-batch.
                if not self._stop_event.is_set():
                    try:
                        self.adapter.advance_cursor(self.adapter.get_cursor())
                    except Exception:
                        logger.error(
                            "PollingRunner[%s]: advance_cursor() raised.",
                            adapter_name,
                            exc_info=True,
                        )

                logger.debug(
                    "PollingRunner[%s]: poll cycle complete — %d record(s) dispatched.",
                    adapter_name,
                    records_seen,
                )

            except Exception:
                logger.error(
                    "PollingRunner[%s]: unexpected exception in poll loop — "
                    "will retry after %.1fs.",
                    adapter_name,
                    self.adapter.poll_interval,
                    exc_info=True,
                )

            # Interruptible sleep: _stop_event.wait() returns True immediately
            # when stop() is called, so the thread exits without waiting the
            # full interval.
            self._stop_event.wait(timeout=self.adapter.poll_interval)

        logger.debug("PollingRunner[%s]: _run() exiting cleanly.", adapter_name)


# ---------------------------------------------------------------------------
# CDCRunner
# ---------------------------------------------------------------------------

class CDCRunner:
    """
    Drives the blocking stream listener for a single :class:`CDCAdapter`.

    ``CacheWatcher`` creates one ``CDCRunner`` per adapter whose ``mode``
    is ``"cdc"``.  The runner owns a background thread that calls
    :meth:`~glia.adapters.cdc.CDCAdapter.listen` on the adapter and
    iterates the resulting generator indefinitely, forwarding each yielded
    :class:`~glia.events.WatcherEvent` to ``CacheWatcher`` via the
    ``dispatch`` callback.

    On stream disconnection the runner retries the connection up to
    ``adapter.reconnect_retries`` times, waiting ``adapter.reconnect_delay``
    seconds between attempts, before giving up and terminating the thread.

    Parameters
    ----------
    adapter : CDCAdapter
        The concrete CDC adapter to drive.  Must already be connected
        (``CacheWatcher.start()`` calls ``adapter.connect()`` first).
    dispatch : DispatchCallback
        Callable provided by ``CacheWatcher``.  Receives the ``source_id``
        and the raw ``WatcherEvent`` for every change event surfaced by
        the CDC stream.

    Attributes
    ----------
    adapter : CDCAdapter
        The adapter instance this runner drives.
    _dispatch : DispatchCallback
        The watcher callback; called once per change event.
    _thread : threading.Thread
        The background thread managed by this runner.
    _stop_event : threading.Event
        Internal shutdown signal.  Set by ``stop()``; the ``listen()``
        generator is expected to observe ``adapter.stop()`` and terminate
        cleanly at its next iteration boundary.
    """

    def __init__(
        self,
        adapter: CDCAdapter,
        dispatch: DispatchCallback,
    ) -> None:
        self.adapter: CDCAdapter = adapter
        self._dispatch: DispatchCallback = dispatch

        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread = threading.Thread(
            target=self._run,
            name=f"glia.CDCRunner[{type(adapter).__name__}]",
            daemon=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the background CDC stream thread.

        Raises
        ------
        RuntimeError
            If ``start()`` is called more than once on the same instance.
        """
        if self._thread.is_alive():
            raise RuntimeError(
                f"CDCRunner for {type(self.adapter).__name__} is already running. "
                "Call stop() before calling start() again."
            )
        self._stop_event.clear()
        self._thread.start()
        logger.debug(
            "CDCRunner[%s]: background thread started.",
            type(self.adapter).__name__,
        )

    def stop(self) -> None:
        """
        Signal the CDC stream to shut down and block until the thread exits.

        Calls ``adapter.stop()`` first so the blocking ``listen()`` generator
        terminates at its next iteration boundary, then sets ``_stop_event``
        and joins the thread.  Idempotent.
        """
        # Tell the adapter's generator to exit before setting our own flag,
        # so the blocking listen() call unblocks and the thread can observe
        # _stop_event.
        try:
            self.adapter.stop()
        except Exception:
            logger.warning(
                "CDCRunner[%s]: adapter.stop() raised — proceeding with thread join.",
                type(self.adapter).__name__,
                exc_info=True,
            )

        self._stop_event.set()

        if self._thread.is_alive():
            self._thread.join()
            logger.debug(
                "CDCRunner[%s]: background thread stopped.",
                type(self.adapter).__name__,
            )

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """
        Body of the background CDC stream thread.

        Enters a retry loop bounded by ``adapter.reconnect_retries``.  On
        each attempt it calls ``adapter.listen()`` and iterates the blocking
        generator.  Each yielded ``WatcherEvent`` whose ``source_id`` is not
        ``None`` is forwarded to ``self._dispatch``.

        If the generator raises (stream drop):
        - If retries remain and ``_stop_event`` is not set, the runner
          waits ``adapter.reconnect_delay`` seconds, calls
          ``adapter.connect()``, and re-enters ``adapter.listen()``.
        - If retries are exhausted, a CRITICAL message is logged and the
          thread exits.

        A clean ``StopIteration`` (adapter called ``stop()``) exits the
        retry loop immediately without consuming a retry slot.
        """
        adapter_name = type(self.adapter).__name__
        retries_remaining = self.adapter.reconnect_retries

        while not self._stop_event.is_set():
            try:
                logger.debug("CDCRunner[%s]: entering listen() generator.", adapter_name)

                for event in self.adapter.listen():
                    if self._stop_event.is_set():
                        return

                    if event is None:
                        # Adapter yielded a heartbeat / irrelevant event.
                        continue

                    source_id: Optional[str] = event.source_id
                    if source_id is None:
                        continue

                    try:
                        self._dispatch(source_id, event)
                    except Exception:
                        logger.error(
                            "CDCRunner[%s]: dispatch raised for source_id=%r.",
                            adapter_name,
                            source_id,
                            exc_info=True,
                        )

                # Generator exhausted cleanly (adapter.stop() was called).
                logger.debug(
                    "CDCRunner[%s]: listen() generator exhausted — exiting.", adapter_name
                )
                return

            except Exception:
                if self._stop_event.is_set():
                    # Shutdown was requested; the exception is expected.
                    return

                logger.error(
                    "CDCRunner[%s]: stream error — %d retry attempt(s) remaining.",
                    adapter_name,
                    retries_remaining,
                    exc_info=True,
                )

                if retries_remaining <= 0:
                    logger.critical(
                        "CDCRunner[%s]: reconnect retries exhausted — "
                        "giving up and terminating thread.",
                        adapter_name,
                    )
                    return

                retries_remaining -= 1

                # Interruptible wait between reconnection attempts.
                if self._stop_event.wait(timeout=self.adapter.reconnect_delay):
                    return  # Stop was requested during the back-off wait.

                # Attempt to re-establish the upstream connection.
                try:
                    logger.info(
                        "CDCRunner[%s]: attempting reconnect (%d retries left).",
                        adapter_name,
                        retries_remaining,
                    )
                    self.adapter.connect()
                except Exception:
                    logger.error(
                        "CDCRunner[%s]: reconnect failed.",
                        adapter_name,
                        exc_info=True,
                    )
                    # The outer while loop will decrement retries again on
                    # the next iteration, so we continue rather than return.

        logger.debug("CDCRunner[%s]: _run() exiting cleanly.", adapter_name)