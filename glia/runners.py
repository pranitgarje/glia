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
        """
        Initialise the runner without starting the background thread.

        Parameters
        ----------
        adapter : PollingAdapter
            The polling adapter to drive.
        dispatch : DispatchCallback
            Callback supplied by ``CacheWatcher._dispatch``.
        """
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

        The thread runs :meth:`_run` in a loop, waking every
        ``adapter.poll_interval`` seconds to call ``adapter.poll()``.

        Raises
        ------
        RuntimeError
            If ``start()`` is called more than once on the same instance.
        """
        ...

    def stop(self) -> None:
        """
        Signal the polling loop to exit and block until the thread terminates.

        Sets the internal ``_stop_event`` so ``_run()`` exits at its next
        sleep boundary, then joins the background thread.  Safe to call
        from any thread, including the main thread.

        This method is idempotent: calling ``stop()`` on an already-stopped
        runner raises no error.
        """
        ...

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """
        Body of the background polling thread.

        Algorithm (to be implemented in Layer 5 work):

        1. Loop until ``_stop_event`` is set.
        2. Call ``adapter.poll()`` and iterate the returned records.
        3. For each record, call ``adapter.map_to_source_id(record)``.
        4. If ``source_id`` is not ``None``, build a ``WatcherEvent`` and
           invoke ``self._dispatch(source_id, event)``.
        5. After exhausting all records, call ``adapter.advance_cursor()``
           with the latest cursor from ``adapter.get_cursor()``.
        6. Sleep for ``adapter.poll_interval`` seconds (interruptible via
           ``_stop_event.wait()`` so ``stop()`` is responsive).
        7. On any unexpected exception, log at ERROR level and continue
           (avoid crashing the background thread on transient failures).

        This method is intentionally left empty at the skeleton stage.
        """
        ...


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
        """
        Initialise the runner without starting the background thread.

        Parameters
        ----------
        adapter : CDCAdapter
            The CDC adapter to drive.
        dispatch : DispatchCallback
            Callback supplied by ``CacheWatcher._dispatch``.
        """
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

        The thread runs :meth:`_run`, which blocks on the generator
        returned by ``adapter.listen()`` and forwards each event to
        ``self._dispatch``.

        Raises
        ------
        RuntimeError
            If ``start()`` is called more than once on the same instance.
        """
        ...

    def stop(self) -> None:
        """
        Signal the CDC stream to shut down and block until the thread exits.

        Calls ``adapter.stop()`` (which instructs the adapter's generator
        to terminate), sets ``_stop_event``, then joins the background thread.

        This method MUST be safe to call from a different thread than the
        one running ``_run()``.  It is idempotent.
        """
        ...

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """
        Body of the background CDC stream thread.

        Algorithm (to be implemented in Layer 5 work):

        1. Enter a retry loop bounded by ``adapter.reconnect_retries``.
        2. Call ``adapter.listen()`` to obtain the blocking generator.
        3. Iterate the generator; for each yielded ``WatcherEvent``:
           a. Extract ``source_id`` from ``event.source_id``.
           b. If ``source_id`` is not ``None``, call
              ``self._dispatch(source_id, event)``.
        4. If the generator raises an exception (stream drop):
           a. Log the error.
           b. If retries remain and ``_stop_event`` is not set, wait
              ``adapter.reconnect_delay`` seconds, call
              ``adapter.connect()``, and re-enter ``adapter.listen()``.
           c. If retries are exhausted, log a CRITICAL message and exit.
        5. If ``_stop_event`` is set, exit the retry loop cleanly.

        This method is intentionally left empty at the skeleton stage.
        """
        ...