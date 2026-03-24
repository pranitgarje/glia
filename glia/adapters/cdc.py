"""
glia/adapters/cdc.py
────────────────────
Layer 2 — Adapter Contracts

Defines CDCAdapter, the abstract contract that any Change-Data-Capture
(CDC) mode adapter must satisfy.

Architectural constraints (BDUF §2.3):
- CDCAdapter extends DatabaseAdapter.
- runners.CDCRunner types ONLY against this interface — it never
  imports a concrete adapter class.
- This file imports ONLY from the Python standard library and
  glia.adapters.base — never from higher-layer glia modules.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Generator

from glia.adapters.base import DatabaseAdapter
from glia.events import WatcherEvent


class CDCAdapter(DatabaseAdapter):
    """
    Abstract contract for CDC (Change-Data-Capture) mode adapters.

    A CDC adapter works by opening a long-lived stream connection to
    the data source and blocking on it.  Change events are pushed to
    the adapter rather than pulled.  ``CDCRunner`` runs the blocking
    loop in a background thread and calls ``listen()`` to consume the
    stream.

    Subclasses MUST implement all abstract methods inherited from
    ``DatabaseAdapter`` (``connect``, ``disconnect``,
    ``map_to_source_id``) in addition to the two methods declared
    here.

    Parameters
    ----------
    reconnect_retries : int
        How many times ``CDCRunner`` should attempt to re-establish
        the stream after an unexpected disconnection before giving up.
    reconnect_delay : float
        Seconds to wait between consecutive reconnection attempts.
    **kwargs
        Forwarded verbatim to ``DatabaseAdapter.__init__()`` so that
        ``mode=`` and ``source_id_field=`` (and any future base
        parameters) are handled transparently.
    """

    def __init__(
        self,
        reconnect_retries: int,
        reconnect_delay: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.reconnect_retries: int = reconnect_retries
        self.reconnect_delay: float = reconnect_delay

    # ------------------------------------------------------------------
    # CDC contract
    # ------------------------------------------------------------------

    @abstractmethod
    def listen(self) -> Generator[WatcherEvent, None, None]:
        """
        Open the CDC stream and yield ``WatcherEvent`` objects indefinitely.

        This is a *blocking* generator.  ``CDCRunner`` calls it inside
        a dedicated background thread and iterates it for the lifetime
        of the watcher.  Each yielded ``WatcherEvent`` is forwarded
        directly to ``CacheWatcher._dispatch()``.

        The generator MUST:
        - Block between events rather than busy-loop.
        - Yield ``None`` or skip silently for irrelevant events so the
          runner can continue without interruption.
        - Raise ``StopIteration`` (or simply ``return``) only when
          ``stop()`` has been called, signalling a clean shutdown.

        Yields
        ------
        WatcherEvent
            Structured change-event objects.  Concrete adapters are
            responsible for mapping raw source events (e.g. a Postgres
            logical-replication message, a Neo4j CDC event dict, a
            Redis keyspace notification) into ``WatcherEvent`` instances
            before yielding them.

        Notes
        -----
        Reconnection on transient failures is the responsibility of
        ``CDCRunner``, which will call ``connect()`` and re-enter
        ``listen()`` up to ``reconnect_retries`` times.  The adapter
        MUST raise an exception (not swallow it) when the stream drops
        so the runner can act on it.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """
        Signal the CDC stream to shut down cleanly.

        Called by ``CDCRunner`` when ``CacheWatcher.stop()`` is
        invoked.  After this call, the generator returned by
        ``listen()`` MUST terminate at its next iteration boundary
        without processing further events.

        This method MUST be safe to call from a different thread than
        the one running ``listen()``.
        """
        raise NotImplementedError