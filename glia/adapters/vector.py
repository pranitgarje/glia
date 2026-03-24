"""
glia/adapters/vector.py
────────────────────────
Layer 4 — Concrete Adapters

Defines VectorDBAdapter, a concrete adapter for vector database sources
(e.g. Pinecone, Weaviate, Qdrant, Chroma).

The adapter is generic: it works with any vector store whose client exposes
the following interface:

  Polling mode
  ────────────
  client.fetch_updated(
      collection      : str,
      since           : Any,   # last_cursor value; None on first run
      timestamp_field : str,   # metadata field used as the watermark
  ) -> List[Dict[str, Any]]
      Returns a list of record dicts.  Each dict must contain at minimum:
          { "<source_id_field>": <str>, "<timestamp_field>": <comparable> }

  CDC mode
  ────────
  client.subscribe(collection: str) -> Iterator[Dict[str, Any]]
      Returns a blocking iterator that yields raw change-event dicts.
      Each dict must contain at minimum:
          {
              "event_type":        "upsert" | "delete",
              "<source_id_field>": <str>,
          }
      The iterator must stop (raise StopIteration or return) once the
      underlying stream is closed — the adapter signals this via stop().

  Connection probe (used by connect())
  ─────────────────────────────────────
  client.describe_collection(collection: str)   # preferred
  client.get_collection(collection: str)         # fallback
      Either is used as a lightweight liveness check.  If the client
      exposes neither, the probe is skipped and errors will surface on
      the first real query.

Architectural constraints:
- Inherits from PollingAdapter and CDCAdapter (dual-inheritance, MRO-safe).
- MUST NOT import from Layer 3 (manager.py, invalidator.py),
  Layer 5 (runners.py, watcher.py), or Layer 6 (__init__.py).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Generator, Iterator, Optional

from glia.adapters.polling import PollingAdapter
from glia.adapters.cdc import CDCAdapter
from glia.events import WatcherEvent
from glia.exceptions import AdapterConnectionError

logger = logging.getLogger(__name__)


class VectorDBAdapter(PollingAdapter, CDCAdapter):
    """
    Concrete adapter for generic vector database sources.

    Supports both polling and CDC execution modes.  Pass ``mode="polling"``
    for timed-interval timestamp / state-hash checks, or ``mode="cdc"`` to
    consume a real-time change stream where the vector store pushes upsert /
    delete events.

    Parameters
    ----------
    client : Any
        Instantiated vector store client.  Not managed by Glia — the
        developer creates and configures it before passing it in.
    collection : str
        Name of the collection / namespace / index within the vector store
        to monitor (e.g. ``"documents"`` in Qdrant, an index name in
        Pinecone).
    timestamp_field : str
        Metadata field whose value acts as the polling watermark
        (e.g. ``"updated_at"``).  Values must be comparable with ``>`` so
        that records newer than ``last_cursor`` can be identified.
        Only used when ``mode="polling"``.
    mode : str
        Execution strategy — ``"polling"`` or ``"cdc"``.
    source_id_field : str
        Metadata field whose value becomes Glia's ``source_id`` TAG
        (e.g. ``"doc_id"``, ``"file_path"``).
    poll_interval : float
        Seconds between consecutive ``poll()`` calls.  Defaults to ``30.0``.
    last_cursor : Any, optional
        Initial watermark.  ``None`` signals a bootstrap run — all records
        in the collection are returned on the first ``poll()`` call.
    reconnect_retries : int
        Maximum reconnection attempts before ``CDCRunner`` gives up.
        Defaults to ``3``.
    reconnect_delay : float
        Seconds between consecutive reconnection attempts.  Defaults to
        ``5.0``.
    **kwargs
        Forwarded to parent initialisers via cooperative ``super()``.
    """

    def __init__(
        self,
        client: Any,
        collection: str,
        timestamp_field: str,
        mode: str,
        source_id_field: str,
        poll_interval: float = 30.0,
        last_cursor: Optional[Any] = None,
        reconnect_retries: int = 3,
        reconnect_delay: float = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            mode=mode,
            source_id_field=source_id_field,
            poll_interval=poll_interval,
            last_cursor=last_cursor,
            reconnect_retries=reconnect_retries,
            reconnect_delay=reconnect_delay,
            **kwargs,
        )
        self.client: Any = client
        self.collection: str = collection
        self.timestamp_field: str = timestamp_field

        # ── Internal state ────────────────────────────────────────────────
        # _connected: idempotency guard for connect() / disconnect().
        self._connected: bool = False

        # _stop_event: set by stop() to signal listen() to exit its loop.
        # threading.Event is the correct primitive here — it is both
        # thread-safe and non-blocking to set, so stop() never blocks the
        # caller even while the background thread is blocked inside the
        # client's subscribe() iterator.
        self._stop_event: threading.Event = threading.Event()

        # _current_cursor: high-water-mark accumulated inside poll().
        # Kept separate from last_cursor so that a partial batch (exception
        # mid-iteration) does not silently advance the committed watermark.
        # advance_cursor() writes both once the full batch is dispatched.
        self._current_cursor: Any = last_cursor

    # ------------------------------------------------------------------
    # DatabaseAdapter contract — connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Verify reachability of the vector store and the target collection.

        Performs a lightweight probe (``describe_collection`` or
        ``get_collection``) to confirm the collection exists before any
        polling or CDC work begins.  Clients that expose neither method are
        assumed to be connection-pooled; errors will surface on the first
        real query instead.

        Idempotent: a second call on an already-connected adapter is a
        no-op.

        Raises
        ------
        AdapterConnectionError
            If the probe raises any exception.
        """
        if self._connected:
            logger.debug(
                "VectorDBAdapter.connect(): already connected to %r — "
                "skipping.",
                self.collection,
            )
            return

        try:
            if hasattr(self.client, "describe_collection"):
                self.client.describe_collection(self.collection)
            elif hasattr(self.client, "get_collection"):
                self.client.get_collection(self.collection)
            # If neither probe method exists we skip the check silently.
        except Exception as exc:
            raise AdapterConnectionError(
                f"VectorDBAdapter: failed to connect to collection "
                f"'{self.collection}': {exc}"
            ) from exc

        self._connected = True
        logger.info(
            "VectorDBAdapter connected (collection=%r).", self.collection
        )

    def disconnect(self) -> None:
        """
        Release resources held by the vector store client.

        Calls ``client.close()`` if the method exists; otherwise the client
        is left in its current state (connection-pooled clients manage their
        own lifecycle).  The ``client`` object itself is not destroyed —
        it was injected by the developer and may be shared.

        Idempotent: safe to call on an already-disconnected adapter.
        """
        if not self._connected:
            return

        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception:
            logger.warning(
                "VectorDBAdapter.disconnect(): exception while closing "
                "client for %r (ignored).",
                self.collection,
                exc_info=True,
            )
        finally:
            self._connected = False
            logger.info(
                "VectorDBAdapter disconnected (collection=%r).",
                self.collection,
            )

    # ------------------------------------------------------------------
    # DatabaseAdapter contract — source_id mapping
    # ------------------------------------------------------------------

    def map_to_source_id(self, record: Any) -> Optional[str]:
        """
        Extract the ``source_id`` string from a raw vector store record.

        Reads ``record[self.source_id_field]`` (dict) or
        ``record.<source_id_field>`` (object).  Returns ``None`` when the
        field is absent, empty, or the record itself is ``None`` — the
        runner will skip those records without triggering invalidation.

        The extracted value is always cast to ``str`` so UUID ``node.id_``
        values and arbitrary string identifiers are handled uniformly.

        Parameters
        ----------
        record : Any
            A raw record dict or object as yielded by ``poll()`` or emitted
            by ``listen()``.

        Returns
        -------
        str or None
        """
        if record is None:
            return None

        if isinstance(record, dict):
            value = record.get(self.source_id_field)
        else:
            value = getattr(record, self.source_id_field, None)

        if not value:
            logger.debug(
                "VectorDBAdapter.map_to_source_id(): field %r absent or "
                "empty — skipping record.",
                self.source_id_field,
            )
            return None

        return str(value)

    # ------------------------------------------------------------------
    # PollingAdapter contract
    # ------------------------------------------------------------------

    def poll(self) -> Iterator[Any]:
        """
        Fetch records whose ``timestamp_field`` value exceeds
        ``last_cursor`` and yield them one at a time.

        On the first run (``last_cursor is None``) the adapter performs a
        full bootstrap sweep — every record in the collection is returned so
        a complete baseline can be established.

        The highest ``timestamp_field`` value seen across the batch is
        stored in ``_current_cursor``.  ``PollingRunner`` reads it via
        ``get_cursor()`` and commits it via ``advance_cursor()`` *after* the
        full batch has been dispatched — ``last_cursor`` is never mutated
        inside this method.

        Yields
        ------
        dict
            One raw record dict per changed / new document.

        Raises
        ------
        AdapterConnectionError
            Wraps any exception raised by the client during the fetch so
            the runner receives a typed error it can act on.
        """
        logger.debug(
            "VectorDBAdapter.poll(): fetching updates since cursor=%r "
            "(collection=%r).",
            self.last_cursor,
            self.collection,
        )

        try:
            records = self.client.fetch_updated(
                self.collection,
                since=self.last_cursor,
                timestamp_field=self.timestamp_field,
            )
        except Exception as exc:
            raise AdapterConnectionError(
                f"VectorDBAdapter: poll() failed for collection "
                f"'{self.collection}': {exc}"
            ) from exc

        if not records:
            logger.debug(
                "VectorDBAdapter.poll(): no new records (collection=%r).",
                self.collection,
            )
            return

        high_water: Any = self._current_cursor

        for record in records:
            # Track the maximum timestamp seen in this batch.
            if isinstance(record, dict):
                ts = record.get(self.timestamp_field)
            else:
                ts = getattr(record, self.timestamp_field, None)

            if ts is not None and (high_water is None or ts > high_water):
                high_water = ts

            yield record

        # Persist the high-water-mark for get_cursor() to return.
        # Do NOT write to last_cursor here — that is advance_cursor()'s job.
        self._current_cursor = high_water

    def get_cursor(self) -> Any:
        """
        Return the highest ``timestamp_field`` value observed during the
        most recent ``poll()`` batch.

        ``PollingRunner`` calls this immediately after draining the
        generator to obtain the value it will pass to ``advance_cursor()``.

        Returns
        -------
        Any
            Current high-water-mark.  ``None`` before the first successful
            poll.
        """
        return self._current_cursor

    def advance_cursor(self, new_cursor: Any) -> None:
        """
        Commit ``new_cursor`` as the lower bound for the next ``poll()``
        call.

        Called by ``PollingRunner`` only after every record in the current
        batch has been successfully dispatched.  Keeping cursor advancement
        here — rather than inside ``poll()`` — guarantees that a
        mid-iteration exception does not silently skip records on the next
        cycle.

        Parameters
        ----------
        new_cursor : Any
            The value returned by ``get_cursor()`` after the batch was
            consumed.
        """
        logger.debug(
            "VectorDBAdapter.advance_cursor(): %r → %r (collection=%r).",
            self.last_cursor,
            new_cursor,
            self.collection,
        )
        self.last_cursor = new_cursor
        self._current_cursor = new_cursor

    # ------------------------------------------------------------------
    # CDCAdapter contract
    # ------------------------------------------------------------------

    def listen(self) -> Generator[WatcherEvent, None, None]:
        """
        Subscribe to the vector store's change stream and yield a
        ``WatcherEvent`` for every upsert or delete event.

        This is a **blocking generator**.  ``CDCRunner`` drives it from a
        dedicated background thread so the main thread is never blocked.

        The generator exits cleanly when ``stop()`` has been called — it
        checks ``_stop_event`` before processing each raw event from the
        stream.  ``_stop_event`` is cleared at entry so that ``listen()``
        is safely re-entrant after a CDCRunner reconnect attempt.

        Raw event dict expected from ``client.subscribe()``:

        .. code-block:: python

            {
                "event_type":        "upsert" | "delete",
                "<source_id_field>": "<document identifier>",
                # any additional fields are forwarded to WatcherEvent.payload
            }

        ``None`` values are treated as heartbeat / keepalive signals and
        are skipped silently.

        Upsert events → ``WatcherEvent(payload={"fs_event_type": "modified"})``
        Delete events → ``WatcherEvent(payload={"fs_event_type": "deleted"})``

        The ``fs_event_type`` key name mirrors the filesystem watcher
        convention used elsewhere in Glia so downstream consumers have a
        single, uniform payload shape regardless of adapter type.

        Yields
        ------
        WatcherEvent
            One structured event per upsert or delete in the monitored
            collection.

        Raises
        ------
        AdapterConnectionError
            If ``client.subscribe()`` raises during stream initialisation.
        """
        logger.info(
            "VectorDBAdapter.listen(): opening CDC stream (collection=%r).",
            self.collection,
        )

        # Clear any previous stop signal so reconnects work correctly.
        self._stop_event.clear()

        try:
            stream = self.client.subscribe(self.collection)
        except Exception as exc:
            raise AdapterConnectionError(
                f"VectorDBAdapter: failed to open CDC stream for collection "
                f"'{self.collection}': {exc}"
            ) from exc

        for raw_event in stream:
            # Primary exit path for a clean shutdown request.
            if self._stop_event.is_set():
                logger.info(
                    "VectorDBAdapter.listen(): stop event set — exiting CDC "
                    "loop (collection=%r).",
                    self.collection,
                )
                return

            # Heartbeat / keepalive — skip without yielding.
            if raw_event is None:
                continue

            # Normalise event_type to the two operations Glia cares about.
            # Any value other than "delete" is treated as "upsert" (the safe
            # default — invalidation is idempotent so a false positive is
            # preferable to a missed delete).
            if isinstance(raw_event, dict):
                raw_type: str = raw_event.get("event_type", "upsert")
            else:
                raw_type = getattr(raw_event, "event_type", "upsert")

            fs_event_type: str = (
                "deleted" if raw_type.lower() == "delete" else "modified"
            )

            source_id: Optional[str] = self.map_to_source_id(raw_event)

            if source_id is None:
                # No usable source_id — skip without triggering invalidation.
                logger.debug(
                    "VectorDBAdapter.listen(): skipping event with no "
                    "source_id (event_type=%r, collection=%r).",
                    raw_type,
                    self.collection,
                )
                continue

            yield WatcherEvent(
                event_type="watcher_event",
                source_id=source_id,
                adapter_type="vector",
                detection_mode="cdc",
                payload={
                    "collection": self.collection,
                    "fs_event_type": fs_event_type,
                },
            )

        # Stream exhausted — the client closed the connection naturally or
        # stop() was honoured by ending the subscribe() iterator.
        logger.info(
            "VectorDBAdapter.listen(): CDC stream closed (collection=%r).",
            self.collection,
        )

    def stop(self) -> None:
        """
        Signal ``listen()`` to exit after the current event is processed.

        Sets ``_stop_event`` so the generator breaks out of its loop at the
        next iteration boundary.  Thread-safe — safe to call from any thread,
        including the main thread while the background CDC thread is blocked
        inside the stream iterator.
        """
        logger.info(
            "VectorDBAdapter.stop(): signalling CDC shutdown "
            "(collection=%r).",
            self.collection,
        )
        self._stop_event.set()