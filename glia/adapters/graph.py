"""
glia/adapters/graph.py
──────────────────────
Layer 4 — Concrete Adapters

Defines GraphDBAdapter, a concrete adapter for graph database sources,
with first-class support for Neo4j (the dominant graph database used in
RAG / knowledge-graph pipelines).

The adapter is generic: it works with any graph database whose driver
exposes the following interface:

  Connection probe (used by connect())
  ─────────────────────────────────────
  driver.verify_connectivity()          # Neo4j Python driver 5.x
  driver.session()                      # fallback — open a session as probe

  Polling mode
  ────────────
  The adapter executes a developer-supplied Cypher query via a driver
  session.  Two sub-strategies are supported:

  Strategy A — developer-supplied ``change_query`` (default)
      The developer provides any Cypher query that returns nodes matching
      the change window.  The query MUST be parameterised with:

          $cursor      — the last committed watermark value (datetime /
                         epoch ms / None on bootstrap)
          $since       — alias for $cursor (either is accepted)

      The adapter passes BOTH so the query author can use whichever name
      they prefer.  Example:

          MATCH (n:Document)
          WHERE n.last_modified > $cursor
          RETURN n
          ORDER BY n.last_modified ASC

  Strategy B — default built-in query (``change_query`` left empty / None)
      When no custom query is supplied the adapter generates:

          MATCH (n)
          WHERE n.`<source_id_field>` IS NOT NULL
            AND ($cursor IS NULL OR n.last_modified > $cursor)
          RETURN n
          ORDER BY n.last_modified ASC

      This covers the common case where every node carries a ``last_modified``
      property and a stable identifier property.

  CDC mode
  ────────
  Neo4j 5.x ships a CDC (Change Data Capture) API that can be queried via
  Cypher or consumed through a dedicated streaming interface.  The adapter
  models this through the driver's session.run() interface using the
  Neo4j CDC query pattern, OR through a ``cdc_stream`` client object that
  wraps a higher-level streaming subscription.

  The driver (or cdc_stream) must support one of:

  Option A — Session-based CDC query (Neo4j 5.x built-in CDC)
      driver.session().run(
          "CALL db.cdc.query($from_id, $limit)",
          from_id=<cursor_id>,  # opaque string; None on bootstrap
          limit=<int>,
      )
      Returns a result whose records carry:
          {
              "id":        str,   # opaque CDC cursor for next query
              "txId":      int,
              "seq":       int,
              "event": {
                  "elementId":  str,     # Neo4j element ID of the affected node
                  "operation":  "c" | "u" | "d",   # create/update/delete
                  "labels":     List[str],
                  "state": {
                      "before": Dict | None,
                      "after":  Dict | None,
                  },
              },
          }

  Option B — Streaming CDC client (e.g. Debezium Neo4j connector)
      driver.cdc_stream(database="neo4j") -> Iterator[Dict[str, Any]]
      Yields raw change event dicts with the same shape as Option A records.

  The adapter detects which option is available at listen() entry and
  uses the first one it can establish.

Architectural constraints:
- Inherits from PollingAdapter and CDCAdapter (dual-inheritance, MRO-safe).
- MUST NOT import from Layer 3 (manager.py, invalidator.py),
  Layer 5 (runners.py, watcher.py), or Layer 6 (__init__.py).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Generator, Iterator, List, Optional

from glia.adapters.polling import PollingAdapter
from glia.adapters.cdc import CDCAdapter
from glia.events import WatcherEvent
from glia.exceptions import AdapterConnectionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Neo4j CDC operation codes → human-readable labels used in WatcherEvent.
_CDC_OPERATION_MAP: Dict[str, str] = {
    "c": "created",    # node / relationship created
    "u": "modified",   # properties updated
    "d": "deleted",    # node / relationship deleted
}

# Operations that warrant cache invalidation.
# Heartbeats, schema events, and unknown operation codes are skipped.
_INVALIDATING_CDC_OPS: frozenset = frozenset({"c", "u", "d"})

# fs_event_type value used in WatcherEvent payload — mirrors the filesystem
# watcher convention so downstream consumers have a uniform payload shape.
_FS_EVENT_MAP: Dict[str, str] = {
    "c": "modified",   # creation of a source node = new data → invalidate
    "u": "modified",
    "d": "deleted",
}

# Maximum number of CDC events to fetch per built-in query cycle.
_CDC_QUERY_PAGE_SIZE: int = 100

# Default Cypher template used when the developer does not supply change_query.
# Both $cursor and $since are passed as parameters so the developer can use
# either name in their own queries without friction.
_DEFAULT_CHANGE_QUERY: str = (
    "MATCH (n) "
    "WHERE n.`{source_id_field}` IS NOT NULL "
    "  AND ($cursor IS NULL OR n.last_modified > $cursor) "
    "RETURN n "
    "ORDER BY n.last_modified ASC"
)


class GraphDBAdapter(PollingAdapter, CDCAdapter):
    """
    Concrete adapter for graph database sources (Neo4j and compatible).

    Supports both polling and CDC execution modes.

    **Polling mode** (``mode="polling"``)
        Periodically executes a Cypher query (developer-supplied via
        ``change_query``, or a built-in default) that returns nodes whose
        ``last_modified`` property exceeds the last committed cursor.
        Each returned node dict is yielded to ``PollingRunner``, which
        calls ``map_to_source_id()`` and forwards the result to
        ``CacheInvalidator.delete_by_tag()``.

    **CDC mode** (``mode="cdc"``)
        Streams node and relationship mutations from Neo4j's built-in CDC
        API (``db.cdc.query``) or a Debezium-style streaming client.
        Each DML event (create / update / delete) on a node that carries
        ``source_id_field`` is mapped to a ``WatcherEvent`` and yielded
        to ``CDCRunner``.

    Parameters
    ----------
    driver : Any
        Instantiated Neo4j ``Driver`` (or compatible graph driver).
        Not managed by Glia — the developer creates and configures it.
    change_query : str, optional
        Cypher query used for polling.  Must be parameterised with
        ``$cursor`` (and/or ``$since``).  If ``None`` or empty, the
        adapter generates a default query using ``source_id_field`` and
        a ``last_modified`` property.
    mode : str
        Execution strategy — ``"polling"`` or ``"cdc"``.
    source_id_field : str
        Node property whose value becomes Glia's ``source_id`` TAG
        (e.g. ``"doc_id"``, ``"file_path"``, ``"node_uuid"``).
    node_label : str, optional
        Neo4j label used to narrow the CDC subscription and the default
        polling query (e.g. ``"Document"``).  Defaults to ``""``
        (match all labels).
    database : str, optional
        Neo4j database name to target.  Defaults to ``"neo4j"``.
    poll_interval : float
        Seconds between consecutive ``poll()`` calls.  Defaults to ``30.0``.
    last_cursor : Any, optional
        Initial watermark (datetime, epoch ms, or opaque CDC cursor string).
        ``None`` triggers a full bootstrap sweep on first poll.
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
        driver: Any,
        change_query: Optional[str],
        mode: str,
        source_id_field: str,
        node_label: str = "",
        database: str = "neo4j",
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
        self.driver: Any = driver
        self.change_query: str = change_query or ""
        self.node_label: str = node_label
        self.database: str = database

        # ── Internal state ────────────────────────────────────────────────
        # _connected: idempotency guard for connect() / disconnect().
        self._connected: bool = False

        # _stop_event: set by stop() to signal the listen() generator to
        # exit its loop safely from any thread.
        self._stop_event: threading.Event = threading.Event()

        # _current_cursor: high-water-mark accumulated inside poll().
        # Kept separate from last_cursor so a partial batch caused by a
        # mid-iteration exception does not silently skip records on the
        # next cycle.  advance_cursor() writes both atomically.
        self._current_cursor: Any = last_cursor

        # _cdc_cursor: opaque CDC cursor string used by the Neo4j
        # db.cdc.query() built-in to page through the transaction log.
        # Updated after each CDC fetch cycle inside listen().
        self._cdc_cursor: Optional[str] = None

    # ------------------------------------------------------------------
    # DatabaseAdapter contract — connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Verify that the Neo4j driver can reach the database.

        Calls ``driver.verify_connectivity()`` (Neo4j Python driver 5.x)
        if available; otherwise opens and immediately closes a session as a
        lightweight liveness probe.

        Idempotent: a second call on an already-connected adapter is a
        no-op.

        Raises
        ------
        AdapterConnectionError
            If the connectivity probe raises any exception.
        """
        if self._connected:
            logger.debug(
                "GraphDBAdapter.connect(): already connected to database=%r "
                "— skipping.",
                self.database,
            )
            return

        try:
            if hasattr(self.driver, "verify_connectivity"):
                # Neo4j Python driver 5.x preferred path — single round-trip.
                self.driver.verify_connectivity()
            elif hasattr(self.driver, "session"):
                # Fallback: open a session, run a trivial query, close it.
                with self.driver.session(database=self.database) as session:
                    session.run("RETURN 1").consume()
            else:
                # Unknown driver type — optimistically assume it is live;
                # errors will surface on the first real query.
                logger.debug(
                    "GraphDBAdapter.connect(): unrecognised driver type %r "
                    "— skipping probe.",
                    type(self.driver).__name__,
                )
        except Exception as exc:
            raise AdapterConnectionError(
                f"GraphDBAdapter: failed to connect to Neo4j database "
                f"'{self.database}': {exc}"
            ) from exc

        self._connected = True
        logger.info(
            "GraphDBAdapter connected (database=%r).", self.database
        )

    def disconnect(self) -> None:
        """
        Close the Neo4j driver cleanly.

        Calls ``driver.close()`` if available.  The driver object is not
        destroyed — it was injected by the developer and may be shared.

        Idempotent: safe to call on an already-disconnected adapter.
        """
        if not self._connected:
            return

        try:
            if hasattr(self.driver, "close"):
                self.driver.close()
        except Exception:
            logger.warning(
                "GraphDBAdapter.disconnect(): exception while closing driver "
                "for database=%r (ignored).",
                self.database,
                exc_info=True,
            )
        finally:
            self._connected = False
            logger.info(
                "GraphDBAdapter disconnected (database=%r).", self.database
            )

    # ------------------------------------------------------------------
    # DatabaseAdapter contract — source_id mapping
    # ------------------------------------------------------------------

    def map_to_source_id(self, record: Any) -> Optional[str]:
        """
        Derive a stable Glia ``source_id`` from a raw graph record.

        The method handles three record shapes that the adapter produces:

        **Polling result — Neo4j node object**
            ``record`` is a dict (or Neo4j ``Node`` proxy) with a ``"n"``
            key whose value is the node, or directly a node dict / object.
            The ``source_id_field`` property is extracted from the node's
            property map.

        **Polling result — flat property dict**
            ``record`` is a plain dict of node properties (as returned when
            the Cypher query uses ``RETURN n {.*}`` projection syntax).
            ``source_id_field`` is read directly.

        **CDC event dict**
            ``record`` is a CDC change-event dict (Neo4j ``db.cdc.query``
            format or Debezium-style).  The ``source_id_field`` property
            is extracted from ``event.state.after`` (create / update) or
            ``event.state.before`` (delete).

        The final ``source_id`` uses a namespaced format::

            "node:<node_label>|id:<property_value>"

        This namespace prevents collisions when multiple node labels share
        the same property value space.  When ``node_label`` is empty the
        segment is omitted::

            "node|id:<property_value>"

        Parameters
        ----------
        record : Any
            Raw record dict or Neo4j result object as yielded by ``poll()``
            or ``listen()``.

        Returns
        -------
        str or None
            Stable ``source_id`` string, or ``None`` to skip the record
            (heartbeats, schema events, nodes missing the target property).
        """
        if record is None:
            return None

        prop_value: Optional[Any] = None

        # ── CDC event dict ───────────────────────────────────────────────
        # Identified by the presence of an "event" key (Neo4j CDC) or an
        # "operation" key (Debezium-style flattened event).
        if isinstance(record, dict) and "event" in record:
            event_block: Dict = record.get("event") or {}
            operation: str = event_block.get("operation", "")

            if operation not in _INVALIDATING_CDC_OPS:
                return None

            state: Dict = event_block.get("state") or {}
            # DELETE → use before-state; CREATE/UPDATE → use after-state.
            row_data: Optional[Dict] = (
                state.get("before") if operation == "d" else state.get("after")
            )
            if not row_data:
                return None

            prop_value = (row_data.get("properties") or {}).get(
                self.source_id_field
            )

        # ── Debezium-style flat CDC event ────────────────────────────────
        elif isinstance(record, dict) and "operation" in record:
            op: str = record.get("operation", "")
            if op not in _INVALIDATING_CDC_OPS:
                return None

            row_data = (
                record.get("before") if op == "d" else record.get("after")
            )
            if not row_data:
                return None

            prop_value = (
                row_data.get(self.source_id_field)
                if isinstance(row_data, dict)
                else getattr(row_data, self.source_id_field, None)
            )

        # ── Polling result — Cypher record with an "n" alias ─────────────
        # Standard pattern: MATCH (n:Document) ... RETURN n
        # The Neo4j driver wraps the record in a dict-like object keyed
        # by query alias.
        elif isinstance(record, dict) and "n" in record:
            node = record["n"]
            # Neo4j Node objects expose a dict-like .items() interface;
            # fallback to direct dict access for mock / stub nodes.
            if hasattr(node, "get"):
                prop_value = node.get(self.source_id_field)
            elif hasattr(node, "_properties"):
                prop_value = node._properties.get(self.source_id_field)
            else:
                prop_value = getattr(node, self.source_id_field, None)

        # ── Polling result — flat property dict ──────────────────────────
        # Pattern: MATCH (n) RETURN n {.*}  or  RETURN n.prop AS prop
        elif isinstance(record, dict):
            prop_value = record.get(self.source_id_field)

        # ── Object-style node (OGM / dataclass) ──────────────────────────
        else:
            prop_value = getattr(record, self.source_id_field, None)

        if prop_value is None:
            logger.debug(
                "GraphDBAdapter.map_to_source_id(): property %r absent or "
                "None in record — skipping.",
                self.source_id_field,
            )
            return None

        label_segment = self.node_label if self.node_label else ""
        if label_segment:
            return f"node:{label_segment}|id:{prop_value}"
        return f"node|id:{prop_value}"

    # ------------------------------------------------------------------
    # PollingAdapter contract
    # ------------------------------------------------------------------

    def poll(self) -> Iterator[Any]:
        """
        Execute the Cypher change query and yield one record per modified
        node.

        The query is parameterised with both ``$cursor`` and ``$since``
        (same value) so developers can use either variable name in their
        custom Cypher.  On bootstrap (``last_cursor is None``) both
        parameters are ``None``, which the default query interprets as
        "return all nodes" via the ``$cursor IS NULL`` guard.

        The method tracks the highest ``last_modified`` value seen across
        the batch in ``_current_cursor``.  ``PollingRunner`` reads this via
        ``get_cursor()`` and commits it via ``advance_cursor()`` only after
        the full batch has been dispatched — ``last_cursor`` is never
        mutated here.

        Yields
        ------
        dict
            Raw Cypher result records (Neo4j ``Record``-like dicts).

        Raises
        ------
        AdapterConnectionError
            Wraps any driver-level exception so ``PollingRunner`` receives
            a typed error.
        """
        cypher = self._resolve_query()
        params: Dict[str, Any] = {
            "cursor": self.last_cursor,
            "since":  self.last_cursor,   # alias — query can use either name
        }

        logger.debug(
            "GraphDBAdapter.poll(): executing change query "
            "(database=%r, cursor=%r).",
            self.database,
            self.last_cursor,
        )

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, params)
                records: List[Any] = list(result)   # materialise inside session
        except Exception as exc:
            raise AdapterConnectionError(
                f"GraphDBAdapter: poll() query failed "
                f"(database='{self.database}'): {exc}"
            ) from exc

        if not records:
            logger.debug(
                "GraphDBAdapter.poll(): no new records (database=%r).",
                self.database,
            )
            return

        high_water: Any = self._current_cursor

        for raw_record in records:
            # Extract last_modified from the node for watermark tracking.
            # Try the "n" alias first (standard RETURN n pattern), then fall
            # back to a top-level key in case the query uses a projection.
            ts = self._extract_last_modified(raw_record)
            if ts is not None and (high_water is None or ts > high_water):
                high_water = ts

            # Yield as a plain dict so map_to_source_id() and PollingRunner
            # always receive a consistent, JSON-serialisable type.
            if hasattr(raw_record, "data"):
                yield raw_record.data()     # Neo4j Record.data() → dict
            elif hasattr(raw_record, "_asdict"):
                yield raw_record._asdict()  # named-tuple style
            else:
                yield dict(raw_record)      # already dict-like

        self._current_cursor = high_water

    def get_cursor(self) -> Any:
        """
        Return the highest ``last_modified`` value observed in the most
        recent ``poll()`` batch.

        ``PollingRunner`` calls this immediately after draining the batch
        generator to obtain the value it will pass to ``advance_cursor()``.

        Returns
        -------
        Any
            Current high-water-mark (datetime or comparable value).
            ``None`` before the first successful poll.
        """
        return self._current_cursor

    def advance_cursor(self, new_cursor: Any) -> None:
        """
        Commit ``new_cursor`` as the lower bound for the next ``poll()``
        call.

        Called by ``PollingRunner`` only after every record in the current
        batch has been successfully dispatched, ensuring that a
        mid-iteration exception does not silently skip records.

        Parameters
        ----------
        new_cursor : Any
            The value returned by ``get_cursor()`` after the batch was
            consumed.
        """
        logger.debug(
            "GraphDBAdapter.advance_cursor(): %r → %r (database=%r).",
            self.last_cursor,
            new_cursor,
            self.database,
        )
        self.last_cursor = new_cursor
        self._current_cursor = new_cursor

    # ------------------------------------------------------------------
    # CDCAdapter contract
    # ------------------------------------------------------------------

    def listen(self) -> Generator[WatcherEvent, None, None]:
        """
        Stream node and relationship mutations from Neo4j's CDC API and
        yield a ``WatcherEvent`` for each invalidating event.

        This is a **blocking generator**.  ``CDCRunner`` drives it from a
        dedicated background thread so the main thread is never blocked.

        **Detection strategy** (selected at listen() entry):

        1. **Built-in Neo4j CDC** — if ``driver.session()`` exposes the
           ``db.cdc.query`` procedure (Neo4j 5.x Enterprise), the generator
           enters a tight page-query loop: it calls ``db.cdc.query`` with
           the current ``_cdc_cursor``, processes the page, advances the
           cursor to the last event ID in the page, and sleeps briefly
           before the next page.  This avoids a busy-loop while still
           reacting to changes within seconds.

        2. **Streaming CDC client** — if ``driver`` exposes a
           ``cdc_stream()`` method (Debezium connector / custom wrapper),
           the generator delegates to its blocking iterator.

        The generator exits cleanly when ``stop()`` has been called — it
        checks ``_stop_event`` at every page boundary.  ``_stop_event`` is
        cleared on entry so that ``listen()`` is safely re-entrant after a
        CDCRunner reconnect attempt.

        **Neo4j CDC event shape** (``db.cdc.query`` result):

        .. code-block:: python

            {
                "id":   "<opaque_cursor_string>",
                "txId": int,
                "seq":  int,
                "event": {
                    "elementId": str,
                    "operation": "c" | "u" | "d",
                    "labels":    List[str],
                    "state": {
                        "before": {"properties": {...}} | None,
                        "after":  {"properties": {...}} | None,
                    },
                },
            }

        Only events where ``event.labels`` contains ``node_label``
        (if set) and where ``event.state.after/before`` contains
        ``source_id_field`` are forwarded.

        Yields
        ------
        WatcherEvent
            One event per invalidating node mutation (create / update /
            delete) on a node that carries ``source_id_field``.

        Raises
        ------
        AdapterConnectionError
            If neither CDC strategy can be established.
        """
        logger.info(
            "GraphDBAdapter.listen(): starting CDC stream (database=%r).",
            self.database,
        )

        # Clear any previous stop signal so reconnects work correctly.
        self._stop_event.clear()

        # Detect which CDC strategy is available.
        if hasattr(self.driver, "cdc_stream"):
            yield from self._listen_via_stream()
        else:
            yield from self._listen_via_cdc_query()

    def _listen_via_cdc_query(self) -> Generator[WatcherEvent, None, None]:
        """
        Internal: poll Neo4j's built-in ``db.cdc.query`` procedure in a
        tight page loop, yielding ``WatcherEvent`` objects.

        Uses ``db.cdc.earliest`` to bootstrap the cursor on the first run
        (``_cdc_cursor is None``), then advances through the log by
        storing the ``id`` of the last event seen in each page.

        The loop sleeps for ``poll_interval / 2`` seconds between pages
        when a page returns fewer events than ``_CDC_QUERY_PAGE_SIZE``,
        backing off gracefully when the log has no new events rather than
        busy-spinning.  When a full page is returned the next page is
        fetched immediately to drain the backlog as quickly as possible.
        """
        import time

        # Bootstrap: fetch the earliest available CDC cursor if we have none.
        if self._cdc_cursor is None:
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run("CALL db.cdc.earliest()")
                    record = result.single()
                    if record:
                        self._cdc_cursor = record["id"]
                        logger.debug(
                            "GraphDBAdapter._listen_via_cdc_query(): "
                            "bootstrapped CDC cursor to %r.",
                            self._cdc_cursor,
                        )
            except Exception as exc:
                raise AdapterConnectionError(
                    f"GraphDBAdapter: failed to bootstrap CDC cursor "
                    f"(database='{self.database}'): {exc}"
                ) from exc

        sleep_secs: float = max(1.0, self.poll_interval / 2)

        while not self._stop_event.is_set():
            try:
                with self.driver.session(database=self.database) as session:
                    result = session.run(
                        "CALL db.cdc.query($from_id, $limit)",
                        from_id=self._cdc_cursor,
                        limit=_CDC_QUERY_PAGE_SIZE,
                    )
                    page: List[Any] = list(result)
            except Exception as exc:
                raise AdapterConnectionError(
                    f"GraphDBAdapter: db.cdc.query failed "
                    f"(database='{self.database}'): {exc}"
                ) from exc

            for raw in page:
                if self._stop_event.is_set():
                    return

                event_record = raw.data() if hasattr(raw, "data") else dict(raw)

                # Advance the opaque CDC cursor to the current event's ID
                # so the next page starts from here even on restart.
                if "id" in event_record:
                    self._cdc_cursor = event_record["id"]

                watcher_event = self._cdc_record_to_watcher_event(event_record)
                if watcher_event is not None:
                    yield watcher_event

            # If the page was not full there are no pending events;
            # wait before checking again to avoid a busy-loop.
            if len(page) < _CDC_QUERY_PAGE_SIZE:
                self._stop_event.wait(timeout=sleep_secs)

        logger.info(
            "GraphDBAdapter._listen_via_cdc_query(): stop event set — "
            "exiting (database=%r).",
            self.database,
        )

    def _listen_via_stream(self) -> Generator[WatcherEvent, None, None]:
        """
        Internal: consume a blocking CDC stream exposed by the driver's
        ``cdc_stream()`` method (Debezium connector or custom wrapper).

        The stream yields raw event dicts in the same shape as
        ``db.cdc.query`` records.  The generator exits when either the
        stream ends naturally or ``_stop_event`` is set.
        """
        try:
            stream = self.driver.cdc_stream(database=self.database)
        except Exception as exc:
            raise AdapterConnectionError(
                f"GraphDBAdapter: failed to open CDC stream "
                f"(database='{self.database}'): {exc}"
            ) from exc

        for raw in stream:
            if self._stop_event.is_set():
                logger.info(
                    "GraphDBAdapter._listen_via_stream(): stop event set — "
                    "exiting (database=%r).",
                    self.database,
                )
                return

            if raw is None:
                # Heartbeat / keepalive — skip silently.
                continue

            event_record = raw.data() if hasattr(raw, "data") else dict(raw)
            watcher_event = self._cdc_record_to_watcher_event(event_record)
            if watcher_event is not None:
                yield watcher_event

        logger.info(
            "GraphDBAdapter._listen_via_stream(): stream closed "
            "(database=%r).",
            self.database,
        )

    def _cdc_record_to_watcher_event(
        self, event_record: Dict[str, Any]
    ) -> Optional[WatcherEvent]:
        """
        Convert a raw Neo4j CDC event dict into a ``WatcherEvent``.

        Returns ``None`` for non-invalidating events (DDL, heartbeats,
        schema changes, events on unmonitored labels, or nodes missing
        ``source_id_field``).

        Parameters
        ----------
        event_record : dict
            Raw dict as produced by ``db.cdc.query`` or ``cdc_stream()``.

        Returns
        -------
        WatcherEvent or None
        """
        event_block: Dict = event_record.get("event") or {}
        operation: str = event_block.get("operation", "")

        if operation not in _INVALIDATING_CDC_OPS:
            return None

        # If node_label is set, filter events to that label only.
        if self.node_label:
            labels: List[str] = event_block.get("labels") or []
            if self.node_label not in labels:
                logger.debug(
                    "GraphDBAdapter: CDC event skipped — label %r not in %r.",
                    self.node_label,
                    labels,
                )
                return None

        source_id: Optional[str] = self.map_to_source_id(event_record)
        if source_id is None:
            return None

        fs_event_type: str = _FS_EVENT_MAP.get(operation, "modified")
        tx_id: Optional[int] = event_record.get("txId")

        return WatcherEvent(
            event_type="watcher_event",
            source_id=source_id,
            adapter_type="graph",
            detection_mode="cdc",
            payload={
                "operation":     _CDC_OPERATION_MAP.get(operation, operation),
                "fs_event_type": fs_event_type,
                "element_id":    event_block.get("elementId"),
                "labels":        event_block.get("labels"),
                "tx_id":         tx_id,
                "database":      self.database,
            },
        )

    def stop(self) -> None:
        """
        Signal ``listen()`` to exit after the current page / event is
        processed.

        Sets ``_stop_event`` so the generator breaks out of its loop at the
        next iteration boundary.  Thread-safe — safe to call from any thread,
        including the main thread while the background CDC thread is blocked
        inside a page query or stream iterator.
        """
        logger.info(
            "GraphDBAdapter.stop(): signalling CDC shutdown (database=%r).",
            self.database,
        )
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_query(self) -> str:
        """
        Return the Cypher query to use for polling.

        If the developer supplied a non-empty ``change_query`` at
        construction, it is returned verbatim.  Otherwise the built-in
        default template is rendered with ``source_id_field`` substituted.
        """
        if self.change_query:
            return self.change_query

        # Render the default template.  We use str.format() rather than
        # f-string so the Cypher $parameter placeholders are preserved.
        label_fragment = f":{self.node_label}" if self.node_label else ""
        return (
            f"MATCH (n{label_fragment}) "
            f"WHERE n.`{self.source_id_field}` IS NOT NULL "
            f"  AND ($cursor IS NULL OR n.last_modified > $cursor) "
            f"RETURN n "
            f"ORDER BY n.last_modified ASC"
        )

    def _extract_last_modified(self, record: Any) -> Optional[Any]:
        """
        Pull the ``last_modified`` property out of a raw Cypher record
        for watermark tracking during polling.

        Tries the ``"n"`` alias first (``RETURN n`` pattern), then looks
        for a top-level ``"last_modified"`` key (projection queries), and
        finally falls back to ``None`` if neither is present.

        Parameters
        ----------
        record : Any
            Raw Cypher result record as a dict or Neo4j Record.

        Returns
        -------
        Any or None
            The ``last_modified`` value, or ``None`` if not found.
        """
        if not isinstance(record, dict):
            return None

        # RETURN n pattern → the node object is under the "n" key.
        node = record.get("n")
        if node is not None:
            if hasattr(node, "get"):
                return node.get("last_modified")
            if hasattr(node, "_properties"):
                return node._properties.get("last_modified")

        # Projection pattern → RETURN n.last_modified AS last_modified
        return record.get("last_modified")