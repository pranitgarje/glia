"""
glia/adapters/relational.py
───────────────────────────
Layer 4 — Concrete Adapters

Defines RelationalDBAdapter, a concrete adapter for relational database
sources (e.g. PostgreSQL, MySQL, SQLite).

The adapter is generic: it works with any relational database whose
connection object exposes the following interface:

  Connection probe (used by connect())
  ─────────────────────────────────────
  connection.cursor() -> cursor
      Standard DB-API 2.0 cursor factory used as a liveness probe.

  Polling mode
  ────────────
  The adapter executes SQL through a DB-API 2.0 cursor.  Two strategies
  are supported and selected automatically at construction time:

  Strategy A — updated_at column (default)
      Queries ``table`` directly, filtering by ``updated_at_col``:

          SELECT * FROM <table>
          WHERE <updated_at_col> > <last_cursor>
          ORDER BY <updated_at_col> ASC

      ``last_cursor`` is a comparable value (datetime, ISO string, epoch
      integer).  ``None`` on first run → WHERE clause is omitted so all
      rows are returned as a bootstrap baseline.

  Strategy B — change-log table (opt-in via ``changelog_table``)
      Queries a dedicated change-log / audit table that already records
      which primary key changed and when.  Expected schema:

          changelog_table (
              id              SERIAL PRIMARY KEY,
              operation       TEXT,        -- 'INSERT' | 'UPDATE' | 'DELETE'
              table_name      TEXT,
              record_pk       TEXT,        -- serialised PK of the changed row
              changed_at      TIMESTAMP,   -- or equivalent watermark column
          )

      The adapter filters by ``changed_at > last_cursor`` and
      ``table_name = self.table``.

  CDC mode
  ────────
  Logical-replication / binlog streaming is modelled through a client
  object that wraps the database driver's streaming API and exposes:

  connection.replication_stream(
      slot_name  : str,          # PostgreSQL replication slot / MySQL server-id
      table      : str,          # table to monitor
  ) -> Iterator[Dict[str, Any]]
      Blocking iterator yielding raw replication messages:
          {
              "operation":  "INSERT" | "UPDATE" | "DELETE" | "DDL" | "HEARTBEAT",
              "table":      str,                  # qualified table name
              "new_row":    Dict[str, Any] | None, # row state after change
              "old_row":    Dict[str, Any] | None, # row state before change
              "lsn":        str | None,            # log-sequence-number (PG)
              "timestamp":  datetime | None,
          }

  The ``new_row`` dict (INSERT / UPDATE) or ``old_row`` dict (DELETE) is
  used to extract ``source_id_field`` and derive the ``source_id``.

Architectural constraints:
- Inherits from PollingAdapter and CDCAdapter (dual-inheritance, MRO-safe).
- MUST NOT import from Layer 3 (manager.py, invalidator.py),
  Layer 5 (runners.py, watcher.py), or Layer 6 (__init__.py).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Iterator, List, Optional

from glia.adapters.polling import PollingAdapter
from glia.adapters.cdc import CDCAdapter
from glia.events import WatcherEvent
from glia.exceptions import AdapterConnectionError

logger = logging.getLogger(__name__)

# Operations that warrant cache invalidation.
# DDL events, heartbeats, and schema migrations do not map to a cached
# document and are silently skipped.
_INVALIDATING_OPERATIONS = frozenset({"INSERT", "UPDATE", "DELETE"})


class RelationalDBAdapter(PollingAdapter, CDCAdapter):
    """
    Concrete adapter for relational database sources
    (e.g. PostgreSQL, MySQL, SQLite).

    Supports both polling and CDC execution modes.

    **Polling mode** (``mode="polling"``)
        Periodically executes a SQL query that returns rows modified since
        the last committed cursor.  Two sub-strategies are available:

        * *updated_at column* (default) — queries the monitored ``table``
          directly using ``updated_at_col`` as the watermark.
        * *change-log table* (opt-in via ``changelog_table``) — queries a
          dedicated audit / change-log table that the database populates
          via triggers or application logic.

    **CDC mode** (``mode="cdc"``)
        Consumes a logical-replication slot (PostgreSQL ``pgoutput`` /
        ``wal2json``) or a binlog stream (MySQL) by calling
        ``connection.replication_stream()``.  Each DML message is mapped
        to a ``WatcherEvent`` and yielded to ``CDCRunner``.

    Parameters
    ----------
    connection : Any
        DB-API 2.0 connection (or connection pool) to the relational
        database.  Not managed by Glia — the developer creates it.
    table : str
        Unqualified or schema-qualified name of the monitored table
        (e.g. ``"documents"`` or ``"public.documents"``).
    updated_at_col : str
        Column used as the polling watermark (e.g. ``"updated_at"``).
        Only relevant when ``mode="polling"`` and ``changelog_table`` is
        not set.
    mode : str
        Execution strategy — ``"polling"`` or ``"cdc"``.
    source_id_field : str
        Column name whose value becomes Glia's ``source_id`` TAG
        (e.g. ``"id"``, ``"document_id"``).
    poll_interval : float
        Seconds between consecutive ``poll()`` calls.  Defaults to ``30.0``.
    last_cursor : Any, optional
        Initial watermark value (datetime, ISO string, epoch int, or LSN
        string).  ``None`` triggers a full-table bootstrap on first poll.
    changelog_table : str, optional
        Name of a dedicated change-log / audit table.  When set, polling
        queries this table instead of ``table`` directly.  Defaults to
        ``None`` (use ``updated_at_col`` strategy).
    replication_slot : str, optional
        PostgreSQL replication slot name or MySQL server-id string.
        Required when ``mode="cdc"``.  Defaults to ``"glia_slot"``.
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
        connection: Any,
        table: str,
        updated_at_col: str,
        mode: str,
        source_id_field: str,
        poll_interval: float = 30.0,
        last_cursor: Optional[Any] = None,
        changelog_table: Optional[str] = None,
        replication_slot: str = "glia_slot",
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
        self.connection: Any = connection
        self.table: str = table
        self.updated_at_col: str = updated_at_col
        self.changelog_table: Optional[str] = changelog_table
        self.replication_slot: str = replication_slot

        # ── Internal state ────────────────────────────────────────────────
        # _connected: idempotency guard for connect() / disconnect().
        self._connected: bool = False

        # _stop_event: set by stop() to signal the listen() generator to
        # exit its loop safely from any thread.
        self._stop_event: threading.Event = threading.Event()

        # _current_cursor: the highest watermark value seen in the most
        # recent poll() batch.  Kept separate from last_cursor so that a
        # partial batch caused by a mid-iteration exception does not silently
        # skip records on the next cycle.
        self._current_cursor: Any = last_cursor

    # ------------------------------------------------------------------
    # DatabaseAdapter contract — connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Verify that the database connection is live and the target table
        is reachable.

        Uses a lightweight DB-API 2.0 ``cursor()`` probe — no data is read.
        If the connection object is a pool (e.g. SQLAlchemy Engine), the
        probe checks out one connection, executes a trivial query, and
        returns it immediately.

        Idempotent: a second call on an already-connected adapter is a
        no-op.

        Raises
        ------
        AdapterConnectionError
            If the cursor probe raises any exception.
        """
        if self._connected:
            logger.debug(
                "RelationalDBAdapter.connect(): already connected to "
                "table=%r — skipping.",
                self.table,
            )
            return

        try:
            # DB-API 2.0 cursor() is the lowest-common-denominator probe.
            # For SQLAlchemy engines we call connect() on the engine and
            # immediately close the borrowed connection.
            if hasattr(self.connection, "cursor"):
                # Raw DB-API 2.0 connection (psycopg2, sqlite3, mysql-connector)
                cursor = self.connection.cursor()
                cursor.close()
            elif hasattr(self.connection, "connect"):
                # SQLAlchemy Engine — borrow and immediately release.
                conn = self.connection.connect()
                conn.close()
            else:
                # Unknown connection type — optimistically assume it is live.
                logger.debug(
                    "RelationalDBAdapter.connect(): unrecognised connection "
                    "type %r — skipping probe.",
                    type(self.connection).__name__,
                )
        except Exception as exc:
            raise AdapterConnectionError(
                f"RelationalDBAdapter: failed to connect to table "
                f"'{self.table}': {exc}"
            ) from exc

        self._connected = True
        logger.info(
            "RelationalDBAdapter connected (table=%r).", self.table
        )

    def disconnect(self) -> None:
        """
        Close the database connection cleanly.

        Calls ``connection.close()`` if the method exists.  The connection
        object itself is not destroyed — it was injected by the developer
        and may be shared with other components.

        Idempotent: safe to call on an already-disconnected adapter.
        """
        if not self._connected:
            return

        try:
            if hasattr(self.connection, "close"):
                self.connection.close()
        except Exception:
            logger.warning(
                "RelationalDBAdapter.disconnect(): exception while closing "
                "connection for table=%r (ignored).",
                self.table,
                exc_info=True,
            )
        finally:
            self._connected = False
            logger.info(
                "RelationalDBAdapter disconnected (table=%r).", self.table
            )

    # ------------------------------------------------------------------
    # DatabaseAdapter contract — source_id mapping
    # ------------------------------------------------------------------

    def map_to_source_id(self, record: Any) -> Optional[str]:
        """
        Derive a stable Glia ``source_id`` from a raw row dict or CDC event.

        **Polling rows**
            ``record`` is a plain dict keyed by column name (as returned by
            a DB-API cursor with ``RealDictCursor`` / ``DictCursor``, or by
            SQLAlchemy's ``RowMapping``).  The primary key value is read
            directly from ``record[self.source_id_field]``.

        **Change-log rows** (``changelog_table`` strategy)
            ``record`` is a row from the audit table.  The primary key of
            the *original* changed row is stored in the ``record_pk`` column
            as a string, so it is used directly without further composition.

        **CDC replication events**
            ``record`` is a dict produced by the replication stream:
            ``{"operation": ..., "new_row": {...}, "old_row": {...}, ...}``.
            DELETE events use ``old_row``; INSERT / UPDATE events use
            ``new_row``.  The primary key is extracted from the appropriate
            sub-dict.

        The final ``source_id`` uses a namespaced format::

            "table:<table>|pk:<primary_key_value>"

        This namespace prevents collisions when multiple tables share the
        same primary key space (e.g. both ``documents`` and ``contracts``
        have ``id = 42``).

        Parameters
        ----------
        record : Any
            Raw row dict or CDC event dict.

        Returns
        -------
        str or None
            Stable ``source_id`` string, or ``None`` to skip the record
            (e.g. DDL events, heartbeats, rows with NULL primary keys).
        """
        if record is None:
            return None

        # ── Change-log table row ─────────────────────────────────────────
        # The changelog stores a pre-serialised string of the PK so we can
        # use it directly without going back to the main table.
        if self.changelog_table and isinstance(record, dict) and "record_pk" in record:
            pk_value = record.get("record_pk")
            if not pk_value:
                logger.debug(
                    "RelationalDBAdapter.map_to_source_id(): 'record_pk' is "
                    "empty in changelog row — skipping."
                )
                return None
            return f"table:{self.table}|pk:{pk_value}"

        # ── CDC replication event ────────────────────────────────────────
        # Replication events carry 'operation' and 'new_row' / 'old_row'.
        if isinstance(record, dict) and "operation" in record:
            operation: str = (record.get("operation") or "").upper()

            if operation not in _INVALIDATING_OPERATIONS:
                # DDL, HEARTBEAT, TRUNCATE etc. — not relevant to cache.
                logger.debug(
                    "RelationalDBAdapter.map_to_source_id(): skipping "
                    "non-invalidating CDC operation %r.",
                    operation,
                )
                return None

            # For DELETE we need the *old* state; INSERT/UPDATE the *new* state.
            row_data: Optional[Dict[str, Any]] = (
                record.get("old_row")
                if operation == "DELETE"
                else record.get("new_row")
            )

            if not row_data:
                logger.debug(
                    "RelationalDBAdapter.map_to_source_id(): CDC event for "
                    "operation=%r has no usable row data — skipping.",
                    operation,
                )
                return None

            pk_value = row_data.get(self.source_id_field)
            if pk_value is None:
                logger.debug(
                    "RelationalDBAdapter.map_to_source_id(): PK field %r "
                    "absent in CDC row data — skipping.",
                    self.source_id_field,
                )
                return None

            return f"table:{self.table}|pk:{pk_value}"

        # ── Plain polling row ────────────────────────────────────────────
        # Record is a flat dict from a direct table query.
        if isinstance(record, dict):
            pk_value = record.get(self.source_id_field)
        else:
            # Object-style rows (SQLAlchemy ORM instances, named tuples).
            pk_value = getattr(record, self.source_id_field, None)

        if pk_value is None:
            logger.debug(
                "RelationalDBAdapter.map_to_source_id(): PK field %r not "
                "found in record — skipping.",
                self.source_id_field,
            )
            return None

        return f"table:{self.table}|pk:{pk_value}"

    # ------------------------------------------------------------------
    # PollingAdapter contract
    # ------------------------------------------------------------------

    def poll(self) -> Iterator[Any]:
        """
        Query the database for rows modified since ``last_cursor`` and
        yield them one at a time.

        **Strategy A — updated_at column (default)**

            Executes:

            .. code-block:: sql

                SELECT *
                FROM   <table>
                WHERE  <updated_at_col> > <last_cursor>   -- omitted on bootstrap
                ORDER  BY <updated_at_col> ASC

            Each row is yielded as a plain ``dict`` (column → value).

        **Strategy B — change-log table (``changelog_table`` is set)**

            Executes:

            .. code-block:: sql

                SELECT *
                FROM   <changelog_table>
                WHERE  table_name  = '<table>'
                  AND  changed_at  > <last_cursor>        -- omitted on bootstrap
                ORDER  BY changed_at ASC

            Each change-log row is yielded as a plain ``dict``.

        In both cases ``last_cursor=None`` performs a full bootstrap sweep
        — the WHERE clause on the watermark column is omitted so every
        existing row is returned and the initial baseline is established.

        The method accumulates the highest watermark value seen across the
        batch in ``_current_cursor``.  ``PollingRunner`` reads this via
        ``get_cursor()`` and commits it via ``advance_cursor()`` *after*
        the full batch has been dispatched, so ``last_cursor`` is never
        mutated here.

        Yields
        ------
        dict
            One row dict per modified record.

        Raises
        ------
        AdapterConnectionError
            Wraps any database error so ``PollingRunner`` receives a typed
            exception it can handle.
        """
        if self.changelog_table:
            yield from self._poll_changelog()
        else:
            yield from self._poll_updated_at()

    def _poll_updated_at(self) -> Iterator[Dict[str, Any]]:
        """
        Internal helper: query the monitored table via its ``updated_at_col``.
        """
        watermark_col = self.updated_at_col

        try:
            cursor = self._open_cursor()
        except Exception as exc:
            raise AdapterConnectionError(
                f"RelationalDBAdapter: could not open cursor for table "
                f"'{self.table}': {exc}"
            ) from exc

        try:
            if self.last_cursor is None:
                # Bootstrap — return all rows ordered by watermark ascending
                # so that _current_cursor captures the true latest timestamp.
                logger.debug(
                    "RelationalDBAdapter._poll_updated_at(): bootstrap sweep "
                    "(table=%r, watermark_col=%r).",
                    self.table,
                    watermark_col,
                )
                cursor.execute(
                    f"SELECT * FROM {self.table} ORDER BY {watermark_col} ASC"
                )
            else:
                logger.debug(
                    "RelationalDBAdapter._poll_updated_at(): incremental poll "
                    "(table=%r, since=%r).",
                    self.table,
                    self.last_cursor,
                )
                cursor.execute(
                    f"SELECT * FROM {self.table} "
                    f"WHERE {watermark_col} > %s "
                    f"ORDER BY {watermark_col} ASC",
                    (self.last_cursor,),
                )

            columns: List[str] = [desc[0] for desc in cursor.description]
            high_water: Any = self._current_cursor

            for raw_row in cursor:
                row: Dict[str, Any] = dict(zip(columns, raw_row))

                ts = row.get(watermark_col)
                if ts is not None and (high_water is None or ts > high_water):
                    high_water = ts

                yield row

            self._current_cursor = high_water

        except AdapterConnectionError:
            raise
        except Exception as exc:
            raise AdapterConnectionError(
                f"RelationalDBAdapter: poll query failed on table "
                f"'{self.table}': {exc}"
            ) from exc
        finally:
            self._close_cursor(cursor)

    def _poll_changelog(self) -> Iterator[Dict[str, Any]]:
        """
        Internal helper: query the change-log table for entries that
        reference the monitored table and are newer than ``last_cursor``.
        """
        watermark_col = "changed_at"

        try:
            cursor = self._open_cursor()
        except Exception as exc:
            raise AdapterConnectionError(
                f"RelationalDBAdapter: could not open cursor for changelog "
                f"table '{self.changelog_table}': {exc}"
            ) from exc

        try:
            if self.last_cursor is None:
                logger.debug(
                    "RelationalDBAdapter._poll_changelog(): bootstrap sweep "
                    "(changelog=%r, table=%r).",
                    self.changelog_table,
                    self.table,
                )
                cursor.execute(
                    f"SELECT * FROM {self.changelog_table} "
                    f"WHERE table_name = %s "
                    f"ORDER BY {watermark_col} ASC",
                    (self.table,),
                )
            else:
                logger.debug(
                    "RelationalDBAdapter._poll_changelog(): incremental poll "
                    "(changelog=%r, table=%r, since=%r).",
                    self.changelog_table,
                    self.table,
                    self.last_cursor,
                )
                cursor.execute(
                    f"SELECT * FROM {self.changelog_table} "
                    f"WHERE table_name = %s "
                    f"  AND {watermark_col} > %s "
                    f"ORDER BY {watermark_col} ASC",
                    (self.table, self.last_cursor),
                )

            columns: List[str] = [desc[0] for desc in cursor.description]
            high_water: Any = self._current_cursor

            for raw_row in cursor:
                row: Dict[str, Any] = dict(zip(columns, raw_row))

                ts = row.get(watermark_col)
                if ts is not None and (high_water is None or ts > high_water):
                    high_water = ts

                yield row

            self._current_cursor = high_water

        except AdapterConnectionError:
            raise
        except Exception as exc:
            raise AdapterConnectionError(
                f"RelationalDBAdapter: changelog poll failed "
                f"(changelog={self.changelog_table!r}, table={self.table!r}): "
                f"{exc}"
            ) from exc
        finally:
            self._close_cursor(cursor)

    def get_cursor(self) -> Any:
        """
        Return the highest watermark value observed in the most recent
        ``poll()`` batch.

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
        batch has been successfully dispatched.  Deferring the write to this
        method (rather than inside ``poll()``) ensures that a mid-iteration
        exception does not silently skip records on the next cycle.

        Parameters
        ----------
        new_cursor : Any
            The value returned by ``get_cursor()`` after the batch was
            consumed.
        """
        logger.debug(
            "RelationalDBAdapter.advance_cursor(): %r → %r (table=%r).",
            self.last_cursor,
            new_cursor,
            self.table,
        )
        self.last_cursor = new_cursor
        self._current_cursor = new_cursor

    # ------------------------------------------------------------------
    # CDCAdapter contract
    # ------------------------------------------------------------------

    def listen(self) -> Generator[WatcherEvent, None, None]:
        """
        Open a logical-replication slot (PostgreSQL) or binlog stream
        (MySQL) and yield a ``WatcherEvent`` for every DML event on the
        monitored table.

        This is a **blocking generator**.  ``CDCRunner`` drives it from a
        dedicated background thread so the main thread is never blocked.

        The generator exits cleanly when ``stop()`` has been called — it
        checks ``_stop_event`` before processing each replication message.
        ``_stop_event`` is cleared on entry so that ``listen()`` is safely
        re-entrant after a CDCRunner reconnect attempt.

        **Replication message format** expected from
        ``connection.replication_stream()``:

        .. code-block:: python

            {
                "operation":  "INSERT" | "UPDATE" | "DELETE" | "DDL" | "HEARTBEAT",
                "table":      str,                   # qualified table name
                "new_row":    Dict[str, Any] | None, # row state after change
                "old_row":    Dict[str, Any] | None, # row state before change
                "lsn":        str | None,            # log-sequence-number (PG)
                "timestamp":  datetime | None,
            }

        Only ``INSERT``, ``UPDATE``, and ``DELETE`` operations on
        ``self.table`` produce a ``WatcherEvent``.  DDL events, heartbeats,
        and events from other tables are skipped silently.

        **Operation → fs_event_type mapping** (mirrors the filesystem watcher
        convention used throughout Glia):

        * ``INSERT`` / ``UPDATE`` → ``"modified"``
        * ``DELETE``              → ``"deleted"``

        Yields
        ------
        WatcherEvent
            One structured event per INSERT, UPDATE, or DELETE on the
            monitored table.

        Raises
        ------
        AdapterConnectionError
            If opening the replication stream raises any exception.
        """
        logger.info(
            "RelationalDBAdapter.listen(): opening replication stream "
            "(table=%r, slot=%r).",
            self.table,
            self.replication_slot,
        )

        # Clear any previous stop signal so reconnects work correctly.
        self._stop_event.clear()

        try:
            stream = self.connection.replication_stream(
                slot_name=self.replication_slot,
                table=self.table,
            )
        except Exception as exc:
            raise AdapterConnectionError(
                f"RelationalDBAdapter: failed to open replication stream "
                f"for table '{self.table}' "
                f"(slot='{self.replication_slot}'): {exc}"
            ) from exc

        for raw_event in stream:
            # Primary exit path for a clean shutdown request.
            if self._stop_event.is_set():
                logger.info(
                    "RelationalDBAdapter.listen(): stop event set — exiting "
                    "CDC loop (table=%r).",
                    self.table,
                )
                return

            # Heartbeat / keepalive frames — skip without yielding.
            if raw_event is None:
                continue

            operation: str = (
                raw_event.get("operation", "") if isinstance(raw_event, dict)
                else getattr(raw_event, "operation", "")
            ).upper()

            if operation not in _INVALIDATING_OPERATIONS:
                # DDL, TRUNCATE, HEARTBEAT, schema-change — not relevant.
                logger.debug(
                    "RelationalDBAdapter.listen(): skipping non-invalidating "
                    "operation=%r (table=%r).",
                    operation,
                    self.table,
                )
                continue

            # Filter events that belong to tables we are not monitoring.
            # Replication slots on PostgreSQL can deliver changes for all
            # tables in a publication; we only care about self.table.
            event_table: str = (
                raw_event.get("table", self.table) if isinstance(raw_event, dict)
                else getattr(raw_event, "table", self.table)
            )
            # Normalise: strip schema qualifier for comparison.
            if event_table.split(".")[-1] != self.table.split(".")[-1]:
                logger.debug(
                    "RelationalDBAdapter.listen(): skipping event for "
                    "unmonitored table=%r (monitoring=%r).",
                    event_table,
                    self.table,
                )
                continue

            source_id: Optional[str] = self.map_to_source_id(raw_event)

            if source_id is None:
                # Row did not carry a usable PK — skip without invalidating.
                logger.debug(
                    "RelationalDBAdapter.listen(): could not derive source_id "
                    "from CDC event (operation=%r, table=%r) — skipping.",
                    operation,
                    event_table,
                )
                continue

            # Map DML operation to the fs_event_type convention used by Glia.
            fs_event_type: str = "deleted" if operation == "DELETE" else "modified"

            # Extract the log-sequence-number for observability.
            lsn: Optional[str] = (
                raw_event.get("lsn") if isinstance(raw_event, dict)
                else getattr(raw_event, "lsn", None)
            )

            yield WatcherEvent(
                event_type="watcher_event",
                source_id=source_id,
                adapter_type="relational",
                detection_mode="cdc",
                payload={
                    "table": event_table,
                    "operation": operation,
                    "fs_event_type": fs_event_type,
                    "lsn": lsn,
                },
            )

        # Stream exhausted — the server closed the replication connection or
        # stop() was honoured by the driver ending its iterator.
        logger.info(
            "RelationalDBAdapter.listen(): replication stream closed "
            "(table=%r, slot=%r).",
            self.table,
            self.replication_slot,
        )

    def stop(self) -> None:
        """
        Signal ``listen()`` to exit after the current event is processed.

        Sets ``_stop_event`` so the generator breaks out of its loop at the
        next iteration boundary.  Thread-safe — safe to call from any thread,
        including the main thread while the background CDC thread is blocked
        inside the replication stream iterator.
        """
        logger.info(
            "RelationalDBAdapter.stop(): signalling CDC shutdown (table=%r).",
            self.table,
        )
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Private cursor helpers
    # ------------------------------------------------------------------

    def _open_cursor(self) -> Any:
        """
        Return a DB-API 2.0 cursor from the active connection.

        Handles both raw DB-API 2.0 connections (``connection.cursor()``)
        and SQLAlchemy engines (``connection.connect().execute()``-style
        usage is handled at the call site; here we just return the raw
        connection object so callers can call ``.execute()`` on it directly
        when dealing with SQLAlchemy).

        For SQLAlchemy ``Engine`` objects the returned "cursor" is actually
        a ``Connection`` context — callers must call ``_close_cursor()``
        unconditionally in a ``finally`` block.
        """
        if hasattr(self.connection, "cursor"):
            # Standard DB-API 2.0 (psycopg2, sqlite3, mysql-connector-python)
            return self.connection.cursor()
        elif hasattr(self.connection, "connect"):
            # SQLAlchemy Engine — return a raw DBAPI connection.
            return self.connection.raw_connection().cursor()
        else:
            raise AdapterConnectionError(
                "RelationalDBAdapter: connection object does not expose "
                "cursor() or connect() — cannot execute queries."
            )

    @staticmethod
    def _close_cursor(cursor: Any) -> None:
        """Close a cursor, suppressing any error during teardown."""
        try:
            if cursor is not None and hasattr(cursor, "close"):
                cursor.close()
        except Exception:
            pass