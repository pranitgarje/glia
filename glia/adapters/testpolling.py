"""
test_polling.py
───────────────
Local polling-mode tests for VectorDBAdapter, GraphDBAdapter, and
RelationalDBAdapter.

Every test uses an in-process mock client — no real database required.
Run with:   python test_polling.py
"""

import sys, os
sys.path.insert(0, "/home/claude")

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from glia.adapters.vector import VectorDBAdapter
from glia.adapters.graph import GraphDBAdapter
from glia.adapters.relational import RelationalDBAdapter
from glia.exceptions import AdapterConnectionError, MissingModeError


# ══════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════

def make_timestamps(n=3):
    """Return n ascending datetime values starting from 2024-01-01."""
    base = datetime(2024, 1, 1)
    return [base + timedelta(hours=i) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════
# VectorDBAdapter — polling
# ══════════════════════════════════════════════════════════════════════

class TestVectorDBAdapterPolling(unittest.TestCase):

    def _make_client(self, records):
        """Mock client whose fetch_updated returns a fixed list."""
        client = MagicMock()
        client.describe_collection.return_value = True
        client.fetch_updated.return_value = records
        return client

    def _make_adapter(self, client, last_cursor=None):
        return VectorDBAdapter(
            client=client,
            collection="docs",
            timestamp_field="updated_at",
            mode="polling",
            source_id_field="doc_id",
            poll_interval=5.0,
            last_cursor=last_cursor,
        )

    # ── construction ──────────────────────────────────────────────────

    def test_init_stores_params(self):
        client = self._make_client([])
        adapter = self._make_adapter(client)
        self.assertEqual(adapter.mode, "polling")
        self.assertEqual(adapter.collection, "docs")
        self.assertEqual(adapter.source_id_field, "doc_id")
        self.assertEqual(adapter.poll_interval, 5.0)
        self.assertIsNone(adapter.last_cursor)

    def test_missing_mode_raises(self):
        with self.assertRaises(MissingModeError):
            VectorDBAdapter(
                client=MagicMock(),
                collection="docs",
                timestamp_field="updated_at",
                mode=None,
                source_id_field="doc_id",
            )

    def test_invalid_mode_raises(self):
        with self.assertRaises(MissingModeError):
            VectorDBAdapter(
                client=MagicMock(),
                collection="docs",
                timestamp_field="updated_at",
                mode="stream",          # invalid
                source_id_field="doc_id",
            )

    # ── connect / disconnect ──────────────────────────────────────────

    def test_connect_calls_describe_collection(self):
        client = self._make_client([])
        adapter = self._make_adapter(client)
        adapter.connect()
        client.describe_collection.assert_called_once_with("docs")
        self.assertTrue(adapter._connected)

    def test_connect_idempotent(self):
        client = self._make_client([])
        adapter = self._make_adapter(client)
        adapter.connect()
        adapter.connect()                        # second call must be no-op
        client.describe_collection.assert_called_once()

    def test_connect_fallback_to_get_collection(self):
        client = MagicMock(spec=["get_collection"])
        client.get_collection.return_value = True
        adapter = self._make_adapter(client)
        adapter.connect()
        client.get_collection.assert_called_once_with("docs")

    def test_connect_raises_on_failure(self):
        client = MagicMock()
        client.describe_collection.side_effect = RuntimeError("unreachable")
        adapter = self._make_adapter(client)
        with self.assertRaises(AdapterConnectionError):
            adapter.connect()

    def test_disconnect_calls_close(self):
        client = self._make_client([])
        adapter = self._make_adapter(client)
        adapter.connect()
        adapter.disconnect()
        client.close.assert_called_once()
        self.assertFalse(adapter._connected)

    def test_disconnect_idempotent(self):
        client = self._make_client([])
        adapter = self._make_adapter(client)
        adapter.connect()
        adapter.disconnect()
        adapter.disconnect()                     # must not raise
        client.close.assert_called_once()

    # ── poll() ────────────────────────────────────────────────────────

    def test_poll_yields_all_records(self):
        ts = make_timestamps(3)
        records = [
            {"doc_id": "a", "updated_at": ts[0]},
            {"doc_id": "b", "updated_at": ts[1]},
            {"doc_id": "c", "updated_at": ts[2]},
        ]
        client = self._make_client(records)
        adapter = self._make_adapter(client)

        yielded = list(adapter.poll())
        self.assertEqual(len(yielded), 3)
        self.assertEqual(yielded[0]["doc_id"], "a")

    def test_poll_passes_last_cursor(self):
        ts = make_timestamps(1)
        client = self._make_client([])
        adapter = self._make_adapter(client, last_cursor=ts[0])

        list(adapter.poll())
        _, kwargs = client.fetch_updated.call_args
        self.assertEqual(kwargs.get("since") or client.fetch_updated.call_args[0][1], ts[0])

    def test_poll_empty_returns_nothing(self):
        client = self._make_client([])
        adapter = self._make_adapter(client)
        yielded = list(adapter.poll())
        self.assertEqual(yielded, [])

    def test_poll_advances_current_cursor(self):
        ts = make_timestamps(3)
        records = [
            {"doc_id": "a", "updated_at": ts[0]},
            {"doc_id": "b", "updated_at": ts[2]},   # highest
            {"doc_id": "c", "updated_at": ts[1]},
        ]
        client = self._make_client(records)
        adapter = self._make_adapter(client)

        list(adapter.poll())
        self.assertEqual(adapter.get_cursor(), ts[2])

    def test_poll_does_not_mutate_last_cursor(self):
        ts = make_timestamps(2)
        records = [{"doc_id": "x", "updated_at": ts[1]}]
        client = self._make_client(records)
        adapter = self._make_adapter(client, last_cursor=ts[0])

        list(adapter.poll())
        # last_cursor unchanged until advance_cursor() is called
        self.assertEqual(adapter.last_cursor, ts[0])

    def test_poll_raises_adapter_error_on_client_failure(self):
        client = MagicMock()
        client.fetch_updated.side_effect = ConnectionError("timeout")
        adapter = self._make_adapter(client)
        with self.assertRaises(AdapterConnectionError):
            list(adapter.poll())

    # ── get_cursor / advance_cursor ───────────────────────────────────

    def test_advance_cursor_updates_both_cursors(self):
        ts = make_timestamps(2)
        client = self._make_client([])
        adapter = self._make_adapter(client, last_cursor=ts[0])

        adapter.advance_cursor(ts[1])
        self.assertEqual(adapter.last_cursor, ts[1])
        self.assertEqual(adapter._current_cursor, ts[1])

    def test_get_cursor_none_before_first_poll(self):
        client = self._make_client([])
        adapter = self._make_adapter(client)
        self.assertIsNone(adapter.get_cursor())

    # ── map_to_source_id ─────────────────────────────────────────────

    def test_map_to_source_id_dict(self):
        client = self._make_client([])
        adapter = self._make_adapter(client)
        sid = adapter.map_to_source_id({"doc_id": "abc-123"})
        self.assertEqual(sid, "abc-123")

    def test_map_to_source_id_object(self):
        client = self._make_client([])
        adapter = self._make_adapter(client)
        rec = MagicMock()
        rec.doc_id = "obj-doc"
        sid = adapter.map_to_source_id(rec)
        self.assertEqual(sid, "obj-doc")

    def test_map_to_source_id_missing_returns_none(self):
        client = self._make_client([])
        adapter = self._make_adapter(client)
        self.assertIsNone(adapter.map_to_source_id({"other_field": "x"}))

    def test_map_to_source_id_none_record(self):
        client = self._make_client([])
        adapter = self._make_adapter(client)
        self.assertIsNone(adapter.map_to_source_id(None))

    # ── end-to-end poll + map flow ────────────────────────────────────

    def test_full_poll_and_map_cycle(self):
        """Simulate what PollingRunner does: poll → map → advance."""
        ts = make_timestamps(2)
        records = [
            {"doc_id": "doc-1", "updated_at": ts[0]},
            {"doc_id": "doc-2", "updated_at": ts[1]},
        ]
        client = self._make_client(records)
        adapter = self._make_adapter(client)

        source_ids = []
        for rec in adapter.poll():
            sid = adapter.map_to_source_id(rec)
            source_ids.append(sid)

        adapter.advance_cursor(adapter.get_cursor())

        self.assertEqual(source_ids, ["doc-1", "doc-2"])
        self.assertEqual(adapter.last_cursor, ts[1])


# ══════════════════════════════════════════════════════════════════════
# GraphDBAdapter — polling
# ══════════════════════════════════════════════════════════════════════

def _make_neo4j_record(doc_id, last_modified):
    """Return a minimal dict that looks like a RETURN n result."""
    node = {"doc_id": doc_id, "last_modified": last_modified}
    # Wrap in the {"n": node} shape that RETURN n produces in Neo4j driver
    mock_record = MagicMock()
    mock_record.data.return_value = {"n": node}
    return mock_record


class TestGraphDBAdapterPolling(unittest.TestCase):

    def _make_driver(self, records=None):
        driver = MagicMock()
        driver.verify_connectivity.return_value = None

        # session context manager yields a mock session
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda s: iter(records or [])
        mock_session.run.return_value = mock_result
        driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return driver, mock_session

    def _make_adapter(self, driver, last_cursor=None, change_query=None):
        return GraphDBAdapter(
            driver=driver,
            change_query=change_query,
            mode="polling",
            source_id_field="doc_id",
            node_label="Document",
            database="neo4j",
            poll_interval=10.0,
            last_cursor=last_cursor,
        )

    # ── construction ──────────────────────────────────────────────────

    def test_init_stores_params(self):
        driver, _ = self._make_driver()
        adapter = self._make_adapter(driver)
        self.assertEqual(adapter.mode, "polling")
        self.assertEqual(adapter.source_id_field, "doc_id")
        self.assertEqual(adapter.node_label, "Document")
        self.assertEqual(adapter.database, "neo4j")

    def test_missing_mode_raises(self):
        driver, _ = self._make_driver()
        with self.assertRaises(MissingModeError):
            GraphDBAdapter(
                driver=driver,
                change_query=None,
                mode=None,
                source_id_field="doc_id",
            )

    # ── connect ───────────────────────────────────────────────────────

    def test_connect_uses_verify_connectivity(self):
        driver, _ = self._make_driver()
        adapter = self._make_adapter(driver)
        adapter.connect()
        driver.verify_connectivity.assert_called_once()
        self.assertTrue(adapter._connected)

    def test_connect_fallback_to_session(self):
        driver = MagicMock(spec=["session", "close"])
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.consume.return_value = None
        mock_session.run.return_value = mock_result
        driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        adapter = self._make_adapter(driver)
        adapter.connect()
        self.assertTrue(adapter._connected)

    def test_connect_raises_on_failure(self):
        driver, _ = self._make_driver()
        driver.verify_connectivity.side_effect = Exception("unreachable")
        adapter = self._make_adapter(driver)
        with self.assertRaises(AdapterConnectionError):
            adapter.connect()

    # ── poll() uses default query when no change_query ────────────────

    def test_poll_uses_default_query_when_no_change_query(self):
        ts = make_timestamps(1)
        records = [_make_neo4j_record("doc-1", ts[0])]
        driver, session = self._make_driver(records)
        adapter = self._make_adapter(driver)

        list(adapter.poll())
        cypher_used = session.run.call_args[0][0]
        self.assertIn("last_modified", cypher_used)
        self.assertIn("Document", cypher_used)

    def test_poll_uses_custom_query_when_provided(self):
        ts = make_timestamps(1)
        records = [_make_neo4j_record("doc-1", ts[0])]
        driver, session = self._make_driver(records)
        custom_q = "MATCH (n:Document) WHERE n.doc_id IS NOT NULL AND ($cursor IS NULL OR n.last_modified > $cursor) RETURN n"
        adapter = self._make_adapter(driver, change_query=custom_q)

        list(adapter.poll())
        cypher_used = session.run.call_args[0][0]
        self.assertEqual(cypher_used, custom_q)

    def test_poll_yields_records(self):
        ts = make_timestamps(2)
        records = [_make_neo4j_record("a", ts[0]), _make_neo4j_record("b", ts[1])]
        driver, _ = self._make_driver(records)
        adapter = self._make_adapter(driver)

        yielded = list(adapter.poll())
        self.assertEqual(len(yielded), 2)

    def test_poll_passes_cursor_as_param(self):
        ts = make_timestamps(1)
        driver, session = self._make_driver([])
        adapter = self._make_adapter(driver, last_cursor=ts[0])

        list(adapter.poll())
        params = session.run.call_args[0][1]
        self.assertEqual(params["cursor"], ts[0])
        self.assertEqual(params["since"], ts[0])    # alias must also be set

    def test_poll_empty_returns_nothing(self):
        driver, _ = self._make_driver([])
        adapter = self._make_adapter(driver)
        self.assertEqual(list(adapter.poll()), [])

    def test_poll_tracks_high_water_mark(self):
        ts = make_timestamps(3)
        records = [
            _make_neo4j_record("a", ts[0]),
            _make_neo4j_record("b", ts[2]),   # highest
            _make_neo4j_record("c", ts[1]),
        ]
        driver, _ = self._make_driver(records)
        adapter = self._make_adapter(driver)

        list(adapter.poll())
        self.assertEqual(adapter.get_cursor(), ts[2])

    def test_poll_does_not_change_last_cursor(self):
        ts = make_timestamps(2)
        records = [_make_neo4j_record("x", ts[1])]
        driver, _ = self._make_driver(records)
        adapter = self._make_adapter(driver, last_cursor=ts[0])

        list(adapter.poll())
        self.assertEqual(adapter.last_cursor, ts[0])

    # ── map_to_source_id ─────────────────────────────────────────────

    def test_map_to_source_id_n_alias(self):
        driver, _ = self._make_driver()
        adapter = self._make_adapter(driver)
        record = {"n": {"doc_id": "node-42"}}
        sid = adapter.map_to_source_id(record)
        self.assertEqual(sid, "node:Document|id:node-42")

    def test_map_to_source_id_flat_dict(self):
        driver, _ = self._make_driver()
        adapter = self._make_adapter(driver)
        record = {"doc_id": "flat-99"}
        sid = adapter.map_to_source_id(record)
        self.assertEqual(sid, "node:Document|id:flat-99")

    def test_map_to_source_id_no_label(self):
        driver, _ = self._make_driver()
        adapter = GraphDBAdapter(
            driver=driver,
            change_query=None,
            mode="polling",
            source_id_field="doc_id",
            node_label="",        # no label
        )
        record = {"doc_id": "unlabeled-1"}
        sid = adapter.map_to_source_id(record)
        self.assertEqual(sid, "node|id:unlabeled-1")

    def test_map_to_source_id_missing_returns_none(self):
        driver, _ = self._make_driver()
        adapter = self._make_adapter(driver)
        self.assertIsNone(adapter.map_to_source_id({"other": "x"}))

    # ── advance_cursor ────────────────────────────────────────────────

    def test_advance_cursor(self):
        ts = make_timestamps(2)
        driver, _ = self._make_driver()
        adapter = self._make_adapter(driver, last_cursor=ts[0])
        adapter.advance_cursor(ts[1])
        self.assertEqual(adapter.last_cursor, ts[1])
        self.assertEqual(adapter._current_cursor, ts[1])

    # ── end-to-end ────────────────────────────────────────────────────

    def test_full_poll_and_map_cycle(self):
        ts = make_timestamps(2)
        records = [_make_neo4j_record("doc-A", ts[0]), _make_neo4j_record("doc-B", ts[1])]
        driver, _ = self._make_driver(records)
        adapter = self._make_adapter(driver)

        source_ids = []
        for rec in adapter.poll():
            sid = adapter.map_to_source_id(rec)
            if sid:
                source_ids.append(sid)
        adapter.advance_cursor(adapter.get_cursor())

        self.assertEqual(source_ids, ["node:Document|id:doc-A", "node:Document|id:doc-B"])
        self.assertEqual(adapter.last_cursor, ts[1])


# ══════════════════════════════════════════════════════════════════════
# RelationalDBAdapter — polling
# ══════════════════════════════════════════════════════════════════════

def _make_db_connection(rows, columns):
    """
    Return a mock DB-API 2.0 connection whose cursor yields `rows`
    with the given column names in cursor.description.
    """
    conn = MagicMock()
    cursor = MagicMock()
    cursor.description = [(col,) for col in columns]
    cursor.__iter__ = MagicMock(return_value=iter(rows))
    conn.cursor.return_value = cursor
    return conn, cursor


class TestRelationalDBAdapterPolling(unittest.TestCase):

    def _make_adapter(self, connection, last_cursor=None, changelog_table=None):
        return RelationalDBAdapter(
            connection=connection,
            table="documents",
            updated_at_col="updated_at",
            mode="polling",
            source_id_field="id",
            poll_interval=10.0,
            last_cursor=last_cursor,
            changelog_table=changelog_table,
        )

    # ── construction ──────────────────────────────────────────────────

    def test_init_stores_params(self):
        conn, _ = _make_db_connection([], [])
        adapter = self._make_adapter(conn)
        self.assertEqual(adapter.mode, "polling")
        self.assertEqual(adapter.table, "documents")
        self.assertEqual(adapter.source_id_field, "id")
        self.assertIsNone(adapter.last_cursor)

    def test_missing_mode_raises(self):
        conn, _ = _make_db_connection([], [])
        with self.assertRaises(MissingModeError):
            RelationalDBAdapter(
                connection=conn,
                table="documents",
                updated_at_col="updated_at",
                mode=None,
                source_id_field="id",
            )

    # ── connect ───────────────────────────────────────────────────────

    def test_connect_opens_and_closes_cursor(self):
        conn, cursor = _make_db_connection([], [])
        adapter = self._make_adapter(conn)
        adapter.connect()
        conn.cursor.assert_called_once()
        cursor.close.assert_called_once()
        self.assertTrue(adapter._connected)

    def test_connect_idempotent(self):
        conn, _ = _make_db_connection([], [])
        adapter = self._make_adapter(conn)
        adapter.connect()
        adapter.connect()
        conn.cursor.assert_called_once()

    def test_connect_raises_on_failure(self):
        conn = MagicMock()
        conn.cursor.side_effect = Exception("connection refused")
        adapter = self._make_adapter(conn)
        with self.assertRaises(AdapterConnectionError):
            adapter.connect()

    def test_disconnect_calls_close(self):
        conn, _ = _make_db_connection([], [])
        adapter = self._make_adapter(conn)
        adapter.connect()
        adapter.disconnect()
        conn.close.assert_called_once()
        self.assertFalse(adapter._connected)

    # ── poll() — updated_at strategy ─────────────────────────────────

    def test_poll_bootstrap_no_where_clause(self):
        ts = make_timestamps(1)
        rows = [(1, ts[0]), (2, ts[0])]
        columns = ["id", "updated_at"]
        conn, cursor = _make_db_connection(rows, columns)
        adapter = self._make_adapter(conn)          # last_cursor=None

        list(adapter.poll())
        sql = cursor.execute.call_args[0][0]
        self.assertNotIn("WHERE updated_at", sql)   # bootstrap: no WHERE

    def test_poll_incremental_uses_where_clause(self):
        ts = make_timestamps(2)
        rows = [(3, ts[1])]
        columns = ["id", "updated_at"]
        conn, cursor = _make_db_connection(rows, columns)
        adapter = self._make_adapter(conn, last_cursor=ts[0])

        list(adapter.poll())
        sql = cursor.execute.call_args[0][0]
        self.assertIn("WHERE updated_at", sql)

    def test_poll_yields_dicts(self):
        ts = make_timestamps(2)
        rows = [(1, ts[0]), (2, ts[1])]
        columns = ["id", "updated_at"]
        conn, _ = _make_db_connection(rows, columns)
        adapter = self._make_adapter(conn)

        results = list(adapter.poll())
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], dict)
        self.assertEqual(results[0]["id"], 1)

    def test_poll_tracks_high_water_mark(self):
        ts = make_timestamps(3)
        rows = [(1, ts[0]), (2, ts[2]), (3, ts[1])]
        columns = ["id", "updated_at"]
        conn, _ = _make_db_connection(rows, columns)
        adapter = self._make_adapter(conn)

        list(adapter.poll())
        self.assertEqual(adapter.get_cursor(), ts[2])

    def test_poll_does_not_mutate_last_cursor(self):
        ts = make_timestamps(2)
        rows = [(5, ts[1])]
        columns = ["id", "updated_at"]
        conn, _ = _make_db_connection(rows, columns)
        adapter = self._make_adapter(conn, last_cursor=ts[0])

        list(adapter.poll())
        self.assertEqual(adapter.last_cursor, ts[0])

    def test_poll_empty(self):
        conn, _ = _make_db_connection([], ["id", "updated_at"])
        adapter = self._make_adapter(conn)
        self.assertEqual(list(adapter.poll()), [])

    def test_poll_raises_adapter_error_on_execute_failure(self):
        conn = MagicMock()
        cursor = MagicMock()
        cursor.description = [("id",), ("updated_at",)]
        cursor.execute.side_effect = Exception("query failed")
        conn.cursor.return_value = cursor
        adapter = self._make_adapter(conn)
        with self.assertRaises(AdapterConnectionError):
            list(adapter.poll())

    # ── poll() — changelog strategy ───────────────────────────────────

    def test_poll_changelog_bootstrap(self):
        ts = make_timestamps(1)
        rows = [("documents", "pk-1", "INSERT", ts[0])]
        columns = ["table_name", "record_pk", "operation", "changed_at"]
        conn, cursor = _make_db_connection(rows, columns)
        adapter = self._make_adapter(conn, changelog_table="changelog")

        list(adapter.poll())
        sql = cursor.execute.call_args[0][0]
        self.assertIn("changelog", sql)
        self.assertIn("table_name", sql)

    def test_poll_changelog_incremental(self):
        ts = make_timestamps(2)
        rows = [("documents", "pk-2", "UPDATE", ts[1])]
        columns = ["table_name", "record_pk", "operation", "changed_at"]
        conn, cursor = _make_db_connection(rows, columns)
        adapter = self._make_adapter(conn, last_cursor=ts[0], changelog_table="changelog")

        list(adapter.poll())
        sql = cursor.execute.call_args[0][0]
        self.assertIn("changed_at", sql)
        self.assertIn(">", sql)

    # ── map_to_source_id ─────────────────────────────────────────────

    def test_map_to_source_id_plain_row(self):
        conn, _ = _make_db_connection([], [])
        adapter = self._make_adapter(conn)
        sid = adapter.map_to_source_id({"id": 42, "updated_at": "2024-01-01"})
        self.assertEqual(sid, "table:documents|pk:42")

    def test_map_to_source_id_changelog_row(self):
        conn, _ = _make_db_connection([], [])
        adapter = self._make_adapter(conn, changelog_table="changelog")
        sid = adapter.map_to_source_id({"record_pk": "doc-99", "table_name": "documents"})
        self.assertEqual(sid, "table:documents|pk:doc-99")

    def test_map_to_source_id_missing_returns_none(self):
        conn, _ = _make_db_connection([], [])
        adapter = self._make_adapter(conn)
        self.assertIsNone(adapter.map_to_source_id({"other_col": "x"}))

    def test_map_to_source_id_none(self):
        conn, _ = _make_db_connection([], [])
        adapter = self._make_adapter(conn)
        self.assertIsNone(adapter.map_to_source_id(None))

    # ── advance_cursor ────────────────────────────────────────────────

    def test_advance_cursor(self):
        ts = make_timestamps(2)
        conn, _ = _make_db_connection([], [])
        adapter = self._make_adapter(conn, last_cursor=ts[0])
        adapter.advance_cursor(ts[1])
        self.assertEqual(adapter.last_cursor, ts[1])
        self.assertEqual(adapter._current_cursor, ts[1])

    # ── end-to-end ────────────────────────────────────────────────────

    def test_full_poll_and_map_cycle_updated_at(self):
        ts = make_timestamps(3)
        rows = [(1, ts[0]), (2, ts[1]), (3, ts[2])]
        columns = ["id", "updated_at"]
        conn, _ = _make_db_connection(rows, columns)
        adapter = self._make_adapter(conn)

        source_ids = []
        for rec in adapter.poll():
            sid = adapter.map_to_source_id(rec)
            if sid:
                source_ids.append(sid)
        adapter.advance_cursor(adapter.get_cursor())

        self.assertEqual(source_ids, [
            "table:documents|pk:1",
            "table:documents|pk:2",
            "table:documents|pk:3",
        ])
        self.assertEqual(adapter.last_cursor, ts[2])

    def test_full_poll_and_map_cycle_changelog(self):
        ts = make_timestamps(2)
        rows = [
            ("documents", "pk-A", "INSERT", ts[0]),
            ("documents", "pk-B", "UPDATE", ts[1]),
        ]
        columns = ["table_name", "record_pk", "operation", "changed_at"]
        conn, _ = _make_db_connection(rows, columns)
        adapter = self._make_adapter(conn, changelog_table="changelog")

        source_ids = []
        for rec in adapter.poll():
            sid = adapter.map_to_source_id(rec)
            if sid:
                source_ids.append(sid)
        adapter.advance_cursor(adapter.get_cursor())

        self.assertEqual(source_ids, [
            "table:documents|pk:pk-A",
            "table:documents|pk:pk-B",
        ])
        self.assertEqual(adapter.last_cursor, ts[1])


# ══════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for cls in [
        TestVectorDBAdapterPolling,
        TestGraphDBAdapterPolling,
        TestRelationalDBAdapterPolling,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)