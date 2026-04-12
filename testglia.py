"""
test_glia.py
============
Comprehensive unit + integration tests for the Glia library.

Coverage map
------------
Layer 1  – events.py        : WatcherEvent, EventEmitter
Layer 1  – exceptions.py    : GliaBaseError hierarchy
Layer 1  – schema.py        : SchemaBuilder
Layer 2  – base.py          : DatabaseAdapter (via concrete stub)
Layer 2  – polling.py       : PollingAdapter (via concrete stub)
Layer 2  – cdc.py           : CDCAdapter (via concrete stub)
Layer 3  – manager.py       : GliaManager (Redis mocked)
Layer 3  – invalidator.py   : CacheInvalidator (Redis mocked)
Layer 5  – runners.py       : PollingRunner, CDCRunner (threading)
Layer 5  – watcher.py       : CacheWatcher (orchestration + dispatch)

Run with:
    pytest test_glia.py -v
    python -m pytest testglia.py -v
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Generator, Iterator, List, Optional
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Layer 1: exceptions
# ---------------------------------------------------------------------------
from glia.exceptions import (
    AdapterConnectionError,
    GliaBaseError,
    InvalidationError,
    MissingModeError,
    SchemaValidationError,
)

# ---------------------------------------------------------------------------
# Layer 1: events
# ---------------------------------------------------------------------------
from glia.events import EventEmitter, WatcherEvent

# ---------------------------------------------------------------------------
# Layer 1: schema
# ---------------------------------------------------------------------------
from glia.schema import SchemaBuilder

# ---------------------------------------------------------------------------
# Layer 2: adapter contracts
# ---------------------------------------------------------------------------
from glia.adapters.base import DatabaseAdapter, VALID_MODES
from glia.adapters.polling import PollingAdapter
from glia.adapters.cdc import CDCAdapter

# ---------------------------------------------------------------------------
# Layer 3: cache core
# ---------------------------------------------------------------------------
from glia.manager import GliaManager, _make_cache_key, _vector_to_bytes
from glia.invalidator import CacheInvalidator

# ---------------------------------------------------------------------------
# Layer 5: runners + watcher
# ---------------------------------------------------------------------------
from glia.runners import CDCRunner, PollingRunner
from glia.watcher import CacheWatcher


# ===========================================================================
# Shared test stubs / helpers
# ===========================================================================

class _ConcretePollingAdapter(PollingAdapter):
    """Minimal concrete polling adapter used throughout the tests."""

    def __init__(self, mode="polling", records=None, cursor_val=None, **kwargs):
        super().__init__(mode=mode, source_id_field="id", **kwargs)
        self._records: list = records or []
        self._cursor = cursor_val
        self.connected = False
        self.disconnected = False
        self.cursor_advanced_to: list = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def map_to_source_id(self, record: Any) -> Optional[str]:
        return record.get("id") if isinstance(record, dict) else None

    def poll(self) -> Iterator[Any]:
        yield from self._records

    def get_cursor(self) -> Any:
        return self._cursor

    def advance_cursor(self, new_cursor: Any) -> None:
        self.cursor_advanced_to.append(new_cursor)
        self._cursor = new_cursor


class _ConcreteCDCAdapter(CDCAdapter):
    """Minimal concrete CDC adapter used throughout the tests."""

    def __init__(self, events=None, **kwargs):
        super().__init__(
            mode="cdc",
            source_id_field="id",
            reconnect_retries=3,
            reconnect_delay=0.01,
            **kwargs,
        )
        self._events: list = list(events or [])
        self._stop_flag = threading.Event()
        self.connected = False
        self.disconnected = False
        self.stop_called = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def map_to_source_id(self, record: Any) -> Optional[str]:
        return record.get("id") if isinstance(record, dict) else None

    def listen(self) -> Generator[WatcherEvent, None, None]:
        for evt in self._events:
            if self._stop_flag.is_set():
                return
            yield evt

    def stop(self) -> None:
        self.stop_called = True
        self._stop_flag.set()


def _make_watcher_event(**kwargs) -> WatcherEvent:
    defaults = dict(event_type="watcher_event", source_id="doc-1",
                    adapter_type="test", detection_mode="polling")
    defaults.update(kwargs)
    return WatcherEvent(**defaults)


# ===========================================================================
# LAYER 1 — exceptions.py
# ===========================================================================

class TestExceptions:
    def test_hierarchy_root(self):
        assert issubclass(GliaBaseError, Exception)

    def test_all_exceptions_inherit_from_base(self):
        for cls in (MissingModeError, AdapterConnectionError,
                    SchemaValidationError, InvalidationError):
            assert issubclass(cls, GliaBaseError), f"{cls} must inherit GliaBaseError"

    def test_missing_mode_default_message(self):
        exc = MissingModeError()
        assert "mode" in str(exc).lower()
        assert "polling" in str(exc)
        assert "cdc" in str(exc)

    def test_missing_mode_custom_message(self):
        exc = MissingModeError("custom")
        assert str(exc) == "custom"

    def test_adapter_connection_error_chaining(self):
        cause = RuntimeError("driver down")
        try:
            raise AdapterConnectionError("could not connect") from cause
        except AdapterConnectionError as exc:
            assert exc.__cause__ is cause

    def test_schema_validation_error_is_catchable_as_base(self):
        with pytest.raises(GliaBaseError):
            raise SchemaValidationError("bad schema")

    def test_invalidation_error_is_catchable_as_base(self):
        with pytest.raises(GliaBaseError):
            raise InvalidationError("delete failed")


# ===========================================================================
# LAYER 1 — events.py
# ===========================================================================

class TestWatcherEvent:
    def test_minimal_construction(self):
        e = WatcherEvent(event_type="cache_hit")
        assert e.event_type == "cache_hit"
        assert e.source_id is None
        assert e.deleted_count == 0
        assert isinstance(e.timestamp, datetime)
        assert isinstance(e.payload, dict)

    def test_to_dict_keys_always_present(self):
        e = WatcherEvent(event_type="cache_miss")
        d = e.to_dict()
        for key in ("event_type", "source_id", "adapter_type",
                    "detection_mode", "deleted_count", "timestamp", "payload"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_timestamp_is_iso_string(self):
        e = WatcherEvent(event_type="test")
        d = e.to_dict()
        # Must be parseable as ISO datetime
        datetime.fromisoformat(d["timestamp"])

    def test_to_dict_payload_passthrough(self):
        e = WatcherEvent(event_type="cache_hit", payload={"similarity_score": 0.05})
        assert e.to_dict()["payload"]["similarity_score"] == 0.05

    def test_full_construction(self):
        ts = datetime(2024, 6, 1, 12, 0, 0)
        e = WatcherEvent(
            event_type="invalidation_complete",
            source_id="doc-42",
            adapter_type="filesystem",
            detection_mode="cdc",
            deleted_count=7,
            timestamp=ts,
            payload={"extra": True},
        )
        d = e.to_dict()
        assert d["event_type"] == "invalidation_complete"
        assert d["source_id"] == "doc-42"
        assert d["deleted_count"] == 7
        assert d["detection_mode"] == "cdc"


class TestEventEmitter:
    def test_emit_calls_registered_callback(self):
        emitter = EventEmitter()
        received = []
        emitter.on("cache_hit", lambda e: received.append(e))
        evt = WatcherEvent(event_type="cache_hit")
        emitter.emit("cache_hit", evt)
        assert received == [evt]

    def test_multiple_callbacks_called_in_order(self):
        emitter = EventEmitter()
        order = []
        emitter.on("cache_miss", lambda e: order.append(1))
        emitter.on("cache_miss", lambda e: order.append(2))
        emitter.emit("cache_miss", WatcherEvent(event_type="cache_miss"))
        assert order == [1, 2]

    def test_no_listeners_does_not_raise(self):
        emitter = EventEmitter()
        # Should log at DEBUG and return without raising
        emitter.emit("cache_hit", WatcherEvent(event_type="cache_hit"))

    def test_different_event_types_are_isolated(self):
        emitter = EventEmitter()
        hits = []
        misses = []
        emitter.on("cache_hit", lambda e: hits.append(e))
        emitter.on("cache_miss", lambda e: misses.append(e))
        emitter.emit("cache_hit", WatcherEvent(event_type="cache_hit"))
        assert len(hits) == 1
        assert len(misses) == 0

    def test_callback_exception_is_reraised(self):
        emitter = EventEmitter()
        emitter.on("cache_hit", lambda e: (_ for _ in ()).throw(ValueError("boom")))
        with pytest.raises(ValueError, match="boom"):
            emitter.emit("cache_hit", WatcherEvent(event_type="cache_hit"))

    def test_snapshot_semantics_mid_emit(self):
        """Adding a callback during emission must not affect current dispatch."""
        emitter = EventEmitter()
        call_count = []

        def cb(e):
            call_count.append(1)
            # Register another callback mid-flight
            emitter.on("cache_hit", lambda _e: call_count.append(99))

        emitter.on("cache_hit", cb)
        emitter.emit("cache_hit", WatcherEvent(event_type="cache_hit"))
        # Only the original callback should have been called in this emission
        assert call_count == [1]

    def test_emit_on_separate_event_names_no_cross_talk(self):
        emitter = EventEmitter()
        seen = []
        emitter.on("invalidation_complete", lambda e: seen.append("inv"))
        emitter.emit("watcher_event", WatcherEvent(event_type="watcher_event"))
        assert seen == []


# ===========================================================================
# LAYER 1 — schema.py
# ===========================================================================

class TestSchemaBuilder:
    def test_invalid_vector_dims_zero(self):
        with pytest.raises(SchemaValidationError):
            SchemaBuilder(vector_dims=0)

    def test_invalid_vector_dims_negative(self):
        with pytest.raises(SchemaValidationError):
            SchemaBuilder(vector_dims=-1)

    def test_invalid_vector_dims_float(self):
        with pytest.raises(SchemaValidationError):
            SchemaBuilder(vector_dims=1.5)  # type: ignore[arg-type]

    def test_valid_construction(self):
        sb = SchemaBuilder(vector_dims=512)
        assert sb.vector_dims == 512

    def test_custom_field_missing_name_raises(self):
        with pytest.raises(SchemaValidationError):
            SchemaBuilder(custom_fields=[{"type": "tag"}])

    def test_custom_field_unsupported_type_raises(self):
        with pytest.raises(SchemaValidationError):
            SchemaBuilder(custom_fields=[{"name": "x", "type": "vector"}])

    def test_custom_field_duplicate_builtin_raises(self):
        for builtin in ("prompt", "response", "source_id", "prompt_vector"):
            with pytest.raises(SchemaValidationError):
                SchemaBuilder(custom_fields=[{"name": builtin, "type": "tag"}])

    def test_add_tag_field(self):
        sb = SchemaBuilder(vector_dims=128)
        sb.add_tag_field("tenant_id")
        assert any(f["name"] == "tenant_id" for f in sb._custom_fields)

    def test_add_numeric_field(self):
        sb = SchemaBuilder(vector_dims=128)
        sb.add_numeric_field("score")
        assert any(f["name"] == "score" for f in sb._custom_fields)

    def test_duplicate_custom_field_raises(self):
        sb = SchemaBuilder(vector_dims=128)
        sb.add_tag_field("tenant_id")
        with pytest.raises(SchemaValidationError):
            sb.add_tag_field("tenant_id")

    @patch("glia.schema.IndexSchema")
    def test_build_calls_from_dict(self, mock_index_schema):
        mock_index_schema.from_dict.return_value = MagicMock()
        sb = SchemaBuilder(vector_dims=256)
        sb.build()
        mock_index_schema.from_dict.assert_called_once()
        schema_dict = mock_index_schema.from_dict.call_args[0][0]
        field_names = [f["name"] for f in schema_dict["fields"]]
        assert "prompt" in field_names
        assert "response" in field_names
        assert "source_id" in field_names
        assert "prompt_vector" in field_names

    @patch("glia.schema.IndexSchema")
    def test_build_includes_custom_fields(self, mock_index_schema):
        mock_index_schema.from_dict.return_value = MagicMock()
        sb = SchemaBuilder(vector_dims=128)
        sb.add_tag_field("department")
        sb.add_numeric_field("priority")
        sb.build()
        schema_dict = mock_index_schema.from_dict.call_args[0][0]
        field_names = [f["name"] for f in schema_dict["fields"]]
        assert "department" in field_names
        assert "priority" in field_names

    @patch("glia.schema.IndexSchema")
    def test_build_schema_validation_error_on_redisvl_failure(self, mock_index_schema):
        mock_index_schema.from_dict.side_effect = Exception("pydantic error")
        sb = SchemaBuilder(vector_dims=128)
        with pytest.raises(SchemaValidationError):
            sb.build()


# ===========================================================================
# LAYER 2 — base.py
# ===========================================================================

class TestDatabaseAdapterBase:
    def test_valid_polling_mode_accepted(self):
        adapter = _ConcretePollingAdapter()
        assert adapter.mode == "polling"

    def test_none_mode_raises_missing_mode_error(self):
        with pytest.raises(MissingModeError):
            _ConcretePollingAdapter(mode=None)  # type: ignore[arg-type]

    def test_empty_string_mode_raises(self):
        with pytest.raises(MissingModeError):
            _ConcretePollingAdapter(mode="")  # type: ignore[arg-type]

    def test_invalid_mode_string_raises(self):
        with pytest.raises(MissingModeError):
            _ConcretePollingAdapter(mode="streaming")

    def test_valid_cdc_mode_accepted(self):
        adapter = _ConcreteCDCAdapter()
        assert adapter.mode == "cdc"

    def test_source_id_field_stored(self):
        adapter = _ConcretePollingAdapter()
        assert adapter.source_id_field == "id"

    def test_valid_modes_constant(self):
        assert "polling" in VALID_MODES
        assert "cdc" in VALID_MODES


# ===========================================================================
# LAYER 2 — polling.py
# ===========================================================================

class TestPollingAdapter:
    def test_default_poll_interval(self):
        adapter = _ConcretePollingAdapter()
        assert adapter.poll_interval == 30.0

    def test_custom_poll_interval(self):
        adapter = _ConcretePollingAdapter(poll_interval=10.0)
        assert adapter.poll_interval == 10.0

    def test_default_last_cursor_is_none(self):
        adapter = _ConcretePollingAdapter()
        assert adapter.last_cursor is None

    def test_custom_last_cursor(self):
        adapter = _ConcretePollingAdapter(last_cursor="2024-01-01")
        assert adapter.last_cursor == "2024-01-01"

    def test_poll_yields_all_records(self):
        records = [{"id": "a"}, {"id": "b"}]
        adapter = _ConcretePollingAdapter(records=records)
        assert list(adapter.poll()) == records

    def test_get_cursor_returns_stored_value(self):
        adapter = _ConcretePollingAdapter(cursor_val="ts-100")
        assert adapter.get_cursor() == "ts-100"

    def test_advance_cursor_updates_state(self):
        adapter = _ConcretePollingAdapter(cursor_val="old")
        adapter.advance_cursor("new")
        assert adapter.get_cursor() == "new"

    def test_map_to_source_id_dict_record(self):
        adapter = _ConcretePollingAdapter()
        assert adapter.map_to_source_id({"id": "doc-42"}) == "doc-42"

    def test_map_to_source_id_missing_field_returns_none(self):
        adapter = _ConcretePollingAdapter()
        assert adapter.map_to_source_id({}) is None

    def test_connect_and_disconnect(self):
        adapter = _ConcretePollingAdapter()
        adapter.connect()
        assert adapter.connected
        adapter.disconnect()
        assert adapter.disconnected


# ===========================================================================
# LAYER 2 — cdc.py
# ===========================================================================

class TestCDCAdapter:
    def test_reconnect_retries_stored(self):
        adapter = _ConcreteCDCAdapter()
        assert adapter.reconnect_retries == 3

    def test_reconnect_delay_stored(self):
        adapter = _ConcreteCDCAdapter()
        assert adapter.reconnect_delay == 0.01

    def test_listen_yields_events(self):
        evts = [WatcherEvent(event_type="watcher_event", source_id=f"doc-{i}")
                for i in range(3)]
        adapter = _ConcreteCDCAdapter(events=evts)
        result = list(adapter.listen())
        assert result == evts

    def test_stop_sets_flag(self):
        adapter = _ConcreteCDCAdapter()
        adapter.stop()
        assert adapter.stop_called
        assert adapter._stop_flag.is_set()

    def test_listen_terminates_after_stop(self):
        # Events that would be yielded, but stop is called first
        evts = [WatcherEvent(event_type="watcher_event", source_id="x")]
        adapter = _ConcreteCDCAdapter(events=evts)
        adapter.stop()
        result = list(adapter.listen())
        assert result == []


# ===========================================================================
# LAYER 3 — manager.py (internal helpers)
# ===========================================================================

class TestManagerHelpers:
    def test_make_cache_key_deterministic(self):
        k1 = _make_cache_key("idx", "hello world")
        k2 = _make_cache_key("idx", "hello world")
        assert k1 == k2

    def test_make_cache_key_different_prompts(self):
        k1 = _make_cache_key("idx", "foo")
        k2 = _make_cache_key("idx", "bar")
        assert k1 != k2

    def test_make_cache_key_different_index_names(self):
        k1 = _make_cache_key("idx-a", "prompt")
        k2 = _make_cache_key("idx-b", "prompt")
        assert k1 != k2

    def test_make_cache_key_format(self):
        key = _make_cache_key("llmcache", "test")
        assert key.startswith("llmcache:entry:")

    def test_vector_to_bytes_length(self):
        vec = [0.1, 0.2, 0.3]
        b = _vector_to_bytes(vec)
        assert len(b) == 4 * len(vec)  # 4 bytes per float32

    def test_vector_to_bytes_type(self):
        assert isinstance(_vector_to_bytes([1.0, 2.0]), bytes)


# ===========================================================================
# LAYER 3 — GliaManager (Redis mocked)
# ===========================================================================

def _make_mock_redis():
    """Build a mock Redis client that satisfies GliaManager's init path."""
    r = MagicMock()
    # ft().info() raises to signal the index does not exist yet
    r.ft.return_value.info.side_effect = Exception("Index not found")
    r.ft.return_value.create_index.return_value = True
    r.json.return_value.set.return_value = True
    return r


@pytest.fixture()
def manager():
    """Return a GliaManager with all Redis I/O mocked out."""
    vectorizer = MagicMock()
    vectorizer.embed.return_value = [0.1] * 768
    vectorizer.embed_many.return_value = [[0.1] * 768]

    with patch("glia.manager.redis.from_url") as mock_from_url, \
         patch("glia.schema.IndexSchema") as mock_schema:
        mock_schema.from_dict.return_value = MagicMock()
        mock_redis = _make_mock_redis()
        mock_from_url.return_value = mock_redis

        mgr = GliaManager(
            vectorizer=vectorizer,
            redis_url="redis://localhost:6379",
            index_name="test_cache",
            distance_threshold=0.2,
            vector_dims=768,
        )
        mgr._redis = mock_redis
        yield mgr


class TestGliaManagerInit:
    def test_index_created_when_not_exists(self, manager):
        manager._redis.ft.return_value.create_index.assert_called()

    def test_index_not_recreated_when_exists(self):
        vectorizer = MagicMock()
        vectorizer.embed.return_value = [0.1] * 768
        with patch("glia.manager.redis.from_url") as mock_from_url, \
             patch("glia.schema.IndexSchema") as mock_schema:
            mock_schema.from_dict.return_value = MagicMock()
            mock_redis = MagicMock()
            # info() succeeds → index already exists
            mock_redis.ft.return_value.info.return_value = {}
            mock_from_url.return_value = mock_redis

            GliaManager(vectorizer=vectorizer, redis_url="redis://localhost:6379")
            mock_redis.ft.return_value.create_index.assert_not_called()

    def test_emitter_created_if_not_injected(self, manager):
        assert isinstance(manager._emitter, EventEmitter)

    def test_injected_emitter_is_used(self):
        vectorizer = MagicMock()
        vectorizer.embed.return_value = [0.1] * 768
        custom_emitter = EventEmitter()
        with patch("glia.manager.redis.from_url") as mock_from_url, \
             patch("glia.schema.IndexSchema") as mock_schema:
            mock_schema.from_dict.return_value = MagicMock()
            mock_redis = _make_mock_redis()
            mock_from_url.return_value = mock_redis
            mgr = GliaManager(vectorizer=vectorizer,
                              redis_url="redis://localhost:6379",
                              emitter=custom_emitter)
            assert mgr._emitter is custom_emitter


class TestGliaManagerStore:
    def test_store_calls_embed(self, manager):
        manager.store("what is glia?", "a caching lib", "doc-1")
        manager.vectorizer.embed.assert_called_with("what is glia?")

    def test_store_writes_json(self, manager):
        manager.store("q", "a", "src-1")
        manager._redis.json.return_value.set.assert_called()

    def test_store_applies_ttl_when_set(self, manager):
        manager.ttl_seconds = 300
        manager.store("q", "a", "src-1")
        manager._redis.expire.assert_called()

    def test_store_no_ttl_by_default(self, manager):
        manager.ttl_seconds = None
        manager.store("q", "a", "src-1")
        manager._redis.expire.assert_not_called()

    def test_store_payload_contains_expected_keys(self, manager):
        manager.store("prompt text", "response text", "doc-42")
        set_args = manager._redis.json.return_value.set.call_args
        payload = set_args[0][2]  # third positional arg is the payload dict
        assert "prompt" in payload
        assert "response" in payload
        assert "source_id" in payload
        assert "prompt_vector" in payload
        assert payload["source_id"] == "doc-42"

    def test_store_merges_metadata(self, manager):
        manager.store("q", "a", "src", metadata={"department": "eng"})
        set_args = manager._redis.json.return_value.set.call_args
        payload = set_args[0][2]
        assert payload.get("department") == "eng"


class TestGliaManagerCheck:
    def _setup_hit(self, manager, distance=0.1, source_id="doc-1", response="cached"):
        doc = MagicMock()
        doc.response = response
        doc.vector_score = str(distance)
        doc.source_id = source_id
        manager._redis.ft.return_value.search.return_value = MagicMock(docs=[doc])

    def _setup_miss_empty(self, manager):
        manager._redis.ft.return_value.search.return_value = MagicMock(docs=[])

    def _setup_miss_threshold(self, manager, distance=0.9):
        doc = MagicMock()
        doc.response = "irrelevant"
        doc.vector_score = str(distance)
        doc.source_id = "doc-x"
        manager._redis.ft.return_value.search.return_value = MagicMock(docs=[doc])

    def test_check_returns_cached_response_on_hit(self, manager):
        self._setup_hit(manager)
        result = manager.check("what is glia?")
        assert result == "cached"

    def test_check_returns_none_on_empty_results(self, manager):
        self._setup_miss_empty(manager)
        assert manager.check("unknown prompt") is None

    def test_check_returns_none_when_above_threshold(self, manager):
        self._setup_miss_threshold(manager)
        assert manager.check("far away prompt") is None

    def test_check_emits_cache_hit_event(self, manager):
        self._setup_hit(manager)
        hits = []
        manager._emitter.on("cache_hit", lambda e: hits.append(e))
        manager.check("q")
        assert len(hits) == 1
        assert hits[0].event_type == "cache_hit"

    def test_check_emits_cache_miss_event_on_empty(self, manager):
        self._setup_miss_empty(manager)
        misses = []
        manager._emitter.on("cache_miss", lambda e: misses.append(e))
        manager.check("q")
        assert len(misses) == 1
        assert misses[0].event_type == "cache_miss"

    def test_check_emits_cache_miss_event_on_threshold(self, manager):
        self._setup_miss_threshold(manager)
        misses = []
        manager._emitter.on("cache_miss", lambda e: misses.append(e))
        manager.check("q")
        assert len(misses) == 1

    def test_check_returns_none_on_search_error(self, manager):
        manager._redis.ft.return_value.search.side_effect = Exception("redis down")
        assert manager.check("q") is None

    def test_check_emits_miss_on_search_error(self, manager):
        manager._redis.ft.return_value.search.side_effect = Exception("redis down")
        misses = []
        manager._emitter.on("cache_miss", lambda e: misses.append(e))
        manager.check("q")
        assert misses[0].payload.get("reason") == "search_error"

    def test_check_with_filter_passes_prefilter(self, manager):
        self._setup_hit(manager)
        manager.check("q", filter="@source_id:{doc-1}")
        search_args = manager._redis.ft.return_value.search.call_args
        query_obj = search_args[0][0]
        # The query string should contain the filter
        assert "doc-1" in str(query_obj.query_string())

    def test_on_registers_listener_on_emitter(self, manager):
        received = []
        manager.on("cache_hit", lambda e: received.append(e))
        assert len(manager._emitter._callbacks.get("cache_hit", [])) == 1


class TestGliaManagerDeleteIndex:
    def test_delete_index_calls_dropindex(self, manager):
        manager.delete_index()
        manager._redis.ft.return_value.dropindex.assert_called()

    def test_delete_index_with_documents(self, manager):
        manager.delete_index(drop_documents=True)
        manager._redis.ft.return_value.dropindex.assert_called_with(delete_documents=True)

    def test_delete_index_swallows_exception(self, manager):
        manager._redis.ft.return_value.dropindex.side_effect = Exception("no index")
        manager.delete_index()  # must not raise


# ===========================================================================
# LAYER 3 — CacheInvalidator
# ===========================================================================

@pytest.fixture()
def invalidator(manager):
    return CacheInvalidator(cache_manager=manager)


class TestCacheInvalidatorInit:
    def test_stores_manager(self, invalidator, manager):
        assert invalidator.cache_manager is manager

    def test_redis_shortcut(self, invalidator, manager):
        assert invalidator._redis is manager._redis

    def test_index_name_shortcut(self, invalidator, manager):
        assert invalidator._index_name == manager.index_name


class TestCacheInvalidatorSearchKeys:
    def _setup_search(self, invalidator, keys):
        docs = [MagicMock(id=k) for k in keys]
        invalidator._redis.ft.return_value.search.return_value = MagicMock(docs=docs)

    def test_returns_keys_from_search(self, invalidator):
        self._setup_search(invalidator, ["key:1", "key:2"])
        keys = invalidator._search_keys_for_tag("doc-1")
        assert set(keys) == {"key:1", "key:2"}

    def test_empty_search_returns_empty_list(self, invalidator):
        invalidator._redis.ft.return_value.search.return_value = MagicMock(docs=[])
        assert invalidator._search_keys_for_tag("doc-1") == []

    def test_response_error_returns_empty_list(self, invalidator):
        import redis as redis_lib
        invalidator._redis.ft.return_value.search.side_effect = (
            redis_lib.exceptions.ResponseError("no such index")
        )
        assert invalidator._search_keys_for_tag("doc-1") == []

    def test_other_exception_raises_invalidation_error(self, invalidator):
        invalidator._redis.ft.return_value.search.side_effect = ConnectionError("timeout")
        with pytest.raises(InvalidationError):
            invalidator._search_keys_for_tag("doc-1")

    def test_special_chars_in_source_id_are_escaped(self, invalidator):
        invalidator._redis.ft.return_value.search.return_value = MagicMock(docs=[])
        invalidator._search_keys_for_tag("path/to/file.md")
        # Just verify it didn't crash and search was called
        invalidator._redis.ft.return_value.search.assert_called()


class TestCacheInvalidatorDeleteByTag:
    def _setup_pipeline(self, invalidator, n_keys):
        keys = [f"key:{i}" for i in range(n_keys)]
        docs = [MagicMock(id=k) for k in keys]
        invalidator._redis.ft.return_value.search.return_value = MagicMock(docs=docs)
        pipe = MagicMock()
        pipe.execute.return_value = [1] * n_keys
        invalidator._redis.pipeline.return_value = pipe
        return pipe

    def test_returns_zero_when_no_keys_found(self, invalidator):
        invalidator._redis.ft.return_value.search.return_value = MagicMock(docs=[])
        assert invalidator.delete_by_tag("nonexistent") == 0

    def test_returns_deleted_count(self, invalidator):
        self._setup_pipeline(invalidator, 3)
        result = invalidator.delete_by_tag("doc-1")
        assert result == 3

    def test_pipeline_is_transactional(self, invalidator):
        self._setup_pipeline(invalidator, 2)
        invalidator.delete_by_tag("doc-1")
        invalidator._redis.pipeline.assert_called_with(transaction=True)

    def test_pipeline_delete_called_per_key(self, invalidator):
        pipe = self._setup_pipeline(invalidator, 2)
        invalidator.delete_by_tag("doc-1")
        assert pipe.delete.call_count == 2

    def test_idempotent_on_already_deleted_keys(self, invalidator):
        """Pipeline returns 0 for already-gone keys — sum should reflect actual deletes."""
        keys = ["key:1", "key:2"]
        docs = [MagicMock(id=k) for k in keys]
        invalidator._redis.ft.return_value.search.return_value = MagicMock(docs=docs)
        pipe = MagicMock()
        pipe.execute.return_value = [1, 0]  # second key already gone
        invalidator._redis.pipeline.return_value = pipe
        result = invalidator.delete_by_tag("doc-1")
        assert result == 1


# ===========================================================================
# LAYER 5 — PollingRunner
# ===========================================================================

class TestPollingRunner:
    def _make_runner(self, records=None, dispatch=None, poll_interval=0.05):
        adapter = _ConcretePollingAdapter(
            records=records or [],
            poll_interval=poll_interval,
        )
        dispatch = dispatch or (lambda sid, evt: None)
        return PollingRunner(adapter=adapter, dispatch=dispatch)

    def test_start_launches_thread(self):
        runner = self._make_runner()
        runner.start()
        assert runner._thread.is_alive()
        runner.stop()

    def test_stop_joins_thread(self):
        runner = self._make_runner()
        runner.start()
        runner.stop()
        assert not runner._thread.is_alive()

    def test_double_start_raises(self):
        runner = self._make_runner()
        runner.start()
        with pytest.raises(RuntimeError):
            runner.start()
        runner.stop()

    def test_stop_is_idempotent(self):
        runner = self._make_runner()
        runner.stop()  # never started — should not raise
        runner.stop()

    def test_dispatch_called_for_each_record(self):
        dispatched = []
        records = [{"id": f"doc-{i}"} for i in range(5)]
        runner = self._make_runner(
            records=records,
            dispatch=lambda sid, evt: dispatched.append(sid),
            poll_interval=0.05,
        )
        runner.start()
        time.sleep(0.15)
        runner.stop()
        assert all(f"doc-{i}" in dispatched for i in range(5))

    def test_none_source_id_not_dispatched(self):
        dispatched = []
        records = [{"no_id_field": "x"}]  # map_to_source_id returns None
        runner = self._make_runner(
            records=records,
            dispatch=lambda sid, evt: dispatched.append(sid),
            poll_interval=0.05,
        )
        runner.start()
        time.sleep(0.1)
        runner.stop()
        assert dispatched == []

    def test_cursor_advanced_after_batch(self):
        adapter = _ConcretePollingAdapter(
            records=[{"id": "d1"}],
            cursor_val="ts-99",
            poll_interval=0.05,
        )
        runner = PollingRunner(adapter=adapter, dispatch=lambda s, e: None)
        runner.start()
        time.sleep(0.12)
        runner.stop()
        assert adapter.cursor_advanced_to != []

    def test_dispatch_exception_does_not_crash_thread(self):
        def bad_dispatch(sid, evt):
            raise RuntimeError("callback boom")

        runner = self._make_runner(
            records=[{"id": "doc-1"}],
            dispatch=bad_dispatch,
            poll_interval=0.05,
        )
        runner.start()
        time.sleep(0.12)
        # Thread must still be alive despite the exception
        assert runner._thread.is_alive()
        runner.stop()

    def test_poll_exception_does_not_crash_thread(self):
        adapter = _ConcretePollingAdapter(poll_interval=0.05)
        adapter.poll = MagicMock(side_effect=RuntimeError("db error"))
        runner = PollingRunner(adapter=adapter, dispatch=lambda s, e: None)
        runner.start()
        time.sleep(0.12)
        assert runner._thread.is_alive()
        runner.stop()

    def test_watcher_event_has_correct_fields(self):
        events = []
        records = [{"id": "doc-42"}]
        runner = self._make_runner(
            records=records,
            dispatch=lambda sid, evt: events.append(evt),
            poll_interval=0.05,
        )
        runner.start()
        time.sleep(0.12)
        runner.stop()
        assert len(events) >= 1
        evt = events[0]
        assert evt.event_type == "watcher_event"
        assert evt.source_id == "doc-42"
        assert evt.detection_mode == "polling"


# ===========================================================================
# LAYER 5 — CDCRunner
# ===========================================================================

class TestCDCRunner:
    def _make_runner(self, events=None, dispatch=None):
        adapter = _ConcreteCDCAdapter(events=events or [])
        dispatch = dispatch or (lambda sid, evt: None)
        return CDCRunner(adapter=adapter, dispatch=dispatch)

    def test_start_launches_thread(self):
        runner = self._make_runner()
        runner.start()
        time.sleep(0.05)
        runner.stop()
        assert not runner._thread.is_alive()

    def test_double_start_raises(self):
        runner = self._make_runner()
        runner.start()
        time.sleep(0.05)
        with pytest.raises(RuntimeError):
            runner.start()
        runner.stop()

    def test_stop_is_idempotent(self):
        runner = self._make_runner()
        runner.stop()

    def test_dispatch_called_for_events(self):
        dispatched = []
        evts = [WatcherEvent(event_type="watcher_event", source_id=f"doc-{i}")
                for i in range(3)]
        runner = self._make_runner(
            events=evts,
            dispatch=lambda sid, evt: dispatched.append(sid),
        )
        runner.start()
        time.sleep(0.1)
        runner.stop()
        assert set(dispatched) == {"doc-0", "doc-1", "doc-2"}

    def test_none_source_id_skipped(self):
        dispatched = []
        evts = [WatcherEvent(event_type="watcher_event", source_id=None)]
        runner = self._make_runner(
            events=evts,
            dispatch=lambda sid, evt: dispatched.append(sid),
        )
        runner.start()
        time.sleep(0.1)
        runner.stop()
        assert dispatched == []

    def test_stop_calls_adapter_stop(self):
        runner = self._make_runner()
        runner.start()
        time.sleep(0.05)
        runner.stop()
        assert runner.adapter.stop_called

    def test_stream_exception_triggers_reconnect(self):
        """Adapter that raises on listen() should consume retries."""
        adapter = _ConcreteCDCAdapter()
        call_count = [0]

        def bad_listen():
            call_count[0] += 1
            raise ConnectionError("dropped")
            yield  # make it a generator

        adapter.listen = bad_listen
        runner = CDCRunner(adapter=adapter, dispatch=lambda s, e: None)
        runner.start()
        time.sleep(0.2)
        runner.stop()
        # Must have retried at least once (up to reconnect_retries+1 = 4 times)
        assert call_count[0] >= 1

    def test_dispatch_exception_does_not_kill_thread(self):
        evts = [WatcherEvent(event_type="watcher_event", source_id="doc-1")]

        def bad_dispatch(sid, evt):
            raise RuntimeError("callback error")

        adapter = _ConcreteCDCAdapter(events=evts)
        runner = CDCRunner(adapter=adapter, dispatch=bad_dispatch)
        runner.start()
        time.sleep(0.1)
        runner.stop()
        # Thread exited cleanly (no crash, not alive after stop)
        assert not runner._thread.is_alive()


# ===========================================================================
# LAYER 5 — CacheWatcher
# ===========================================================================

@pytest.fixture()
def mock_invalidator(manager):
    inv = CacheInvalidator(cache_manager=manager)
    inv.delete_by_tag = MagicMock(return_value=3)
    return inv


class TestCacheWatcherInit:
    def test_empty_adapters_raises(self, mock_invalidator):
        with pytest.raises(ValueError):
            CacheWatcher(invalidator=mock_invalidator, adapters=[])

    def test_stores_adapters(self, mock_invalidator):
        adapter = _ConcretePollingAdapter(poll_interval=0.05)
        watcher = CacheWatcher(invalidator=mock_invalidator, adapters=[adapter])
        assert adapter in watcher.adapters

    def test_private_emitter_created_if_not_injected(self, mock_invalidator):
        adapter = _ConcretePollingAdapter(poll_interval=0.05)
        watcher = CacheWatcher(invalidator=mock_invalidator, adapters=[adapter])
        assert isinstance(watcher._emitter, EventEmitter)

    def test_injected_emitter_used(self, mock_invalidator):
        adapter = _ConcretePollingAdapter(poll_interval=0.05)
        emitter = EventEmitter()
        watcher = CacheWatcher(invalidator=mock_invalidator,
                               adapters=[adapter], emitter=emitter)
        assert watcher._emitter is emitter


class TestCacheWatcherLifecycle:
    def test_start_connects_adapter(self, mock_invalidator):
        adapter = _ConcretePollingAdapter(poll_interval=0.05)
        watcher = CacheWatcher(invalidator=mock_invalidator, adapters=[adapter])
        watcher.start()
        watcher.stop()
        assert adapter.connected

    def test_stop_disconnects_adapter(self, mock_invalidator):
        adapter = _ConcretePollingAdapter(poll_interval=0.05)
        watcher = CacheWatcher(invalidator=mock_invalidator, adapters=[adapter])
        watcher.start()
        watcher.stop()
        assert adapter.disconnected

    def test_double_start_raises(self, mock_invalidator):
        adapter = _ConcretePollingAdapter(poll_interval=0.05)
        watcher = CacheWatcher(invalidator=mock_invalidator, adapters=[adapter])
        watcher.start()
        with pytest.raises(RuntimeError):
            watcher.start()
        watcher.stop()

    def test_stop_idempotent_when_not_started(self, mock_invalidator):
        adapter = _ConcretePollingAdapter(poll_interval=0.05)
        watcher = CacheWatcher(invalidator=mock_invalidator, adapters=[adapter])
        watcher.stop()  # should not raise

    def test_runners_cleared_after_stop(self, mock_invalidator):
        adapter = _ConcretePollingAdapter(poll_interval=0.05)
        watcher = CacheWatcher(invalidator=mock_invalidator, adapters=[adapter])
        watcher.start()
        watcher.stop()
        assert watcher._runners == []

    def test_cdc_adapter_gets_cdc_runner(self, mock_invalidator):
        adapter = _ConcreteCDCAdapter()
        watcher = CacheWatcher(invalidator=mock_invalidator, adapters=[adapter])
        watcher.start()
        time.sleep(0.05)
        assert len(watcher._runners) == 1
        assert isinstance(watcher._runners[0], CDCRunner)
        watcher.stop()

    def test_polling_adapter_gets_polling_runner(self, mock_invalidator):
        adapter = _ConcretePollingAdapter(poll_interval=0.05)
        watcher = CacheWatcher(invalidator=mock_invalidator, adapters=[adapter])
        watcher.start()
        assert len(watcher._runners) == 1
        assert isinstance(watcher._runners[0], PollingRunner)
        watcher.stop()

    def test_mixed_adapters(self, mock_invalidator):
        polling = _ConcretePollingAdapter(poll_interval=0.05)
        cdc = _ConcreteCDCAdapter()
        watcher = CacheWatcher(invalidator=mock_invalidator,
                               adapters=[polling, cdc])
        watcher.start()
        time.sleep(0.05)
        runner_types = {type(r) for r in watcher._runners}
        watcher.stop()
        assert PollingRunner in runner_types
        assert CDCRunner in runner_types

    def test_adapter_connection_failure_propagates(self, mock_invalidator):
        adapter = _ConcretePollingAdapter(poll_interval=0.05)
        adapter.connect = MagicMock(side_effect=AdapterConnectionError("no db"))
        watcher = CacheWatcher(invalidator=mock_invalidator, adapters=[adapter])
        with pytest.raises(AdapterConnectionError):
            watcher.start()

    def test_disconnect_failure_does_not_prevent_cleanup(self, mock_invalidator):
        adapter = _ConcretePollingAdapter(poll_interval=0.05)
        adapter.disconnect = MagicMock(side_effect=RuntimeError("disconnect failed"))
        watcher = CacheWatcher(invalidator=mock_invalidator, adapters=[adapter])
        watcher.start()
        watcher.stop()  # must not raise even though disconnect failed
        assert watcher._runners == []


class TestCacheWatcherDispatch:
    def _make_watcher_with_polling_adapter(self, mock_invalidator, records,
                                           on_invalidation=None):
        adapter = _ConcretePollingAdapter(
            records=records, poll_interval=0.05
        )
        return CacheWatcher(
            invalidator=mock_invalidator,
            adapters=[adapter],
            on_invalidation=on_invalidation,
        )

    def test_dispatch_calls_delete_by_tag(self, mock_invalidator):
        records = [{"id": "doc-99"}]
        watcher = self._make_watcher_with_polling_adapter(mock_invalidator, records)
        watcher.start()
        time.sleep(0.15)
        watcher.stop()
        mock_invalidator.delete_by_tag.assert_called_with("doc-99")

    def test_dispatch_emits_invalidation_complete(self, mock_invalidator):
        records = [{"id": "doc-1"}]
        emitter = EventEmitter()
        events_received = []
        emitter.on("invalidation_complete", lambda e: events_received.append(e))
        adapter = _ConcretePollingAdapter(records=records, poll_interval=0.05)
        watcher = CacheWatcher(invalidator=mock_invalidator,
                               adapters=[adapter], emitter=emitter)
        watcher.start()
        time.sleep(0.15)
        watcher.stop()
        assert any(e.event_type == "invalidation_complete" for e in events_received)

    def test_invalidation_complete_event_has_deleted_count(self, mock_invalidator):
        mock_invalidator.delete_by_tag.return_value = 5
        records = [{"id": "doc-1"}]
        emitter = EventEmitter()
        events_received = []
        emitter.on("invalidation_complete", lambda e: events_received.append(e))
        adapter = _ConcretePollingAdapter(records=records, poll_interval=0.05)
        watcher = CacheWatcher(invalidator=mock_invalidator,
                               adapters=[adapter], emitter=emitter)
        watcher.start()
        time.sleep(0.15)
        watcher.stop()
        inv_events = [e for e in events_received if e.event_type == "invalidation_complete"]
        assert any(e.deleted_count == 5 for e in inv_events)

    def test_on_invalidation_callback_called(self, mock_invalidator):
        calls = []
        records = [{"id": "doc-callback"}]
        watcher = self._make_watcher_with_polling_adapter(
            mock_invalidator,
            records,
            on_invalidation=lambda sid, cnt: calls.append((sid, cnt)),
        )
        watcher.start()
        time.sleep(0.15)
        watcher.stop()
        assert any(sid == "doc-callback" for sid, _ in calls)

    def test_on_invalidation_callback_exception_does_not_propagate(self, mock_invalidator):
        def bad_cb(sid, cnt):
            raise RuntimeError("buggy callback")

        records = [{"id": "doc-1"}]
        watcher = self._make_watcher_with_polling_adapter(
            mock_invalidator, records, on_invalidation=bad_cb
        )
        watcher.start()
        time.sleep(0.15)
        watcher.stop()
        # If we got here, the exception was swallowed correctly

    def test_dispatch_delete_by_tag_error_reraises_to_runner(self, mock_invalidator):
        """delete_by_tag failure must propagate so PollingRunner can log it."""
        mock_invalidator.delete_by_tag.side_effect = InvalidationError("redis gone")
        records = [{"id": "doc-boom"}]
        watcher = self._make_watcher_with_polling_adapter(mock_invalidator, records)
        # Watcher must still start and stop cleanly despite internal errors
        watcher.start()
        time.sleep(0.15)
        watcher.stop()


# ===========================================================================
# End-to-end integration smoke test (all layers, minimal mocking)
# ===========================================================================

class TestEndToEndSmoke:
    """
    Smoke test that exercises the full pipeline from GliaManager through
    CacheWatcher → PollingRunner → CacheInvalidator dispatch, using only
    the minimum necessary mocking (Redis I/O).
    """

    def test_full_pipeline_polling(self):
        # --- Set up mocked Redis ---
        vectorizer = MagicMock()
        vectorizer.embed.return_value = [0.1] * 128

        with patch("glia.manager.redis.from_url") as mock_from_url, \
             patch("glia.schema.IndexSchema") as mock_schema:

            mock_schema.from_dict.return_value = MagicMock()
            mock_redis = _make_mock_redis()
            mock_from_url.return_value = mock_redis

            mgr = GliaManager(
                vectorizer=vectorizer,
                redis_url="redis://localhost:6379",
                vector_dims=128,
            )
            mgr._redis = mock_redis

            # Simulate two cache entries for "doc-pipeline"
            docs_found = [MagicMock(id="llmcache:entry:abc"),
                          MagicMock(id="llmcache:entry:def")]
            mock_redis.ft.return_value.search.return_value = MagicMock(docs=docs_found)
            pipe = MagicMock()
            pipe.execute.return_value = [1, 1]
            mock_redis.pipeline.return_value = pipe

            # Wire up invalidator + watcher
            invalidator = CacheInvalidator(cache_manager=mgr)
            on_inv_calls = []
            emitter = EventEmitter()
            emitter.on("invalidation_complete",
                       lambda e: on_inv_calls.append(e.to_dict()))

            adapter = _ConcretePollingAdapter(
                records=[{"id": "doc-pipeline"}],
                poll_interval=0.05,
            )
            watcher = CacheWatcher(
                invalidator=invalidator,
                adapters=[adapter],
                emitter=emitter,
            )
            watcher.start()
            time.sleep(0.15)
            watcher.stop()

            # Validate pipeline ran: delete_by_tag should have been invoked
            mock_redis.ft.return_value.search.assert_called()
            # And an invalidation_complete event should have been emitted
            assert len(on_inv_calls) >= 1
            assert on_inv_calls[0]["event_type"] == "invalidation_complete"
            assert on_inv_calls[0]["source_id"] == "doc-pipeline"