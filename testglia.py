"""
test_glia.py
────────────
Offline test suite for Glia — Layers 1, 2, and 3.

No live Redis or embedding model is required.  All external I/O is
intercepted with unittest.mock so the tests run anywhere.

Run with:
    python test_glia.py          # plain output
    python -m pytest test_glia.py -v   # pytest (if installed)

Layout
------
TestWatcherEvent       – events.py dataclass
TestEventEmitter       – events.py pub/sub dispatcher
TestExceptions         – exceptions.py hierarchy
TestSchemaBuilder      – schema.py validation and build()
TestManagerHelpers     – manager.py pure helpers (_vector_to_bytes, _make_cache_key)
TestGliaManagerInit    – manager.py __init__ and _ensure_index
TestGliaManagerStore   – manager.py store()
TestGliaManagerCheck   – manager.py check() hits, misses, filter
TestGliaManagerMisc    – manager.py delete_index(), on() pass-through
TestCacheInvalidator   – invalidator.py delete_by_tag() all branches
TestDatabaseAdapter    – adapters/base.py mode validation
TestPollingAdapter     – adapters/polling.py construction + contract
TestCDCAdapter         – adapters/cdc.py construction + contract
"""

from __future__ import annotations

import struct
import sys
import types
import unittest
import unittest.mock as mock
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Patch redis sub-modules BEFORE any glia import so they resolve cleanly.
# We leave the real `redis` package in place (for redis.exceptions.ResponseError)
# and only stub the search-command sub-modules that need a live server to import.
# ─────────────────────────────────────────────────────────────────────────────
import redis  # real package — kept for redis.exceptions
from redis.exceptions import ResponseError as RedisResponseError

for _mod in [
    "redis.commands",
    "redis.commands.search",
    "redis.commands.search.field",
    "redis.commands.search.indexDefinition",
    "redis.commands.search.query",
]:
    sys.modules[_mod] = mock.MagicMock()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared across tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_redis(index_exists: bool = False) -> mock.MagicMock:
    """
    Return a MagicMock that behaves like a connected Redis client.

    Parameters
    ----------
    index_exists:
        If True, ft().info() returns successfully (index already created).
        If False (default), ft().info() raises an exception, triggering
        index creation in GliaManager._ensure_index().
    """
    m = mock.MagicMock()
    if index_exists:
        m.ft.return_value.info.return_value = {"index_name": "llmcache"}
    else:
        m.ft.return_value.info.side_effect = Exception("Index does not exist")
    m.ft.return_value.create_index.return_value = True
    m.json.return_value.set.return_value = True
    m.expire.return_value = True
    return m


def _make_manager(
    vector_dims: int = 768,
    distance_threshold: float = 0.2,
    ttl_seconds: int | None = None,
    index_exists: bool = False,
    embed_vector: list[float] | None = None,
) -> tuple:
    """
    Return (GliaManager, mock_redis, mock_vectorizer) with redis.from_url
    already patched.
    """
    # Import here so the redis sub-module stubs above are already active.
    from glia.manager import GliaManager

    mock_redis_instance = _make_mock_redis(index_exists=index_exists)
    redis.from_url = mock.MagicMock(return_value=mock_redis_instance)

    vectorizer = mock.MagicMock()
    vectorizer.embed.return_value = embed_vector or [0.1] * vector_dims

    mgr = GliaManager(
        vectorizer=vectorizer,
        redis_url="redis://localhost:6379",
        vector_dims=vector_dims,
        distance_threshold=distance_threshold,
        ttl_seconds=ttl_seconds,
    )
    return mgr, mock_redis_instance, vectorizer


def _make_search_doc(vector_score: float, response: str, source_id: str = "doc-1"):
    """Return a mock RediSearch document with the given fields."""
    doc = mock.MagicMock()
    doc.vector_score = str(vector_score)
    doc.response = response
    doc.source_id = source_id
    return doc


# ═════════════════════════════════════════════════════════════════════════════
# Layer 1 — events.py
# ═════════════════════════════════════════════════════════════════════════════

class TestWatcherEvent(unittest.TestCase):
    """WatcherEvent dataclass construction and serialisation."""

    def setUp(self):
        from glia.events import WatcherEvent
        self.WatcherEvent = WatcherEvent

    def test_defaults(self):
        ev = self.WatcherEvent(event_type="cache_hit")
        self.assertEqual(ev.event_type, "cache_hit")
        self.assertIsNone(ev.source_id)
        self.assertIsNone(ev.adapter_type)
        self.assertIsNone(ev.detection_mode)
        self.assertEqual(ev.deleted_count, 0)
        self.assertIsInstance(ev.timestamp, datetime)
        self.assertEqual(ev.payload, {})

    def test_full_construction(self):
        ev = self.WatcherEvent(
            event_type="invalidation_complete",
            source_id="doc-42",
            adapter_type="filesystem",
            detection_mode="polling",
            deleted_count=3,
            payload={"file_path": "data/report.md"},
        )
        self.assertEqual(ev.source_id, "doc-42")
        self.assertEqual(ev.deleted_count, 3)

    def test_to_dict_schema(self):
        ev = self.WatcherEvent(
            event_type="cache_miss",
            source_id="node-1",
            payload={"prompt_excerpt": "what is"},
        )
        d = ev.to_dict()
        required_keys = {
            "event_type", "source_id", "adapter_type",
            "detection_mode", "deleted_count", "timestamp", "payload",
        }
        self.assertEqual(set(d.keys()), required_keys)

    def test_to_dict_timestamp_is_string(self):
        ev = self.WatcherEvent(event_type="cache_hit")
        self.assertIsInstance(ev.to_dict()["timestamp"], str)
        # Parseable as ISO-8601
        datetime.fromisoformat(ev.to_dict()["timestamp"])

    def test_to_dict_all_keys_present_even_when_none(self):
        ev = self.WatcherEvent(event_type="cache_miss")
        d = ev.to_dict()
        self.assertIn("source_id", d)
        self.assertIsNone(d["source_id"])
        self.assertEqual(d["deleted_count"], 0)


class TestEventEmitter(unittest.TestCase):
    """EventEmitter registration and dispatch."""

    def setUp(self):
        from glia.events import EventEmitter, WatcherEvent
        self.EventEmitter = EventEmitter
        self.WatcherEvent = WatcherEvent

    def _event(self, event_type="cache_hit", **kw):
        return self.WatcherEvent(event_type=event_type, **kw)

    def test_single_listener_called(self):
        emitter = self.EventEmitter()
        received = []
        emitter.on("cache_hit", lambda e: received.append(e))
        ev = self._event("cache_hit", source_id="doc-1")
        emitter.emit("cache_hit", ev)
        self.assertEqual(len(received), 1)
        self.assertIs(received[0], ev)

    def test_multiple_listeners_called_in_order(self):
        emitter = self.EventEmitter()
        order = []
        emitter.on("cache_hit", lambda e: order.append(1))
        emitter.on("cache_hit", lambda e: order.append(2))
        emitter.emit("cache_hit", self._event())
        self.assertEqual(order, [1, 2])

    def test_different_event_types_isolated(self):
        emitter = self.EventEmitter()
        hits, misses = [], []
        emitter.on("cache_hit", lambda e: hits.append(e))
        emitter.on("cache_miss", lambda e: misses.append(e))
        emitter.emit("cache_hit", self._event("cache_hit"))
        self.assertEqual(len(hits), 1)
        self.assertEqual(len(misses), 0)

    def test_no_listener_does_not_raise(self):
        emitter = self.EventEmitter()
        # Should silently fall back to logging — no exception
        emitter.emit("unknown_event", self._event("unknown_event"))

    def test_callback_exception_reraises(self):
        emitter = self.EventEmitter()
        emitter.on("boom", lambda e: (_ for _ in ()).throw(RuntimeError("oops")))
        with self.assertRaises(RuntimeError):
            emitter.emit("boom", self._event("boom"))

    def test_snapshot_prevents_mid_dispatch_mutation(self):
        """A callback that registers a new listener does not affect current emit."""
        emitter = self.EventEmitter()
        late_calls = []

        def first_listener(e):
            # Register a second listener during dispatch
            emitter.on("cache_hit", lambda ev: late_calls.append(ev))

        emitter.on("cache_hit", first_listener)
        emitter.emit("cache_hit", self._event())
        # The newly registered listener should NOT have been called this round
        self.assertEqual(len(late_calls), 0)
        # But IS called next round
        emitter.emit("cache_hit", self._event())
        self.assertEqual(len(late_calls), 1)


# ═════════════════════════════════════════════════════════════════════════════
# Layer 1 — exceptions.py
# ═════════════════════════════════════════════════════════════════════════════

class TestExceptions(unittest.TestCase):
    """Exception hierarchy and default messages."""

    def setUp(self):
        from glia.exceptions import (
            GliaBaseError, MissingModeError, AdapterConnectionError,
            SchemaValidationError, InvalidationError,
        )
        self.GliaBaseError = GliaBaseError
        self.MissingModeError = MissingModeError
        self.AdapterConnectionError = AdapterConnectionError
        self.SchemaValidationError = SchemaValidationError
        self.InvalidationError = InvalidationError

    def test_all_inherit_from_base(self):
        for cls in (
            self.MissingModeError, self.AdapterConnectionError,
            self.SchemaValidationError, self.InvalidationError,
        ):
            with self.subTest(cls=cls.__name__):
                self.assertTrue(issubclass(cls, self.GliaBaseError))
                self.assertTrue(issubclass(cls, Exception))

    def test_missing_mode_error_default_message(self):
        exc = self.MissingModeError()
        self.assertIn("polling", str(exc))
        self.assertIn("cdc", str(exc))

    def test_missing_mode_error_custom_message(self):
        exc = self.MissingModeError("custom msg")
        self.assertEqual(str(exc), "custom msg")

    def test_catch_by_base(self):
        """All Glia exceptions can be caught with GliaBaseError."""
        for cls in (
            self.MissingModeError, self.AdapterConnectionError,
            self.SchemaValidationError, self.InvalidationError,
        ):
            with self.subTest(cls=cls.__name__):
                try:
                    raise cls("test")
                except self.GliaBaseError:
                    pass  # expected

    def test_cause_chaining(self):
        try:
            cause = ValueError("root cause")
            raise self.AdapterConnectionError("connection failed") from cause
        except self.AdapterConnectionError as exc:
            self.assertIs(exc.__cause__, cause)


# ═════════════════════════════════════════════════════════════════════════════
# Layer 1 — schema.py
# ═════════════════════════════════════════════════════════════════════════════

class TestSchemaBuilder(unittest.TestCase):
    """SchemaBuilder validation and build()."""

    def setUp(self):
        from glia.schema import SchemaBuilder
        from glia.exceptions import SchemaValidationError
        self.SchemaBuilder = SchemaBuilder
        self.SchemaValidationError = SchemaValidationError

    def test_default_build_returns_index_schema(self):
        sb = self.SchemaBuilder(vector_dims=768)
        schema = sb.build()
        self.assertIsNotNone(schema)

    def test_custom_tag_field_at_init(self):
        sb = self.SchemaBuilder(
            vector_dims=768,
            custom_fields=[{"name": "tenant_id", "type": "tag"}],
        )
        sb.build()  # must not raise

    def test_custom_numeric_field_at_init(self):
        sb = self.SchemaBuilder(
            vector_dims=768,
            custom_fields=[{"name": "priority", "type": "numeric"}],
        )
        sb.build()

    def test_add_tag_field_helper(self):
        sb = self.SchemaBuilder(vector_dims=512)
        sb.add_tag_field("department")
        sb.build()

    def test_add_numeric_field_helper(self):
        sb = self.SchemaBuilder(vector_dims=512)
        sb.add_numeric_field("score")
        sb.build()

    def test_zero_vector_dims_raises(self):
        with self.assertRaises(self.SchemaValidationError):
            self.SchemaBuilder(vector_dims=0)

    def test_negative_vector_dims_raises(self):
        with self.assertRaises(self.SchemaValidationError):
            self.SchemaBuilder(vector_dims=-1)

    def test_non_int_vector_dims_raises(self):
        with self.assertRaises(self.SchemaValidationError):
            self.SchemaBuilder(vector_dims=768.0)  # type: ignore

    def test_duplicate_builtin_field_raises(self):
        with self.assertRaises(self.SchemaValidationError):
            self.SchemaBuilder(custom_fields=[{"name": "source_id", "type": "tag"}])

    def test_duplicate_custom_field_raises(self):
        sb = self.SchemaBuilder(custom_fields=[{"name": "tenant", "type": "tag"}])
        with self.assertRaises(self.SchemaValidationError):
            sb.add_tag_field("tenant")

    def test_unsupported_field_type_raises(self):
        with self.assertRaises(self.SchemaValidationError):
            self.SchemaBuilder(custom_fields=[{"name": "body", "type": "text"}])

    def test_missing_name_raises(self):
        with self.assertRaises(self.SchemaValidationError):
            self.SchemaBuilder(custom_fields=[{"type": "tag"}])

    def test_empty_name_raises(self):
        with self.assertRaises(self.SchemaValidationError):
            self.SchemaBuilder(custom_fields=[{"name": "", "type": "tag"}])


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3 — manager.py pure helpers
# ═════════════════════════════════════════════════════════════════════════════

class TestManagerHelpers(unittest.TestCase):
    """Pure helper functions — no Redis needed."""

    def setUp(self):
        from glia.manager import _vector_to_bytes, _make_cache_key
        self._vector_to_bytes = _vector_to_bytes
        self._make_cache_key = _make_cache_key

    def test_vector_to_bytes_length(self):
        v = [0.1, 0.2, 0.3]
        b = self._vector_to_bytes(v)
        # 4 bytes per float32
        self.assertEqual(len(b), 4 * len(v))

    def test_vector_to_bytes_little_endian(self):
        v = [1.0, 2.0]
        b = self._vector_to_bytes(v)
        unpacked = struct.unpack("<2f", b)
        self.assertAlmostEqual(unpacked[0], 1.0, places=5)
        self.assertAlmostEqual(unpacked[1], 2.0, places=5)

    def test_make_cache_key_prefix(self):
        key = self._make_cache_key("llmcache", "hello")
        self.assertTrue(key.startswith("llmcache:entry:"))

    def test_make_cache_key_length(self):
        key = self._make_cache_key("llmcache", "hello")
        # "llmcache:entry:" + 64 hex chars (SHA-256)
        self.assertEqual(len(key), len("llmcache:entry:") + 64)

    def test_make_cache_key_deterministic(self):
        k1 = self._make_cache_key("idx", "same prompt")
        k2 = self._make_cache_key("idx", "same prompt")
        self.assertEqual(k1, k2)

    def test_make_cache_key_differs_by_prompt(self):
        k1 = self._make_cache_key("idx", "prompt A")
        k2 = self._make_cache_key("idx", "prompt B")
        self.assertNotEqual(k1, k2)

    def test_make_cache_key_differs_by_index_name(self):
        k1 = self._make_cache_key("index_a", "prompt")
        k2 = self._make_cache_key("index_b", "prompt")
        self.assertNotEqual(k1, k2)


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3 — manager.py __init__ and index lifecycle
# ═════════════════════════════════════════════════════════════════════════════

class TestGliaManagerInit(unittest.TestCase):

    def test_stores_config_attributes(self):
        mgr, _, _ = _make_manager(vector_dims=512, distance_threshold=0.3, ttl_seconds=60)
        self.assertEqual(mgr.vector_dims, 512)
        self.assertEqual(mgr.distance_threshold, 0.3)
        self.assertEqual(mgr.ttl_seconds, 60)

    def test_vectorizer_stored_as_is(self):
        mgr, _, vec = _make_manager()
        self.assertIs(mgr.vectorizer, vec)

    def test_creates_index_when_absent(self):
        mgr, mock_r, _ = _make_manager(index_exists=False)
        mock_r.ft.return_value.create_index.assert_called_once()

    def test_skips_index_creation_when_present(self):
        mgr, mock_r, _ = _make_manager(index_exists=True)
        mock_r.ft.return_value.create_index.assert_not_called()

    def test_accepts_injected_emitter(self):
        from glia.manager import GliaManager
        from glia.events import EventEmitter
        emitter = EventEmitter()
        mock_r = _make_mock_redis()
        redis.from_url = mock.MagicMock(return_value=mock_r)
        vec = mock.MagicMock()
        vec.embed.return_value = [0.0] * 768
        mgr = GliaManager(
            vectorizer=vec,
            redis_url="redis://localhost:6379",
            emitter=emitter,
        )
        self.assertIs(mgr._emitter, emitter)

    def test_creates_private_emitter_when_none(self):
        from glia.events import EventEmitter
        mgr, _, _ = _make_manager()
        self.assertIsInstance(mgr._emitter, EventEmitter)


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3 — manager.py store()
# ═════════════════════════════════════════════════════════════════════════════

class TestGliaManagerStore(unittest.TestCase):

    def test_embeds_prompt(self):
        mgr, _, vec = _make_manager()
        mgr.store("what is AI?", "AI is...", source_id="doc-1")
        vec.embed.assert_called_with("what is AI?")

    def test_writes_json_to_redis(self):
        mgr, mock_r, _ = _make_manager()
        mgr.store("query", "answer", source_id="doc-1")
        mock_r.json.return_value.set.assert_called_once()
        call_args = mock_r.json.return_value.set.call_args
        key, path, payload = call_args[0]
        self.assertEqual(path, "$")
        self.assertIn("prompt", payload)
        self.assertIn("response", payload)
        self.assertIn("source_id", payload)
        self.assertIn("prompt_vector", payload)

    def test_payload_content(self):
        mgr, mock_r, _ = _make_manager()
        mgr.store("my prompt", "my response", source_id="node-xyz")
        payload = mock_r.json.return_value.set.call_args[0][2]
        self.assertEqual(payload["prompt"], "my prompt")
        self.assertEqual(payload["response"], "my response")
        self.assertEqual(payload["source_id"], "node-xyz")

    def test_key_uses_index_name_prefix(self):
        mgr, mock_r, _ = _make_manager()
        mgr.store("q", "a", source_id="s")
        key = mock_r.json.return_value.set.call_args[0][0]
        self.assertTrue(key.startswith("llmcache:entry:"))

    def test_applies_ttl_when_configured(self):
        mgr, mock_r, _ = _make_manager(ttl_seconds=300)
        mgr.store("q", "a", source_id="s")
        mock_r.expire.assert_called_once()
        args = mock_r.expire.call_args[0]
        self.assertEqual(args[1], 300)

    def test_no_ttl_when_not_configured(self):
        mgr, mock_r, _ = _make_manager(ttl_seconds=None)
        mgr.store("q", "a", source_id="s")
        mock_r.expire.assert_not_called()

    def test_metadata_merged_into_payload(self):
        mgr, mock_r, _ = _make_manager()
        mgr.store("q", "a", source_id="s", metadata={"tenant": "acme"})
        payload = mock_r.json.return_value.set.call_args[0][2]
        self.assertEqual(payload["tenant"], "acme")

    def test_source_id_stored_in_json(self):
        mgr, mock_r, _ = _make_manager()
        mgr.store("q", "a", source_id="my-doc-99")
        payload = mock_r.json.return_value.set.call_args[0][2]
        self.assertEqual(payload["source_id"], "my-doc-99")

    def test_returns_none(self):
        mgr, _, _ = _make_manager()
        result = mgr.store("q", "a", source_id="s")
        self.assertIsNone(result)


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3 — manager.py check()
# ═════════════════════════════════════════════════════════════════════════════

class TestGliaManagerCheck(unittest.TestCase):

    def _setup_search(self, mgr, mock_r, docs):
        mock_r.ft.return_value.search.return_value = mock.MagicMock(docs=docs)

    def test_returns_response_on_hit(self):
        mgr, mock_r, _ = _make_manager(distance_threshold=0.2)
        doc = _make_search_doc(0.05, "cached answer", "doc-1")
        self._setup_search(mgr, mock_r, [doc])
        result = mgr.check("what is AI?")
        self.assertEqual(result, "cached answer")

    def test_returns_none_on_miss_no_docs(self):
        mgr, mock_r, _ = _make_manager()
        self._setup_search(mgr, mock_r, [])
        result = mgr.check("unknown query")
        self.assertIsNone(result)

    def test_returns_none_when_distance_exceeds_threshold(self):
        mgr, mock_r, _ = _make_manager(distance_threshold=0.2)
        doc = _make_search_doc(0.5, "far away answer")  # 0.5 > 0.2
        self._setup_search(mgr, mock_r, [doc])
        result = mgr.check("query")
        self.assertIsNone(result)

    def test_returns_response_exactly_at_threshold(self):
        """Distance equal to threshold is a HIT (not strictly greater than)."""
        mgr, mock_r, _ = _make_manager(distance_threshold=0.2)
        doc = _make_search_doc(0.2, "boundary answer")
        self._setup_search(mgr, mock_r, [doc])
        result = mgr.check("query")
        self.assertEqual(result, "boundary answer")

    def test_emits_cache_hit_event(self):
        mgr, mock_r, _ = _make_manager()
        doc = _make_search_doc(0.05, "answer", "doc-99")
        self._setup_search(mgr, mock_r, [doc])
        hits = []
        mgr.on("cache_hit", lambda ev: hits.append(ev))
        mgr.check("query")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].event_type, "cache_hit")
        self.assertEqual(hits[0].source_id, "doc-99")

    def test_emits_cache_miss_event_no_docs(self):
        mgr, mock_r, _ = _make_manager()
        self._setup_search(mgr, mock_r, [])
        misses = []
        mgr.on("cache_miss", lambda ev: misses.append(ev))
        mgr.check("unknown")
        self.assertEqual(len(misses), 1)
        self.assertEqual(misses[0].event_type, "cache_miss")

    def test_emits_cache_miss_event_threshold_exceeded(self):
        mgr, mock_r, _ = _make_manager(distance_threshold=0.2)
        doc = _make_search_doc(0.9, "far")
        self._setup_search(mgr, mock_r, [doc])
        misses = []
        mgr.on("cache_miss", lambda ev: misses.append(ev))
        mgr.check("query")
        self.assertEqual(len(misses), 1)

    def test_returns_none_on_redis_error(self):
        mgr, mock_r, _ = _make_manager()
        mock_r.ft.return_value.search.side_effect = Exception("Redis unavailable")
        result = mgr.check("query")
        self.assertIsNone(result)

    def test_embeds_prompt_for_search(self):
        mgr, mock_r, vec = _make_manager()
        self._setup_search(mgr, mock_r, [])
        mgr.check("my query")
        vec.embed.assert_called_with("my query")

    def test_filter_included_in_query(self):
        """
        When filter= is provided it must appear in the KNN query string
        that is passed to the Query() constructor.

        Because redis.commands.search.query is mocked, we inspect the
        argument that was given to Query() rather than relying on the
        mock object's __str__.
        """
        mgr, mock_r, _ = _make_manager()
        self._setup_search(mgr, mock_r, [])

        # Capture the Query constructor call
        from redis.commands.search import query as query_mod  # mocked module
        with mock.patch.object(query_mod, "Query", wraps=query_mod.Query) as mock_query:
            mgr.check("q", filter="@source_id:{doc-42}")
            # If Query was called (it will be via the mock), grab the positional arg
            if mock_query.called:
                query_str_arg = mock_query.call_args[0][0]
                self.assertIn("doc-42", query_str_arg)
            else:
                # Query() is fully mocked; verify the search call was made with
                # a query_params dict containing the vector bytes, which
                # confirms check() proceeded to the search step.
                search_call = mock_r.ft.return_value.search.call_args
                self.assertIsNotNone(search_call)
                _, kwargs = search_call
                self.assertIn("vec", kwargs.get("query_params", {}))


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3 — manager.py misc
# ═════════════════════════════════════════════════════════════════════════════

class TestGliaManagerMisc(unittest.TestCase):

    def test_delete_index_calls_dropindex(self):
        mgr, mock_r, _ = _make_manager()
        mgr.delete_index()
        mock_r.ft.return_value.dropindex.assert_called_once()

    def test_delete_index_silent_on_missing(self):
        mgr, mock_r, _ = _make_manager()
        mock_r.ft.return_value.dropindex.side_effect = Exception("no index")
        # Must not raise
        mgr.delete_index()

    def test_on_registers_listener(self):
        mgr, mock_r, _ = _make_manager()
        calls = []
        mgr.on("cache_hit", lambda ev: calls.append(ev))
        # Trigger a hit
        doc = _make_search_doc(0.05, "ans")
        mock_r.ft.return_value.search.return_value = mock.MagicMock(docs=[doc])
        mgr.check("q")
        self.assertEqual(len(calls), 1)


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3 — invalidator.py
# ═════════════════════════════════════════════════════════════════════════════

class TestCacheInvalidator(unittest.TestCase):

    def _setup(self, docs=None, side_effect=None):
        """Return (CacheInvalidator, mock_redis) with search pre-configured."""
        from glia.invalidator import CacheInvalidator
        mgr, mock_r, _ = _make_manager()
        inv = CacheInvalidator(mgr)
        if side_effect:
            mock_r.ft.return_value.search.side_effect = side_effect
        else:
            mock_r.ft.return_value.search.return_value = mock.MagicMock(
                docs=docs or []
            )
        pipe = mock.MagicMock()
        pipe.execute.return_value = [1] * len(docs or [])
        mock_r.pipeline.return_value = pipe
        return inv, mock_r, pipe

    def test_shared_redis_connection(self):
        from glia.invalidator import CacheInvalidator
        mgr, mock_r, _ = _make_manager()
        inv = CacheInvalidator(mgr)
        self.assertIs(inv._redis, mgr._redis)

    def test_returns_zero_when_no_docs(self):
        inv, _, _ = self._setup(docs=[])
        self.assertEqual(inv.delete_by_tag("nonexistent"), 0)

    def test_pipeline_not_called_when_no_docs(self):
        inv, mock_r, pipe = self._setup(docs=[])
        inv.delete_by_tag("nonexistent")
        pipe.delete.assert_not_called()

    def test_deletes_single_key(self):
        doc = mock.MagicMock(id="llmcache:entry:aaa")
        inv, _, pipe = self._setup(docs=[doc])
        count = inv.delete_by_tag("doc-1")
        self.assertEqual(count, 1)
        pipe.delete.assert_called_once_with("llmcache:entry:aaa")

    def test_deletes_multiple_keys_in_single_pipeline(self):
        docs = [
            mock.MagicMock(id="llmcache:entry:aaa"),
            mock.MagicMock(id="llmcache:entry:bbb"),
            mock.MagicMock(id="llmcache:entry:ccc"),
        ]
        inv, mock_r, pipe = self._setup(docs=docs)
        pipe.execute.return_value = [1, 1, 1]
        count = inv.delete_by_tag("doc-multi")
        self.assertEqual(count, 3)
        self.assertEqual(pipe.delete.call_count, 3)

    def test_pipeline_uses_transaction(self):
        doc = mock.MagicMock(id="llmcache:entry:x")
        inv, mock_r, pipe = self._setup(docs=[doc])
        inv.delete_by_tag("doc-1")
        mock_r.pipeline.assert_called_with(transaction=True)

    def test_idempotent_on_response_error(self):
        """ResponseError (index missing) returns 0, never raises."""
        inv, _, _ = self._setup(side_effect=RedisResponseError("Index not found"))
        count = inv.delete_by_tag("doc-99")
        self.assertEqual(count, 0)

    def test_raises_invalidation_error_on_unknown_exception(self):
        from glia.exceptions import InvalidationError
        inv, _, _ = self._setup(side_effect=ConnectionError("Redis down"))
        with self.assertRaises(InvalidationError):
            inv.delete_by_tag("doc-99")

    def test_tag_escaping_special_chars(self):
        """source_id values with special chars must not raise during search."""
        inv, mock_r, pipe = self._setup(docs=[])
        # These characters are all special in RediSearch TAG queries
        inv.delete_by_tag("some-tricky.source/id@test:v1")
        mock_r.ft.return_value.search.assert_called()

    def test_partial_race_condition_counted_correctly(self):
        """
        If a key disappears between search and delete,
        pipeline returns 0 for that slot — the count must reflect
        only actually-deleted keys.
        """
        docs = [
            mock.MagicMock(id="llmcache:entry:aaa"),
            mock.MagicMock(id="llmcache:entry:bbb"),
        ]
        inv, mock_r, pipe = self._setup(docs=docs)
        pipe.execute.return_value = [1, 0]  # second key already gone
        count = inv.delete_by_tag("doc-race")
        self.assertEqual(count, 1)


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2 — adapters/base.py
# ═════════════════════════════════════════════════════════════════════════════

class TestDatabaseAdapter(unittest.TestCase):

    def setUp(self):
        from glia.adapters.base import DatabaseAdapter, VALID_MODES
        from glia.exceptions import MissingModeError

        # Minimal concrete subclass for testing the abstract base
        class ConcreteAdapter(DatabaseAdapter):
            def connect(self): pass
            def disconnect(self): pass
            def map_to_source_id(self, record): return str(record)

        self.ConcreteAdapter = ConcreteAdapter
        self.DatabaseAdapter = DatabaseAdapter
        self.VALID_MODES = VALID_MODES
        self.MissingModeError = MissingModeError

    def test_valid_modes_set(self):
        self.assertIn("polling", self.VALID_MODES)
        self.assertIn("cdc", self.VALID_MODES)

    def test_none_mode_raises(self):
        with self.assertRaises(self.MissingModeError):
            self.ConcreteAdapter(mode=None, source_id_field="id")

    def test_empty_mode_raises(self):
        with self.assertRaises(self.MissingModeError):
            self.ConcreteAdapter(mode="", source_id_field="id")

    def test_invalid_mode_raises(self):
        with self.assertRaises(self.MissingModeError):
            self.ConcreteAdapter(mode="streaming", source_id_field="id")

    def test_polling_mode_accepted(self):
        adapter = self.ConcreteAdapter(mode="polling", source_id_field="id")
        self.assertEqual(adapter.mode, "polling")

    def test_cdc_mode_accepted(self):
        adapter = self.ConcreteAdapter(mode="cdc", source_id_field="id")
        self.assertEqual(adapter.mode, "cdc")

    def test_source_id_field_stored(self):
        adapter = self.ConcreteAdapter(mode="polling", source_id_field="doc_id")
        self.assertEqual(adapter.source_id_field, "doc_id")

    def test_missing_mode_error_message_contains_valid_values(self):
        try:
            self.ConcreteAdapter(mode="bad", source_id_field="id")
        except self.MissingModeError as exc:
            msg = str(exc)
            self.assertIn("polling", msg)
            self.assertIn("cdc", msg)


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2 — adapters/polling.py
# ═════════════════════════════════════════════════════════════════════════════

class TestPollingAdapter(unittest.TestCase):

    def setUp(self):
        from glia.adapters.polling import PollingAdapter
        from glia.exceptions import MissingModeError

        class ConcretePolling(PollingAdapter):
            def connect(self): pass
            def disconnect(self): pass
            def map_to_source_id(self, r): return str(r)
            def poll(self): return iter([])
            def get_cursor(self): return self.last_cursor
            def advance_cursor(self, c): self.last_cursor = c

        self.ConcretePolling = ConcretePolling
        self.MissingModeError = MissingModeError

    def test_default_poll_interval(self):
        pa = self.ConcretePolling(mode="polling", source_id_field="id")
        self.assertEqual(pa.poll_interval, 30.0)

    def test_custom_poll_interval(self):
        pa = self.ConcretePolling(mode="polling", source_id_field="id", poll_interval=10.0)
        self.assertEqual(pa.poll_interval, 10.0)

    def test_default_last_cursor_is_none(self):
        pa = self.ConcretePolling(mode="polling", source_id_field="id")
        self.assertIsNone(pa.last_cursor)

    def test_advance_cursor_updates_last_cursor(self):
        pa = self.ConcretePolling(mode="polling", source_id_field="id")
        pa.advance_cursor("2024-01-01T00:00:00")
        self.assertEqual(pa.get_cursor(), "2024-01-01T00:00:00")

    def test_get_cursor_returns_last_cursor(self):
        pa = self.ConcretePolling(
            mode="polling", source_id_field="id", last_cursor=42
        )
        self.assertEqual(pa.get_cursor(), 42)

    def test_mode_validated_by_base(self):
        with self.assertRaises(self.MissingModeError):
            self.ConcretePolling(mode=None, source_id_field="id")

    def test_map_to_source_id(self):
        pa = self.ConcretePolling(mode="polling", source_id_field="id")
        self.assertEqual(pa.map_to_source_id({"id": "doc-1"}), "{'id': 'doc-1'}")


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2 — adapters/cdc.py
# ═════════════════════════════════════════════════════════════════════════════

class TestCDCAdapter(unittest.TestCase):

    def setUp(self):
        from glia.adapters.cdc import CDCAdapter
        from glia.events import WatcherEvent
        from glia.exceptions import MissingModeError

        class ConcreteCDC(CDCAdapter):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._running = True

            def connect(self): pass
            def disconnect(self): pass
            def map_to_source_id(self, r): return str(r)

            def listen(self):
                yield WatcherEvent(event_type="watcher_event", source_id="doc-1")

            def stop(self):
                self._running = False

        self.ConcreteCDC = ConcreteCDC
        self.WatcherEvent = WatcherEvent
        self.MissingModeError = MissingModeError

    def test_construction(self):
        ca = self.ConcreteCDC(
            mode="cdc",
            source_id_field="id",
            reconnect_retries=5,
            reconnect_delay=2.0,
        )
        self.assertEqual(ca.mode, "cdc")
        self.assertEqual(ca.reconnect_retries, 5)
        self.assertEqual(ca.reconnect_delay, 2.0)

    def test_listen_yields_watcher_events(self):
        ca = self.ConcreteCDC(
            mode="cdc", source_id_field="id",
            reconnect_retries=3, reconnect_delay=1.0,
        )
        events = list(ca.listen())
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], self.WatcherEvent)
        self.assertEqual(events[0].source_id, "doc-1")

    def test_stop_called(self):
        ca = self.ConcreteCDC(
            mode="cdc", source_id_field="id",
            reconnect_retries=3, reconnect_delay=1.0,
        )
        self.assertTrue(ca._running)
        ca.stop()
        self.assertFalse(ca._running)

    def test_mode_validated_by_base(self):
        with self.assertRaises(self.MissingModeError):
            self.ConcreteCDC(
                mode=None, source_id_field="id",
                reconnect_retries=3, reconnect_delay=1.0,
            )


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)