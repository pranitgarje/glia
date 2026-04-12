"""
runner/benchmark.py
───────────────────
Symmetric Comparative Benchmark — Dual-Track Execution Engine

Track A: Standard RAG  — always queries the DB + calls LLM (simulated).
Track B: Glia RAG      — checks semantic cache first; falls back to DB+LLM
                         and stores the result for future hits.

Telemetry is collected via Glia's EventEmitter and a lightweight per-query
BenchmarkPacket, then aggregated into a comparison report.

Usage:
    python -m runner.benchmark [--queries 1000] [--skip-seed]
"""
from __future__ import annotations

import json
import random
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, "/home/claude/glia_benchmark")

from config.settings import (
    BENCHMARK_QUERY_COUNT, BENCHMARK_WORKERS,
    DOCUMENT_CATEGORIES, QUERY_TEMPLATES,
    REDIS_URL, GLIA_INDEX_NAME, GLIA_DISTANCE_THRESHOLD,
    GLIA_VECTOR_DIMS, GLIA_TTL_SECONDS,
    POLL_INTERVAL_SECONDS, RECONNECT_RETRIES, RECONNECT_DELAY,
    POSTGRES_DSN, POSTGRES_TABLE, POSTGRES_CHANGELOG_TABLE, POSTGRES_PK_FIELD, POSTGRES_WATERMARK_COL,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, NEO4J_NODE_LABEL, NEO4J_PK_FIELD,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, QDRANT_PK_FIELD, QDRANT_TIMESTAMP_FIELD,
    DASHBOARD_REDIS_CHANNEL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Simulated LLM call (avoids real API costs during benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def _simulated_llm_call(prompt: str, context: str) -> Tuple[str, int]:
    """
    Simulates an LLM call with realistic latency distribution (50–250ms).
    Returns (response_text, tokens_used).
    In production replace this with your actual LLM client call.
    """
    # Simulate variable LLM network latency
    time.sleep(random.uniform(0.05, 0.25))
    tokens = len(prompt.split()) + len(context.split()) + random.randint(50, 150)
    response = (
        f"[LLM] Based on the provided context: {context[:150]}… "
        f"The answer to '{prompt[:60]}' is: {context[150:250]}."
    )
    return response, tokens


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark packet
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkPacket:
    """Single-query telemetry unit. One per (track × query)."""
    timestamp:     float           # UTC epoch
    track:         str             # "glia" | "standard"
    db_type:       str             # "relational" | "graph" | "vector"
    query_text:    str
    latency_ms:    float
    cache_result:  Optional[str]   # "hit" | "miss" | None
    tokens_used:   int
    source_id:     Optional[str]
    is_stale:      bool = False    # True if response reflects pre-update state


# ─────────────────────────────────────────────────────────────────────────────
# Per-DB retrieval stubs (replace with real DB client calls)
# ─────────────────────────────────────────────────────────────────────────────

class RelationalRetriever:
    def __init__(self):
        self._conn = None
        try:
            import psycopg2
            self._conn = psycopg2.connect(POSTGRES_DSN)
            self._conn.autocommit = True
        except Exception:
            pass

    def fetch(self, doc_id: str) -> Optional[Dict]:
        if self._conn is None:
            return {"id": doc_id, "content": "fallback content", "title": "Fallback"}
        try:
            cur = self._conn.cursor()
            cur.execute(f"SELECT id, title, content FROM {POSTGRES_TABLE} WHERE id=%s", (doc_id,))
            row = cur.fetchone()
            cur.close()
            if row:
                return {"id": row[0], "title": row[1], "content": row[2]}
        except Exception:
            pass
        return None

    def close(self):
        if self._conn:
            self._conn.close()


class GraphRetriever:
    def __init__(self):
        self._driver = None
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        except Exception:
            pass

    def fetch(self, doc_id: str) -> Optional[Dict]:
        if self._driver is None:
            return {"doc_id": doc_id, "content": "fallback content", "title": "Fallback"}
        try:
            with self._driver.session(database=NEO4J_DATABASE) as s:
                result = s.run(
                    f"MATCH (n:{NEO4J_NODE_LABEL} {{doc_id: $id}}) "
                    "RETURN n.doc_id AS id, n.title AS title, n.content AS content",
                    id=doc_id,
                )
                rec = result.single()
                if rec:
                    return dict(rec)
        except Exception:
            pass
        return None

    def close(self):
        if self._driver:
            self._driver.close()


class VectorRetriever:
    def __init__(self):
        self._client = None
        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception:
            pass

    def fetch(self, doc_id: str) -> Optional[Dict]:
        if self._client is None:
            return {"doc_id": doc_id, "content": "fallback content", "title": "Fallback"}
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            results = self._client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=Filter(must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]),
                limit=1,
                with_payload=True,
            )
            if results[0]:
                p = results[0][0].payload
                return {"doc_id": p.get("doc_id"), "title": p.get("title",""),
                        "content": p.get("title","") + " document content"}
        except Exception:
            pass
        return None

    def close(self):
        if self._client:
            pass  # qdrant-client handles pool lifecycle


# ─────────────────────────────────────────────────────────────────────────────
# Glia adapter builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_relational_adapter(connection):
    from glia.adapters.relational import RelationalDBAdapter
    return RelationalDBAdapter(
        connection=connection,
        table=POSTGRES_TABLE,
        updated_at_col=POSTGRES_WATERMARK_COL,
        mode="polling",
        source_id_field=POSTGRES_PK_FIELD,
        poll_interval=POLL_INTERVAL_SECONDS,
        changelog_table=POSTGRES_CHANGELOG_TABLE,
        reconnect_retries=RECONNECT_RETRIES,
        reconnect_delay=RECONNECT_DELAY,
    )


def _build_graph_adapter(driver):
    from glia.adapters.graph import GraphDBAdapter
    return GraphDBAdapter(
        driver=driver,
        change_query=None,
        mode="polling",
        source_id_field=NEO4J_PK_FIELD,
        node_label=NEO4J_NODE_LABEL,
        database=NEO4J_DATABASE,
        poll_interval=POLL_INTERVAL_SECONDS,
        reconnect_retries=RECONNECT_RETRIES,
        reconnect_delay=RECONNECT_DELAY,
    )


def _build_vector_adapter(client):
    from glia.adapters.vector import VectorDBAdapter
    return VectorDBAdapter(
        client=client,
        collection=QDRANT_COLLECTION,
        timestamp_field=QDRANT_TIMESTAMP_FIELD,
        mode="polling",
        source_id_field=QDRANT_PK_FIELD,
        poll_interval=POLL_INTERVAL_SECONDS,
        reconnect_retries=RECONNECT_RETRIES,
        reconnect_delay=RECONNECT_DELAY,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry aggregator
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryStore:
    """Thread-safe packet accumulator with live aggregation."""

    def __init__(self):
        self._lock   = threading.Lock()
        self._packets: List[BenchmarkPacket] = []

        # Invalidation latency tracking: source_id → (mutation_ts, invalidation_ts)
        self.invalidation_events: Dict[str, Dict] = {}

    def record(self, packet: BenchmarkPacket) -> None:
        with self._lock:
            self._packets.append(packet)

    def record_invalidation(self, source_id: str, mutation_ts: Optional[float], inv_ts: float) -> None:
        with self._lock:
            self.invalidation_events[source_id] = {
                "mutation_ts":      mutation_ts,
                "invalidation_ts":  inv_ts,
                "latency_ms":       (inv_ts - mutation_ts) * 1000 if mutation_ts else None,
            }

    def snapshot(self) -> List[BenchmarkPacket]:
        with self._lock:
            return list(self._packets)

    def aggregate(self) -> Dict:
        packets = self.snapshot()
        if not packets:
            return {}

        def stats(values):
            if not values:
                return {"mean": 0, "p50": 0, "p95": 0, "p99": 0}
            s = sorted(values)
            n = len(s)
            return {
                "mean": sum(s) / n,
                "p50":  s[int(n * 0.50)],
                "p95":  s[int(n * 0.95)],
                "p99":  s[int(n * 0.99)],
                "count": n,
            }

        result = {}
        for track in ("glia", "standard"):
            result[track] = {}
            for db in ("relational", "graph", "vector"):
                sub = [p for p in packets if p.track == track and p.db_type == db]
                latencies = [p.latency_ms for p in sub]
                tokens    = [p.tokens_used for p in sub]
                hits      = [p for p in sub if p.cache_result == "hit"]
                misses    = [p for p in sub if p.cache_result == "miss"]
                result[track][db] = {
                    "latency":       stats(latencies),
                    "tokens":        stats(tokens),
                    "total_tokens":  sum(tokens),
                    "hit_rate":      len(hits) / max(len(sub), 1),
                    "miss_rate":     len(misses) / max(len(sub), 1),
                    "query_count":   len(sub),
                }

        # Invalidation latency summary
        inv_lats = [
            v["latency_ms"] for v in self.invalidation_events.values()
            if v.get("latency_ms") is not None
        ]
        result["invalidation"] = {
            "count":    len(inv_lats),
            "latency":  stats(inv_lats),
        }

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    Orchestrates the full symmetric comparative benchmark.

    1. Initialises GliaManager + three Glia adapters.
    2. Subscribes to cache_hit, cache_miss, invalidation_complete events.
    3. Starts CacheWatcher (background invalidation).
    4. Starts the VolatilitySimulator.
    5. Runs dual-track query loops across all three DB types.
    6. Collects telemetry and emits it to the Redis channel for the dashboard.
    7. Produces a structured comparison report.
    """

    def __init__(
        self,
        docs: List[Dict],
        vectorizer,
        query_count: int = BENCHMARK_QUERY_COUNT,
        redis_publisher=None,
    ) -> None:
        self._docs         = docs
        self._vectorizer   = vectorizer
        self._query_count  = query_count
        self._redis_pub    = redis_publisher
        self._telemetry    = TelemetryStore()

        # Build GliaManager
        from glia.manager import GliaManager
        from glia.events  import EventEmitter

        self._emitter = EventEmitter()
        self._manager = GliaManager(
            vectorizer=vectorizer,
            redis_url=REDIS_URL,
            index_name=GLIA_INDEX_NAME,
            distance_threshold=GLIA_DISTANCE_THRESHOLD,
            vector_dims=GLIA_VECTOR_DIMS,
            ttl_seconds=GLIA_TTL_SECONDS,
            emitter=self._emitter,
        )

        # Wire event listeners
        self._emitter.on("cache_hit",  self._on_cache_hit)
        self._emitter.on("cache_miss", self._on_cache_miss)
        self._emitter.on("invalidation_complete", self._on_invalidation)

        # Retrievers (standard track)
        self._rel_retriever = RelationalRetriever()
        self._gph_retriever = GraphRetriever()
        self._vec_retriever = VectorRetriever()

        self._watcher   = None
        self._simulator = None

    # ── Event listeners ──────────────────────────────────────────────────────

    def _on_cache_hit(self, event) -> None:
        self._publish("cache_hit", {
            "source_id": event.source_id,
            "payload":   event.payload,
        })

    def _on_cache_miss(self, event) -> None:
        self._publish("cache_miss", {"payload": event.payload})

    def _on_invalidation(self, event) -> None:
        source_id = event.source_id
        inv_ts    = time.time()
        from simulator.volatility import MUTATION_LOG
        mut_ts = MUTATION_LOG.get(source_id)
        self._telemetry.record_invalidation(source_id, mut_ts, inv_ts)
        lat_ms = (inv_ts - mut_ts) * 1000 if mut_ts else None
        self._publish("invalidation", {
            "source_id":    source_id,
            "deleted_count": event.deleted_count,
            "latency_ms":   lat_ms,
        })

    def _publish(self, event_type: str, data: Dict) -> None:
        if self._redis_pub is None:
            return
        try:
            payload = json.dumps({"type": event_type, "ts": time.time(), **data})
            self._redis_pub.publish(DASHBOARD_REDIS_CHANNEL, payload)
        except Exception:
            pass

    # ── Watcher setup ─────────────────────────────────────────────────────────

    def _start_watcher(self) -> None:
        from glia.invalidator  import CacheInvalidator
        from glia.watcher      import CacheWatcher

        invalidator = CacheInvalidator(self._manager)
        adapters    = []

        # PostgreSQL adapter
        try:
            import psycopg2
            conn = psycopg2.connect(POSTGRES_DSN)
            adapters.append(_build_relational_adapter(conn))
        except Exception as e:
            print(f"  [Watcher] PG adapter skipped: {e}")

        # Neo4j adapter
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            adapters.append(_build_graph_adapter(driver))
        except Exception as e:
            print(f"  [Watcher] Neo4j adapter skipped: {e}")

        if adapters:
            self._watcher = CacheWatcher(
                invalidator=invalidator,
                adapters=adapters,
                emitter=self._emitter,
            )
            self._watcher.start()
            print(f"  [Watcher] Started with {len(adapters)} adapter(s).")
        else:
            print("  [Watcher] No adapters available — skipping CacheWatcher.")

    # ── Query execution ───────────────────────────────────────────────────────

    def _run_query_standard(self, doc: Dict, db_type: str) -> BenchmarkPacket:
        """Track A: always hit the DB and call the LLM."""
        t0 = time.perf_counter()
        query = self._make_query(doc)

        retriever = {"relational": self._rel_retriever,
                     "graph":      self._gph_retriever,
                     "vector":     self._vec_retriever}[db_type]

        context_doc = retriever.fetch(doc.get("id") or doc.get("doc_id", ""))
        context = context_doc.get("content", "") if context_doc else ""

        _, tokens = _simulated_llm_call(query, context)
        latency_ms = (time.perf_counter() - t0) * 1000

        return BenchmarkPacket(
            timestamp=time.time(), track="standard", db_type=db_type,
            query_text=query, latency_ms=latency_ms,
            cache_result=None, tokens_used=tokens,
            source_id=doc.get("id") or doc.get("doc_id"),
        )

    def _run_query_glia(self, doc: Dict, db_type: str) -> BenchmarkPacket:
        """Track B: check cache first; fall back to DB+LLM on miss."""
        t0 = time.perf_counter()
        query = self._make_query(doc)
        source_id = doc.get("id") or doc.get("doc_id", "")

        cached = self._manager.check(query)
        if cached is not None:
            latency_ms = (time.perf_counter() - t0) * 1000
            return BenchmarkPacket(
                timestamp=time.time(), track="glia", db_type=db_type,
                query_text=query, latency_ms=latency_ms,
                cache_result="hit", tokens_used=0,
                source_id=source_id,
            )

        # Cache miss — retrieve from DB + LLM
        retriever = {"relational": self._rel_retriever,
                     "graph":      self._gph_retriever,
                     "vector":     self._vec_retriever}[db_type]

        context_doc = retriever.fetch(source_id)
        context = context_doc.get("content", "") if context_doc else ""
        response, tokens = _simulated_llm_call(query, context)

        # Store in cache
        try:
            self._manager.store(prompt=query, response=response, source_id=source_id)
        except Exception:
            pass

        latency_ms = (time.perf_counter() - t0) * 1000

        return BenchmarkPacket(
            timestamp=time.time(), track="glia", db_type=db_type,
            query_text=query, latency_ms=latency_ms,
            cache_result="miss", tokens_used=tokens,
            source_id=source_id,
        )

    def _make_query(self, doc: Dict) -> str:
        cat = doc.get("category", "research")
        templates = QUERY_TEMPLATES.get(cat, QUERY_TEMPLATES["research"])
        return random.choice(templates).format(title=doc.get("title", "the document"))

    # ── Benchmark loop ────────────────────────────────────────────────────────

    def run(self) -> Dict:
        print(f"\n{'═'*60}")
        print(f"  GLIA SYMMETRIC COMPARATIVE BENCHMARK")
        print(f"  {self._query_count} queries × 2 tracks × 3 DB types")
        print(f"{'═'*60}\n")

        self._start_watcher()

        from simulator.volatility import VolatilitySimulator
        doc_ids = [d.get("id") or d.get("doc_id", "") for d in self._docs]
        self._simulator = VolatilitySimulator(
            all_doc_ids=doc_ids,
            on_cycle=lambda c, n: self._publish("simulator_cycle",
                                                 {"cycle": c, "mutations": n}),
        )
        self._simulator.start()

        db_types = ["relational", "graph", "vector"]
        sample   = random.sample(self._docs, min(self._query_count, len(self._docs)))

        print(f"  Running {len(sample)} queries per track per DB type …\n")
        t_start = time.perf_counter()
        completed = 0

        for doc in sample:
            for db_type in db_types:
                # Standard track
                pkt_std = self._run_query_standard(doc, db_type)
                self._telemetry.record(pkt_std)

                # Glia track
                pkt_glia = self._run_query_glia(doc, db_type)
                self._telemetry.record(pkt_glia)

            completed += 1
            if completed % 100 == 0:
                agg = self._telemetry.aggregate()
                self._publish("progress", {
                    "completed": completed,
                    "total":     len(sample),
                    "aggregate": agg,
                })
                print(f"  Progress: {completed}/{len(sample)} queries …", end="\r")

        elapsed = time.perf_counter() - t_start
        print(f"\n  ✓ Benchmark loop complete in {elapsed:.1f}s")

        # Teardown
        self._simulator.stop()
        if self._watcher:
            self._watcher.stop()

        self._rel_retriever.close()
        self._gph_retriever.close()
        self._vec_retriever.close()

        report = self._build_report(elapsed)
        self._publish("final_report", report)
        return report

    def _build_report(self, elapsed_s: float) -> Dict:
        agg = self._telemetry.aggregate()
        report = {
            "meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(elapsed_s, 2),
                "query_count": self._query_count,
                "simulator_cycles":   self._simulator.cycles_completed if self._simulator else 0,
                "simulator_mutations": self._simulator.total_mutations if self._simulator else 0,
            },
            "results": agg,
        }
        return report
