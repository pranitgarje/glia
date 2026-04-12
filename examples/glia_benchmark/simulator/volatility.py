"""
simulator/volatility.py
───────────────────────
Synthetic Volatility Simulator — Actor A

Continuously mutates SIMULATOR_UPDATE_FRACTION of the seeded documents
at SIMULATOR_CYCLE_SECONDS intervals across all three databases, forcing
Glia's CacheWatcher to detect and invalidate stale cache entries.

The simulator emits precise "mutation timestamps" into a shared dict so
the benchmark runner can compute per-source-id invalidation latency
(time from DB UPDATE to `invalidation_complete` event).

Usage (standalone):
    python -m simulator.volatility --cycles 10
"""
from __future__ import annotations

import random
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

sys.path.insert(0, "/home/claude/glia_benchmark")

from config.settings import (
    SIMULATOR_UPDATE_FRACTION,
    SIMULATOR_CYCLE_SECONDS,
    SIMULATOR_MAX_CYCLES,
    POSTGRES_DSN, POSTGRES_TABLE,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, NEO4J_NODE_LABEL,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
)


# Shared dict: source_id → UTC epoch float of the most recent mutation.
# The benchmark runner reads this to compute invalidation latency.
MUTATION_LOG: Dict[str, float] = {}
_mutation_lock = threading.Lock()


def record_mutation(source_id: str) -> None:
    with _mutation_lock:
        MUTATION_LOG[source_id] = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# Per-database mutators
# ─────────────────────────────────────────────────────────────────────────────

def _mutate_postgres(doc_ids: List[str]) -> int:
    """UPDATE content + updated_at for a batch of doc IDs."""
    try:
        import psycopg2
    except ImportError:
        return 0

    conn = psycopg2.connect(POSTGRES_DSN)
    conn.autocommit = True
    cur = conn.cursor()

    mutated = 0
    now = datetime.now(timezone.utc)
    for doc_id in doc_ids:
        new_content = (
            f"[UPDATED {now.isoformat()}] This document has been revised "
            f"during the benchmark volatility cycle. ref={random.randint(1000,9999)}"
        )
        cur.execute(
            f"UPDATE {POSTGRES_TABLE} "
            "SET content=%(c)s, version=version+1, updated_at=%(ts)s "
            "WHERE id=%(id)s",
            {"c": new_content, "ts": now, "id": doc_id},
        )
        record_mutation(doc_id)
        mutated += 1

    cur.close()
    conn.close()
    return mutated


def _mutate_neo4j(doc_ids: List[str]) -> int:
    """SET content + last_modified on Document nodes."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return 0

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    now_str = datetime.now(timezone.utc).isoformat()
    mutated = 0

    with driver.session(database=NEO4J_DATABASE) as session:
        for doc_id in doc_ids:
            session.run(
                f"MATCH (n:{NEO4J_NODE_LABEL} {{doc_id: $id}}) "
                "SET n.content       = $content, "
                "    n.last_modified = $ts, "
                "    n.version       = coalesce(n.version, 1) + 1",
                id=doc_id,
                content=(
                    f"[UPDATED {now_str}] Graph node revised. "
                    f"ref={random.randint(1000,9999)}"
                ),
                ts=now_str,
            )
            record_mutation(doc_id)
            mutated += 1

    driver.close()
    return mutated


def _mutate_qdrant(doc_ids: List[str]) -> int:
    """Update payload metadata in Qdrant (simulates a document re-index)."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import SetPayload, Filter, FieldCondition, MatchValue
    except ImportError:
        return 0

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    now_ts = time.time()
    mutated = 0

    for doc_id in doc_ids:
        try:
            client.set_payload(
                collection_name=QDRANT_COLLECTION,
                payload={
                    "updated_at": now_ts,
                    "mutated":    True,
                    "mutation_ref": random.randint(1000, 9999),
                },
                points=Filter(
                    must=[FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )]
                ),
            )
            record_mutation(doc_id)
            mutated += 1
        except Exception:
            pass

    return mutated


# ─────────────────────────────────────────────────────────────────────────────
# Simulator class
# ─────────────────────────────────────────────────────────────────────────────

class VolatilitySimulator:
    """
    Background simulator that continuously mutates a fraction of the seeded
    documents to test Glia's cache invalidation pipeline under realistic load.

    Parameters
    ----------
    all_doc_ids : List[str]
        Full list of document IDs seeded into the databases.
    on_cycle : Callable[[int, int], None], optional
        Callback invoked at the end of each cycle with
        (cycle_number, total_mutations_this_cycle).
    max_cycles : int
        Stop after this many cycles (default from settings).
    cycle_seconds : float
        Wait between cycles (default from settings).
    """

    def __init__(
        self,
        all_doc_ids: List[str],
        on_cycle: Optional[Callable[[int, int], None]] = None,
        max_cycles: int = SIMULATOR_MAX_CYCLES,
        cycle_seconds: float = SIMULATOR_CYCLE_SECONDS,
    ) -> None:
        self._doc_ids    = all_doc_ids
        self._on_cycle   = on_cycle
        self._max_cycles = max_cycles
        self._cycle_secs = cycle_seconds
        self._stop_evt   = threading.Event()
        self._thread     = threading.Thread(
            target=self._run,
            name="glia.VolatilitySimulator",
            daemon=True,
        )
        self.cycles_completed = 0
        self.total_mutations  = 0

    def start(self) -> None:
        self._stop_evt.clear()
        self._thread.start()
        print("[Simulator] Volatility simulator started.")

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        print(f"[Simulator] Stopped after {self.cycles_completed} cycles, "
              f"{self.total_mutations} total mutations.")

    def _run(self) -> None:
        for cycle in range(1, self._max_cycles + 1):
            if self._stop_evt.is_set():
                break

            # Select random subset to mutate this cycle
            n_mutate = max(1, int(len(self._doc_ids) * SIMULATOR_UPDATE_FRACTION))
            targets  = random.sample(self._doc_ids, n_mutate)

            # Split evenly across all three DB types
            chunk = max(1, n_mutate // 3)
            pg_ids  = targets[:chunk]
            n4j_ids = targets[chunk:chunk*2]
            qd_ids  = targets[chunk*2:]

            m_pg  = _mutate_postgres(pg_ids)
            m_n4j = _mutate_neo4j(n4j_ids)
            m_qd  = _mutate_qdrant(qd_ids)

            cycle_total = m_pg + m_n4j + m_qd
            self.total_mutations  += cycle_total
            self.cycles_completed += 1

            print(
                f"[Simulator] Cycle {cycle:03d}: "
                f"PG={m_pg} Neo4j={m_n4j} Qdrant={m_qd} "
                f"(total {cycle_total} mutations)"
            )

            if self._on_cycle:
                try:
                    self._on_cycle(cycle, cycle_total)
                except Exception:
                    pass

            # Interruptible sleep
            self._stop_evt.wait(timeout=self._cycle_secs)

        print("[Simulator] Max cycles reached — exiting.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=5)
    args = p.parse_args()

    # Quick smoke-test with 100 fake IDs
    fake_ids = [f"doc-{i:06d}" for i in range(100)]
    sim = VolatilitySimulator(fake_ids, max_cycles=args.cycles, cycle_seconds=2.0)
    sim.start()
    sim._thread.join()
    print(f"Mutation log sample: {list(MUTATION_LOG.items())[:5]}")
