"""
runner/seeder.py
────────────────
Massive Seed Phase — injects SEED_RECORD_COUNT records into PostgreSQL,
Neo4j, and Qdrant simultaneously using threaded batch writes.

Also runs the Cache Priming Phase: executes PRIME_QUERY_COUNT queries
through Glia so the Redis semantic cache is warm before benchmarking.

Usage:
    python -m runner.seeder
"""
from __future__ import annotations

import random
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Any

sys.path.insert(0, "/home/claude/glia_benchmark")

from config.settings import (
    SEED_RECORD_COUNT, SEED_BATCH_SIZE, PRIME_QUERY_COUNT,
    DOCUMENT_CATEGORIES, QUERY_TEMPLATES,
    POSTGRES_DSN, POSTGRES_TABLE,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, NEO4J_NODE_LABEL,
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, QDRANT_VECTOR_DIMS if hasattr(__import__("config.settings", fromlist=["QDRANT_VECTOR_DIMS"]), "QDRANT_VECTOR_DIMS") else None,
    REDIS_URL, GLIA_INDEX_NAME, GLIA_DISTANCE_THRESHOLD, GLIA_VECTOR_DIMS, GLIA_TTL_SECONDS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Document factory
# ─────────────────────────────────────────────────────────────────────────────

_LOREM_WORDS = (
    "analysis report system network protocol data model architecture service"
    " compliance risk framework integration pipeline schema latency throughput"
    " benchmark evaluation metric performance optimisation security audit policy"
    " document version release deployment configuration infrastructure cluster"
).split()


def _lorem(n_words: int = 60) -> str:
    return " ".join(random.choices(_LOREM_WORDS, k=n_words)).capitalize() + "."


def generate_documents(n: int) -> List[Dict[str, Any]]:
    """Generate n synthetic document dicts."""
    docs = []
    now = datetime.now(timezone.utc)
    for i in range(n):
        cat = random.choice(DOCUMENT_CATEGORIES)
        doc_id = f"doc-{uuid.uuid4().hex[:12]}"
        title  = f"{cat.title()} Report {i:06d}"
        docs.append({
            "id":         doc_id,
            "title":      title,
            "content":    _lorem(random.randint(40, 100)),
            "category":   cat,
            "version":    1,
            "updated_at": now,
        })
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# PostgreSQL seeder
# ─────────────────────────────────────────────────────────────────────────────

def seed_postgres(docs: List[Dict[str, Any]]) -> int:
    """Bulk-insert docs into PostgreSQL. Returns inserted count."""
    try:
        import psycopg2
        from psycopg2.extras import execute_batch
    except ImportError:
        print("[Seeder/PG] psycopg2 not installed — skipping PostgreSQL seeding.")
        return 0

    conn = psycopg2.connect(POSTGRES_DSN)
    conn.autocommit = False
    cur = conn.cursor()

    sql = (
        "INSERT INTO documents (id, title, content, category, version, updated_at) "
        "VALUES (%(id)s, %(title)s, %(content)s, %(category)s, %(version)s, %(updated_at)s) "
        "ON CONFLICT (id) DO UPDATE SET "
        "  title=EXCLUDED.title, content=EXCLUDED.content, "
        "  version=EXCLUDED.version, updated_at=EXCLUDED.updated_at"
    )

    inserted = 0
    for i in range(0, len(docs), SEED_BATCH_SIZE):
        batch = docs[i : i + SEED_BATCH_SIZE]
        execute_batch(cur, sql, batch, page_size=SEED_BATCH_SIZE)
        conn.commit()
        inserted += len(batch)
        print(f"  [PG] {inserted}/{len(docs)} rows inserted …", end="\r")

    cur.close()
    conn.close()
    print(f"\n  [PG] ✓ {inserted} documents seeded.")
    return inserted


# ─────────────────────────────────────────────────────────────────────────────
# Neo4j seeder
# ─────────────────────────────────────────────────────────────────────────────

def seed_neo4j(docs: List[Dict[str, Any]]) -> int:
    """Bulk-create Document nodes in Neo4j. Returns created count."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[Seeder/Neo4j] neo4j driver not installed — skipping.")
        return 0

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    created = 0

    cypher = (
        "UNWIND $batch AS d "
        f"MERGE (n:{NEO4J_NODE_LABEL} {{doc_id: d.id}}) "
        "SET n.title        = d.title, "
        "    n.content      = d.content, "
        "    n.category     = d.category, "
        "    n.version      = d.version, "
        "    n.last_modified = d.updated_at_str"
    )

    with driver.session(database=NEO4J_DATABASE) as session:
        for i in range(0, len(docs), SEED_BATCH_SIZE):
            batch = [
                {**d, "updated_at_str": d["updated_at"].isoformat()}
                for d in docs[i : i + SEED_BATCH_SIZE]
            ]
            session.run(cypher, batch=batch)
            created += len(batch)
            print(f"  [Neo4j] {created}/{len(docs)} nodes …", end="\r")

    driver.close()
    print(f"\n  [Neo4j] ✓ {created} Document nodes seeded.")
    return created


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant seeder
# ─────────────────────────────────────────────────────────────────────────────

def seed_qdrant(docs: List[Dict[str, Any]], vectorizer) -> int:
    """Upsert docs as vectors into Qdrant. Returns upserted count."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, PointStruct,
        )
    except ImportError:
        print("[Seeder/Qdrant] qdrant-client not installed — skipping.")
        return 0

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Ensure collection exists
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=GLIA_VECTOR_DIMS, distance=Distance.COSINE),
        )
        print(f"  [Qdrant] Created collection '{QDRANT_COLLECTION}'.")

    upserted = 0
    for i in range(0, len(docs), SEED_BATCH_SIZE):
        batch = docs[i : i + SEED_BATCH_SIZE]
        texts  = [f"{d['title']} {d['content']}" for d in batch]
        embeds = vectorizer.embed_many(texts)

        points = [
            PointStruct(
                id=abs(hash(d["id"])) % (2**53),   # Qdrant needs uint64
                vector=vec,
                payload={
                    "doc_id":     d["id"],
                    "title":      d["title"],
                    "category":   d["category"],
                    "updated_at": d["updated_at"].timestamp(),
                },
            )
            for d, vec in zip(batch, embeds)
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        upserted += len(batch)
        print(f"  [Qdrant] {upserted}/{len(docs)} vectors …", end="\r")

    print(f"\n  [Qdrant] ✓ {upserted} vectors seeded.")
    return upserted


# ─────────────────────────────────────────────────────────────────────────────
# Cache priming phase
# ─────────────────────────────────────────────────────────────────────────────

def prime_cache(docs: List[Dict[str, Any]], manager) -> int:
    """
    Run PRIME_QUERY_COUNT queries through Glia to warm the semantic cache.
    Simulates real usage: query → check (miss) → store.
    """
    sample = random.sample(docs, min(PRIME_QUERY_COUNT, len(docs)))
    primed = 0

    for doc in sample:
        templates = QUERY_TEMPLATES.get(doc["category"], QUERY_TEMPLATES["research"])
        query = random.choice(templates).format(title=doc["title"])

        # Simulate cache miss → store
        existing = manager.check(query)
        if existing is None:
            fake_response = (
                f"Based on {doc['title']}: {doc['content'][:200]}"
            )
            manager.store(
                prompt=query,
                response=fake_response,
                source_id=doc["id"],
            )
            primed += 1

        if primed % 50 == 0 and primed > 0:
            print(f"  [Prime] {primed}/{PRIME_QUERY_COUNT} cache entries written …", end="\r")

    print(f"\n  [Prime] ✓ {primed} cache entries primed.")
    return primed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_seed_phase(vectorizer=None, manager=None) -> List[Dict[str, Any]]:
    """
    Full seeding pipeline. Returns the generated document list for reuse
    by the benchmark runner and simulator.

    Parameters
    ----------
    vectorizer : optional
        If None, imports BenchmarkVectorizer automatically.
    manager : optional
        GliaManager instance for cache priming. If None, priming is skipped.
    """
    if vectorizer is None:
        from runner.vectorizer import BenchmarkVectorizer
        vectorizer = BenchmarkVectorizer()

    print(f"\n{'─'*60}")
    print(f"  GLIA BENCHMARK — SEED PHASE ({SEED_RECORD_COUNT:,} records)")
    print(f"{'─'*60}")

    t0 = time.perf_counter()
    docs = generate_documents(SEED_RECORD_COUNT)
    print(f"  Generated {len(docs):,} synthetic documents in "
          f"{time.perf_counter()-t0:.2f}s")

    # Parallel seeding across all three databases
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(seed_postgres, docs): "PostgreSQL",
            pool.submit(seed_neo4j,    docs): "Neo4j",
            pool.submit(seed_qdrant,   docs, vectorizer): "Qdrant",
        }
        for fut in as_completed(futures):
            db = futures[fut]
            try:
                count = fut.result()
                print(f"  ✓ {db}: {count:,} records confirmed.")
            except Exception as exc:
                print(f"  ✗ {db}: seeding failed — {exc}")

    if manager is not None:
        print("\n  Starting cache priming phase …")
        prime_cache(docs, manager)

    elapsed = time.perf_counter() - t0
    print(f"\n  Seed phase complete in {elapsed:.1f}s")
    print(f"{'─'*60}\n")

    return docs


if __name__ == "__main__":
    run_seed_phase()
