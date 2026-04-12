"""
run_benchmark.py
────────────────
Main entry point for the Glia Symmetric Comparative Benchmark.

Usage:
    python run_benchmark.py [options]

Options:
    --queries     N      Number of queries per track per DB (default: 1000)
    --skip-seed          Skip the seeding phase (uses existing DB data)
    --skip-prime         Skip the cache priming phase
    --report-only        Just print the last stored report from Redis
    --workers     N      Concurrent query workers (default: 4)
    --output      PATH   Write JSON report to file (default: report.json)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/claude/glia_benchmark")


def _wait_for_services(timeout: int = 60) -> bool:
    """Poll until Redis is reachable (other DBs checked by seeder)."""
    import redis as _redis
    from config.settings import REDIS_URL

    client = _redis.from_url(REDIS_URL)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client.ping()
            print("  ✓ Redis reachable.")
            return True
        except Exception:
            print("  … waiting for Redis …", end="\r")
            time.sleep(2)
    return False


def _print_report(report: dict) -> None:
    """Pretty-print a structured comparison report to stdout."""
    meta = report.get("meta", {})
    res  = report.get("results", {})

    print(f"\n{'═'*70}")
    print(f"  GLIA BENCHMARK REPORT  |  {meta.get('generated_at','')}")
    print(f"  Elapsed: {meta.get('elapsed_seconds',0):.1f}s  |  "
          f"Queries: {meta.get('query_count',0):,}  |  "
          f"Simulator mutations: {meta.get('simulator_mutations',0):,}")
    print(f"{'═'*70}")

    db_types = ["relational", "graph", "vector"]
    tracks   = ["standard", "glia"]

    # ── Latency table ─────────────────────────────────────────────────────────
    print(f"\n  {'LATENCY (ms)':30s}  {'Standard':>12}  {'Glia':>12}  {'Δ':>10}")
    print(f"  {'─'*66}")
    for db in db_types:
        std_lat  = res.get("standard",{}).get(db,{}).get("latency",{}).get("mean",0)
        glia_lat = res.get("glia",   {}).get(db,{}).get("latency",{}).get("mean",0)
        delta    = glia_lat - std_lat
        sign     = "↑" if delta > 0 else "↓"
        print(f"  {db.upper()+' mean':30s}  {std_lat:>12.1f}  {glia_lat:>12.1f}  "
              f"{sign}{abs(delta):>8.1f}")

    # ── Token savings table ───────────────────────────────────────────────────
    print(f"\n  {'TOKEN CONSUMPTION':30s}  {'Standard':>12}  {'Glia':>12}  {'Saved':>10}")
    print(f"  {'─'*66}")
    for db in db_types:
        std_tok  = res.get("standard",{}).get(db,{}).get("total_tokens",0)
        glia_tok = res.get("glia",   {}).get(db,{}).get("total_tokens",0)
        saved    = std_tok - glia_tok
        print(f"  {db.upper()+' total':30s}  {std_tok:>12,}  {glia_tok:>12,}  "
              f"{saved:>+10,}")

    # ── Cache hit/miss table ──────────────────────────────────────────────────
    print(f"\n  {'GLIA CACHE METRICS':30s}  {'Hit Rate':>12}  {'Miss Rate':>12}  {'Queries':>10}")
    print(f"  {'─'*66}")
    for db in db_types:
        gd = res.get("glia",{}).get(db,{})
        print(f"  {db.upper():30s}  {gd.get('hit_rate',0)*100:>11.1f}%  "
              f"{gd.get('miss_rate',0)*100:>11.1f}%  "
              f"{gd.get('query_count',0):>10,}")

    # ── Invalidation latency ─────────────────────────────────────────────────
    inv = res.get("invalidation", {})
    inv_lat = inv.get("latency", {})
    if inv.get("count", 0) > 0:
        print(f"\n  INVALIDATION LATENCY  (n={inv['count']} events)")
        print(f"  Mean:  {inv_lat.get('mean',0):.1f} ms")
        print(f"  P95:   {inv_lat.get('p95',0):.1f} ms")
        print(f"  P99:   {inv_lat.get('p99',0):.1f} ms")

    print(f"\n{'═'*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Glia Symmetric Comparative Benchmark")
    parser.add_argument("--queries",     type=int,  default=1000)
    parser.add_argument("--skip-seed",   action="store_true")
    parser.add_argument("--skip-prime",  action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--output",      default="report.json")
    args = parser.parse_args()

    # ── Report-only mode ──────────────────────────────────────────────────────
    if args.report_only:
        from config.settings import REDIS_URL
        import redis
        r = redis.from_url(REDIS_URL)
        raw = r.get("glia:benchmark:last_report")
        if raw:
            _print_report(json.loads(raw))
        else:
            print("No report found in Redis. Run the benchmark first.")
        return

    # ── Service health check ──────────────────────────────────────────────────
    print("\n[Startup] Checking service connectivity …")
    if not _wait_for_services():
        print("ERROR: Redis not reachable. Start with: docker compose up -d")
        sys.exit(1)

    # ── Vectorizer ────────────────────────────────────────────────────────────
    from runner.vectorizer import BenchmarkVectorizer
    vectorizer = BenchmarkVectorizer()

    # ── GliaManager (for priming) ─────────────────────────────────────────────
    from glia.manager import GliaManager
    from config.settings import (
        REDIS_URL, GLIA_INDEX_NAME, GLIA_DISTANCE_THRESHOLD,
        GLIA_VECTOR_DIMS, GLIA_TTL_SECONDS,
    )
    manager = GliaManager(
        vectorizer=vectorizer,
        redis_url=REDIS_URL,
        index_name=GLIA_INDEX_NAME,
        distance_threshold=GLIA_DISTANCE_THRESHOLD,
        vector_dims=GLIA_VECTOR_DIMS,
        ttl_seconds=GLIA_TTL_SECONDS,
    )

    # ── Seed phase ────────────────────────────────────────────────────────────
    docs = []
    if not args.skip_seed:
        from runner.seeder import run_seed_phase
        prime_manager = None if args.skip_prime else manager
        docs = run_seed_phase(vectorizer=vectorizer, manager=prime_manager)
    else:
        # Reconstruct a minimal doc list from Redis cache keys for query generation
        print("[Startup] Seed skipped — generating placeholder document list.")
        from config.settings import SEED_RECORD_COUNT, DOCUMENT_CATEGORIES, QUERY_TEMPLATES
        import random, uuid
        for i in range(min(args.queries * 2, SEED_RECORD_COUNT)):
            cat = random.choice(DOCUMENT_CATEGORIES)
            docs.append({
                "id": f"doc-{uuid.uuid4().hex[:12]}",
                "title": f"{cat.title()} Report {i:06d}",
                "content": "placeholder",
                "category": cat,
            })

    # ── Redis publisher for dashboard ─────────────────────────────────────────
    import redis as _redis
    pub = _redis.from_url(REDIS_URL)

    # ── Run benchmark ─────────────────────────────────────────────────────────
    from runner.benchmark import BenchmarkRunner
    runner = BenchmarkRunner(
        docs=docs,
        vectorizer=vectorizer,
        query_count=args.queries,
        redis_publisher=pub,
    )

    report = runner.run()

    # ── Persist report ────────────────────────────────────────────────────────
    pub.set("glia:benchmark:last_report", json.dumps(report, default=str))

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"  JSON report written to: {output_path.resolve()}")

    _print_report(report)


if __name__ == "__main__":
    main()
