"""
run_validation.py
─────────────────
Main entry point for the Glia Closed-Loop Validation System.

Runs all three actors in a single process:
  • Actor B (GliaApplication) — started in the foreground.
  • Actor A (DataSimulator)   — runs in a background thread (optional,
                                for continuous simulation mode).
  • Actor C (ClosedLoopVerifier) — orchestrates and asserts.

Usage
─────
# Full automated validation (actors A, B, C together):
    python run_validation.py

# Simulation-only mode (Actor A updates PostgreSQL continuously):
    python run_validation.py --simulate-only --order-id 1 --updates 5

# Pre-flight connectivity check:
    python run_validation.py --check-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [ORCHESTRATOR]          %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestrator")


def check_connectivity() -> bool:
    """Quick pre-flight: verify Redis and PostgreSQL are reachable."""
    from config import REDIS_URL, PG_DSN
    import redis
    import psycopg2

    ok = True

    # Redis
    try:
        r = redis.from_url(REDIS_URL, socket_connect_timeout=3)
        r.ping()
        log.info("✅ Redis is reachable at %s", REDIS_URL)
    except Exception as exc:
        log.error("❌ Redis not reachable: %s", exc)
        log.error("   Make sure Docker services are running: docker compose up -d")
        ok = False

    # PostgreSQL
    try:
        conn = psycopg2.connect(PG_DSN, connect_timeout=3)
        conn.close()
        log.info("✅ PostgreSQL is reachable.")
    except Exception as exc:
        log.error("❌ PostgreSQL not reachable: %s", exc)
        log.error("   Make sure Docker services are running: docker compose up -d")
        ok = False

    return ok


def run_full_validation() -> int:
    """Run the complete closed-loop validation and return exit code."""
    from actors.actor_c_verifier import ClosedLoopVerifier

    log.info("=" * 60)
    log.info("  GLIA CLOSED-LOOP VALIDATION — FULL RUN")
    log.info("=" * 60)

    verifier = ClosedLoopVerifier()
    verifier.run_all()
    return 0 if all(r.passed for r in verifier.results) else 1


def run_simulation_only(order_id: int, updates: int, delay: float) -> int:
    """Run Actor A standalone — just simulate data changes."""
    from actors.actor_a_simulator import run_simulation
    run_simulation(doc_id=order_id, updates=updates, delay_between=delay)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Glia Closed-Loop Validation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full validation (default):
  python run_validation.py

  # Check that Docker services are up:
  python run_validation.py --check-only

  # Simulate data changes only (no cache validation):
  python run_validation.py --simulate-only --order-id 1 --updates 5 --delay 10
        """,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify connectivity to Redis and PostgreSQL, then exit.",
    )
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help="Run Actor A (Data Simulator) only — no cache validation.",
    )
    parser.add_argument("--order-id", type=int, default=1,   help="Document ID to simulate (default: 1)")
    parser.add_argument("--updates",  type=int, default=3,   help="Number of UPDATE operations (default: 3)")
    parser.add_argument("--delay",    type=float, default=8, help="Seconds between updates (default: 8)")
    parser.add_argument("--verbose",  action="store_true",   help="Enable DEBUG-level logging.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Pre-flight check ──────────────────────────────────────────────────
    log.info("Running connectivity check…")
    if not check_connectivity():
        return 1

    if args.check_only:
        log.info("All services reachable — exiting (--check-only).")
        return 0

    if args.simulate_only:
        return run_simulation_only(
            order_id=args.order_id,
            updates=args.updates,
            delay=args.delay,
        )

    return run_full_validation()


if __name__ == "__main__":
    sys.exit(main())
