"""
actors/actor_a_simulator.py
────────────────────────────
Actor A — The Data Simulator ("The Updater")

Connects directly to PostgreSQL and performs UPDATE operations on the
documents table at randomised intervals, mimicking an external system
modifying source data.

Usage (standalone):
    python actors/actor_a_simulator.py

The script accepts an optional --order-id argument to target a specific
document row (defaults to 1, which corresponds to "Order 123").
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import psycopg2
import psycopg2.extras

# Allow imports from parent directory when run standalone.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PG_DSN, PHASE_DELAY_SECONDS

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [ACTOR-A / SIMULATOR]  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("actor_a")

# ── Status progressions that simulate real order lifecycle changes ─────────
_STATUS_UPDATES = [
    ("Processing",        "Order received. Warehouse picking in progress."),
    ("Quality Check",     "Items verified against order manifest."),
    ("Packed",            "Order packed and awaiting carrier pickup."),
    ("Shipped",           "Order handed to carrier. Tracking: TRK-{rand}."),
    ("Out for Delivery",  "With local courier. Expected delivery today."),
    ("Delivered",         "Package delivered and signed for."),
]


def connect_pg(dsn: str):
    """Return a psycopg2 connection, retrying up to 10 times."""
    for attempt in range(1, 11):
        try:
            conn = psycopg2.connect(dsn)
            conn.autocommit = True
            log.info("Connected to PostgreSQL (attempt %d).", attempt)
            return conn
        except psycopg2.OperationalError as exc:
            log.warning("PG not ready (attempt %d): %s", attempt, exc)
            time.sleep(3)
    log.error("Could not connect to PostgreSQL after 10 attempts — exiting.")
    sys.exit(1)


def fetch_document(conn, doc_id: int) -> dict | None:
    """Return the current row for *doc_id*, or None if not found."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
        return cur.fetchone()


def update_document(conn, doc_id: int, new_status: str, new_content: str) -> None:
    """UPDATE the documents row and let the trigger write to the changelog."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE documents SET status = %s, content = %s WHERE id = %s",
            (new_status, new_content, doc_id),
        )
    log.info(
        "✏️  Updated document id=%d → status=%r | content=%r",
        doc_id, new_status, new_content[:60],
    )


def run_simulation(doc_id: int, updates: int, delay_between: float) -> list[int]:
    """
    Perform *updates* sequential UPDATEs on *doc_id*, waiting
    *delay_between* seconds between each.

    Returns a list of the document IDs that were modified (for the
    verifier to check against).
    """
    conn = connect_pg(PG_DSN)

    doc = fetch_document(conn, doc_id)
    if doc is None:
        log.error("Document id=%d not found — check that init.sql ran correctly.", doc_id)
        sys.exit(1)

    log.info("📄 Starting simulation for document id=%d (title=%r).", doc_id, doc["title"])
    log.info("⏳ Waiting %.1fs before first update (Actor B needs time to warm up)…", PHASE_DELAY_SECONDS)
    time.sleep(PHASE_DELAY_SECONDS)

    modified_ids: list[int] = []

    for i in range(updates):
        status_label, content_tmpl = _STATUS_UPDATES[i % len(_STATUS_UPDATES)]
        rand_tracking = random.randint(100_000, 999_999)
        new_content = content_tmpl.format(rand=rand_tracking)

        update_document(conn, doc_id, status_label, new_content)
        modified_ids.append(doc_id)

        if i < updates - 1:
            sleep_secs = delay_between + random.uniform(0, 2)
            log.info("💤 Sleeping %.1fs before next update…", sleep_secs)
            time.sleep(sleep_secs)

    log.info("✅ Simulation complete — %d update(s) issued for doc id=%d.", updates, doc_id)
    conn.close()
    return modified_ids


# ── CLI entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glia validation — Actor A (Data Simulator)")
    parser.add_argument("--order-id",  type=int, default=1,   help="documents.id to update (default: 1)")
    parser.add_argument("--updates",   type=int, default=3,   help="Number of UPDATE operations (default: 3)")
    parser.add_argument("--delay",     type=float, default=8, help="Seconds between updates (default: 8)")
    args = parser.parse_args()

    run_simulation(doc_id=args.order_id, updates=args.updates, delay_between=args.delay)
