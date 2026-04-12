"""
actors/actor_c_verifier.py
───────────────────────────
Actor C — The Verifier ("The Monitor")

Orchestrates the full closed-loop validation workflow:

  Phase 1 — Establish Baseline
    • Store a cache entry tagged with a specific source_id.
    • Verify subsequent check() returns a HIT.

  Phase 2 — Trigger the Update
    • Update the corresponding row in PostgreSQL (via Actor A logic).
    • Wait for the invalidation_complete event from CacheWatcher.

  Phase 3 — Prove the Miss
    • Re-check the same prompt.
    • Assert the result is a MISS (cache was cleared).

Prints a colour-coded summary report at the end.

Usage (standalone — requires Docker services to be running):
    python actors/actor_c_verifier.py
"""

from __future__ import annotations

import logging
import sys
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Allow imports from parent directory.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import INVALIDATION_TIMEOUT_SECONDS, PG_DSN, PHASE_DELAY_SECONDS
from actors.actor_b_application import GliaApplication

import psycopg2

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [ACTOR-C / VERIFIER]   %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("actor_c")

# ANSI colours for terminal output.
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# ── Test scenario definition ──────────────────────────────────────────────

@dataclass
class TestScenario:
    """Describes one closed-loop validation run."""
    name: str
    doc_id: int                    # PK in the documents table
    prompt: str                    # The LLM query to cache
    initial_response: str          # Simulated LLM answer stored in cache
    update_status: str             # New status to write to the documents row
    update_content: str            # New content for the documents row


@dataclass
class PhaseResult:
    phase: str
    passed: bool
    detail: str
    duration_ms: float


@dataclass
class ScenarioResult:
    scenario: TestScenario
    phases: list[PhaseResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(p.passed for p in self.phases)


# ── Verifier ──────────────────────────────────────────────────────────────

class ClosedLoopVerifier:
    """
    Orchestrates Actor A (simulation) and Actor B (application) to run
    the three-phase validation sequence for each test scenario.
    """

    SCENARIOS: list[TestScenario] = [
        TestScenario(
            name="Order Status Invalidation",
            doc_id=1,
            prompt="What is the current status of Order 123?",
            initial_response="Order 123 is currently Processing. Estimated delivery: 3 days.",
            update_status="Shipped",
            update_content="Order 123 has been shipped! Tracking number: TRK-555777.",
        ),
        TestScenario(
            name="Order Delivery Invalidation",
            doc_id=2,
            prompt="Has Order 456 been delivered yet?",
            initial_response="Order 456 has been shipped. Tracking: TRK-987654.",
            update_status="Delivered",
            update_content="Order 456 has been delivered and signed for.",
        ),
    ]

    def __init__(self) -> None:
        self.app = GliaApplication()
        self._pg_conn = None
        self.results: list[ScenarioResult] = []

    def _connect_pg(self):
        if self._pg_conn is None or self._pg_conn.closed:
            for attempt in range(1, 11):
                try:
                    self._pg_conn = psycopg2.connect(PG_DSN)
                    self._pg_conn.autocommit = True
                    return
                except psycopg2.OperationalError as exc:
                    log.warning("PG not ready (attempt %d): %s", attempt, exc)
                    time.sleep(3)
            raise RuntimeError("Cannot connect to PostgreSQL.")

    def _update_document(self, doc_id: int, status: str, content: str) -> None:
        """Perform a direct SQL UPDATE to trigger the changelog watcher."""
        self._connect_pg()
        with self._pg_conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET status = %s, content = %s WHERE id = %s",
                (status, content, doc_id),
            )
        log.info(
            "✏️  Wrote UPDATE to PostgreSQL: doc_id=%d, status=%r", doc_id, status
        )

    def _make_source_id(self, doc_id: int) -> str:
        """Produce the source_id format that RelationalDBAdapter generates."""
        return f"table:documents|pk:{doc_id}"

    def run_scenario(self, scenario: TestScenario) -> ScenarioResult:
        result = ScenarioResult(scenario=scenario)
        source_id = self._make_source_id(scenario.doc_id)

        _bar = f"{CYAN}{'─' * 60}{RESET}"
        log.info("%s", _bar)
        log.info(
            "%s🔬 Running scenario: %s%s%s", BOLD, CYAN, scenario.name, RESET
        )
        log.info("   source_id = %r", source_id)
        log.info("%s", _bar)

        # ── Phase 1: Establish Baseline ───────────────────────────────────
        log.info("%s▶ Phase 1: Establish Baseline%s", BOLD, RESET)

        t0 = time.monotonic()
        self.app.store(
            prompt=scenario.prompt,
            response=scenario.initial_response,
            source_id=source_id,
        )
        cached, hit = self.app.ask(scenario.prompt)
        phase1_ms = (time.monotonic() - t0) * 1000

        p1_passed = hit and cached == scenario.initial_response
        result.phases.append(PhaseResult(
            phase="Phase 1 — Baseline (cache hit after store)",
            passed=p1_passed,
            detail=(
                f"Expected HIT with exact response — "
                + ("GOT IT ✓" if p1_passed else f"FAILED (hit={hit}, cached={cached!r})")
            ),
            duration_ms=phase1_ms,
        ))
        _log_phase(1, p1_passed, result.phases[-1].detail)

        if not p1_passed:
            log.error("Phase 1 failed — aborting scenario early.")
            return result

        # ── Phase 2: Trigger the Update ───────────────────────────────────
        log.info("%s▶ Phase 2: Trigger Data Update%s", BOLD, RESET)
        log.info("   Sleeping %.1fs before update (gives watcher bootstrap time)…", PHASE_DELAY_SECONDS)
        time.sleep(PHASE_DELAY_SECONDS)

        t0 = time.monotonic()
        self._update_document(
            doc_id=scenario.doc_id,
            status=scenario.update_status,
            content=scenario.update_content,
        )

        log.info("   Waiting for CacheWatcher to detect the change…")
        invalidated = self.app.wait_for_invalidation(
            source_id=source_id,
            timeout=INVALIDATION_TIMEOUT_SECONDS,
        )
        phase2_ms = (time.monotonic() - t0) * 1000

        p2_passed = invalidated
        result.phases.append(PhaseResult(
            phase="Phase 2 — Update detected & invalidated",
            passed=p2_passed,
            detail=(
                "CacheWatcher detected change and deleted cache entries ✓"
                if p2_passed
                else f"Timeout after {INVALIDATION_TIMEOUT_SECONDS}s — watcher did not fire"
            ),
            duration_ms=phase2_ms,
        ))
        _log_phase(2, p2_passed, result.phases[-1].detail)

        if not p2_passed:
            log.error("Phase 2 failed — watcher did not invalidate in time.")
            return result

        # ── Phase 3: Prove the Miss ───────────────────────────────────────
        log.info("%s▶ Phase 3: Prove Cache Miss%s", BOLD, RESET)

        t0 = time.monotonic()
        cached_after, hit_after = self.app.ask(scenario.prompt)
        phase3_ms = (time.monotonic() - t0) * 1000

        p3_passed = not hit_after and cached_after is None
        result.phases.append(PhaseResult(
            phase="Phase 3 — Cache miss after invalidation",
            passed=p3_passed,
            detail=(
                "Cache correctly returned MISS for invalidated entry ✓"
                if p3_passed
                else f"FAILED — cache still returned a response: {cached_after!r}"
            ),
            duration_ms=phase3_ms,
        ))
        _log_phase(3, p3_passed, result.phases[-1].detail)

        return result

    def run_all(self) -> None:
        """Start the application, run every scenario, then print the report."""
        self.app.start()

        # Brief pause so the watcher's bootstrap poll cycle completes before
        # we start storing entries (avoids the watcher invalidating baseline
        # entries that were stored before the first cursor was established).
        log.info("⏳ Waiting %ds for CacheWatcher bootstrap cycle to complete…", PHASE_DELAY_SECONDS)
        time.sleep(PHASE_DELAY_SECONDS)

        try:
            for scenario in self.SCENARIOS:
                r = self.run_scenario(scenario)
                self.results.append(r)
                # Small gap between scenarios so cursors don't overlap.
                time.sleep(2)
        finally:
            self.app.stop()
            if self._pg_conn and not self._pg_conn.closed:
                self._pg_conn.close()

        self._print_report()

    def _print_report(self) -> None:
        """Print a formatted summary of all scenario results."""
        total_phases = sum(len(r.phases) for r in self.results)
        passed_phases = sum(p.passed for r in self.results for p in r.phases)
        all_pass = all(r.passed for r in self.results)

        print()
        print(f"{BOLD}{CYAN}{'═' * 70}{RESET}")
        print(f"{BOLD}{CYAN}  GLIA CLOSED-LOOP VALIDATION REPORT{RESET}")
        print(f"{BOLD}{CYAN}{'═' * 70}{RESET}")
        print()

        for r in self.results:
            status = f"{GREEN}PASS{RESET}" if r.passed else f"{RED}FAIL{RESET}"
            print(f"  {BOLD}Scenario: {r.scenario.name}{RESET}  [{status}]")
            for p in r.phases:
                colour = GREEN if p.passed else RED
                tick   = "✓" if p.passed else "✗"
                print(f"    {colour}{tick}{RESET}  {p.phase}")
                print(f"       {p.detail}  ({p.duration_ms:.0f} ms)")
            print()

        print(f"  Phases passed : {passed_phases}/{total_phases}")
        print(f"  Overall result: ", end="")
        if all_pass:
            print(f"{GREEN}{BOLD}ALL TESTS PASSED ✅{RESET}")
        else:
            print(f"{RED}{BOLD}SOME TESTS FAILED ❌{RESET}")
        print()
        print(f"{BOLD}{CYAN}{'═' * 70}{RESET}")
        print()

        # Print the full Glia event log for debugging.
        print(f"{BOLD}Event log ({len(self.app.event_log)} events):{RESET}")
        for i, evt in enumerate(self.app.event_log, 1):
            et = evt["event_type"]
            sid = evt.get("source_id") or "—"
            ts  = evt.get("timestamp", "")[-12:]
            d   = evt.get("deleted_count", 0)
            print(f"  {i:3}. [{ts}]  {et:<25}  source_id={sid:<40}  deleted={d}")
        print()


# ── Helper ────────────────────────────────────────────────────────────────

def _log_phase(num: int, passed: bool, detail: str) -> None:
    colour = GREEN if passed else RED
    label  = "PASS" if passed else "FAIL"
    log.info("   %s[Phase %d: %s]%s  %s", colour, num, label, RESET, detail)


# ── CLI entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    verifier = ClosedLoopVerifier()
    verifier.run_all()
    # Exit with non-zero code if any test failed (useful in CI).
    sys.exit(0 if all(r.passed for r in verifier.results) else 1)
