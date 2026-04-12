# Glia — Closed-Loop Validation System

A self-contained test harness that proves Glia's cache invalidation pipeline
works end-to-end in a realistic production-like environment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Layer                                 │
│   ┌──────────────────────┐    ┌───────────────────────────────┐ │
│   │  Redis Stack         │    │  PostgreSQL 16                │ │
│   │  :6379  (cache)      │    │  :5432  (source DB)           │ │
│   │  :8001  (RedisInsight)│   │  table: documents             │ │
│   └──────────────────────┘    │  table: document_changelog    │ │
│                                └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐   ┌──────────────────────────────────────┐
│  Actor A             │   │  Actor B                             │
│  (Data Simulator)    │──▶│  (Glia Application)                  │
│                      │   │                                      │
│  • Connects to PG    │   │  ┌─────────────────────────────────┐ │
│  • Issues UPDATE     │   │  │  GliaManager                    │ │
│    statements at     │   │  │  • store() / check()            │ │
│    random intervals  │   │  │  • RediSearch KNN cache         │ │
└──────────────────────┘   │  └─────────────────────────────────┘ │
                           │  ┌─────────────────────────────────┐ │
                           │  │  CacheWatcher (background)      │ │
                           │  │  • RelationalDBAdapter          │ │
                           │  │  • Polls document_changelog     │ │
                           │  │  • Calls CacheInvalidator       │ │
                           │  └─────────────────────────────────┘ │
                           └──────────────────────────────────────┘
                                          │ events
                                          ▼
                           ┌──────────────────────────────────────┐
                           │  Actor C                             │
                           │  (Verifier / Monitor)                │
                           │                                      │
                           │  Phase 1: store → assert HIT         │
                           │  Phase 2: trigger UPDATE → wait for  │
                           │           invalidation_complete       │
                           │  Phase 3: assert MISS                 │
                           └──────────────────────────────────────┘
```

---

## Quick Start

### 1. Start the Docker services

```bash
cd glia-validation
docker compose up -d
```

Wait for both services to be healthy (usually ~10 seconds):

```bash
docker compose ps
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you installed Glia from source with `pip install -e .`, that
> already satisfies the `glia` requirement — just install the rest.

### 3. Run the connectivity check

```bash
python run_validation.py --check-only
```

Expected output:
```
✅ Redis is reachable at redis://localhost:6379
✅ PostgreSQL is reachable.
All services reachable — exiting (--check-only).
```

### 4. Run the full validation

```bash
python run_validation.py
```

---

## Expected Output (abridged)

```
09:12:01  [ORCHESTRATOR]          INFO      Running connectivity check…
09:12:01  [ORCHESTRATOR]          INFO      ✅ Redis is reachable
09:12:01  [ORCHESTRATOR]          INFO      ✅ PostgreSQL is reachable
09:12:01  [ORCHESTRATOR]          INFO      GLIA CLOSED-LOOP VALIDATION — FULL RUN

09:12:02  [ACTOR-B / APPLICATION] INFO      🚀 GliaApplication starting…
09:12:02  [ACTOR-B / APPLICATION] INFO      ✅ GliaManager connected to Redis
09:12:02  [ACTOR-B / APPLICATION] INFO      ✅ CacheWatcher started

09:12:04  [ACTOR-C / VERIFIER]    INFO      🔬 Running scenario: Order Status Invalidation

▶ Phase 1: Establish Baseline
09:12:04  [ACTOR-B / APPLICATION] INFO      💾 Stored cache entry — source_id='table:documents|pk:1'
09:12:04  [ACTOR-B / APPLICATION] INFO      🔍 Cache HIT ✅ for prompt='What is the current status...'
   [Phase 1: PASS] Expected HIT with exact response — GOT IT ✓

▶ Phase 2: Trigger Data Update
09:12:06  [ACTOR-C / VERIFIER]    INFO      ✏️  Wrote UPDATE to PostgreSQL: doc_id=1, status='Shipped'
09:12:06  [ACTOR-C / VERIFIER]    INFO      ⏳ Waiting up to 30s for invalidation...
09:12:11  [ACTOR-B / APPLICATION] INFO         🗑️  invalidation_complete  source_id='table:documents|pk:1'  deleted=1
09:12:11  [ACTOR-C / VERIFIER]    INFO      🔔 Invalidation confirmed for source_id='table:documents|pk:1'
   [Phase 2: PASS] CacheWatcher detected change and deleted cache entries ✓

▶ Phase 3: Prove Cache Miss
09:12:11  [ACTOR-B / APPLICATION] INFO      🔍 Cache MISS ❌ for prompt='What is the current status...'
   [Phase 3: PASS] Cache correctly returned MISS for invalidated entry ✓

══════════════════════════════════════════════════════════════════════
  GLIA CLOSED-LOOP VALIDATION REPORT
══════════════════════════════════════════════════════════════════════

  Scenario: Order Status Invalidation  [PASS]
    ✓  Phase 1 — Baseline (cache hit after store)        (12 ms)
    ✓  Phase 2 — Update detected & invalidated           (5213 ms)
    ✓  Phase 3 — Cache miss after invalidation           (8 ms)

  Phases passed : 6/6
  Overall result: ALL TESTS PASSED ✅
```

---

## Running Actors Individually

### Actor A — Data Simulator only

Update document 1 three times with 8-second gaps:

```bash
python actors/actor_a_simulator.py --order-id 1 --updates 3 --delay 8
```

### Actor B — Glia Application only (runs forever)

```bash
python actors/actor_b_application.py
```

### Actor C — Full validation only (requires Actor B to be running)

```bash
python actors/actor_c_verifier.py
```

---

## Project Structure

```
glia-validation/
├── docker-compose.yml          # Redis Stack + PostgreSQL
├── requirements.txt
├── config.py                   # Shared settings (URLs, thresholds, etc.)
├── mock_vectorizer.py          # Deterministic embedding provider (no GPU needed)
├── run_validation.py           # Main orchestration entry point
│
├── actors/
│   ├── __init__.py
│   ├── actor_a_simulator.py    # Writes UPDATEs to PostgreSQL
│   ├── actor_b_application.py  # GliaManager + CacheWatcher application
│   └── actor_c_verifier.py     # Orchestrates and asserts state transitions
│
└── mock_db/
    └── init.sql                # Creates tables, trigger, and seed data
```

---

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379` | Redis Stack connection URL |
| `PG_DSN` | `host=localhost ...` | PostgreSQL connection string |
| `CACHE_INDEX_NAME` | `glia_validation_cache` | RediSearch index name |
| `DISTANCE_THRESHOLD` | `0.15` | Cosine distance for cache hits |
| `POLL_INTERVAL_SECONDS` | `5` | How often RelationalDBAdapter polls |
| `VECTOR_DIMS` | `64` | Embedding dimensions (MockVectorizer) |
| `PHASE_DELAY_SECONDS` | `2` | Pause between validation phases |
| `INVALIDATION_TIMEOUT_SECONDS` | `30` | Max wait for invalidation event |

---

## How the Mock Works

**No real LLM or embedding model is required.** The `MockVectorizer` derives
deterministic vectors from character-level hashing, producing consistent
outputs so that `store()` followed by `check()` always results in a cache
hit. The test harness proves the *plumbing* (watcher → invalidator → Redis)
rather than the embedding quality.

**No real CDC driver is required.** The adapter runs in `polling` mode,
querying the `document_changelog` table (populated by a PostgreSQL trigger)
every `POLL_INTERVAL_SECONDS` seconds. This exercises the full
`PollingRunner → RelationalDBAdapter → CacheInvalidator` chain without
needing a logical replication slot.

---

## Resetting Between Runs

```bash
# Clear the Redis cache index:
docker exec glia_redis redis-cli FT.DROPINDEX glia_validation_cache DD

# Reset PostgreSQL to seed data:
docker exec -it glia_postgres psql -U glia -d gliadb \
  -c "DELETE FROM document_changelog; UPDATE documents SET status='active', updated_at=NOW();"
```

Or tear down and recreate everything:

```bash
docker compose down -v && docker compose up -d
```
