"""
config.py
─────────
Shared configuration for the Glia Closed-Loop Validation System.

Edit the values here to match your environment; all three actors
import from this module so there is a single source of truth.
"""

# ── Redis ─────────────────────────────────────────────────────────────────
REDIS_URL = "redis://localhost:6379"
CACHE_INDEX_NAME = "glia_validation_cache"
DISTANCE_THRESHOLD = 0.15   # cosine distance — lower = stricter matching

# ── PostgreSQL ────────────────────────────────────────────────────────────
PG_DSN = "host=localhost port=5432 dbname=gliadb user=glia password=gliapassword"

# ── Polling ───────────────────────────────────────────────────────────────
# How often (seconds) RelationalDBAdapter polls the changelog table.
# Kept short for demo purposes; production values are typically 30–300 s.
POLL_INTERVAL_SECONDS = 5

# ── Embedding ─────────────────────────────────────────────────────────────
# Dimensionality must match MockVectorizer.embed() output below.
VECTOR_DIMS = 64

# ── Simulation timing ─────────────────────────────────────────────────────
# Seconds between the "baseline hit" phase and the "data update" phase.
PHASE_DELAY_SECONDS = 2
# Maximum seconds to wait for the invalidation_complete event after an update.
INVALIDATION_TIMEOUT_SECONDS = 30
