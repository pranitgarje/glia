"""
config/settings.py
──────────────────
Central configuration for the Glia Symmetric Comparative Benchmark.
All DB credentials, tuning knobs, and benchmark parameters live here.
"""
from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure endpoints
# ─────────────────────────────────────────────────────────────────────────────

REDIS_URL       = "redis://localhost:6379"

POSTGRES_DSN    = "host=localhost port=5432 dbname=benchmark user=glia password=gliapass"
POSTGRES_TABLE  = "documents"
POSTGRES_CHANGELOG_TABLE = "doc_changelog"
POSTGRES_PK_FIELD       = "id"
POSTGRES_WATERMARK_COL  = "updated_at"

NEO4J_URI       = "bolt://localhost:7687"
NEO4J_USER      = "neo4j"
NEO4J_PASSWORD  = "gliapass123"
NEO4J_DATABASE  = "neo4j"
NEO4J_NODE_LABEL = "Document"
NEO4J_PK_FIELD  = "doc_id"

QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333
QDRANT_COLLECTION = "benchmark_docs"
QDRANT_PK_FIELD   = "doc_id"
QDRANT_TIMESTAMP_FIELD = "updated_at"

# ─────────────────────────────────────────────────────────────────────────────
# Glia / Redis cache settings
# ─────────────────────────────────────────────────────────────────────────────

GLIA_INDEX_NAME         = "glia_benchmark_cache"
GLIA_DISTANCE_THRESHOLD = 0.18   # tighter threshold for benchmark precision
GLIA_VECTOR_DIMS        = 384    # matches sentence-transformers all-MiniLM-L6-v2
GLIA_TTL_SECONDS        = 3600   # 1-hour entry TTL as safety net

# ─────────────────────────────────────────────────────────────────────────────
# Seeding parameters
# ─────────────────────────────────────────────────────────────────────────────

SEED_RECORD_COUNT       = 10_000   # records injected into each DB
SEED_BATCH_SIZE         = 500      # records per batch insert
PRIME_QUERY_COUNT       = 500      # cache-warming queries

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark run parameters
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_QUERY_COUNT   = 1_000   # total queries per track per DB type
BENCHMARK_WORKERS       = 4       # concurrent query workers

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic volatility simulator
# ─────────────────────────────────────────────────────────────────────────────

SIMULATOR_UPDATE_FRACTION   = 0.07     # 7% of records mutated per cycle
SIMULATOR_CYCLE_SECONDS     = 5.0      # seconds between mutation waves
SIMULATOR_MAX_CYCLES        = 120      # stop after N cycles (~10 min)

# ─────────────────────────────────────────────────────────────────────────────
# Adapter polling / CDC settings
# ─────────────────────────────────────────────────────────────────────────────

POLL_INTERVAL_SECONDS   = 3.0     # short interval for benchmark visibility
RECONNECT_RETRIES       = 3
RECONNECT_DELAY         = 2.0

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_PORT          = 5050
DASHBOARD_REDIS_CHANNEL = "glia:benchmark:events"

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic document categories (used by seeder)
# ─────────────────────────────────────────────────────────────────────────────

DOCUMENT_CATEGORIES = [
    "finance", "medicine", "legal", "engineering",
    "research", "policy", "product", "security",
]

# Query templates per category — used to generate realistic semantic queries
QUERY_TEMPLATES = {
    "finance":     ["What is the revenue breakdown for {title}?",
                    "Summarise the financial risk factors in {title}.",
                    "What are the key metrics discussed in {title}?"],
    "medicine":    ["What treatment protocols are described in {title}?",
                    "What are the contraindications mentioned in {title}?",
                    "Summarise the clinical findings of {title}."],
    "legal":       ["What are the liability clauses in {title}?",
                    "What jurisdiction governs {title}?",
                    "List the key obligations under {title}."],
    "engineering": ["What are the technical specifications in {title}?",
                    "Describe the architecture outlined in {title}.",
                    "What failure modes are documented in {title}?"],
    "research":    ["What methodology was used in {title}?",
                    "What are the conclusions of {title}?",
                    "What future work is proposed in {title}?"],
    "policy":      ["What are the compliance requirements in {title}?",
                    "Who does {title} apply to?",
                    "What penalties are described in {title}?"],
    "product":     ["What features are listed in {title}?",
                    "What is the target audience of {title}?",
                    "What pricing model is described in {title}?"],
    "security":    ["What vulnerabilities are covered in {title}?",
                    "What mitigations are proposed in {title}?",
                    "What is the threat model in {title}?"],
}
