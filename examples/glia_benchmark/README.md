# Glia — Symmetric Comparative Benchmark

A production-grade framework for benchmarking **Glia's cache invalidation pipeline** against standard RAG retrieval across three database paradigms simultaneously.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  SYMMETRIC COMPARATIVE FRAMEWORK                │
│                                                                 │
│  ┌──────────────────────┐      ┌──────────────────────────────┐ │
│  │  TRACK A — STANDARD  │      │    TRACK B — GLIA-ENHANCED   │ │
│  │                      │      │                              │ │
│  │  Query → DB lookup   │      │  Query → GliaManager.check() │ │
│  │       → LLM call     │      │     HIT  → return cached     │ │
│  │       → response     │      │     MISS → DB + LLM + store  │ │
│  └──────────────────────┘      └──────────────────────────────┘ │
│           │                               │                     │
│           └──────────── TELEMETRY ────────┘                     │
│                            │                                    │
│              ┌─────────────▼────────────┐                       │
│              │    BenchmarkPacket       │                       │
│              │  track / db_type /       │                       │
│              │  latency_ms / tokens /   │                       │
│              │  cache_result            │                       │
│              └─────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   HETEROGENEOUS DATA TIER                       │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │  PostgreSQL  │  │    Neo4j     │  │       Qdrant          │ │
│  │  (Relational)│  │   (Graph)    │  │      (Vector)         │ │
│  │              │  │              │  │                       │ │
│  │ RelationalDB │  │  GraphDB     │  │  VectorDB             │ │
│  │ Adapter      │  │  Adapter     │  │  Adapter              │ │
│  │ (changelog   │  │  (Cypher     │  │  (metadata poll)      │ │
│  │  polling)    │  │   polling)   │  │                       │ │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬────────────┘ │
│         └─────────────────┼──────────────────────┘             │
│                           ▼                                     │
│              ┌────────────────────────┐                         │
│              │     CacheWatcher       │                         │
│              │   (background thread)  │                         │
│              │                        │                         │
│              │  PollingRunner × 3     │                         │
│              └────────────┬───────────┘                         │
│                           │                                     │
│              ┌────────────▼───────────┐                         │
│              │   CacheInvalidator     │                         │
│              │  delete_by_tag(id)     │                         │
│              └────────────┬───────────┘                         │
│                           │                                     │
│              ┌────────────▼───────────┐                         │
│              │   Redis Stack          │                         │
│              │   (Glia cache +        │                         │
│              │    RediSearch index)   │                         │
│              └────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│               VOLATILITY SIMULATOR (Actor A)                    │
│                                                                 │
│  Every 5s:  randomly mutates 7% of seeded records               │
│             across PG + Neo4j + Qdrant                          │
│             records exact mutation timestamp per source_id      │
│             → enables invalidation latency measurement          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Start all databases

```bash
cd glia_benchmark
docker compose up -d

# Verify all services are healthy:
docker compose ps
```

Services started:
| Service | Port | UI |
|---|---|---|
| Redis Stack | 6379 | RedisInsight: http://localhost:8001 |
| PostgreSQL | 5432 | — |
| Neo4j | 7474 / 7687 | Browser: http://localhost:7474 |
| Qdrant | 6333 | Dashboard: http://localhost:6333/dashboard |
| Dashboard | 5050 | http://localhost:5050 |

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full benchmark

```bash
# Full benchmark (10k seed + 500 cache prime + 1000 queries × 2 tracks × 3 DBs)
python run_benchmark.py

# Faster smoke test
python run_benchmark.py --queries 100

# Skip seeding (use existing DB data)
python run_benchmark.py --skip-seed --queries 500

# Custom output path
python run_benchmark.py --output results/my_run.json
```

### 4. View live dashboard

Open **http://localhost:5050** in your browser while the benchmark runs.

---

## Benchmark Parameters

All knobs are in `config/settings.py`:

| Parameter | Default | Description |
|---|---|---|
| `SEED_RECORD_COUNT` | 10,000 | Records injected per DB |
| `PRIME_QUERY_COUNT` | 500 | Cache warm-up queries |
| `BENCHMARK_QUERY_COUNT` | 1,000 | Queries per track per DB |
| `SIMULATOR_UPDATE_FRACTION` | 7% | Records mutated per cycle |
| `SIMULATOR_CYCLE_SECONDS` | 5.0 | Mutation interval |
| `GLIA_DISTANCE_THRESHOLD` | 0.18 | Semantic cache threshold |
| `POLL_INTERVAL_SECONDS` | 3.0 | Adapter poll frequency |

---

## Metrics Measured

| Metric | Description |
|---|---|
| **End-to-End Latency** | Time from query to response (ms), per track and DB type |
| **Token Consumption** | Total LLM input+output tokens; Glia hits use 0 tokens |
| **Cache Hit/Miss Rate** | % of queries served from cache vs. full retrieval |
| **Invalidation Latency** | ms from DB `UPDATE` → `invalidation_complete` event |
| **Staleness Detection** | Whether Glia detects mutations before serving cached answers |

---

## File Structure

```
glia_benchmark/
├── docker-compose.yml          # All infrastructure (Redis, PG, Neo4j, Qdrant)
├── run_benchmark.py            # Main entry point
├── requirements.txt
│
├── config/
│   ├── settings.py             # All tuning parameters
│   └── postgres_init.sql       # Schema + changelog trigger
│
├── runner/
│   ├── benchmark.py            # Dual-track execution engine
│   ├── seeder.py               # 10k-record seed + cache prime
│   └── vectorizer.py           # Local sentence-transformers embedder
│
├── simulator/
│   └── volatility.py           # Background data mutation simulator
│
└── dashboard/
    ├── app.py                  # Flask SSE server + live HTML dashboard
    ├── Dockerfile
    └── requirements.txt
```

---

## Extending the Benchmark

**Add a new DB type**: implement a retriever class (see `RelationalRetriever` pattern) and a Glia adapter, then add it to `BenchmarkRunner.run()`.

**Use a real LLM**: replace `_simulated_llm_call()` in `runner/benchmark.py` with your OpenAI/Anthropic/Cohere client call. Token counting will be automatic.

**Use real embeddings**: replace `BenchmarkVectorizer` with your production embedding provider. It must expose `embed(text) -> List[float]` and `embed_many(texts) -> List[List[float]]`.

**Enable CDC mode**: change `mode="polling"` to `mode="cdc"` in the adapter builder functions in `runner/benchmark.py`. Requires a PostgreSQL logical replication slot or Neo4j 5.x Enterprise.
