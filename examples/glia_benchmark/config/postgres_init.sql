-- ─────────────────────────────────────────────────────────────────────────────
-- Glia Benchmark — PostgreSQL Schema
-- ─────────────────────────────────────────────────────────────────────────────

-- ── Documents table (monitored by RelationalDBAdapter) ───────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    title       TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    category    TEXT        NOT NULL DEFAULT 'general',
    version     INTEGER     NOT NULL DEFAULT 1,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Changelog table (polling strategy B) ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS doc_changelog (
    id          SERIAL      PRIMARY KEY,
    table_name  TEXT        NOT NULL,
    record_pk   TEXT        NOT NULL,
    operation   TEXT        NOT NULL,   -- INSERT | UPDATE | DELETE
    changed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Trigger: auto-populate changelog on every DML ────────────────────────────
CREATE OR REPLACE FUNCTION trg_doc_changelog()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO doc_changelog (table_name, record_pk, operation, changed_at)
    VALUES (
        TG_TABLE_NAME,
        COALESCE(NEW.id, OLD.id),
        TG_OP,
        NOW()
    );
    RETURN COALESCE(NEW, OLD);
END;
$$;

DROP TRIGGER IF EXISTS trg_documents_changelog ON documents;
CREATE TRIGGER trg_documents_changelog
AFTER INSERT OR UPDATE OR DELETE ON documents
FOR EACH ROW EXECUTE FUNCTION trg_doc_changelog();

-- ── Benchmark telemetry table ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS benchmark_telemetry (
    id              SERIAL      PRIMARY KEY,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    track           TEXT        NOT NULL,   -- 'glia' | 'standard'
    db_type         TEXT        NOT NULL,   -- 'relational' | 'graph' | 'vector'
    query_text      TEXT        NOT NULL,
    latency_ms      FLOAT       NOT NULL,
    cache_result    TEXT,                   -- 'hit' | 'miss' | NULL
    tokens_used     INTEGER     DEFAULT 0,
    source_id       TEXT
);

-- Index for fast dashboard aggregations
CREATE INDEX IF NOT EXISTS idx_telemetry_track_db
    ON benchmark_telemetry (track, db_type, recorded_at DESC);

-- ── Seed with a small warm-up set (full seed done by Python seeder) ──────────
INSERT INTO documents (id, title, content, category) VALUES
  ('doc-seed-001', 'Seed Record Alpha',
   'This is a baseline seed document for benchmark initialisation.',
   'seed'),
  ('doc-seed-002', 'Seed Record Beta',
   'Another baseline document used to verify the changelog trigger is active.',
   'seed')
ON CONFLICT (id) DO NOTHING;
