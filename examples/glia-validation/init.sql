-- ── Schema ────────────────────────────────────────────────────────────────
-- documents: the primary source table that GliaManager caches responses for.
-- document_changelog: audit table populated by the trigger below.
--   RelationalDBAdapter uses this in changelog-strategy polling mode.

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    title       TEXT        NOT NULL,
    content     TEXT        NOT NULL,
    status      TEXT        NOT NULL DEFAULT 'active',
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS document_changelog (
    id          SERIAL PRIMARY KEY,
    table_name  TEXT        NOT NULL,
    record_pk   TEXT        NOT NULL,
    operation   TEXT        NOT NULL,   -- INSERT | UPDATE | DELETE
    changed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Trigger: write to changelog on every documents change ─────────────────
CREATE OR REPLACE FUNCTION log_document_change()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO document_changelog (table_name, record_pk, operation, changed_at)
        VALUES ('documents', OLD.id::TEXT, 'DELETE', NOW());
        RETURN OLD;
    ELSE
        NEW.updated_at = NOW();
        INSERT INTO document_changelog (table_name, record_pk, operation, changed_at)
        VALUES ('documents', NEW.id::TEXT, TG_OP, NOW());
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_document_changelog ON documents;
CREATE TRIGGER trg_document_changelog
AFTER INSERT OR UPDATE OR DELETE ON documents
FOR EACH ROW EXECUTE FUNCTION log_document_change();

-- ── Seed data ─────────────────────────────────────────────────────────────
INSERT INTO documents (title, content, status) VALUES
    ('Order 123', 'Status: Processing. Estimated delivery: 3 days.', 'active'),
    ('Order 456', 'Status: Shipped. Tracking: TRK-987654.',           'active'),
    ('Order 789', 'Status: Delivered. Received on 2024-01-10.',        'active');
