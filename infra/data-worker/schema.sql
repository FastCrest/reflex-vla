-- D1 schema for Reflex episode data upload tracking.
-- Apply with: wrangler d1 execute reflex-data --file=schema.sql

CREATE TABLE IF NOT EXISTS uploads (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id        TEXT NOT NULL,
    contributor_hash  TEXT NOT NULL,
    file_hash         TEXT NOT NULL,
    r2_key            TEXT NOT NULL,
    file_size         INTEGER NOT NULL DEFAULT 0,
    uploaded_at       TEXT NOT NULL DEFAULT (datetime('now')),
    processed         INTEGER NOT NULL DEFAULT 0,
    processed_at      TEXT DEFAULT NULL
);

-- Common query indexes:
-- 1. "Uploads by contributor" (stats page)
CREATE INDEX IF NOT EXISTS uploads_contributor ON uploads (contributor_hash, uploaded_at);
-- 2. "Unprocessed uploads" (batch processing pipeline)
CREATE INDEX IF NOT EXISTS uploads_unprocessed ON uploads (processed, uploaded_at);
-- 3. "Dedup by episode_id + contributor"
CREATE UNIQUE INDEX IF NOT EXISTS uploads_episode_contributor ON uploads (episode_id, contributor_hash);
