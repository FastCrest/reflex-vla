-- D1 schema for Reflex Pro telemetry heartbeats.
-- Apply with: wrangler d1 execute reflex-telemetry --file=schema.sql

CREATE TABLE IF NOT EXISTS heartbeats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    license_id      TEXT NOT NULL,
    org_hash        TEXT NOT NULL,
    vla_family      TEXT NOT NULL,
    hardware_tier   TEXT NOT NULL,
    reflex_version  TEXT NOT NULL,
    client_timestamp TEXT NOT NULL,
    server_timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Common query indexes:
-- 1. "How many unique deployments today?" (count distinct org_hash recently)
CREATE INDEX IF NOT EXISTS heartbeats_server_ts ON heartbeats (server_timestamp);
-- 2. "Heartbeats for license X over time"
CREATE INDEX IF NOT EXISTS heartbeats_license ON heartbeats (license_id);
-- 3. "Active deployment count by org"
CREATE INDEX IF NOT EXISTS heartbeats_org ON heartbeats (org_hash, server_timestamp);
