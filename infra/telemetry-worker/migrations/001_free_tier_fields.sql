-- Migration: add free-tier telemetry fields to heartbeats table.
-- Apply with: wrangler d1 execute reflex-telemetry --file=migrations/001_free_tier_fields.sql

ALTER TABLE heartbeats ADD COLUMN model_name TEXT DEFAULT 'unknown';
ALTER TABLE heartbeats ADD COLUMN hardware_detail TEXT DEFAULT 'unknown';
ALTER TABLE heartbeats ADD COLUMN latency_p50 REAL DEFAULT NULL;
ALTER TABLE heartbeats ADD COLUMN latency_p95 REAL DEFAULT NULL;
ALTER TABLE heartbeats ADD COLUMN latency_p99 REAL DEFAULT NULL;
ALTER TABLE heartbeats ADD COLUMN error_count_24h INTEGER DEFAULT 0;
ALTER TABLE heartbeats ADD COLUMN safety_violation_count_24h INTEGER DEFAULT 0;
ALTER TABLE heartbeats ADD COLUMN episode_count_24h INTEGER DEFAULT 0;
ALTER TABLE heartbeats ADD COLUMN action_dim INTEGER DEFAULT NULL;
ALTER TABLE heartbeats ADD COLUMN embodiment TEXT DEFAULT 'unknown';
ALTER TABLE heartbeats ADD COLUMN denoise_steps INTEGER DEFAULT NULL;
ALTER TABLE heartbeats ADD COLUMN inference_mode TEXT DEFAULT 'unknown';
ALTER TABLE heartbeats ADD COLUMN tier TEXT DEFAULT 'pro';

-- Index for tier-based queries (e.g., "how many free-tier deployments?")
CREATE INDEX IF NOT EXISTS heartbeats_tier ON heartbeats (tier, server_timestamp);
