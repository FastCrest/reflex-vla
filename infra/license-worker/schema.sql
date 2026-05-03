-- D1 schema for Reflex Pro license server.
-- Apply with: wrangler d1 execute reflex-licenses --file=schema.sql

-- Master Ed25519 keypair tracking. Private key never stored here — that lives
-- in Cloudflare Secrets (PRIVATE_KEY env binding). Public key is bundled into
-- the reflex package + served via GET /v1/pubkey for online verification.
CREATE TABLE IF NOT EXISTS master_keys (
    key_id          TEXT PRIMARY KEY,
    public_key_b64  TEXT NOT NULL,
    generated_at    TEXT NOT NULL,
    retired_at      TEXT
);

-- All issued licenses. revoked_at is a denormalized cache of revocation_list
-- for fast list queries; the source of truth for revocation checks is the
-- revocation_list table.
CREATE TABLE IF NOT EXISTS licenses (
    license_id      TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL,
    tier            TEXT NOT NULL,
    issued_at       TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    max_seats       INTEGER NOT NULL DEFAULT 1,
    signature       TEXT NOT NULL,
    key_id          TEXT NOT NULL,
    notes           TEXT,
    license_json    TEXT NOT NULL,
    revoked_at      TEXT,
    FOREIGN KEY (key_id) REFERENCES master_keys(key_id)
);

CREATE INDEX IF NOT EXISTS licenses_customer ON licenses (customer_id);
CREATE INDEX IF NOT EXISTS licenses_issued ON licenses (issued_at);

-- One-time activation codes. Customer redeems via GET /v1/activation/<code>.
-- 24h TTL, single-use.
CREATE TABLE IF NOT EXISTS activation_codes (
    code            TEXT PRIMARY KEY,
    license_id      TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    used            INTEGER NOT NULL DEFAULT 0,
    used_at         TEXT,
    FOREIGN KEY (license_id) REFERENCES licenses(license_id)
);

CREATE INDEX IF NOT EXISTS activation_codes_license ON activation_codes (license_id);

-- Revocation list. Heartbeat checks this on every call; revocation propagates
-- to a deployed customer within their next heartbeat (≤ 24h).
CREATE TABLE IF NOT EXISTS revocation_list (
    license_id      TEXT PRIMARY KEY,
    revoked_at      TEXT NOT NULL,
    reason          TEXT NOT NULL DEFAULT 'unspecified',
    FOREIGN KEY (license_id) REFERENCES licenses(license_id)
);

-- Heartbeat log. Privacy posture: we do NOT store Cf-Connecting-IP, only the
-- country code (which Cloudflare gives us via Cf-IPCountry). hardware_fingerprint
-- is a customer-side hash; does not contain raw machine identifiers.
CREATE TABLE IF NOT EXISTS heartbeats (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    license_id              TEXT NOT NULL,
    hardware_fingerprint    TEXT NOT NULL,
    ip_country              TEXT NOT NULL DEFAULT '??',
    reflex_version          TEXT NOT NULL,
    server_timestamp        TEXT NOT NULL,
    FOREIGN KEY (license_id) REFERENCES licenses(license_id)
);

CREATE INDEX IF NOT EXISTS heartbeats_license_ts ON heartbeats (license_id, server_timestamp);
CREATE INDEX IF NOT EXISTS heartbeats_fp ON heartbeats (hardware_fingerprint);

-- Anti-abuse signals: sharing detected, geographic anomalies, velocity spikes.
-- One row per detection event; populated by the worker's detectSharing() and
-- offline analytics queries.
CREATE TABLE IF NOT EXISTS abuse_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    license_id      TEXT NOT NULL,
    signal_type     TEXT NOT NULL,
    details         TEXT NOT NULL,
    detected_at     TEXT NOT NULL,
    FOREIGN KEY (license_id) REFERENCES licenses(license_id)
);

CREATE INDEX IF NOT EXISTS abuse_signals_license ON abuse_signals (license_id, detected_at);

-- Override list: licenses marked as "known good" so detectSharing skips them
-- (e.g., university research lab with rotating student hardware, where 10
-- distinct fingerprints is normal).
CREATE TABLE IF NOT EXISTS override_list (
    license_id      TEXT PRIMARY KEY,
    marked_safe_at  TEXT NOT NULL,
    reason          TEXT NOT NULL,
    FOREIGN KEY (license_id) REFERENCES licenses(license_id)
);
