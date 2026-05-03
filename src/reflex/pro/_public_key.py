"""Bundled Ed25519 public key for offline license verification.

After running POST /admin/init on the deployed license worker, paste the
``public_key_b64`` field from the response into ``BUNDLED_PUBLIC_KEY_B64``
below, then commit + release a new package version. Customers verify
license signatures against this key on every load (offline; no network
call required for signature verification — only the heartbeat needs the
network, and only daily).

Phase 2 will support multiple trusted keys (current + previous N) for key
rotation. Today we ship a single bundled key.

The PUBLIC key in this file is intentional and safe to publish — that's
what public keys are for. The PRIVATE key lives only in the Cloudflare
Worker's PRIVATE_KEY secret and never appears in this codebase.
"""
from __future__ import annotations

# Replace this placeholder with the public_key_b64 from POST /admin/init.
# Until you do, license verification will fail loudly with a helpful error
# pointing operators at the deploy steps.
BUNDLED_PUBLIC_KEY_B64 = ""

# Key ID of the bundled public key. Returned by POST /admin/init alongside
# the public key. Used to verify the signature was made with a key the
# client knows about.
BUNDLED_KEY_ID = ""
