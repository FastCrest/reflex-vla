"""Pro-tier telemetry heartbeat — opt-out, license-bound, anonymized.

Sends a small JSON heartbeat to the FastCrest telemetry endpoint once per
24h while a Pro license is loaded and active. The aggregate data lets us
estimate the gap between active Pro deployments and paying licenses
(bypass-population detection at scale, per the OSS leak monitoring
runbook in reflex_context/04_product/oss_leak_monitoring.md).

Privacy posture (locked Phase 1)
--------------------------------
What gets sent (5 fields total):

- ``license_id``   — the JWT ``customer_id`` field. Lets us match against
                     the paying-customer DB.
- ``org_hash``     — ``SHA256(customer_id)[:16]``. Anonymized identity for
                     unique-deployment counting without identification.
- ``workload``     — ``{vla_family, hardware_tier}``. Aggregated stats only
                     (e.g., "20% of Pro deployments serve pi0.5 on A100").
- ``reflex_version`` — ``__version__``. Lets us see version-distribution.
- ``timestamp``    — ISO-8601 UTC. Recency / heartbeat staleness only.

What does NOT get sent:

- /act payloads (images, instructions, state)
- Robot trajectories or actions
- Model weights or embeddings
- Customer org name (only the SHA256 tag)
- IP addresses (Cloudflare Worker logs Cf-Connecting-IP separately; we
  do not write it to telemetry storage)

Opt-out
-------
Default behavior on a Pro license: telemetry ON.

Customers can disable via:

- ``REFLEX_NO_TELEMETRY=1`` environment variable (one-shot)
- ``--no-telemetry`` CLI flag (Phase 1.5 wiring)
- License-level opt-out (Phase 2 — per-customer flag in the JWT)

Free-tier deployments (no Pro license) NEVER call telemetry. The
emit() function is a no-op without a valid license.

Endpoint
--------
``https://telemetry.fastcrest.workers.dev/v1/heartbeat`` (Cloudflare
Worker; backend stub at ``infra/telemetry-worker/``). Override via
``REFLEX_TELEMETRY_ENDPOINT`` env var for testing or air-gapped
deployments.

Failure mode
------------
Network unreachable / endpoint down → fail silently. Never blocks the
license heartbeat path. Telemetry failure logs at DEBUG level only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Override via REFLEX_TELEMETRY_ENDPOINT for testing / air-gapped deploys.
DEFAULT_TELEMETRY_ENDPOINT = "https://telemetry.fastcrest.workers.dev/v1/heartbeat"

# Two-second timeout. Telemetry is best-effort; we never block startup
# on a slow telemetry endpoint.
_REQUEST_TIMEOUT_S = 2.0

# Schema version. Bumped on a breaking field-shape change. Worker stub
# accepts v1 only and rejects future versions with HTTP 400 so old
# clients see a clean failure rather than silently wrong data.
HEARTBEAT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HeartbeatPayload:
    """The exact payload sent to the telemetry endpoint. Locked Phase 1.

    See module docstring for the privacy posture and field-by-field
    explanations.
    """

    schema_version: int
    license_id: str
    org_hash: str
    workload: dict[str, str]
    reflex_version: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_disabled() -> bool:
    """True iff the user has opted out of telemetry."""
    return os.environ.get("REFLEX_NO_TELEMETRY", "").lower() in ("1", "true", "yes", "on")


def _org_hash(customer_id: str) -> str:
    """SHA256[:16] of customer_id. Anonymized counting tag."""
    return hashlib.sha256(customer_id.encode("utf-8")).hexdigest()[:16]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_payload(
    *,
    customer_id: str,
    vla_family: str,
    hardware_tier: str,
    reflex_version: str,
) -> HeartbeatPayload:
    """Build (but don't send) a heartbeat payload.

    Separated from emit() so tests can inspect the exact payload without
    triggering an HTTP call.
    """
    return HeartbeatPayload(
        schema_version=HEARTBEAT_SCHEMA_VERSION,
        license_id=customer_id,
        org_hash=_org_hash(customer_id),
        workload={"vla_family": vla_family, "hardware_tier": hardware_tier},
        reflex_version=reflex_version,
        timestamp=_utc_now_iso(),
    )


def emit(
    *,
    customer_id: str,
    vla_family: str = "unknown",
    hardware_tier: str = "unknown",
    reflex_version: str = "unknown",
    endpoint: str | None = None,
) -> bool:
    """Send a heartbeat to the telemetry endpoint.

    Returns True if the heartbeat was POSTed and the endpoint returned 2xx,
    False on opt-out / network failure / non-2xx. Never raises.

    Caller is responsible for invoking this no more than once per 24h
    (the license heartbeat path in ``pro.license.load_license`` is the
    natural integration point).
    """
    if _is_disabled():
        logger.debug("Telemetry disabled via REFLEX_NO_TELEMETRY env; skipping.")
        return False
    if not customer_id:
        logger.debug("Telemetry skipped: no customer_id (free tier).")
        return False

    payload = build_payload(
        customer_id=customer_id,
        vla_family=vla_family,
        hardware_tier=hardware_tier,
        reflex_version=reflex_version,
    )
    url = endpoint or os.environ.get("REFLEX_TELEMETRY_ENDPOINT", DEFAULT_TELEMETRY_ENDPOINT)

    # Lazy httpx import — Reflex's [serve] extra includes httpx; bare
    # `pip install reflex-vla` includes it as a base dep, so this is safe.
    try:
        import httpx
    except ImportError:
        logger.debug("Telemetry skipped: httpx not available.")
        return False

    try:
        resp = httpx.post(
            url,
            json=payload.to_dict(),
            timeout=_REQUEST_TIMEOUT_S,
            headers={"User-Agent": f"reflex-vla/{reflex_version}"},
        )
        if 200 <= resp.status_code < 300:
            logger.debug("Telemetry heartbeat posted: %s", payload.org_hash)
            return True
        logger.debug(
            "Telemetry endpoint returned %d for %s", resp.status_code, payload.org_hash
        )
        return False
    except Exception as exc:  # noqa: BLE001 — telemetry is best-effort
        logger.debug("Telemetry POST failed: %s", exc)
        return False


__all__ = [
    "DEFAULT_TELEMETRY_ENDPOINT",
    "HEARTBEAT_SCHEMA_VERSION",
    "HeartbeatPayload",
    "build_payload",
    "emit",
]
