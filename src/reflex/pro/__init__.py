"""Reflex Pro — $99/mo continuous-learning loop.

Per ADR 2026-04-25-self-distilling-serve-architecture: 4-stage loop
(collect → distill → 9-gate eval → swap) with HW-bound JWT licensing
and customer-disk-only data residency.

Public surface (Phase 1):
- ProDataCollector (data_collection.py): bounded-queue parquet writer
- (Day 2+) ProConsent, ProLicense, DistillScheduler, EvalGate,
  PostSwapMonitor, RollbackHandler, HfHubClient, WeeklyReport,
  DriftDetector

Customer entry: `reflex serve --pro --collect-data` and the related
CLI flags (Day 4+ wiring).
"""
from __future__ import annotations

from reflex.pro.consent import (
    ConsentMismatch,
    ConsentReceipt,
    ConsentRequired,
    PIIOptions,
    ProConsent,
)
from reflex.pro.data_collection import ProDataCollector
from reflex.pro.distill_scheduler import (
    DistillScheduler,
    KickDecision,
    SchedulerConfig,
    SchedulerState,
)
from reflex.pro.eval_gate import (
    EvalGate,
    EvalReport,
    EvalSample,
    GateResult,
    GateThresholds,
    InsufficientEpisodes,
)
from reflex.pro.hf_hub import (
    HfHubAuthFailure,
    HfHubClient,
    HfHubDown,
    HfHubError,
    HfHubMissingToken,
    HfPullOutcome,
    HfPushOutcome,
    HfRepoSpec,
)
from reflex.pro.post_swap_monitor import (
    MonitorConfig,
    PostSwapMonitor,
    TripDecision,
)
from reflex.pro.rollback import (
    RollbackHandler,
    RollbackOutcome,
)
from reflex.pro.drift_detection import (
    DriftDetector,
    DriftReport,
    JointDriftScore,
)
from reflex.pro.weekly_report import (
    RollbackEntry,
    TaskDelta,
    WeeklyReport,
    render_cli,
    render_json,
    send_email,
    send_slack,
)
from reflex.pro.license import (
    HardwareFingerprintLite,
    LicenseCorrupt,
    LicenseError,
    LicenseExpired,
    LicenseHardwareMismatch,
    LicenseHeartbeatStale,
    LicenseMissing,
    ProLicense,
    issue_dev_license,
    load_license,
)
from reflex.pro.fingerprint import (
    Fingerprint,
    compute_fingerprint,
    verify_fingerprint,
)
from reflex.pro.telemetry import (
    HeartbeatPayload,
    build_payload as build_telemetry_payload,
    emit as emit_telemetry,
)
from reflex.pro.signature import (
    LicenseSignatureError,
    verify_license_signature,
)
from reflex.pro.activate import (
    ActivationCodeError,
    ActivationError,
    ActivationNetworkError,
    activate_license,
    heartbeat_fingerprint,
    probe_hardware_binding,
)
from reflex.pro.heartbeat import (
    LicenseExpiredAtServer,
    LicenseRevokedError,
    send_heartbeat,
)

__all__ = [
    "ConsentMismatch",
    "ConsentReceipt",
    "ConsentRequired",
    "DistillScheduler",
    "DriftDetector",
    "DriftReport",
    "EvalGate",
    "EvalReport",
    "EvalSample",
    "Fingerprint",
    "GateResult",
    "GateThresholds",
    "HardwareFingerprintLite",
    "HeartbeatPayload",
    "HfHubAuthFailure",
    "HfHubClient",
    "HfHubDown",
    "HfHubError",
    "HfHubMissingToken",
    "HfPullOutcome",
    "HfPushOutcome",
    "HfRepoSpec",
    "InsufficientEpisodes",
    "JointDriftScore",
    "KickDecision",
    "LicenseCorrupt",
    "LicenseError",
    "LicenseExpired",
    "LicenseHardwareMismatch",
    "LicenseHeartbeatStale",
    "LicenseMissing",
    "MonitorConfig",
    "PIIOptions",
    "PostSwapMonitor",
    "ProConsent",
    "ProDataCollector",
    "ProLicense",
    "RollbackEntry",
    "RollbackHandler",
    "RollbackOutcome",
    "SchedulerConfig",
    "SchedulerState",
    "TaskDelta",
    "TripDecision",
    "WeeklyReport",
    "ActivationCodeError",
    "ActivationError",
    "ActivationNetworkError",
    "LicenseExpiredAtServer",
    "LicenseRevokedError",
    "LicenseSignatureError",
    "activate_license",
    "build_telemetry_payload",
    "compute_fingerprint",
    "emit_telemetry",
    "heartbeat_fingerprint",
    "issue_dev_license",
    "load_license",
    "probe_hardware_binding",
    "render_cli",
    "render_json",
    "send_email",
    "send_heartbeat",
    "send_slack",
    "verify_fingerprint",
    "verify_license_signature",
]
