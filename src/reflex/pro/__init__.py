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
from reflex.pro.post_swap_monitor import (
    MonitorConfig,
    PostSwapMonitor,
    TripDecision,
)
from reflex.pro.rollback import (
    RollbackHandler,
    RollbackOutcome,
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

__all__ = [
    "ConsentMismatch",
    "ConsentReceipt",
    "ConsentRequired",
    "DistillScheduler",
    "EvalGate",
    "EvalReport",
    "EvalSample",
    "GateResult",
    "GateThresholds",
    "HardwareFingerprintLite",
    "InsufficientEpisodes",
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
    "RollbackHandler",
    "RollbackOutcome",
    "SchedulerConfig",
    "SchedulerState",
    "TripDecision",
    "issue_dev_license",
    "load_license",
]
