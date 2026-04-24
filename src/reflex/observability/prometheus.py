"""Prometheus metrics for `reflex serve`.

12 metrics scoped to the cardinality budget — every label key is a
bounded enum (embodiment, model_id, cache_type, violation_kind,
slo_kind, fallback_target). Free-form per-request labels (instruction,
request_id, user_id, timestamp) are FORBIDDEN — they explode cardinality.

Cardinality budget: 3 embodiments × 6 models × ~5 sub-labels max ≈ 90
series. Within Prometheus single-instance comfort zone (target < 10K).

Uses a dedicated CollectorRegistry (not the global default) for test
isolation + clean cross-process export.

Spec: features/01_serve/subfeatures/_ecosystem/prometheus-grafana.md
Plan: features/01_serve/subfeatures/_ecosystem/prometheus-grafana_plan.md
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Dedicated registry — downstream features import this, NOT the global default.
REGISTRY = CollectorRegistry()

# Prometheus text-format media type (operators: serve /metrics with this header).
METRICS_CONTENT_TYPE = CONTENT_TYPE_LATEST  # "text/plain; version=0.0.4; charset=utf-8"


# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------

# /act request latency. Buckets span Jetson 5ms through cloud-A100 5s tail.
_LATENCY_BUCKETS = (
    0.005, 0.010, 0.020, 0.050, 0.100,
    0.200, 0.500, 1.0, 2.0, 5.0,
)
reflex_act_latency_seconds = Histogram(
    "reflex_act_latency_seconds",
    "End-to-end /act handler wall-clock latency",
    labelnames=("embodiment", "model_id"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

# ONNX session load time (cold start). Buckets span small CPU loads through
# 60s monolithic-pi0 builds.
_LOAD_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
reflex_onnx_load_time_seconds = Histogram(
    "reflex_onnx_load_time_seconds",
    "ONNX session creation + warmup wall-clock",
    labelnames=("model_id",),
    buckets=_LOAD_BUCKETS,
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

reflex_cache_hit_total = Counter(
    "reflex_cache_hit_total",
    "Cache hits, partitioned by cache type",
    labelnames=("embodiment", "cache_type"),  # action_chunk | vlm_prefix
    registry=REGISTRY,
)

reflex_cache_miss_total = Counter(
    "reflex_cache_miss_total",
    "Cache misses, partitioned by cache type",
    labelnames=("embodiment", "cache_type"),
    registry=REGISTRY,
)

reflex_denoise_steps_total = Counter(
    "reflex_denoise_steps_total",
    "Total denoise iterations executed (sum across all /act calls)",
    labelnames=("embodiment",),
    registry=REGISTRY,
)

reflex_safety_violations_total = Counter(
    "reflex_safety_violations_total",
    "Safety/guard violations partitioned by kind",
    labelnames=("embodiment", "violation_kind"),  # nan | velocity_clamp | torque_clamp | workspace_breach
    registry=REGISTRY,
)

reflex_slo_violations_total = Counter(
    "reflex_slo_violations_total",
    "SLO threshold violations (per-call, observed at /act)",
    labelnames=("embodiment", "slo_kind"),  # p95_latency | p99_latency
    registry=REGISTRY,
)

reflex_fallback_invocations_total = Counter(
    "reflex_fallback_invocations_total",
    "Fallback path invocations (deadline miss, error recovery)",
    labelnames=("embodiment", "fallback_target"),  # previous_chunk | hold_position | abort
    registry=REGISTRY,
)

reflex_model_swaps_total = Counter(
    "reflex_model_swaps_total",
    "Hot-swap events (recorded at swap-complete)",
    labelnames=("embodiment", "from_model", "to_model"),
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------

reflex_in_flight_requests = Gauge(
    "reflex_in_flight_requests",
    "/act requests currently being processed",
    labelnames=("embodiment",),
    registry=REGISTRY,
)

reflex_episodes_active = Gauge(
    "reflex_episodes_active",
    "Distinct episode_ids seen in the last rolling window",
    labelnames=("embodiment",),
    registry=REGISTRY,
)

reflex_server_up = Gauge(
    "reflex_server_up",
    "Server liveness signal — 1 when serving /metrics, 0 on shutdown",
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers — typed call-sites keep the surface searchable
# ---------------------------------------------------------------------------


def record_act_latency(seconds: float, embodiment: str, model_id: str) -> None:
    reflex_act_latency_seconds.labels(
        embodiment=embodiment, model_id=model_id
    ).observe(seconds)


def observe_onnx_load_time(seconds: float, model_id: str) -> None:
    reflex_onnx_load_time_seconds.labels(model_id=model_id).observe(seconds)


def inc_cache_hit(embodiment: str, cache_type: str) -> None:
    reflex_cache_hit_total.labels(
        embodiment=embodiment, cache_type=cache_type
    ).inc()


def inc_cache_miss(embodiment: str, cache_type: str) -> None:
    reflex_cache_miss_total.labels(
        embodiment=embodiment, cache_type=cache_type
    ).inc()


def inc_denoise_steps(embodiment: str, n_steps: int = 1) -> None:
    reflex_denoise_steps_total.labels(embodiment=embodiment).inc(n_steps)


def inc_safety_violation(embodiment: str, kind: str) -> None:
    reflex_safety_violations_total.labels(
        embodiment=embodiment, violation_kind=kind
    ).inc()


def inc_slo_violation(embodiment: str, kind: str) -> None:
    reflex_slo_violations_total.labels(
        embodiment=embodiment, slo_kind=kind
    ).inc()


def inc_fallback_invocation(embodiment: str, target: str) -> None:
    reflex_fallback_invocations_total.labels(
        embodiment=embodiment, fallback_target=target
    ).inc()


def inc_model_swap(embodiment: str, from_model: str, to_model: str) -> None:
    reflex_model_swaps_total.labels(
        embodiment=embodiment, from_model=from_model, to_model=to_model
    ).inc()


def set_server_up(value: int) -> None:
    """1 when serving, 0 on graceful shutdown."""
    reflex_server_up.set(value)


def set_episodes_active(embodiment: str, value: int) -> None:
    reflex_episodes_active.labels(embodiment=embodiment).set(value)


@contextmanager
def track_in_flight(embodiment: str) -> Iterator[None]:
    """Context manager increments/decrements in-flight gauge for safe
    try/finally semantics. Use:

        with track_in_flight(embodiment="franka"):
            result = await predict(...)
    """
    reflex_in_flight_requests.labels(embodiment=embodiment).inc()
    try:
        yield
    finally:
        reflex_in_flight_requests.labels(embodiment=embodiment).dec()


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render_metrics() -> bytes:
    """Generate Prometheus text-format payload for the /metrics endpoint."""
    return generate_latest(REGISTRY)
