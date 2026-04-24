"""Reflex serve observability — Prometheus /metrics + helpers.

Public API re-exported here. Downstream features (webhooks, SLO
enforcement, all-metrics dashboards) should import from this module,
NOT define their own metrics — that breaks cardinality budgets +
splits the registry.

See features/01_serve/subfeatures/_ecosystem/prometheus-grafana.md
for the canonical spec.
"""
from __future__ import annotations

from reflex.observability.prometheus import (
    REGISTRY,
    METRICS_CONTENT_TYPE,
    inc_cache_hit,
    inc_cache_miss,
    inc_denoise_steps,
    inc_fallback_invocation,
    inc_model_swap,
    inc_safety_violation,
    inc_slo_violation,
    observe_onnx_load_time,
    record_act_latency,
    render_metrics,
    set_episodes_active,
    set_server_up,
    track_in_flight,
)

__all__ = [
    "REGISTRY",
    "METRICS_CONTENT_TYPE",
    "render_metrics",
    "record_act_latency",
    "observe_onnx_load_time",
    "inc_cache_hit",
    "inc_cache_miss",
    "inc_denoise_steps",
    "inc_safety_violation",
    "inc_slo_violation",
    "inc_fallback_invocation",
    "inc_model_swap",
    "track_in_flight",
    "set_server_up",
    "set_episodes_active",
]
