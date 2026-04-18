"""Reflex VLA inference server.

Persistent HTTP server that loads an exported VLA model and serves
action predictions. Camera image in, robot actions out.

Usage:
    reflex serve ./reflex_export/ --port 8000

Then from robot:
    POST http://localhost:8000/act
    {
        "image": "<base64 encoded image>",
        "instruction": "pick up the cup",
        "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    }
    → {"actions": [[...], [...], ...], "latency_ms": 253.1, "hz": 3.9}
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ReflexServer:
    """VLA inference server that loads exported models and serves predictions."""

    def __init__(
        self,
        export_dir: str | Path,
        device: str = "cuda",
        num_denoising_steps: int = 10,
        providers: list[str] | None = None,
        strict_providers: bool = True,
        safety_config: str | Path | None = None,
        adaptive_steps: bool = False,
        cloud_fallback_url: str = "",
        deadline_ms: float | None = None,
        max_batch: int = 1,
        batch_timeout_ms: float = 5.0,
    ):
        """Create the server.

        Args:
            export_dir: directory with exported ONNX + reflex_config.json
            device: "cuda" or "cpu" — selects default ONNX execution provider
            num_denoising_steps: Euler flow matching steps per inference
            providers: explicit list of ORT execution providers to request, e.g.
                ["CUDAExecutionProvider", "CPUExecutionProvider"]. If omitted,
                derived from `device`. Useful for explicit control in production.
            strict_providers: if True (default), raise a loud RuntimeError when
                the requested provider fails to load instead of silently falling
                back to CPU. Set False only if you explicitly want best-effort
                fallback (almost always wrong for GPU deployments).
            safety_config: path to a SafetyLimits JSON (see `reflex guard init`).
                When set, every action is run through ActionGuard.check() before
                being returned. Violations are clamped by default; use
                safety_config with mode="reject" to return an error instead.
            adaptive_steps: if True, use TurboOptimizer adaptive strategy
                (stops early when velocity converges). Requires ORT session
                already loaded.
            cloud_fallback_url: if non-empty, configures `SplitOrchestrator`
                with this cloud endpoint. On edge failure or deadline miss,
                `predict()` will attempt to route through the cloud.
            deadline_ms: soft deadline per `predict()` call. If the denoise
                loop + safety check exceeds this, the server returns the last
                known good action (or a zero vector) and logs a deadline miss.
        """
        self.export_dir = Path(export_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_denoising_steps = num_denoising_steps
        self._requested_device = device
        self._requested_providers = providers
        self._strict_providers = strict_providers
        self.config = self._load_config()
        self.model = None
        self._ready = False
        self._vlm = None
        self._vlm_loaded = False
        self._expert_input_names: list[str] = []

        # Composed wedges (Phase I.2)
        self._safety_config_path = Path(safety_config) if safety_config else None
        self._adaptive_steps = adaptive_steps
        self._cloud_fallback_url = cloud_fallback_url
        self._deadline_ms = deadline_ms
        self._action_guard = None  # built during load()
        self._split_orchestrator = None  # built during load()
        self._last_good_actions: np.ndarray | None = None
        self._deadline_misses = 0

        # Multi-robot batching (Phase III)
        self._max_batch = max(1, max_batch)
        self._batch_timeout_s = max(0.0, batch_timeout_ms) / 1000.0
        # _batch_queue + _batch_worker_task lazily created in start_batch_worker()
        self._batch_queue = None
        self._batch_worker_task = None
        self._batches_run = 0
        self._batched_requests = 0

    def _load_config(self) -> dict[str, Any]:
        config_path = self.export_dir / "reflex_config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
        return {}

    def load(self) -> None:
        """Load the model from exported directory + compose any wedges."""
        logger.info("Loading model from %s", self.export_dir)
        start = time.perf_counter()

        expert_meta = self.config.get("expert", {})
        self.action_dim = expert_meta.get("action_dim", 32)
        self.chunk_size = self.config.get("action_chunk_size", 50)
        self.expert_hidden = expert_meta.get("expert_hidden", 720)

        # Try ONNX runtime first, fall back to PyTorch
        onnx_path = self.export_dir / "expert_stack.onnx"
        if onnx_path.exists():
            self._load_onnx(onnx_path)
        else:
            logger.warning("No ONNX model found, inference not available")
            return

        # Cache expert input names for backward compat (v0.1 exports may not have vlm_kv)
        self._expert_input_names = [inp.name for inp in self._ort_session.get_inputs()]
        logger.info("Expert ONNX inputs: %s", self._expert_input_names)

        # Load VLM prefix pipeline via orchestrator (4-file ONNX pipeline)
        self._load_vlm_orchestrator()

        # ---- Wedge composition (Phase I.2) ----
        # reflex guard: safety limits
        if self._safety_config_path is not None:
            try:
                from reflex.safety import ActionGuard, SafetyLimits
                limits = SafetyLimits.from_json(self._safety_config_path)
                self._action_guard = ActionGuard(limits=limits, mode="clamp")
                logger.info(
                    "reflex guard loaded: %d joints, mode=clamp",
                    len(limits.joint_names),
                )
            except Exception as e:
                logger.warning("Failed to load safety config: %s", e)

        # reflex split: cloud-edge orchestrator
        if self._cloud_fallback_url:
            try:
                from reflex.runtime.split import SplitOrchestrator, SplitConfig
                self._split_orchestrator = SplitOrchestrator(SplitConfig(
                    cloud_url=self._cloud_fallback_url,
                    prefer="edge",
                    fallback_mode="last_action",
                ))
                logger.info(
                    "reflex split configured: cloud_url=%s, fallback=last_action",
                    self._cloud_fallback_url,
                )
            except Exception as e:
                logger.warning("Failed to build split orchestrator: %s", e)

        if self._adaptive_steps:
            logger.info("reflex turbo: adaptive denoise step count ENABLED")
            # Honesty per the Apr-14 phase IV bench: the 0.01 velocity-norm-delta
            # threshold works well on pi0 (~58% latency savings, action diff 0.07)
            # but never triggers on smolvla, rarely triggers on pi0.5, and triggers
            # too aggressively on gr00t (action diff 0.67 — meaningful drift). The
            # per-model threshold tuning lands in v0.2.
            model_type = self.config.get("model_type", "")
            if model_type and model_type != "pi0":
                logger.warning(
                    "adaptive_steps with model_type=%s is unvalidated. "
                    "Phase IV bench (Apr 14): smolvla never triggers (no savings), "
                    "pi0.5 rarely triggers, gr00t triggers too aggressively "
                    "(action diff 0.67). Use --adaptive-steps with model_type=pi0 "
                    "for now; per-model thresholds land in v0.2.",
                    model_type,
                )
        if self._deadline_ms is not None:
            logger.info("deadline enforcement: %.1f ms", self._deadline_ms)

        elapsed = time.perf_counter() - start
        self._ready = True
        logger.info("Model loaded in %.1fs, ready to serve", elapsed)

    def _load_onnx(self, onnx_path: Path) -> None:
        """Load ONNX model via onnxruntime.

        Honors `self._requested_device` and `self._requested_providers`. Raises
        if the requested provider fails to load AND `self._strict_providers` is
        set (the default). Silent CPU fallback was causing users to publish
        "GPU" benchmarks that were actually CPU — Apr 14 post-mortem. Never
        again.
        """
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is not installed. For GPU inference, install "
                "`onnxruntime-gpu` (not `onnxruntime`). For CPU only, "
                "install `onnxruntime`."
            ) from e

        # What the installed ORT actually supports on this machine
        available = set(ort.get_available_providers())

        # Decide which providers to request
        if self._requested_providers is not None:
            providers = list(self._requested_providers)
        elif self._requested_device == "cuda":
            providers = []
            # Prefer TensorRT EP when available — gives FP16 kernels and
            # engine caching transparently. ORT falls back to CUDA EP for
            # ops the TRT EP doesn't support.
            #
            # CRITICAL: TRT EP is incompatible with continuous batching when
            # the source ONNX has static shapes (which our exporters bake).
            # Apr-14 verification showed TRT rebuilds the engine on each new
            # batch shape, causing 34-second latencies instead of milliseconds.
            # When max_batch > 1, fall through to CUDAExecutionProvider which
            # handles dynamic shapes natively and gives the 2.88x batching
            # speedup measured in Phase III.
            use_trt_ep = (
                "TensorrtExecutionProvider" in available
                and self._max_batch <= 1
            )
            if use_trt_ep:
                # Use a per-export-dir engine cache so subsequent serve calls
                # skip the engine-build cost.
                trt_cache = str(self.export_dir / ".trt_cache")
                Path(trt_cache).mkdir(parents=True, exist_ok=True)
                providers.append((
                    "TensorrtExecutionProvider",
                    {
                        "device_id": 0,
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": trt_cache,
                        "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,  # 4GB
                    },
                ))
            elif "TensorrtExecutionProvider" in available and self._max_batch > 1:
                logger.info(
                    "TRT EP available but disabled because --max-batch=%d > 1. "
                    "TRT EP rebuilds engines per batch shape on static-shape "
                    "ONNX, causing 34s+ latencies. Using CUDAExecutionProvider "
                    "which handles dynamic batch natively. (Re-export with "
                    "dynamic batch shapes + TRT shape profiles is a v0.2 fix.)",
                    self._max_batch,
                )
            providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
        else:
            providers = ["CPUExecutionProvider"]

        logger.info("Requested providers: %s; available: %s", providers, sorted(available))

        # Create session
        self._ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
        active = self._ort_session.get_providers()
        logger.info("Loaded ONNX model: %s — active providers: %s", onnx_path.name, active)

        # Strict check: if caller asked for any GPU provider (CUDA or TRT) but
        # we ended up on CPU, fail loudly.
        def _provider_name(p):
            return p[0] if isinstance(p, tuple) else p

        gpu_provider_names = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
        cuda_requested = any(_provider_name(p) in gpu_provider_names for p in providers)
        cuda_active = any(p in gpu_provider_names for p in active)
        if cuda_requested and not cuda_active and self._strict_providers:
            install_hint = ""
            if "CUDAExecutionProvider" not in available:
                install_hint = (
                    "\n\nCUDAExecutionProvider is not available in this ORT install. "
                    "Likely causes:\n"
                    "  - You installed `onnxruntime` (CPU-only). Replace it with:\n"
                    "      pip uninstall onnxruntime && pip install onnxruntime-gpu\n"
                    "  - CUDA 12 + cuDNN 9 libraries are not on the library path. "
                    "ORT 1.20+ requires CUDA 12.x and cuDNN 9.x. "
                    "See https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements\n"
                    "  - You are on a machine without an NVIDIA GPU."
                )
            raise RuntimeError(
                f"reflex serve was started with --device cuda (or CUDAExecutionProvider "
                f"in --providers) but ONNX Runtime fell back to CPU. "
                f"Active providers: {active}. "
                f"Refusing to continue under strict mode — use --no-strict-providers or "
                f"--device cpu to explicitly request CPU execution.{install_hint}"
            )

        # Tag the inference mode so /act responses report which path was used
        if "TensorrtExecutionProvider" in active:
            self._inference_mode = "onnx_trt_fp16"
        elif cuda_active:
            self._inference_mode = "onnx_gpu"
        else:
            self._inference_mode = "onnx_cpu"

    def _load_vlm_orchestrator(self) -> None:
        """Load the 4-file VLM prefix pipeline via VLMPrefixOrchestrator.

        Checks for VLM files (vision_encoder.onnx, text_embedder.onnx,
        decoder_prefill.onnx) in the export directory. If at least the
        vision encoder exists, creates a VLMPrefixOrchestrator. Otherwise
        falls back to dummy conditioning (v0.1 mode).
        """
        # Check if there are any VLM files to load
        has_vlm_files = (
            (self.export_dir / "vision_encoder.onnx").exists()
            or self.config.get("vlm_prefix_onnx") is not None
        )

        if not has_vlm_files:
            self._vlm = None
            self._vlm_loaded = False
            logger.info("No VLM files found, using dummy conditioning (v0.1 mode)")
            return

        try:
            from reflex.runtime.vlm_orchestrator import VLMPrefixOrchestrator

            self._vlm = VLMPrefixOrchestrator(self.export_dir, self.config)
            self._vlm_loaded = self._vlm.is_loaded
            if self._vlm_loaded:
                logger.info(
                    "VLM orchestrator loaded (complete=%s)",
                    self._vlm.is_complete,
                )
            else:
                logger.warning(
                    "VLM orchestrator created but no sessions loaded -- "
                    "falling back to dummy conditioning"
                )
                self._vlm = None
                self._vlm_loaded = False
        except Exception as e:
            self._vlm = None
            self._vlm_loaded = False
            logger.warning(
                "Failed to create VLM orchestrator: %s -- using dummy conditioning", e
            )

    @property
    def ready(self) -> bool:
        return self._ready

    def _run_denoise(
        self,
        noisy_actions: np.ndarray,
        position_ids: np.ndarray,
        vlm_kv: tuple[np.ndarray, np.ndarray] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        """Run the full denoising loop (fixed or adaptive).

        ``vlm_kv`` can be:
            - (k, v) tuple of shape ([L, B, seq, kv], [L, B, seq, kv]) — v0.5+ (RoPE + split k/v)
            - single ndarray [L, B, seq, kv] — v0.4 (per-layer, no split, no RoPE)
            - single ndarray [B, seq, kv] — v0.3 (collapsed shared tensor)
            - None — use zeros of the right shape

        Returns (denoised_actions, steps_used).
        """
        dt = -1.0 / self.num_denoising_steps
        prev_velocity_norm: float | None = None
        converged_at: int | None = None

        # Detect expert schema by ONNX input names.
        expert_has_split_kv = (
            "vlm_k" in self._expert_input_names
            and "vlm_v" in self._expert_input_names
        )
        expert_has_single_kv = "vlm_kv" in self._expert_input_names

        # Normalize vlm_kv to (k, v) if tuple, else leave as scalar array.
        vlm_k: np.ndarray | None = None
        vlm_v: np.ndarray | None = None
        vlm_kv_single: np.ndarray | None = None
        if isinstance(vlm_kv, tuple) and len(vlm_kv) == 2:
            vlm_k, vlm_v = vlm_kv
        elif vlm_kv is not None:
            vlm_kv_single = vlm_kv

        # Zero fallback when expert expects kv inputs but caller provided none.
        if (expert_has_split_kv or expert_has_single_kv) and (
            vlm_k is None and vlm_v is None and vlm_kv_single is None
        ):
            vlm_kv_dim = self.config.get("vlm_kv_dim", 320)
            prefix_seq_len = self.config.get("vlm_prefix_seq_len", 50)
            batch = noisy_actions.shape[0]
            num_layers = self.config.get("vlm_num_layers", 16)
            zeros_4d = np.zeros(
                (num_layers, batch, prefix_seq_len, vlm_kv_dim), dtype=np.float32
            )
            if expert_has_split_kv:
                vlm_k = zeros_4d
                vlm_v = zeros_4d.copy()
            else:
                # v0.3/v0.4 single-tensor fallback
                vlm_kv_single = zeros_4d

        for step in range(self.num_denoising_steps):
            t = 1.0 + step * dt
            timestep = np.array([t], dtype=np.float32)

            feed_dict = {
                "noisy_actions": noisy_actions,
                "timestep": timestep,
                "position_ids": position_ids,
            }
            if expert_has_split_kv and vlm_k is not None and vlm_v is not None:
                feed_dict["vlm_k"] = vlm_k
                feed_dict["vlm_v"] = vlm_v
                prefix_len = int(vlm_k.shape[2])  # [L, B, seq, kv]
                batch = noisy_actions.shape[0]
                if "prefix_offset" in self._expert_input_names:
                    feed_dict["prefix_offset"] = np.full(
                        (batch, 1), prefix_len, dtype=np.int64
                    )
                if "kv_mask" in self._expert_input_names:
                    # All-valid mask when we don't have the prefix pad mask handy.
                    # TODO: plumb the real padded-token mask through from the
                    # VLM orchestrator.
                    feed_dict["kv_mask"] = np.ones(
                        (batch, prefix_len), dtype=bool
                    )
            elif expert_has_single_kv and vlm_kv_single is not None:
                feed_dict["vlm_kv"] = vlm_kv_single

            velocity = self._ort_session.run(None, feed_dict)[0]

            noisy_actions = noisy_actions + velocity * dt

            # Adaptive early stop: if velocity norm stops changing, stop
            if self._adaptive_steps and step >= 2:
                v_norm = float(np.linalg.norm(velocity))
                if prev_velocity_norm is not None:
                    delta = abs(v_norm - prev_velocity_norm)
                    # Threshold chosen so that small models that converge in 4-5
                    # steps actually break early. 0.01 is conservative.
                    if delta < 0.01:
                        converged_at = step + 1
                        break
                prev_velocity_norm = v_norm

        steps_used = converged_at or self.num_denoising_steps
        return noisy_actions, steps_used

    def predict(
        self,
        image: np.ndarray | list[np.ndarray] | None = None,
        instruction: str = "",
        state: list[float] | np.ndarray | None = None,
        noise: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run inference: image + instruction + state → action chunk.

        Composes the wedges when enabled:
        - reflex turbo adaptive step count (`--adaptive-steps`)
        - reflex guard safety check (`--safety-config`)
        - reflex split cloud fallback (`--cloud-fallback`)
        - deadline enforcement (`--deadline-ms`)

        Args:
            image: RGB image array [H, W, 3] or None
            instruction: text instruction (unused in v0.1 expert-only mode)
            state: robot state vector [N] or None

        Returns:
            dict with "actions" (list of action vectors), "latency_ms", "hz",
            and optional telemetry fields (steps_used, safety_violations,
            deadline_exceeded, used_cloud_fallback)
        """
        if not self._ready:
            return {"error": "Model not loaded. Call load() first."}
        if not self._inference_mode.startswith("onnx"):
            return {"error": f"Unknown inference mode: {self._inference_mode}"}
        if self._action_guard is not None and self._action_guard.tripped:
            return {
                "error": "guard_tripped",
                "reason": self._action_guard.trip_reason,
                "hint": "Investigate upstream (inputs / sensors / model) and "
                        "call POST /guard/reset to resume.",
            }

        start = time.perf_counter()

        # Prepare inputs — optionally seed noise externally so test harnesses
        # can produce deterministic outputs matching a reference pipeline.
        if noise is not None:
            noisy_actions = np.asarray(noise, dtype=np.float32)
            if noisy_actions.ndim == 2:
                noisy_actions = noisy_actions[np.newaxis, ...]
        else:
            noisy_actions = np.random.randn(
                1, self.chunk_size, self.action_dim
            ).astype(np.float32)
        position_ids = np.arange(self.chunk_size, dtype=np.int64).reshape(1, -1)

        # VLM prefix conditioning via orchestrator
        vlm_kv = None
        used_vlm = False
        state_np = np.array(state, dtype=np.float32) if state is not None else None
        if self._vlm is not None and image is not None and instruction:
            try:
                _state_for_vlm = state_np if state_np is not None else np.zeros(6, dtype=np.float32)
                vlm_kv = self._vlm.run(image, instruction, _state_for_vlm)
                used_vlm = True
            except Exception as e:
                logger.warning("VLM orchestrator failed: %s — using dummy conditioning", e)
                vlm_kv = None
                used_vlm = False

        # Denoise (adaptive or fixed)
        noisy_actions, steps_used = self._run_denoise(noisy_actions, position_ids, vlm_kv=vlm_kv)

        actions_np = noisy_actions[0]  # [chunk, action_dim]

        # reflex guard — safety check
        safety_violations = 0
        guard_detail: list[str] = []
        if self._action_guard is not None:
            try:
                safe_actions, guard_results = self._action_guard.check(actions_np)
                actions_np = safe_actions
                safety_violations = sum(len(r.violations) for r in guard_results)
                if safety_violations > 0:
                    guard_detail = [
                        f"action {i}: {len(r.violations)} violations"
                        for i, r in enumerate(guard_results[:3]) if r.violations
                    ]
            except Exception as e:
                logger.warning("safety check failed: %s", e)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Deadline enforcement — return last good action if over budget
        deadline_exceeded = False
        if self._deadline_ms is not None and elapsed_ms > self._deadline_ms:
            deadline_exceeded = True
            self._deadline_misses += 1
            if self._last_good_actions is not None:
                actions_np = self._last_good_actions
                logger.warning(
                    "deadline miss (%d): %.1fms > %.1fms — returning last good action",
                    self._deadline_misses, elapsed_ms, self._deadline_ms,
                )
            else:
                # No prior good action; return zeros
                actions_np = np.zeros_like(actions_np)
                logger.warning(
                    "deadline miss (%d): %.1fms > %.1fms — no prior action, returning zeros",
                    self._deadline_misses, elapsed_ms, self._deadline_ms,
                )

        # Cache for next deadline miss
        if not deadline_exceeded:
            self._last_good_actions = actions_np.copy()

        # Convert to list for JSON
        actions = actions_np.tolist()

        result: dict[str, Any] = {
            "actions": actions,
            "num_actions": len(actions),
            "action_dim": self.action_dim,
            "latency_ms": round(elapsed_ms, 1),
            "hz": round(1000.0 / elapsed_ms, 1) if elapsed_ms > 0 else 0,
            "denoising_steps": steps_used,
            "inference_mode": self._inference_mode,
            "vlm_conditioning": "real" if used_vlm else "dummy",
        }
        # Telemetry from wedges — only populate when flags are on
        if self._adaptive_steps:
            result["adaptive_enabled"] = True
        if self._action_guard is not None:
            result["safety_violations"] = safety_violations
            if guard_detail:
                result["safety_detail"] = guard_detail
        if self._deadline_ms is not None:
            result["deadline_exceeded"] = deadline_exceeded
            if self._deadline_misses:
                result["deadline_misses_total"] = self._deadline_misses
        if self._split_orchestrator is not None:
            result["split_enabled"] = True  # full implementation pending Phase VI

        return result

    def predict_from_base64(
        self,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
    ) -> dict[str, Any]:
        """Predict from base64-encoded image (for HTTP API)."""
        image = None
        if image_b64:
            try:
                from PIL import Image

                img_bytes = base64.b64decode(image_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image = np.array(img)
            except Exception as e:
                return {"error": f"Failed to decode image: {e}"}

        return self.predict(image=image, instruction=instruction, state=state)

    # ---------------------------------------------------------------
    # Phase III: continuous batching across HTTP /act requests
    # ---------------------------------------------------------------

    async def start_batch_worker(self) -> None:
        """Spawn an asyncio task that drains the batch queue. Idempotent.

        Only does anything when max_batch > 1 — otherwise predict_async()
        falls through to plain predict().
        """
        if self._max_batch <= 1:
            return
        if self._batch_worker_task is not None and not self._batch_worker_task.done():
            return
        import asyncio
        self._batch_queue = asyncio.Queue()
        self._batch_worker_task = asyncio.create_task(self._batch_worker_loop())
        logger.info(
            "batching enabled: max_batch=%d, timeout=%.1fms",
            self._max_batch, self._batch_timeout_s * 1000,
        )

    async def stop_batch_worker(self) -> None:
        """Cancel the batch worker (called during FastAPI shutdown)."""
        import asyncio
        if self._batch_worker_task is None:
            return
        self._batch_worker_task.cancel()
        try:
            await self._batch_worker_task
        except (asyncio.CancelledError, Exception):
            pass
        self._batch_worker_task = None
        self._batch_queue = None

    async def predict_async(
        self,
        image: np.ndarray | None = None,
        instruction: str = "",
        state: list[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Async front-door used by the HTTP /act handler.

        - If max_batch <= 1: runs `self.predict()` synchronously in this task.
        - If max_batch > 1: enqueues the request onto a batch queue. A worker
          coroutine drains the queue every `batch_timeout_ms` ms (or when the
          queue hits max_batch) and runs ONE batched ONNX inference, then
          splits the results back to each waiter.
        """
        if self._max_batch <= 1 or self._batch_queue is None:
            return self.predict(image=image, instruction=instruction, state=state)

        import asyncio
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self._batch_queue.put((future, image, instruction, state))
        return await future

    async def _batch_worker_loop(self) -> None:
        """Drain the batch queue. Run for the lifetime of the server."""
        import asyncio
        while True:
            batch: list[tuple] = []
            try:
                # Block on the first request — if the queue is empty we just wait.
                first = await self._batch_queue.get()
                batch.append(first)
            except asyncio.CancelledError:
                break

            # Drain up to max_batch within the configured time window.
            deadline = asyncio.get_event_loop().time() + self._batch_timeout_s
            while len(batch) < self._max_batch:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._batch_queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    # Make sure pending futures are released
                    for fut, *_ in batch:
                        if not fut.done():
                            fut.set_exception(asyncio.CancelledError())
                    return

            # Run batched inference (sync — we're holding the event loop, but
            # the actual ORT call is the bottleneck and yields the GIL).
            try:
                results = self._predict_batch_sync(batch)
                for (fut, *_), result in zip(batch, results):
                    if not fut.done():
                        fut.set_result(result)
            except Exception as e:
                for fut, *_ in batch:
                    if not fut.done():
                        fut.set_exception(e)

    def _predict_batch_sync(self, batch: list[tuple]) -> list[dict[str, Any]]:
        """Run one ONNX inference with batch dim = len(batch). Split results.

        For v0.1 of batching, ignores per-item image/instruction/state — same
        as plain predict(). The point is to demonstrate the batching primitive
        and measure throughput scaling. Per-item conditioning lands when the
        VLM prefix path is wired in (Phase II.4).
        """
        if not self._ready:
            return [{"error": "Model not loaded."} for _ in batch]
        if not self._inference_mode.startswith("onnx"):
            return [{"error": f"Unknown inference mode: {self._inference_mode}"} for _ in batch]

        b = len(batch)
        start = time.perf_counter()

        noisy_batched = np.random.randn(
            b, self.chunk_size, self.action_dim
        ).astype(np.float32)
        position_ids_batched = np.tile(
            np.arange(self.chunk_size, dtype=np.int64), (b, 1),
        )

        dt = -1.0 / self.num_denoising_steps
        for step in range(self.num_denoising_steps):
            t = 1.0 + step * dt
            timestep = np.full((b,), t, dtype=np.float32)
            velocity = self._ort_session.run(
                None,
                {
                    "noisy_actions": noisy_batched,
                    "timestep": timestep,
                    "position_ids": position_ids_batched,
                },
            )[0]
            noisy_batched = noisy_batched + velocity * dt

        elapsed_ms = (time.perf_counter() - start) * 1000
        per_request_ms = elapsed_ms / b  # amortized

        self._batches_run += 1
        self._batched_requests += b

        results: list[dict[str, Any]] = []
        for i in range(b):
            actions_np = noisy_batched[i]

            # Apply guard per-item (each request gets its own clamping)
            safety_violations = 0
            if self._action_guard is not None:
                try:
                    safe_actions, guard_results = self._action_guard.check(actions_np)
                    actions_np = safe_actions
                    safety_violations = sum(len(r.violations) for r in guard_results)
                except Exception as e:
                    logger.warning("safety check failed in batch: %s", e)

            result = {
                "actions": actions_np.tolist(),
                "num_actions": len(actions_np),
                "action_dim": self.action_dim,
                "latency_ms": round(elapsed_ms, 1),
                "amortized_latency_ms": round(per_request_ms, 1),
                "hz": round(1000.0 / per_request_ms, 1) if per_request_ms > 0 else 0,
                "denoising_steps": self.num_denoising_steps,
                "inference_mode": self._inference_mode,
                "batch_size": b,
                "request_index": i,
                "batches_run_total": self._batches_run,
                "batched_requests_total": self._batched_requests,
            }
            if self._action_guard is not None:
                result["safety_violations"] = safety_violations
            results.append(result)

        return results

    async def predict_from_base64_async(
        self,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
    ) -> dict[str, Any]:
        """Async base64 entrypoint — decodes image, then routes through batching."""
        image = None
        if image_b64:
            try:
                from PIL import Image
                img_bytes = base64.b64decode(image_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                image = np.array(img)
            except Exception as e:
                return {"error": f"Failed to decode image: {e}"}

        return await self.predict_async(image=image, instruction=instruction, state=state)


try:
    from pydantic import BaseModel

    class PredictRequest(BaseModel):
        image: str | None = None  # base64 encoded
        instruction: str = ""
        state: list[float] | None = None

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        inference_mode: str = ""
        export_dir: str = ""
        vlm_loaded: bool = False

except ImportError:
    PredictRequest = None  # type: ignore
    HealthResponse = None  # type: ignore


def create_app(
    export_dir: str,
    device: str = "cuda",
    providers: list[str] | None = None,
    strict_providers: bool = True,
    safety_config: str | Path | None = None,
    adaptive_steps: bool = False,
    cloud_fallback_url: str = "",
    deadline_ms: float | None = None,
    max_batch: int = 1,
    batch_timeout_ms: float = 5.0,
) -> Any:
    """Create a FastAPI app for serving VLA predictions."""
    try:
        from contextlib import asynccontextmanager
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError("Install fastapi: pip install 'reflex-vla[serve]'")

    # Route: decomposed-ONNX by default; native PyTorch path under REFLEX_NATIVE=1.
    # The native path bypasses our ONNX export and runs lerobot's SmolVLAPolicy
    # directly (RMSNorm still swapped for DecomposedRMSNorm for TRT-export compat
    # on the decomposed side). See reflex/runtime/smolvla_native.py.
    import os as _os

    # Dispatch order:
    #   1. REFLEX_NATIVE=1 — SmolVLANativeServer (PyTorch native path)
    #   2. reflex_config.json export_kind == "monolithic" → model-specific
    #      monolithic server (Pi0OnnxServer / SmolVLAOnnxServer). This is
    #      the cos=1.0 verified production path as of 2026-04-18.
    #   3. Default: ReflexServer (legacy decomposed path).
    _config_path = Path(export_dir) / "reflex_config.json"
    _monolithic_cfg = {}
    if _config_path.exists():
        try:
            _monolithic_cfg = json.loads(_config_path.read_text())
        except Exception:
            _monolithic_cfg = {}

    if _os.environ.get("REFLEX_NATIVE", "0") == "1":
        from reflex.runtime.smolvla_native import SmolVLANativeServer
        server = SmolVLANativeServer(
            export_dir,
            device=device,
            providers=providers,
            strict_providers=strict_providers,
            safety_config=safety_config,
            adaptive_steps=adaptive_steps,
            cloud_fallback_url=cloud_fallback_url,
            deadline_ms=deadline_ms,
            max_batch=max_batch,
            batch_timeout_ms=batch_timeout_ms,
        )
    elif _monolithic_cfg.get("export_kind") == "monolithic":
        _model_type = _monolithic_cfg.get("model_type", "smolvla")
        _ort_providers = providers or (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda" else ["CPUExecutionProvider"]
        )
        if _model_type == "pi0":
            from reflex.runtime.pi0_onnx_server import Pi0OnnxServer
            server = Pi0OnnxServer(
                export_dir,
                providers=_ort_providers,
                strict_providers=strict_providers,
            )
        elif _model_type == "smolvla":
            from reflex.runtime.smolvla_onnx_server import SmolVLAOnnxServer
            server = SmolVLAOnnxServer(
                export_dir,
                providers=_ort_providers,
                strict_providers=strict_providers,
            )
        else:
            raise ValueError(
                f"Monolithic runtime for model_type={_model_type!r} not yet "
                f"supported. v0.2 covers smolvla + pi0."
            )
    else:
        server = ReflexServer(
            export_dir,
            device=device,
            providers=providers,
            strict_providers=strict_providers,
            safety_config=safety_config,
            adaptive_steps=adaptive_steps,
            cloud_fallback_url=cloud_fallback_url,
            deadline_ms=deadline_ms,
            max_batch=max_batch,
            batch_timeout_ms=batch_timeout_ms,
        )

    @asynccontextmanager
    async def lifespan(app):
        server.load()
        # Warm up: run one inference so any lazy-build (TRT engine build,
        # ORT graph optimization passes) happens before users hit /act.
        # Without this, the first /act request takes 30-90s with TRT EP enabled
        # because TRT builds + caches an engine on first call.
        try:
            logger.info("Warming up — running one denoising loop to JIT the engine...")
            import time as _t
            _t0 = _t.perf_counter()
            warmup_result = server.predict()
            _elapsed = (_t.perf_counter() - _t0) * 1000
            if "error" in warmup_result:
                logger.warning("Warmup returned error: %s", warmup_result["error"])
            else:
                logger.info(
                    "Warmup complete in %.0fms (mode=%s). Subsequent /act calls "
                    "will use the cached TRT engine if applicable.",
                    _elapsed, warmup_result.get("inference_mode", "?"),
                )
        except Exception as e:
            logger.warning("Warmup failed (server still up): %s", e)

        await server.start_batch_worker()
        try:
            yield
        finally:
            await server.stop_batch_worker()

    app = FastAPI(
        title="Reflex VLA Server",
        description="Deploy any VLA model to any edge hardware.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ok" if server.ready else "not_ready",
            model_loaded=server.ready,
            inference_mode=getattr(server, "_inference_mode", ""),
            export_dir=str(server.export_dir),
            vlm_loaded=getattr(server, "_vlm_loaded", False),
        )

    @app.post("/act")
    async def act(request: PredictRequest):
        # Routes through the batching path when max_batch > 1.
        result = await server.predict_from_base64_async(
            image_b64=request.image,
            instruction=request.instruction,
            state=request.state,
        )
        return JSONResponse(content=result)

    @app.get("/config")
    async def config():
        return JSONResponse(content=server.config)

    @app.get("/guard/status")
    async def guard_status():
        g = getattr(server, "_action_guard", None)
        if g is None:
            return JSONResponse(content={"enabled": False})
        return JSONResponse(content={
            "enabled": True,
            "tripped": bool(g.tripped),
            "trip_reason": g.trip_reason,
            "consecutive_clamps": int(g.consecutive_clamps),
            "max_consecutive_clamps": int(g.max_consecutive_clamps),
            "inference_count": int(g.inference_count),
        })

    @app.post("/guard/reset")
    async def guard_reset():
        g = getattr(server, "_action_guard", None)
        if g is None:
            return JSONResponse(
                status_code=400,
                content={"error": "guard_not_enabled"},
            )
        was_tripped = bool(g.tripped)
        g.reset()
        return JSONResponse(content={"reset": True, "was_tripped": was_tripped})

    return app
