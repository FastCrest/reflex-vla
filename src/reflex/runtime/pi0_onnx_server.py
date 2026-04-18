"""Pi0OnnxServer — composed 5-stage ONNX inference for pi0.

Loads the 5 ONNX files produced by `export_pi0_prefix()` and runs the
pi0 forward pass in host Python:

    1. pixel_values            -> SigLIP vision_encoder        -> vision_features
    2. vision_features         -> multi_modal_projector        -> projected_vision
    3. input_ids               -> text_embedder                -> text_features
    4. concat(proj_vision, text) -> decoder_prefill(Gemma)     -> prefix_hidden + per-layer KV
    5. noise + 10-step Euler loop:
         for step in range(num_steps):
            velocity = expert_stack(noisy_actions, timestep, pos_ids, prefix_k, prefix_v)
            noisy_actions = noisy_actions + dt * velocity
       -> final_actions

Matches lerobot's PI0Pytorch.sample_actions structure. Target cos >= 0.999
vs PyTorch reference under shared-noise discipline.

Prerequisites:
    reflex export lerobot/pi0_base --target desktop -o /tmp/pi0_export/
    (or call src.reflex.exporters.pi0_prefix_exporter.export_pi0_prefix)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_NUM_DENOISE_STEPS = 10
DEFAULT_ACTION_CHUNK = 50


class Pi0OnnxServer:
    """Compose the 5 pi0 ONNX stages for end-to-end inference."""

    def __init__(self, export_dir: str | Path, providers: list[str] | None = None):
        self.export_dir = Path(export_dir)
        self.providers = providers or ["CPUExecutionProvider"]
        self._sessions: dict[str, Any] = {}
        self.config: dict[str, Any] = {}
        self._ready = False

    def load(self) -> None:
        """Load reflex_config.json + 5 ORT sessions."""
        import onnxruntime as ort

        cfg_path = self.export_dir / "reflex_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing {cfg_path}")
        self.config = json.loads(cfg_path.read_text())

        def _load(key: str, filename: str | None = None) -> Any:
            rel = self.config["components"].get(key) or filename
            if rel is None:
                raise KeyError(f"Component {key} not in reflex_config.json")
            path = Path(rel)
            if not path.is_absolute():
                path = self.export_dir / rel
            logger.info("Loading ONNX session: %s -> %s", key, path)
            return ort.InferenceSession(str(path), providers=self.providers)

        self._sessions["vision_encoder"] = _load("vision_encoder")
        self._sessions["multi_modal_projector"] = _load("multi_modal_projector")
        self._sessions["text_embedder"] = _load("text_embedder")
        self._sessions["decoder_prefill"] = _load("decoder_prefill")
        self._sessions["expert_stack"] = _load("expert_stack")

        # Cache expert metadata (dims) for flow-matching loop
        expert_meta = self.config.get("metadata", {}).get("expert", {})
        self.n_expert_layers = int(expert_meta.get("num_layers", 18))
        self.expert_nkv = int(expert_meta.get("n_kv_heads", 1))
        self.expert_head_dim = int(expert_meta.get("head_dim", 256))
        self.action_dim = int(expert_meta.get("action_dim", 32))
        # Text embedder / projector dims
        te_meta = self.config.get("metadata", {}).get("text_embedder", {})
        self.vocab_size = int(te_meta.get("vocab_size", 257152))
        self.text_hidden = int(te_meta.get("hidden_size", 2048))
        proj_meta = self.config.get("metadata", {}).get("projector", {})
        self.vision_in = int(proj_meta.get("in_dim", 1152))
        self.vision_out = int(proj_meta.get("out_dim", 2048))

        self._ready = True
        logger.info(
            "Pi0OnnxServer ready: %d expert layers, nkv=%d, head_dim=%d, action_dim=%d",
            self.n_expert_layers, self.expert_nkv, self.expert_head_dim, self.action_dim,
        )

    @property
    def ready(self) -> bool:
        return self._ready

    def _run_vision(self, pixel_values: np.ndarray) -> np.ndarray:
        """Stage 1: pixel_values [B, 3, 224, 224] -> vision features [B, 256, 1152]."""
        out = self._sessions["vision_encoder"].run(None, {"pixel_values": pixel_values})[0]
        return out

    def _run_projector(self, vision: np.ndarray) -> np.ndarray:
        """Stage 2: [B, 256, 1152] -> [B, 256, 2048]."""
        out = self._sessions["multi_modal_projector"].run(None, {"vision_features": vision})[0]
        return out

    def _run_text(self, input_ids: np.ndarray) -> np.ndarray:
        """Stage 3: [B, seq] int64 -> [B, seq, 2048]."""
        out = self._sessions["text_embedder"].run(None, {"input_ids": input_ids})[0]
        return out

    def _run_decoder_prefill(
        self,
        inputs_embeds: np.ndarray,
        attention_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stage 4: [B, prefix_len, 2048] -> last_hidden + per-layer prefix_k/v.

        Uses our GemmaFromEmbeds ONNX (takes inputs_embeds directly).
        Returns:
          last_hidden_state: [B, prefix_len, 2048]
          prefix_k: [num_layers, B, prefix_len, nkv, head_dim]
          prefix_v: same shape
        """
        if attention_mask is None:
            B, seq = inputs_embeds.shape[:2]
            attention_mask = np.ones((B, seq), dtype=np.int64)
        ort_outs = self._sessions["decoder_prefill"].run(None, {
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "attention_mask": attention_mask.astype(np.int64),
        })
        last_hidden = ort_outs[0]
        # Remaining outputs alternate present.{i}.key, present.{i}.value
        # Each of shape [B, nkv, seq, head_dim]
        num_layers = self.n_expert_layers
        k_list, v_list = [], []
        for i in range(num_layers):
            k_list.append(ort_outs[1 + 2 * i])
            v_list.append(ort_outs[2 + 2 * i])
        # Stack -> [L, B, nkv, seq, head_dim]
        prefix_k = np.stack(k_list, axis=0)
        prefix_v = np.stack(v_list, axis=0)
        # Expert expects prefix_k shape [L, B, prefix_len, nkv, head_dim]
        # Gemma KV comes as [B, nkv, seq, hd]; transpose to [B, seq, nkv, hd]
        prefix_k = np.transpose(prefix_k, (0, 1, 3, 2, 4))
        prefix_v = np.transpose(prefix_v, (0, 1, 3, 2, 4))
        return last_hidden, prefix_k, prefix_v

    def _run_expert(
        self,
        noisy_actions: np.ndarray,
        timestep: np.ndarray,
        position_ids: np.ndarray,
        prefix_k: np.ndarray,
        prefix_v: np.ndarray,
    ) -> np.ndarray:
        """Stage 5 (per-step): denoise with prefix-KV concat attention."""
        out = self._sessions["expert_stack"].run(None, {
            "noisy_actions": noisy_actions.astype(np.float32),
            "timestep": timestep.astype(np.float32),
            "position_ids": position_ids.astype(np.int64),
            "prefix_k": prefix_k.astype(np.float32),
            "prefix_v": prefix_v.astype(np.float32),
        })[0]
        return out

    def predict(
        self,
        *,
        pixel_values: np.ndarray,
        input_ids: np.ndarray,
        state: np.ndarray | None = None,
        noise: np.ndarray | None = None,
        num_steps: int = DEFAULT_NUM_DENOISE_STEPS,
        chunk_size: int = DEFAULT_ACTION_CHUNK,
    ) -> dict[str, Any]:
        """Run full pi0 forward pass.

        Inputs:
            pixel_values: [B, 3, 224, 224] float32
            input_ids:    [B, text_len] int64 (PaliGemma tokenization)
            state:        [B, state_dim] float32 (optional; random if None)
            noise:        [B, chunk_size, action_dim] float32 (optional seeded)
            num_steps:    flow-matching Euler steps (default 10)
            chunk_size:   action chunk length (default 50)

        Returns dict with 'actions' (denoised chunk) + telemetry.
        """
        if not self._ready:
            raise RuntimeError("Call .load() first")

        B = pixel_values.shape[0]

        # Stage 1-2: vision + project
        vision_feats = self._run_vision(pixel_values.astype(np.float32))
        projected_vision = self._run_projector(vision_feats)  # [B, 256, 2048]

        # Stage 3: text embed
        text_feats = self._run_text(input_ids.astype(np.int64))  # [B, text_len, 2048]

        # Stage 4: decoder prefill -> real per-layer KV via GemmaFromEmbeds
        # Scale vision features by sqrt(hidden) — matches pi0's embed_image
        # which does `features * hidden_size**0.5`. Applied to projected_vision
        # so it lives on the same scale as text embeddings before concat.
        scale = float(self.vision_out) ** 0.5
        vision_scaled = projected_vision * scale
        prefix_embeds = np.concatenate([vision_scaled, text_feats], axis=1)  # [B, 256+text_len, 2048]
        prefix_mask = np.ones(prefix_embeds.shape[:2], dtype=np.int64)
        _, prefix_k, prefix_v = self._run_decoder_prefill(prefix_embeds, prefix_mask)

        # Stage 5: flow matching Euler loop
        if noise is None:
            noise = np.random.RandomState(0).randn(B, chunk_size, self.action_dim).astype(np.float32)
        x_t = noise.copy()
        dt = -1.0 / num_steps
        pos_ids = np.arange(chunk_size, dtype=np.int64)[None, :].repeat(B, 0)

        for step in range(num_steps):
            t = 1.0 + step * dt
            t_arr = np.full((B,), t, dtype=np.float32)
            velocity = self._run_expert(x_t, t_arr, pos_ids, prefix_k, prefix_v)
            x_t = x_t + dt * velocity

        # Output
        return {
            "actions": x_t,  # [B, chunk, action_dim]
            "num_actions": x_t.shape[1],
            "action_dim": x_t.shape[2],
            "num_denoising_steps": num_steps,
            "inference_mode": "pi0_onnx_composed",
            "note": "prefix_kv is zero-init in v0.2 scaffold; real conditioning requires v0.3 Gemma inputs_embeds wrapper",
        }
