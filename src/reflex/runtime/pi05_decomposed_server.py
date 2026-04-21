"""Pi05DecomposedInference — run pi0.5 through the decomposed
``vlm_prefix.onnx`` + ``expert_denoise.onnx`` pair, with optional
cross-query VLM prefix cache.

Design doc: ``reflex_context/reflex_vla/01_architecture/prefix_kv_cache_reuse_design.md``

The cache hashes the VLM input (images + language tokens) per call and
reuses the last ``past_kv`` output when the hash matches inside a
staleness window. During cache-hit the expensive VLM forward (90% of
compute for a 1-NFE student) is skipped; only the tiny expert-denoise
graph runs.

Callers typically use this through:

- the LIBERO-eval harness via ``--decomposed-dir`` flag, or
- ``reflex serve <export_dir>`` when ``reflex_config.json`` declares
  ``"export_kind": "decomposed"`` (wiring lives in
  ``reflex.runtime.server``).

The class exposes the same ``predict_action_chunk`` contract as
``Pi0OnnxServer.predict`` so downstream harnesses don't need to know
which export pattern is active.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """One VLM output we keep around for potential reuse."""
    past_kv: list[np.ndarray]            # flat [k_0, v_0, ..., k_17, v_17]
    prefix_pad_masks: np.ndarray
    image_phashes: tuple[bytes, ...]     # per-camera perceptual hash
    lang_hash: bytes                     # exact md5 of language tokens
    timestamp: float


@dataclass
class CacheStats:
    """Cumulative cache metrics — exposed via get_stats() so callers
    can log hit rate alongside LIBERO task success."""
    hits: int = 0
    misses: int = 0
    evictions_ttl: int = 0
    evictions_lang: int = 0
    evictions_phash: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total": self.total,
            "hit_rate": self.hit_rate,
            "evictions_ttl": self.evictions_ttl,
            "evictions_lang": self.evictions_lang,
            "evictions_phash": self.evictions_phash,
        }


class Pi05DecomposedInference:
    """Two-ONNX pi0.5 inference with optional prefix-cache reuse.

    Parameters
    ----------
    export_dir
        Directory containing ``vlm_prefix.onnx``, ``expert_denoise.onnx``,
        and ``reflex_config.json`` (written by ``export_pi05_decomposed``).
    providers
        onnxruntime execution providers (default CPU).
    enable_cache
        When False, every call runs the VLM. When True, the cache matches
        on perceptual-image-hash + exact-language-hash + TTL.
    cache_ttl_sec
        Seconds after which a cache entry is considered stale regardless
        of match. Default 0.2s — caps error bound at ~5 frames of latency.
    phash_hamming_threshold
        Per-image perceptual-hash distance allowed for a cache hit.
        Default 6 — tuned for typical manipulation sensor noise (tune
        per-deployment via telemetry).
    """

    PHASH_SIZE: int = 8  # 8x8 phash → 64-bit hash → hamming ≤ 64

    def __init__(
        self,
        export_dir: str | Path,
        providers: list[str] | None = None,
        enable_cache: bool = True,
        cache_ttl_sec: float = 0.2,
        phash_hamming_threshold: int = 6,
    ):
        import onnxruntime as ort

        self.export_dir = Path(export_dir)
        self.enable_cache = enable_cache
        self.cache_ttl_sec = cache_ttl_sec
        self.phash_hamming_threshold = phash_hamming_threshold
        self._providers = providers or ["CPUExecutionProvider"]

        cfg_path = self.export_dir / "reflex_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"reflex_config.json missing in {self.export_dir}")
        self.config: dict[str, Any] = json.loads(cfg_path.read_text())
        if self.config.get("export_kind") != "decomposed":
            raise ValueError(
                f"{cfg_path} has export_kind={self.config.get('export_kind')!r}; "
                "Pi05DecomposedInference requires 'decomposed'"
            )
        self._past_kv_names: list[str] = self.config["decomposed"]["past_kv_tensor_names"]
        self._n_layers: int = self.config["decomposed"]["paligemma_layers"]

        prefix_path = self.export_dir / self.config["decomposed"]["vlm_prefix_onnx"]
        expert_path = self.export_dir / self.config["decomposed"]["expert_denoise_onnx"]
        logger.info("loading vlm_prefix: %s", prefix_path)
        self._sess_prefix = ort.InferenceSession(str(prefix_path), providers=self._providers)
        self._prefix_input_names = [i.name for i in self._sess_prefix.get_inputs()]
        self._prefix_output_names = [o.name for o in self._sess_prefix.get_outputs()]
        logger.info("loading expert_denoise: %s", expert_path)
        self._sess_expert = ort.InferenceSession(str(expert_path), providers=self._providers)
        self._expert_input_names = [i.name for i in self._sess_expert.get_inputs()]

        self._cache: CacheEntry | None = None
        self._stats = CacheStats()

    # ---- Public API -------------------------------------------------

    def predict_action_chunk(
        self,
        *,
        img_base: np.ndarray,
        img_wrist_l: np.ndarray,
        img_wrist_r: np.ndarray,
        mask_base: np.ndarray,
        mask_wrist_l: np.ndarray,
        mask_wrist_r: np.ndarray,
        lang_tokens: np.ndarray,
        lang_masks: np.ndarray,
        noise: np.ndarray,
    ) -> np.ndarray:
        """Run one pi0.5 forward, returning ``actions`` of shape
        ``(B, chunk_size, action_dim)``.

        Uses the prefix cache when enabled + hashes match + TTL valid.
        Returns float32 regardless of the ONNX internal dtype."""
        image_phashes = (
            self._phash(img_base),
            self._phash(img_wrist_l),
            self._phash(img_wrist_r),
        )
        lang_hash = self._lang_hash(lang_tokens)

        past_kv, prefix_pad = self._get_or_run_prefix(
            img_base=img_base,
            img_wrist_l=img_wrist_l,
            img_wrist_r=img_wrist_r,
            mask_base=mask_base,
            mask_wrist_l=mask_wrist_l,
            mask_wrist_r=mask_wrist_r,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            image_phashes=image_phashes,
            lang_hash=lang_hash,
        )

        expert_feed = {name: past_kv[i] for i, name in enumerate(self._past_kv_names)}
        expert_feed["prefix_pad_masks"] = prefix_pad
        expert_feed["noise"] = noise.astype(np.float32, copy=False)

        actions = self._sess_expert.run(["actions"], expert_feed)[0]
        if actions.dtype != np.float32:
            actions = actions.astype(np.float32)
        return actions

    def reset_cache(self) -> None:
        """Drop any cached VLM output. Call between episodes so cross-task
        phash collisions can't bridge unrelated observations."""
        self._cache = None

    def get_stats(self) -> dict[str, Any]:
        return self._stats.as_dict()

    # ---- Cache machinery --------------------------------------------

    def _get_or_run_prefix(
        self,
        *,
        img_base, img_wrist_l, img_wrist_r,
        mask_base, mask_wrist_l, mask_wrist_r,
        lang_tokens, lang_masks,
        image_phashes: tuple[bytes, ...],
        lang_hash: bytes,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        now = time.time()

        if self.enable_cache and self._cache is not None:
            entry = self._cache
            if now - entry.timestamp > self.cache_ttl_sec:
                self._stats.evictions_ttl += 1
                self._cache = None
            elif entry.lang_hash != lang_hash:
                self._stats.evictions_lang += 1
                self._cache = None
            elif not self._phashes_match(entry.image_phashes, image_phashes):
                self._stats.evictions_phash += 1
                self._cache = None
            else:
                self._stats.hits += 1
                return entry.past_kv, entry.prefix_pad_masks

        # Cache miss (or disabled): run VLM
        self._stats.misses += 1
        prefix_feed = {
            "img_base": img_base.astype(np.float32, copy=False),
            "img_wrist_l": img_wrist_l.astype(np.float32, copy=False),
            "img_wrist_r": img_wrist_r.astype(np.float32, copy=False),
            "mask_base": mask_base.astype(np.bool_, copy=False),
            "mask_wrist_l": mask_wrist_l.astype(np.bool_, copy=False),
            "mask_wrist_r": mask_wrist_r.astype(np.bool_, copy=False),
            "lang_tokens": lang_tokens.astype(np.int64, copy=False),
            "lang_masks": lang_masks.astype(np.bool_, copy=False),
        }
        outputs = self._sess_prefix.run(self._prefix_output_names, prefix_feed)
        out_dict = dict(zip(self._prefix_output_names, outputs))
        past_kv = [out_dict[n] for n in self._past_kv_names]
        prefix_pad = out_dict["prefix_pad_masks"]

        if self.enable_cache:
            self._cache = CacheEntry(
                past_kv=past_kv,
                prefix_pad_masks=prefix_pad,
                image_phashes=image_phashes,
                lang_hash=lang_hash,
                timestamp=now,
            )
        return past_kv, prefix_pad

    def _phashes_match(
        self,
        a: tuple[bytes, ...],
        b: tuple[bytes, ...],
    ) -> bool:
        if len(a) != len(b):
            return False
        for ha, hb in zip(a, b):
            if self._hamming(ha, hb) > self.phash_hamming_threshold:
                return False
        return True

    @staticmethod
    def _hamming(a: bytes, b: bytes) -> int:
        return sum(bin(x ^ y).count("1") for x, y in zip(a, b))

    @classmethod
    def _phash(cls, img: np.ndarray) -> bytes:
        """Average-hash perceptual hash. ``img`` is (B, 3, H, W) float
        in arbitrary range — we downsample to ``PHASH_SIZE`` and compare
        each pixel to the mean. Robust to small sensor noise and slight
        camera-motion. Pure numpy so no optional deps.

        Returns a ``PHASH_SIZE*PHASH_SIZE//8`` byte string (8 bytes for
        8×8 = 64 bits).
        """
        if img.ndim != 4:
            raise ValueError(f"expected (B,3,H,W) image, got shape {img.shape}")
        # Take first batch item + mean across channels → (H, W) gray
        gray = img[0].mean(axis=0)
        h, w = gray.shape
        step_h = max(1, h // cls.PHASH_SIZE)
        step_w = max(1, w // cls.PHASH_SIZE)
        # Integer downsample — coarse but fast + dependency-free
        small = gray[:step_h * cls.PHASH_SIZE, :step_w * cls.PHASH_SIZE]
        small = small.reshape(cls.PHASH_SIZE, step_h, cls.PHASH_SIZE, step_w).mean(axis=(1, 3))
        bits = small > small.mean()
        bits_flat = bits.flatten()
        # Pack bits into bytes
        out = bytearray()
        for byte_idx in range(0, len(bits_flat), 8):
            byte = 0
            for bit_idx, bit in enumerate(bits_flat[byte_idx : byte_idx + 8]):
                if bit:
                    byte |= 1 << bit_idx
            out.append(byte)
        return bytes(out)

    @staticmethod
    def _lang_hash(lang_tokens: np.ndarray) -> bytes:
        return hashlib.md5(lang_tokens.tobytes()).digest()


__all__ = ["Pi05DecomposedInference", "CacheStats", "CacheEntry"]
