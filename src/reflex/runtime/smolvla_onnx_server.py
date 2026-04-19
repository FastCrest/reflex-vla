"""SmolVLAOnnxServer — serves SmolVLA via the monolithic ONNX exported by
`reflex export --monolithic lerobot/smolvla_base ...`
(or `scripts/modal_smolvla_monolithic_export.py`).

num_steps is baked in at export time. SmolVLA monolithic at num_steps=10
is verified at machine precision (max_abs 5.96e-07 vs PyTorch; see
reflex_context/measured_numbers.md). num_steps=1 is also available.

Interface mirrors Pi0OnnxServer so `reflex serve` can dispatch by
config["model_type"] without case analysis in the endpoints.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SmolVLAOnnxServer:
    """Monolithic SmolVLA ONNX runtime."""

    def __init__(
        self,
        export_dir: str | Path,
        onnx_path: str | Path | None = None,
        providers: list[str] | None = None,
        strict_providers: bool = True,
    ):
        self.export_dir = Path(export_dir)
        self._explicit_onnx_path = Path(onnx_path) if onnx_path else None
        self.providers = providers or ["CPUExecutionProvider"]
        self._session: Any = None
        self.config: dict[str, Any] = {}
        self._ready = False
        self._inference_mode = "smolvla_onnx_monolithic"

    def _find_onnx_path(self) -> Path:
        if self._explicit_onnx_path is not None:
            return self._explicit_onnx_path
        candidates = [
            self.export_dir / "model.onnx",
            self.export_dir / "smolvla_monolithic" / "model.onnx",
            self.export_dir / "smolvla_monolithic.onnx",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No SmolVLA monolithic ONNX found in {self.export_dir}. "
            f"Tried: {candidates}"
        )

    def load(self) -> None:
        import onnxruntime as ort

        onnx_path = self._find_onnx_path()
        logger.info("Loading SmolVLA monolithic ONNX: %s", onnx_path)
        start = time.perf_counter()
        self._session = ort.InferenceSession(str(onnx_path), providers=self.providers)
        elapsed = time.perf_counter() - start

        cfg_path = self.export_dir / "reflex_config.json"
        if cfg_path.exists():
            self.config = json.loads(cfg_path.read_text())

        self._input_names = [i.name for i in self._session.get_inputs()]
        logger.info(
            "SmolVLAOnnxServer ready in %.2fs (inputs: %s)",
            elapsed, self._input_names,
        )
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def _get_tokenizer(self):
        """Load SmolLM2 tokenizer with pad_token set. Cached per instance.

        Without pad_token set, `tokenizer(padding="max_length")` raises
        `Asking to pad but the tokenizer does not have a padding token`,
        and the silent-fallback path zeros out the instruction. Customer
        dogfood 2026-04-19 caught this silent failure.
        """
        if getattr(self, "_tokenizer", None) is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
            # SmolLM2 ships without a pad_token. Set it to eos_token (standard HF pattern).
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self._tokenizer = tok
        return self._tokenizer

    def predict(
        self,
        image: np.ndarray | list[np.ndarray] | None = None,
        instruction: str = "",
        state: list[float] | np.ndarray | None = None,
        noise: np.ndarray | None = None,
        lang_tokens: np.ndarray | None = None,
        lang_masks: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run one SmolVLA forward pass. Accepts a single image or a list of 3."""
        if not self._ready:
            return {"error": "Model not loaded. Call load() first."}

        t0 = time.perf_counter()

        # Accept 1 or 3 cameras; replicate to 3
        images_list = (
            image if isinstance(image, list)
            else [image] if image is not None
            else []
        )
        while len(images_list) < 3:
            images_list.append(
                images_list[-1] if images_list else np.zeros((512, 512, 3), np.uint8)
            )
        images_list = images_list[:3]

        def _prep_img(img: np.ndarray) -> np.ndarray:
            arr = np.asarray(img, dtype=np.float32)
            if arr.max() > 1.5:  # uint8-ish → scale to [-1, 1]
                arr = arr / 255.0
                arr = arr * 2.0 - 1.0
            if arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)[None, :]  # [1, 3, H, W]
            return arr.astype(np.float32)

        img_cam1 = _prep_img(images_list[0])
        img_cam2 = _prep_img(images_list[1])
        img_cam3 = _prep_img(images_list[2])

        mask = np.ones((1,), dtype=np.bool_)

        # Lang: tokenize (SmolLM2 vocab ~49152) or use provided tokens
        if lang_tokens is None:
            try:
                tok = self._get_tokenizer()
                enc = tok(
                    instruction or " ",
                    return_tensors="np", padding="max_length",
                    max_length=16, truncation=True,
                )
                lang_tokens = enc["input_ids"].astype(np.int64)
                lang_masks = enc["attention_mask"].astype(np.bool_)
            except Exception as e:
                # Fall back silently BUT LOUDLY — this breaks instruction-following
                # so we want it painfully visible in the log.
                logger.error(
                    "SEVERE: tokenizer failed (%s). Instruction '%s' has NO effect "
                    "on the output — actions are a function of state+images only. "
                    "Fix the tokenizer before trusting /act responses.", e, instruction,
                )
                lang_tokens = np.zeros((1, 16), dtype=np.int64)
                lang_masks = np.ones((1, 16), dtype=np.bool_)
        if lang_masks is None:
            lang_masks = np.ones_like(lang_tokens, dtype=np.bool_)

        # State: pad to config's max_state_dim (default 32)
        state_dim = int(self.config.get("max_state_dim", 32))
        if state is None:
            state_arr = np.zeros((1, state_dim), dtype=np.float32)
        else:
            state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
            if state_arr.shape[0] < state_dim:
                state_arr = np.concatenate(
                    [state_arr, np.zeros(state_dim - state_arr.shape[0])]
                )
            elif state_arr.shape[0] > state_dim:
                state_arr = state_arr[:state_dim]
            state_arr = state_arr[None, :].astype(np.float32)

        # Noise
        if noise is None:
            chunk = int(self.config.get("chunk_size", 50))
            action_dim = int(self.config.get("action_dim", 32))
            noise = np.random.RandomState(0).randn(1, chunk, action_dim).astype(np.float32)
        noise = np.asarray(noise, dtype=np.float32)

        ort_inputs = {
            "img_cam1": img_cam1,
            "img_cam2": img_cam2,
            "img_cam3": img_cam3,
            "mask_cam1": mask,
            "mask_cam2": mask,
            "mask_cam3": mask,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "state": state_arr,
            "noise": noise,
        }
        ort_inputs = {k: v for k, v in ort_inputs.items() if k in self._input_names}

        actions = self._session.run(None, ort_inputs)[0]  # [B, chunk, action_dim]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        actions_out = actions[0]

        # `denoising_steps` is the README-documented field; `num_denoising_steps`
        # is the internal config key name. Emit both for backwards compat.
        steps = int(self.config.get("num_denoising_steps", 1))
        return {
            "actions": actions_out.tolist(),
            "num_actions": int(actions_out.shape[0]),
            "action_dim": int(actions_out.shape[1]),
            "latency_ms": round(elapsed_ms, 1),
            "hz": round(1000.0 / elapsed_ms, 1) if elapsed_ms > 0 else 0,
            "inference_mode": self._inference_mode,
            "denoising_steps": steps,
            "num_denoising_steps": steps,
        }

    # --- create_app lifespan compat ---------------------------------------

    async def predict_from_base64_async(
        self,
        image_b64: str | None = None,
        instruction: str = "",
        state: list[float] | None = None,
    ) -> dict[str, Any]:
        import base64
        import io
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

    async def start_batch_worker(self) -> None:
        return None

    async def stop_batch_worker(self) -> None:
        return None
