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
    ):
        self.export_dir = Path(export_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_denoising_steps = num_denoising_steps
        self.config = self._load_config()
        self.model = None
        self._ready = False

    def _load_config(self) -> dict[str, Any]:
        config_path = self.export_dir / "reflex_config.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
        return {}

    def load(self) -> None:
        """Load the model from exported directory."""
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

        elapsed = time.perf_counter() - start
        self._ready = True
        logger.info("Model loaded in %.1fs, ready to serve", elapsed)

    def _load_onnx(self, onnx_path: Path) -> None:
        """Load ONNX model via onnxruntime."""
        try:
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
            self._inference_mode = "onnx"
            logger.info("Loaded ONNX model: %s (%s)", onnx_path.name, self._ort_session.get_providers())
        except ImportError:
            logger.warning("onnxruntime not installed, falling back to CPU-only")
            import onnxruntime as ort

            self._ort_session = ort.InferenceSession(str(onnx_path))
            self._inference_mode = "onnx_cpu"

    @property
    def ready(self) -> bool:
        return self._ready

    def predict(
        self,
        image: np.ndarray | None = None,
        instruction: str = "",
        state: list[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run inference: image + instruction + state → action chunk.

        For v0.1, runs the expert stack only (no VLM prefix encoding).
        VLM integration comes in a future version.

        Args:
            image: RGB image array [H, W, 3] or None
            instruction: text instruction (unused in v0.1 expert-only mode)
            state: robot state vector [N] or None

        Returns:
            dict with "actions" (list of action vectors), "latency_ms", "hz"
        """
        if not self._ready:
            return {"error": "Model not loaded. Call load() first."}

        start = time.perf_counter()

        # Prepare inputs for expert stack denoising
        noisy_actions = np.random.randn(1, self.chunk_size, self.action_dim).astype(np.float32)
        position_ids = np.arange(self.chunk_size, dtype=np.int64).reshape(1, -1)

        # Euler denoise loop
        dt = -1.0 / self.num_denoising_steps
        for step in range(self.num_denoising_steps):
            t = 1.0 + step * dt
            timestep = np.array([t], dtype=np.float32)

            if self._inference_mode.startswith("onnx"):
                velocity = self._ort_session.run(
                    None,
                    {
                        "noisy_actions": noisy_actions,
                        "timestep": timestep,
                        "position_ids": position_ids,
                    },
                )[0]
            else:
                return {"error": f"Unknown inference mode: {self._inference_mode}"}

            noisy_actions = noisy_actions + velocity * dt

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Convert to list for JSON serialization
        actions = noisy_actions[0].tolist()

        return {
            "actions": actions,
            "num_actions": len(actions),
            "action_dim": self.action_dim,
            "latency_ms": round(elapsed_ms, 1),
            "hz": round(1000.0 / elapsed_ms, 1) if elapsed_ms > 0 else 0,
            "denoising_steps": self.num_denoising_steps,
            "inference_mode": self._inference_mode,
        }

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


def create_app(export_dir: str, device: str = "cuda") -> Any:
    """Create a FastAPI app for serving VLA predictions."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("Install fastapi: pip install 'reflex-vla[serve]'")

    app = FastAPI(
        title="Reflex VLA Server",
        description="Deploy any VLA model to any edge hardware.",
        version="0.1.0",
    )

    server = ReflexServer(export_dir, device=device)

    class PredictRequest(BaseModel):
        image: str | None = None  # base64 encoded
        instruction: str = ""
        state: list[float] | None = None

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        inference_mode: str = ""
        export_dir: str = ""

    @app.on_event("startup")
    async def startup():
        server.load()

    @app.get("/health")
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if server.ready else "not_ready",
            model_loaded=server.ready,
            inference_mode=getattr(server, "_inference_mode", ""),
            export_dir=str(server.export_dir),
        )

    @app.post("/act")
    async def act(request: PredictRequest) -> JSONResponse:
        result = server.predict_from_base64(
            image_b64=request.image,
            instruction=request.instruction,
            state=request.state,
        )
        return JSONResponse(content=result)

    @app.get("/config")
    async def config():
        return JSONResponse(content=server.config)

    return app
