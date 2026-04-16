"""Tests for VLM prefix pipeline: export, expert wiring, server integration.

Covers the full pipeline from vlm_prefix export through server predict(),
using mocks/stubs to avoid loading real 450M checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from reflex.exporters.vlm_prefix_exporter import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_INSTRUCTION_SEQ_LEN,
    DEFAULT_PREFIX_SEQ_LEN,
    DEFAULT_VLM_KV_DIM,
    VLMPrefixEncoder,
    export_vlm_prefix,
)
from reflex.exporters.smolvla_exporter import ExpertGQALayer, ExpertStack
from reflex.runtime.server import ReflexServer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_expert_stack():
    """Build a minimal ExpertStack with 2 layers, 1 cross-attention layer."""
    hidden = 64
    action_dim = 8
    nq, nkv, hd = 4, 2, 16
    inter = 128
    vlm_kv_dim = 32
    cross_indices = [1]  # layer 1 is cross-attention

    layers = []
    for i in range(2):
        kv_in = vlm_kv_dim if i in cross_indices else None
        layers.append(ExpertGQALayer(hidden, nq, nkv, hd, inter, kv_in=kv_in))

    suffix_weights = {
        "in_w": torch.randn(hidden, action_dim),
        "in_b": torch.randn(hidden),
        "t_in_w": torch.randn(hidden, hidden * 2),
        "t_in_b": torch.randn(hidden),
        "t_out_w": torch.randn(hidden, hidden),
        "t_out_b": torch.randn(hidden),
    }
    action_proj_weights = {
        "w": torch.randn(action_dim, hidden),
        "b": torch.randn(action_dim),
    }
    final_norm_weight = torch.ones(hidden)

    stack = ExpertStack(
        layers=layers,
        expert_hidden=hidden,
        action_dim=action_dim,
        cross_indices=cross_indices,
        vlm_kv_dim=vlm_kv_dim,
        suffix_weights=suffix_weights,
        action_proj_weights=action_proj_weights,
        final_norm_weight=final_norm_weight,
    )
    stack.eval()
    return stack, {"hidden": hidden, "action_dim": action_dim, "vlm_kv_dim": vlm_kv_dim}


@pytest.fixture
def v02_export_dir(tmp_path):
    """Create a mock v0.2 export dir with both expert_stack.onnx and vlm_prefix.onnx."""
    config = {
        "model_id": "lerobot/smolvla_base",
        "target": "desktop",
        "action_chunk_size": 10,
        "export_version": "0.2",
        "vlm_prefix_onnx": "vlm_prefix.onnx",
        "vlm_image_size": [384, 384],
        "vlm_kv_dim": 960,
        "vlm_prefix_seq_len": 50,
        "expert": {
            "expert_hidden": 720,
            "action_dim": 32,
            "num_layers": 16,
        },
    }
    (tmp_path / "reflex_config.json").write_text(json.dumps(config))
    # Create placeholder ONNX files (real loading is mocked)
    (tmp_path / "expert_stack.onnx").write_bytes(b"fake-onnx")
    (tmp_path / "vlm_prefix.onnx").write_bytes(b"fake-onnx")
    return tmp_path


@pytest.fixture
def v01_export_dir(tmp_path):
    """Create a mock v0.1 export dir (no vlm_prefix)."""
    config = {
        "model_id": "lerobot/smolvla_base",
        "target": "desktop",
        "action_chunk_size": 10,
        "expert": {
            "expert_hidden": 720,
            "action_dim": 32,
            "num_layers": 16,
        },
    }
    (tmp_path / "reflex_config.json").write_text(json.dumps(config))
    (tmp_path / "expert_stack.onnx").write_bytes(b"fake-onnx")
    return tmp_path


def _make_mock_ort_session(input_names, output_shape):
    """Create a mock ORT InferenceSession that returns random data."""
    session = MagicMock()
    inputs = []
    for name in input_names:
        inp = MagicMock()
        inp.name = name
        inputs.append(inp)
    session.get_inputs.return_value = inputs
    session.get_providers.return_value = ["CPUExecutionProvider"]

    def mock_run(output_names, feed_dict):
        return [np.random.randn(*output_shape).astype(np.float32)]

    session.run.side_effect = mock_run
    return session


# ---------------------------------------------------------------------------
# Test 1: VLM prefix export produces valid ONNX
# ---------------------------------------------------------------------------

class TestVLMPrefixExport:
    def test_vlm_prefix_export_produces_valid_onnx(self, tmp_path):
        """Export VLMPrefixEncoder, verify ONNX loads in ORT with correct I/O."""
        import onnxruntime as ort

        # Build and export the stub encoder directly (bypass checkpoint loading)
        encoder = VLMPrefixEncoder(vlm_kv_dim=DEFAULT_VLM_KV_DIM)
        encoder.eval()

        onnx_path = tmp_path / "vlm_prefix.onnx"
        dummy_image = torch.randn(1, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3)
        dummy_ids = torch.randint(0, 1000, (1, DEFAULT_INSTRUCTION_SEQ_LEN), dtype=torch.int64)

        with torch.no_grad():
            torch.onnx.export(
                encoder,
                (dummy_image, dummy_ids),
                str(onnx_path),
                input_names=["image", "instruction_ids"],
                output_names=["prefix_kv"],
                dynamic_axes={
                    "image": {0: "batch"},
                    "instruction_ids": {0: "batch", 1: "seq"},
                    "prefix_kv": {0: "batch"},
                },
                opset_version=19,
            )

        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

        # Load in ORT
        sess = ort.InferenceSession(str(onnx_path))
        input_names = [inp.name for inp in sess.get_inputs()]
        output_names = [out.name for out in sess.get_outputs()]
        assert "image" in input_names
        assert "instruction_ids" in input_names
        assert "prefix_kv" in output_names

        # Run and check output shape
        result = sess.run(None, {
            "image": dummy_image.numpy(),
            "instruction_ids": dummy_ids.numpy(),
        })[0]
        assert result.shape == (1, DEFAULT_PREFIX_SEQ_LEN, DEFAULT_VLM_KV_DIM)

        # Numerical validation: PyTorch vs ORT
        with torch.no_grad():
            torch_out = encoder(dummy_image, dummy_ids).numpy()
        max_diff = float(np.abs(torch_out - result).max())
        assert max_diff < 1e-4, f"ONNX/PyTorch max_diff={max_diff} exceeds 1e-4"


# ---------------------------------------------------------------------------
# Test 2: Expert with real vlm_kv differs from zeros
# ---------------------------------------------------------------------------

class TestExpertVLMKV:
    def test_expert_with_real_vlm_kv_differs_from_zeros(self, tiny_expert_stack):
        """Cross-attention actually uses vlm_kv: real vs zeros should differ."""
        stack, meta = tiny_expert_stack
        vlm_kv_dim = meta["vlm_kv_dim"]
        action_dim = meta["action_dim"]

        noisy_actions = torch.randn(1, 5, action_dim)
        timestep = torch.tensor([0.5])
        position_ids = torch.arange(5).unsqueeze(0)

        # Real vlm_kv
        real_vlm_kv = torch.randn(1, 10, vlm_kv_dim)
        with torch.no_grad():
            out_real = stack(noisy_actions, timestep, position_ids, vlm_kv=real_vlm_kv)

        # Zero vlm_kv (explicit)
        zero_vlm_kv = torch.zeros(1, 10, vlm_kv_dim)
        with torch.no_grad():
            out_zero = stack(noisy_actions, timestep, position_ids, vlm_kv=zero_vlm_kv)

        l2_diff = torch.norm(out_real - out_zero).item()
        assert l2_diff > 1e-3, (
            f"Expert outputs should differ with real vs zero vlm_kv but L2={l2_diff}"
        )

    def test_expert_vlm_kv_none_fallback(self, tiny_expert_stack):
        """When vlm_kv=None, expert falls back to zeros without error."""
        stack, meta = tiny_expert_stack
        noisy_actions = torch.randn(1, 5, meta["action_dim"])
        timestep = torch.tensor([0.5])
        position_ids = torch.arange(5).unsqueeze(0)

        with torch.no_grad():
            out = stack(noisy_actions, timestep, position_ids, vlm_kv=None)

        assert out.shape == (1, 5, meta["action_dim"])


# ---------------------------------------------------------------------------
# Test 3: Server predict() with VLM returns vlm_conditioning="real"
# ---------------------------------------------------------------------------

class TestServerVLMConditioning:
    def test_server_vlm_conditioning_real(self, v02_export_dir):
        """Server with vlm_prefix.onnx returns vlm_conditioning='real'."""
        expert_session = _make_mock_ort_session(
            ["noisy_actions", "timestep", "position_ids", "vlm_kv"],
            (1, 10, 32),
        )
        vlm_session = _make_mock_ort_session(
            ["image", "instruction_ids"],
            (1, 50, 960),
        )

        server = ReflexServer(v02_export_dir, device="cpu")
        # Manually wire the server instead of calling load() which does real ort import
        server.action_dim = 32
        server.chunk_size = 10
        server.expert_hidden = 720
        server._inference_mode = "onnx_cpu"
        server._ready = True
        server._ort_session = expert_session
        server._vlm_session = vlm_session
        server._vlm_loaded = True
        server._expert_input_names = ["noisy_actions", "timestep", "position_ids", "vlm_kv"]

        # Provide image and instruction to trigger VLM path
        fake_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = server.predict(image=fake_image, instruction="pick up the cup")

        assert result["vlm_conditioning"] == "real"
        assert "actions" in result
        assert vlm_session.run.call_count == 1


# ---------------------------------------------------------------------------
# Test 4: Backward compat — v0.1 export dir (no vlm_prefix)
# ---------------------------------------------------------------------------

class TestServerBackwardCompat:
    def test_server_backward_compat_v01(self, v01_export_dir):
        """Server with v0.1 export dir returns vlm_conditioning='dummy'."""
        expert_session = _make_mock_ort_session(
            ["noisy_actions", "timestep", "position_ids"],
            (1, 10, 32),
        )

        server = ReflexServer(v01_export_dir, device="cpu")
        # Wire manually
        server.action_dim = 32
        server.chunk_size = 10
        server.expert_hidden = 720
        server._inference_mode = "onnx_cpu"
        server._ready = True
        server._ort_session = expert_session
        server._vlm_session = None
        server._vlm_loaded = False
        server._expert_input_names = ["noisy_actions", "timestep", "position_ids"]

        result = server.predict()

        assert result["vlm_conditioning"] == "dummy"
        assert "actions" in result
        assert "error" not in result

    def test_v01_with_image_still_dummy(self, v01_export_dir):
        """Even with image+instruction, v0.1 server stays dummy (no VLM session)."""
        expert_session = _make_mock_ort_session(
            ["noisy_actions", "timestep", "position_ids"],
            (1, 10, 32),
        )

        server = ReflexServer(v01_export_dir, device="cpu")
        server.action_dim = 32
        server.chunk_size = 10
        server.expert_hidden = 720
        server._inference_mode = "onnx_cpu"
        server._ready = True
        server._ort_session = expert_session
        server._vlm_session = None
        server._vlm_loaded = False
        server._expert_input_names = ["noisy_actions", "timestep", "position_ids"]

        fake_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = server.predict(image=fake_image, instruction="pick up the cup")

        assert result["vlm_conditioning"] == "dummy"


# ---------------------------------------------------------------------------
# Test 5: Config v0.2 schema loads with all new fields
# ---------------------------------------------------------------------------

class TestConfigV02Schema:
    def test_config_v02_schema(self, tmp_path):
        """Config with v0.2 fields loads and has correct types."""
        config = {
            "model_id": "lerobot/smolvla_base",
            "export_version": "0.2",
            "vlm_prefix_onnx": "vlm_prefix.onnx",
            "vlm_image_size": [384, 384],
            "vlm_kv_dim": 960,
            "vlm_prefix_seq_len": 50,
            "expert": {
                "expert_hidden": 720,
                "action_dim": 32,
            },
        }
        config_path = tmp_path / "reflex_config.json"
        config_path.write_text(json.dumps(config))

        loaded = json.loads(config_path.read_text())

        assert loaded["export_version"] == "0.2"
        assert loaded["vlm_prefix_onnx"] == "vlm_prefix.onnx"
        assert isinstance(loaded["vlm_image_size"], list)
        assert len(loaded["vlm_image_size"]) == 2
        assert all(isinstance(x, int) for x in loaded["vlm_image_size"])
        assert isinstance(loaded["vlm_kv_dim"], int)
        assert loaded["vlm_kv_dim"] == 960
        assert isinstance(loaded["vlm_prefix_seq_len"], int)

    def test_v01_config_missing_vlm_fields(self, tmp_path):
        """v0.1 config loads fine — missing VLM fields default gracefully."""
        config = {
            "model_id": "lerobot/smolvla_base",
            "expert": {"action_dim": 32},
        }
        config_path = tmp_path / "reflex_config.json"
        config_path.write_text(json.dumps(config))

        loaded = json.loads(config_path.read_text())

        # These should be absent, and consumers use .get() with defaults
        assert "vlm_prefix_onnx" not in loaded
        assert "export_version" not in loaded
        assert loaded.get("vlm_prefix_onnx") is None
        assert loaded.get("vlm_kv_dim", 960) == 960


# ---------------------------------------------------------------------------
# Test 6: export_vlm_prefix updates config file
# ---------------------------------------------------------------------------

class TestVLMPrefixExporterUpdatesConfig:
    def test_vlm_prefix_exporter_updates_config(self, tmp_path):
        """export_vlm_prefix writes vlm_image_size, vlm_kv_dim, vlm_prefix_onnx to config."""
        # Pre-populate a v0.1 config
        initial_config = {
            "model_id": "lerobot/smolvla_base",
            "expert": {"action_dim": 32},
        }
        (tmp_path / "reflex_config.json").write_text(json.dumps(initial_config))

        # Mock load_checkpoint to return a fake state_dict + config
        fake_state_dict = {"some.weight": torch.randn(10, 10)}
        fake_model_config = {"vision_config": {"image_size": 384}}

        with patch("reflex.exporters.vlm_prefix_exporter.load_checkpoint") as mock_load:
            mock_load.return_value = (fake_state_dict, fake_model_config)

            onnx_path = export_vlm_prefix("fake/checkpoint", tmp_path)

        assert onnx_path.exists()
        assert onnx_path.name == "vlm_prefix.onnx"

        # Verify config was updated
        updated_config = json.loads((tmp_path / "reflex_config.json").read_text())
        assert updated_config["vlm_prefix_onnx"] == "vlm_prefix.onnx"
        assert updated_config["vlm_image_size"] == [384, 384]
        assert updated_config["vlm_kv_dim"] == DEFAULT_VLM_KV_DIM
        assert updated_config["export_version"] == "0.2"
        assert updated_config["vlm_prefix_seq_len"] == DEFAULT_PREFIX_SEQ_LEN

        # Original fields preserved
        assert updated_config["model_id"] == "lerobot/smolvla_base"
        assert updated_config["expert"]["action_dim"] == 32
