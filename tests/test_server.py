"""Tests for the VLA inference server."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from reflex.runtime.server import ReflexServer


@pytest.fixture
def mock_export_dir(tmp_path):
    """Create a mock export directory with config."""
    config = {
        "model_id": "lerobot/smolvla_base",
        "target": "desktop",
        "action_chunk_size": 50,
        "expert": {
            "expert_hidden": 720,
            "action_dim": 32,
            "num_layers": 16,
        },
    }
    config_path = tmp_path / "reflex_config.json"
    config_path.write_text(json.dumps(config))
    return tmp_path


class TestReflexServer:
    def test_loads_config(self, mock_export_dir):
        server = ReflexServer(mock_export_dir, device="cpu")
        assert server.config["model_id"] == "lerobot/smolvla_base"
        assert server.config["expert"]["action_dim"] == 32

    def test_not_ready_before_load(self, mock_export_dir):
        server = ReflexServer(mock_export_dir, device="cpu")
        assert not server.ready

    def test_predict_before_load_returns_error(self, mock_export_dir):
        server = ReflexServer(mock_export_dir, device="cpu")
        result = server.predict()
        assert "error" in result

    def test_loads_with_missing_onnx(self, mock_export_dir):
        server = ReflexServer(mock_export_dir, device="cpu")
        server.load()
        assert not server.ready  # No ONNX file, so not ready


class TestReflexServerWithMockORT:
    def test_predict_returns_actions(self, mock_export_dir):
        server = ReflexServer(mock_export_dir, device="cpu")
        server.action_dim = 32
        server.chunk_size = 50
        server.expert_hidden = 720
        server._inference_mode = "onnx"
        server._ready = True

        # Mock ORT session
        mock_session = MagicMock()
        mock_session.run.return_value = [np.random.randn(1, 50, 32).astype(np.float32)]
        server._ort_session = mock_session

        result = server.predict()

        assert "actions" in result
        assert result["num_actions"] == 50
        assert result["action_dim"] == 32
        assert result["latency_ms"] > 0
        assert result["hz"] > 0
        assert result["denoising_steps"] == 10
        assert mock_session.run.call_count == 10  # 10 denoising steps

    def test_predict_action_shape(self, mock_export_dir):
        server = ReflexServer(mock_export_dir, device="cpu")
        server.action_dim = 6
        server.chunk_size = 20
        server.expert_hidden = 720
        server._inference_mode = "onnx"
        server._ready = True

        mock_session = MagicMock()
        mock_session.run.return_value = [np.random.randn(1, 20, 6).astype(np.float32)]
        server._ort_session = mock_session

        result = server.predict()
        assert len(result["actions"]) == 20
        assert len(result["actions"][0]) == 6


class TestCreateApp:
    def test_app_creates(self, mock_export_dir):
        try:
            from reflex.runtime.server import create_app
            app = create_app(str(mock_export_dir), device="cpu")
            assert app is not None
            assert app.title == "Reflex VLA Server"
        except ImportError:
            pytest.skip("fastapi not installed")
