import pytest
import subprocess

from tether.eval.checkpoints import CheckpointError, CheckpointSpec, resolve_checkpoint
from tether.eval.libero import LiberoSuiteConfig
from tether.eval.local_runner import run_local_libero
from tether.eval.modal_runner import run_libero_on_modal


def test_lora_identity_includes_adapter_files_and_base(tmp_path):
    (tmp_path / "adapter_config.json").write_text('{"base_model_name_or_path":"org/base"}')
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    spec = resolve_checkpoint(tmp_path)
    assert spec.kind == "smolvla-lora"
    assert spec.base == "org/base"
    assert spec.identity.startswith("sha256:")
    assert {item["path"] for item in spec.files} == {"adapter_config.json", "adapter_model.safetensors"}


def test_remote_checkpoint_requires_revision():
    with pytest.raises(CheckpointError, match="revision"):
        resolve_checkpoint("org/model")
    assert resolve_checkpoint("org/model", revision="abc123").identity == "hf:org/model@abc123"


def test_local_runner_passes_exact_cases_and_preserves_real_outcomes():
    captured = {}
    checkpoint = CheckpointSpec("full", "org/model", "hf:org/model@abc", revision="abc")

    def loader(spec):
        assert spec is checkpoint
        return object(), object(), object()

    def rollout(**kwargs):
        captured.update(kwargs)
        return {"per_task": [{"task_idx": 2, "episodes": [
            {"ep": 0, "success": True, "steps": 12},
            {"ep": 1, "success": False, "steps": 220},
        ]}], "errors": []}

    config = LiberoSuiteConfig(tasks=("libero_spatial",), task_indices=(2,), num_episodes=2, seed=41, runtime="local")
    report = run_local_libero(config, checkpoint, loader=loader, rollout=rollout)
    assert captured["task_indices"] == [2]
    assert captured["seed"] == 41
    assert [episode.success for episode in report.results[0].episodes] == [True, False]
    assert report.results[0].episodes[1].n_steps == 220


def test_modal_command_names_selected_adapter_and_never_uses_reference(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "modal_libero_lerobot_native.py").write_text("# fixture")
    captured = []

    def invoke(command, timeout):
        captured.append(command)
        return subprocess.CompletedProcess(command, 1, "", "fixture stop")

    spec = resolve_checkpoint("org/candidate", kind="smolvla-lora", base="org/base", revision="adapter-rev")
    run_libero_on_modal(
        config=LiberoSuiteConfig(tasks=("libero_10",), task_indices=(2,), num_episodes=1),
        checkpoint=spec, repo_root=tmp_path, modal_invoker=invoke,
    )
    command = captured[0]
    assert command[command.index("--model-id") + 1] == "org/candidate"
    assert command[command.index("--adapter-path") + 1] == "org/candidate"
    assert command[command.index("--adapter-base") + 1] == "org/base"
    assert "HuggingFaceVLA/smolvla_libero" not in command
