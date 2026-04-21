"""Modal: decomposed pi0.5 export (vlm_prefix.onnx + expert_denoise.onnx).

Design: reflex_context/reflex_vla/01_architecture/prefix_kv_cache_reuse_design.md

Usage:
    # SnapFlow student (num_steps=1, target_time=1 path):
    modal run scripts/modal_export_pi05_decomposed.py \\
      --student-checkpoint /onnx_out/distill_v031_pi05_libero_r4/training/checkpoints/00010000/pretrained_model \\
      --output-subdir distill_v031_pi05_libero_r4/decomposed

    # pi0.5 teacher (num_steps=10):
    modal run scripts/modal_export_pi05_decomposed.py \\
      --model-id lerobot/pi05_libero_finetuned_v044 \\
      --num-steps 10 \\
      --output-subdir pi05_libero_finetuned_v044/decomposed_num_steps_10
"""
import os
import subprocess
import modal

app = modal.App("reflex-export-pi05-decomposed")


def _hf_secret():
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    try:
        return modal.Secret.from_name("huggingface")
    except Exception:
        return modal.Secret.from_dict({})


def _repo_head_sha() -> str:
    try:
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "main"


def _build_bust() -> str:
    import time
    return str(int(time.time()))


_HEAD = _repo_head_sha()
_BUST = _build_bust()

hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"
ONNX_OUT = "/onnx_out"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "lerobot==0.5.1",
        "transformers==5.3.0",
        "num2words",
        "safetensors>=0.4.0",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "onnx-diagnostic>=0.9",
        "optree",
        "scipy",
        "numpy",
        "accelerate",
        "draccus",
    )
    .run_commands(
        f'echo "build_bust={_BUST}"',
        f'pip install "reflex-vla[monolithic] @ git+https://x-access-token:$GITHUB_TOKEN@github.com/rylinjames/reflex-vla@{_HEAD}"',
        secrets=[modal.Secret.from_name("github-token")],
    )
    .env({
        "HF_HOME": HF_CACHE,
        "TRANSFORMERS_CACHE": f"{HF_CACHE}/transformers",
    })
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=10800,
    volumes={HF_CACHE: hf_cache, ONNX_OUT: onnx_output},
    secrets=[_hf_secret()],
)
def export_decomposed_modal(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    output_subdir: str = "pi05_decomposed_smoke",
    num_steps: int = 1,
    student_checkpoint: str = "",
):
    """Run the decomposed export on Modal."""
    import logging
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from reflex.exporters.decomposed import export_pi05_decomposed

    out = Path(ONNX_OUT) / output_subdir
    out.mkdir(parents=True, exist_ok=True)

    student = Path(student_checkpoint) if student_checkpoint else None
    if student and not student.exists():
        raise FileNotFoundError(f"student checkpoint missing: {student}")

    result = export_pi05_decomposed(
        model_id=model_id,
        output_dir=str(out),
        num_steps=num_steps,
        student_checkpoint=str(student) if student else None,
    )
    onnx_output.commit()

    import onnx
    for name in ("vlm_prefix.onnx", "expert_denoise.onnx"):
        m = onnx.load(str(out / name), load_external_data=False)
        print(f"[{name}] inputs: {[i.name for i in m.graph.input][:6]}...")
        print(f"[{name}] outputs: {[o.name for o in m.graph.output][:6]}...")
    return result


@app.local_entrypoint()
def main(
    model_id: str = "lerobot/pi05_libero_finetuned_v044",
    output_subdir: str = "pi05_decomposed_smoke",
    num_steps: int = 1,
    student_checkpoint: str = "",
):
    r = export_decomposed_modal.remote(
        model_id=model_id,
        output_subdir=output_subdir,
        num_steps=num_steps,
        student_checkpoint=student_checkpoint,
    )
    print("\n=== RESULT ===")
    for k, v in r.items():
        print(f"  {k}: {v}")
