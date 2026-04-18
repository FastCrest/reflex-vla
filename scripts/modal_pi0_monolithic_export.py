"""Modal: pi0 monolithic ONNX export via onnx-diagnostic torch_export_patches.

Uses A10G + persistent Volume to cache pi0_base weights between runs.
onnx-diagnostic's `torch_export_patches(patch_transformers=True)` teaches
torch.export.export how to flatten DynamicCache as PyTree nodes, which is
the research-identified path for Gemma-family DynamicCache ONNX export.

Iteration: ~1-2 min/run after first (pi0_base cached in Volume).

Usage:
    # One-time setup: register your HF token as a Modal secret
    modal secret create hf-token HF_TOKEN=hf_xxxxx
    # Or export locally: `export HF_TOKEN=hf_xxxxx` (picked up as fallback)

    modal run scripts/modal_pi0_monolithic_export.py

Output: /tmp/pi0_monolithic_onnx/model.onnx in the Modal container. Fetch
via modal.Volume or print verdict + cos number for parity test.
"""
import os
import modal

app = modal.App("pi0-monolithic-export")


def _hf_secret():
    """Return a Modal Secret carrying HF_TOKEN.

    Order: local `HF_TOKEN` env var (dev convenience) → Modal named secret
    `hf-token` (production). Never hardcode the token in source.
    """
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name("hf-token")

# Persistent volume for HF cache (pi0_base is ~14GB)
hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
# Persistent volume for ONNX outputs
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)

HF_CACHE_PATH = "/root/.cache/huggingface"
ONNX_OUTPUT_PATH = "/onnx_out"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # Install lerobot first — it pins torch>=2.7 and hub>=1.0
        "lerobot==0.5.1",
        "num2words",
        "safetensors>=0.4.0",
        "onnx>=1.16",
        "onnxruntime>=1.20",
        "onnxscript>=0.1",
        "onnx-diagnostic>=0.9",
        "optree",  # onnx-diagnostic soft-dep
        "scipy",   # onnx-diagnostic patches_transformers_qwen2_5 transitively
        "numpy",
        "accelerate",
        "draccus",
    )
    # transformers 5.x accepts huggingface-hub>=1.0 (which lerobot pins)
    .pip_install("transformers>=5.0,<6.0")
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={
        HF_CACHE_PATH: hf_cache,
        ONNX_OUTPUT_PATH: onnx_output,
    },
    secrets=[_hf_secret()],
)
def export_pi0_monolithic_modal(
    model_id: str = "lerobot/pi0_base",
    num_steps: int = 10,
):
    """Export pi0 monolithic ONNX using onnx-diagnostic torch_export_patches."""
    import os
    import sys
    import types
    import time
    from pathlib import Path

    import torch
    import torch.nn as nn

    # Python 3.13 + lerobot 0.5.1 compat shim (same as local)
    for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
        stub = types.ModuleType(_mod)
        stub.GrootPolicy = None
        stub.GR00TN15 = None
        sys.modules[_mod] = stub

    # ---- Transformers 4.57+ compat monkey-patches ----
    from lerobot.policies.pi0 import modeling_pi0
    from transformers import masking_utils
    from lerobot.policies import pi_gemma as _pg

    _orig_embed = modeling_pi0.PaliGemmaWithExpertModel.embed_image

    def _patched_embed_image(self, image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        out = self.paligemma.model.get_image_features(image)
        if hasattr(out, "pooler_output"):
            features = out.pooler_output
        else:
            features = out
        features = features * self.paligemma.config.text_config.hidden_size ** 0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    modeling_pi0.PaliGemmaWithExpertModel.embed_image = _patched_embed_image

    _orig_ccm = masking_utils.create_causal_mask

    def _ccm_shim(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return _orig_ccm(*args, **kwargs)

    masking_utils.create_causal_mask = _ccm_shim
    if hasattr(_pg, "create_causal_mask"):
        _pg.create_causal_mask = _ccm_shim

    # Patch denoise_step to skip copy.deepcopy(past_key_values) — the deepcopy
    # fails under torch.export FakeTensor tracing ("Cannot access data pointer").
    # Safe to skip: use_cache=False means the forward doesn't mutate the cache.
    import copy as _copy
    _orig_deepcopy = _copy.deepcopy

    def _safe_deepcopy(obj, *args, **kwargs):
        # Pass-through for DynamicCache during tracing — we're read-only anyway
        from transformers.cache_utils import DynamicCache
        if isinstance(obj, DynamicCache):
            return obj
        return _orig_deepcopy(obj, *args, **kwargs)

    _copy.deepcopy = _safe_deepcopy

    # ---- Load policy ----
    print(f"[modal] Loading {model_id}...")
    t0 = time.time()
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    policy = PI0Policy.from_pretrained(model_id).eval().to("cpu").to(torch.float32)
    print(f"[modal] Loaded in {time.time() - t0:.1f}s")

    # ---- Build wrapper ----
    class Pi0MonolithicWrapper(nn.Module):
        def __init__(self, pi0_pytorch_model, num_steps=10):
            super().__init__()
            self.model = pi0_pytorch_model
            self.num_steps = num_steps

        def forward(
            self,
            img_base, img_wrist_l, img_wrist_r,
            mask_base, mask_wrist_l, mask_wrist_r,
            lang_tokens, lang_masks,
            state, noise,
        ):
            images = [img_base, img_wrist_l, img_wrist_r]
            img_masks = [mask_base, mask_wrist_l, mask_wrist_r]
            return self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state,
                noise=noise, num_steps=self.num_steps,
            )

    wrapper = Pi0MonolithicWrapper(policy.model, num_steps=num_steps).eval()

    # ---- Dummy inputs ----
    cfg = policy.config
    B = 1
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim
    state_dim = getattr(cfg, "max_state_dim", 32)

    dummy = dict(
        img_base=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        img_wrist_l=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        img_wrist_r=torch.randn(B, 3, 224, 224, dtype=torch.float32),
        mask_base=torch.ones(B, dtype=torch.bool),
        mask_wrist_l=torch.ones(B, dtype=torch.bool),
        mask_wrist_r=torch.ones(B, dtype=torch.bool),
        lang_tokens=torch.randint(0, 257152, (B, 16), dtype=torch.long),
        lang_masks=torch.ones(B, 16, dtype=torch.bool),
        state=torch.randn(B, state_dim, dtype=torch.float32),
        noise=torch.randn(B, chunk, action_dim, dtype=torch.float32),
    )

    # ---- Export via onnx-diagnostic torch_export_patches ----
    output_dir = Path(ONNX_OUTPUT_PATH) / "monolithic"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    print("[modal] torch_export_patches(patch_transformers=True) + torch.export.export...")
    t0 = time.time()

    # onnx-diagnostic 0.9.x expects MambaCache which was removed in transformers 5.x.
    # Stub it before onnx-diagnostic imports.
    try:
        from transformers.models.mamba.modeling_mamba import MambaCache as _MambaCache  # noqa
    except ImportError:
        import transformers.cache_utils as _cu
        if not hasattr(_cu, "MambaCache"):
            class _MambaCacheStub:
                pass
            _cu.MambaCache = _MambaCacheStub
            try:
                import transformers.models.mamba.modeling_mamba as _mm
                if not hasattr(_mm, "MambaCache"):
                    _mm.MambaCache = _MambaCacheStub
            except ImportError:
                pass

    try:
        from onnx_diagnostic.torch_export_patches import torch_export_patches
    except ImportError as e:
        print(f"[modal] onnx-diagnostic not available: {e}")
        raise

    # Dynamic shapes declaration (torch.export style)
    dynamic_shapes = None  # start simple — try static first, then add Dim if needed

    with torch_export_patches(patch_transformers=True):
        ep = torch.export.export(
            wrapper,
            tuple(dummy.values()),
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
    print(f"[modal] torch.export took {time.time() - t0:.1f}s")

    # ---- Convert ExportedProgram to ONNX ----
    t0 = time.time()
    input_names = list(dummy.keys())
    try:
        # Prefer the newer ONNX conversion path
        onnx_program = torch.onnx.export(
            ep,
            tuple(dummy.values()),
            str(onnx_path),
            input_names=input_names,
            output_names=["actions"],
            opset_version=19,
        )
    except Exception as e:
        print(f"[modal] ONNX conversion failed: {e}")
        raise
    print(f"[modal] ONNX conversion took {time.time() - t0:.1f}s")

    # Commit to volume
    onnx_output.commit()

    # Report
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / 1e6
        print(f"[modal] SUCCESS: {onnx_path} ({size_mb:.1f}MB)")
        # Also check for external data
        data_files = list(output_dir.glob("*.data"))
        total_mb = sum(f.stat().st_size for f in data_files) / 1e6 + size_mb
        print(f"[modal] Total on disk: {total_mb:.1f}MB (incl {len(data_files)} data files)")
        return {
            "status": "ok",
            "onnx_path": str(onnx_path),
            "size_mb": total_mb,
        }
    else:
        return {"status": "fail", "reason": "onnx file not created"}


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={
        HF_CACHE_PATH: hf_cache,
        ONNX_OUTPUT_PATH: onnx_output,
    },
    secrets=[_hf_secret()],
)
def parity_test_monolithic(model_id: str = "lerobot/pi0_base"):
    """Parity test: PyTorch pi0 vs monolithic ONNX at num_steps=1.

    Feeds identical seeded inputs to both PyTorch sample_actions(num_steps=1)
    and our Modal-exported ONNX. Target: cos >= 0.999.
    """
    import sys, types
    for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
        stub = types.ModuleType(_mod); stub.GrootPolicy = None; stub.GR00TN15 = None
        sys.modules[_mod] = stub

    import numpy as np
    import torch
    import onnxruntime as ort
    from pathlib import Path

    # Apply same patches as export (so PyTorch ref runs the same way)
    from lerobot.policies.pi0 import modeling_pi0
    from transformers import masking_utils
    from lerobot.policies import pi_gemma as _pg

    _orig_embed = modeling_pi0.PaliGemmaWithExpertModel.embed_image
    def _patched_embed_image(self, image):
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        out = self.paligemma.model.get_image_features(image)
        features = out.pooler_output if hasattr(out, "pooler_output") else out
        features = features * self.paligemma.config.text_config.hidden_size ** 0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features
    modeling_pi0.PaliGemmaWithExpertModel.embed_image = _patched_embed_image

    _orig_ccm = masking_utils.create_causal_mask
    def _ccm_shim(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return _orig_ccm(*args, **kwargs)
    masking_utils.create_causal_mask = _ccm_shim
    if hasattr(_pg, "create_causal_mask"):
        _pg.create_causal_mask = _ccm_shim

    # Load PyTorch reference
    print("[parity] Loading PyTorch pi0...")
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    policy = PI0Policy.from_pretrained(model_id).eval().to(torch.float32).to("cpu")

    # Build same dummy inputs as export
    B = 1
    cfg = policy.config
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim
    state_dim = getattr(cfg, "max_state_dim", 32)

    torch.manual_seed(42)
    img = torch.randn(B, 3, 224, 224, dtype=torch.float32)
    mask = torch.ones(B, dtype=torch.bool)
    lang_tokens = torch.randint(0, 257152, (B, 16), dtype=torch.long)
    lang_masks = torch.ones(B, 16, dtype=torch.bool)
    state = torch.randn(B, state_dim, dtype=torch.float32)
    noise = torch.randn(B, chunk, action_dim, dtype=torch.float32)

    # Run PyTorch ref at num_steps=1
    print("[parity] Running PyTorch ref (num_steps=1)...")
    images = [img, img, img]
    img_masks = [mask, mask, mask]
    with torch.no_grad():
        pt_actions = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            noise=noise, num_steps=1,
        )
    pt_np = pt_actions.cpu().numpy()
    print(f"[parity] pt actions: {pt_np.shape}, first: {pt_np[0, 0, :5]}")

    # Run ONNX
    print("[parity] Running ONNX...")
    onnx_path = Path(ONNX_OUTPUT_PATH) / "monolithic" / "model.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "img_base": img.numpy(),
        "img_wrist_l": img.numpy(),
        "img_wrist_r": img.numpy(),
        "mask_base": mask.numpy(),
        "mask_wrist_l": mask.numpy(),
        "mask_wrist_r": mask.numpy(),
        "lang_tokens": lang_tokens.numpy().astype(np.int64),
        "lang_masks": lang_masks.numpy(),
        "state": state.numpy(),
        "noise": noise.numpy(),
    }
    ort_out = sess.run(None, ort_inputs)[0]
    print(f"[parity] onnx actions: {ort_out.shape}, first: {ort_out[0, 0, :5]}")

    # Compare
    pt0 = pt_np[0, 0]
    on0 = ort_out[0, 0]
    max_abs = float(np.abs(pt0 - on0).max())
    cos = float(np.dot(pt0, on0) / (np.linalg.norm(pt0) * np.linalg.norm(on0) + 1e-8))
    full_cos = float(
        np.dot(pt_np.flatten(), ort_out.flatten())
        / (np.linalg.norm(pt_np) * np.linalg.norm(ort_out) + 1e-8)
    )
    full_max = float(np.abs(pt_np - ort_out).max())

    print(f"\n====== PARITY (num_steps=1) ======")
    print(f"  first-action max_abs: {max_abs:.4e}")
    print(f"  first-action cos:     {cos:+.6f}")
    print(f"  full chunk max_abs:   {full_max:.4e}")
    print(f"  full chunk cos:       {full_cos:+.6f}")
    passed = full_cos >= 0.999 and full_max < 0.1
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    return {
        "status": "ok",
        "first_cos": cos,
        "first_max_abs": max_abs,
        "full_cos": full_cos,
        "full_max_abs": full_max,
        "passed": passed,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache},
    secrets=[
        _hf_secret(),
    ],
)
def parity_native_pi0(model_id: str = "lerobot/pi0_base"):
    """Native-path parity: PI0Policy.predict_action_chunk (wrapper) vs
    PI0Pytorch.sample_actions (raw) on same seeded inputs.

    Closes the cross-framework moat claim for pi0 — SmolVLA already has
    this via `native-path-parity` item #1 (DecomposedRMSNorm swap path).
    For pi0, no swap needed, so this is trivially cos=1.0 but worth
    recording as a measured number.
    """
    import sys, types
    for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
        stub = types.ModuleType(_mod); stub.GrootPolicy = None; stub.GR00TN15 = None
        sys.modules[_mod] = stub

    import numpy as np
    import torch

    print("[native] Loading PI0Policy...")
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch
    from huggingface_hub import snapshot_download

    policy = PI0Policy.from_pretrained(model_id).eval().to(torch.float32).to("cpu")
    repo = snapshot_download(model_id)
    pre = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=repo,
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={"device_processor": {"device": "cpu"}},
    )

    # Build realistic batch
    cfg = policy.config
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim

    rng = np.random.RandomState(42)
    img_np = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    state = torch.from_numpy(rng.randn(14).astype(np.float32) * 0.1)

    batch_raw = {
        "observation.images.base_0_rgb": img_t.unsqueeze(0),
        "observation.images.left_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.images.right_wrist_0_rgb": img_t.unsqueeze(0),
        "observation.state": state.unsqueeze(0),
        "task": ["pick up the red bowl"],
    }
    batch = pre(batch_raw)
    noise = torch.from_numpy(np.random.RandomState(99).randn(1, chunk, action_dim).astype(np.float32))

    # Path 1: wrapper
    print("[native] Running PI0Policy.predict_action_chunk (wrapper)...")
    with torch.no_grad():
        a_wrapper = policy.predict_action_chunk(batch, noise=noise).cpu().numpy()

    # Path 2: raw — reproduce what the wrapper does internally
    print("[native] Running PI0Pytorch.sample_actions (raw)...")
    images, img_masks = policy._preprocess_images(batch)
    lang_tokens = batch["observation.language.tokens"]
    lang_masks = batch["observation.language.attention_mask"]
    state_padded = policy.prepare_state(batch)
    with torch.no_grad():
        a_raw = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state_padded, noise=noise,
        ).cpu().numpy()
    # Wrapper slices to original action_dim
    orig_action_dim = a_wrapper.shape[-1]
    a_raw_sliced = a_raw[:, :, :orig_action_dim]

    # Compare
    def _cos(a, b):
        af, bf = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
        return float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
    first_cos = _cos(a_wrapper[0, 0], a_raw_sliced[0, 0])
    first_max = float(np.max(np.abs(a_wrapper[0, 0] - a_raw_sliced[0, 0])))
    full_cos = _cos(a_wrapper, a_raw_sliced)
    full_max = float(np.max(np.abs(a_wrapper - a_raw_sliced)))
    print(f"[native] wrapper first: {a_wrapper[0, 0, :5]}")
    print(f"[native] raw     first: {a_raw_sliced[0, 0, :5]}")
    print(f"[native]   first_cos={first_cos:.10f}, first_max={first_max:.3e}")
    print(f"[native]   full_cos ={full_cos:.10f}, full_max ={full_max:.3e}")
    passed = first_cos >= 0.999 and full_cos >= 0.999
    print(f"[native] VERDICT: {'PASS' if passed else 'FAIL'}")
    return {
        "first_cos": first_cos, "first_max_abs": first_max,
        "full_cos": full_cos, "full_max_abs": full_max,
        "passed": passed,
    }


@app.local_entrypoint()
def main(num_steps: int = 1, parity: bool = False, native: bool = False):
    """Default to num_steps=1 (the only working config currently).

    Usage:
        modal run scripts/modal_pi0_monolithic_export.py              # export
        modal run scripts/modal_pi0_monolithic_export.py --parity     # ONNX parity
        modal run scripts/modal_pi0_monolithic_export.py --native     # native parity
    """
    if native:
        result = parity_native_pi0.remote()
    elif parity:
        result = parity_test_monolithic.remote()
    else:
        result = export_pi0_monolithic_modal.remote(num_steps=num_steps)
    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
