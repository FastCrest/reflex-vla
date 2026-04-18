"""Modal: SmolVLA monolithic ONNX export via onnx-diagnostic torch_export_patches.

Port of scripts/modal_pi0_monolithic_export.py for SmolVLA. Same recipe:
wrap sample_actions end-to-end, export via torch.export + onnx-diagnostic,
single monolithic ONNX with num_steps baked in.

SmolVLA-specific deltas from pi0:
  - SmolVLAPolicy (not PI0Policy), model_id lerobot/smolvla_base
  - SmolVLM2 + SmolLM2 backbone (not PaliGemma + Gemma)
  - Image size 512x512 (not 224x224)
  - sample_actions doesn't take num_steps kwarg — override config.num_steps
  - No GemmaAttention reshape patch needed (SmolVLA uses different attn)
  - Shared generic patches (create_causal_mask, deepcopy bypass, MambaCache stub)

Usage:
    modal run scripts/modal_smolvla_monolithic_export.py                # export
    modal run scripts/modal_smolvla_monolithic_export.py --parity       # parity

Output: /onnx_out/smolvla_monolithic/model.onnx on Modal Volume.
"""
import os
import modal

app = modal.App("smolvla-monolithic-export")


def _hf_secret():
    """HF_TOKEN via local env var (dev) or Modal named secret `hf-token` (prod)."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return modal.Secret.from_dict({"HF_TOKEN": token})
    return modal.Secret.from_name("hf-token")

# Reuse pi0's HF cache volume — same HF models may be pulled; saves storage
hf_cache = modal.Volume.from_name("pi0-hf-cache", create_if_missing=True)
onnx_output = modal.Volume.from_name("pi0-onnx-outputs", create_if_missing=True)

HF_CACHE_PATH = "/root/.cache/huggingface"
ONNX_OUTPUT_PATH = "/onnx_out"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "lerobot==0.5.1",
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
    # Pin exactly what lerobot 0.5.1 pins. transformers 5.4+ introduced
    # a q_length-scalar regression in masking_utils.sdpa_mask (PR #44181,
    # 2026-03-04). 5.3.0 is the last clean version. onnx-diagnostic 0.9.x
    # patches target transformers <=5.3 — aligns perfectly.
    .pip_install("transformers==5.3.0")
    .env({
        "HF_HOME": HF_CACHE_PATH,
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/transformers",
    })
)


def _apply_patches():
    """Shared patches for lerobot 0.5.1 + transformers 5.x + onnx-diagnostic."""
    import sys
    import types
    import torch

    # Stub GR00T imports to avoid Python 3.13 dataclass issue (harmless on 3.12 too)
    for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
        stub = types.ModuleType(_mod)
        stub.GrootPolicy = None
        stub.GR00TN15 = None
        sys.modules[_mod] = stub

    # MambaCache stub for onnx-diagnostic compat with transformers 5.x
    try:
        from transformers.models.mamba.modeling_mamba import MambaCache  # noqa
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

    # create_causal_mask kwarg rename (inputs_embeds -> input_embeds)
    from transformers import masking_utils
    _orig_ccm = masking_utils.create_causal_mask

    def _ccm_shim(*args, **kwargs):
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return _orig_ccm(*args, **kwargs)
    masking_utils.create_causal_mask = _ccm_shim
    try:
        from lerobot.policies import pi_gemma as _pg
        if hasattr(_pg, "create_causal_mask"):
            _pg.create_causal_mask = _ccm_shim
    except ImportError:
        pass

    # copy.deepcopy bypass for DynamicCache (fails under FakeTensor tracing)
    import copy as _copy
    _orig_deepcopy = _copy.deepcopy

    def _safe_deepcopy(obj, *args, **kwargs):
        from transformers.cache_utils import DynamicCache
        if isinstance(obj, DynamicCache):
            return obj
        return _orig_deepcopy(obj, *args, **kwargs)
    _copy.deepcopy = _safe_deepcopy

    # NOTE: NO wrap-patching of onnx-diagnostic's patched_*_mask functions.
    # Earlier runs wrapped them, which corrupted onnx-diagnostic's internal
    # registry (caught by its unpatch sanity check — "corrupted function
    # 'eager_mask'"). With transformers==5.3.0 pinned exactly, onnx-diagnostic
    # 0.9.3's native patches target the correct signature. No wrap needed.

    # Replace lerobot's pad_tensor AND pad_vector (both use in-place slice
    # assignment `padded[...] = tensor`) with torch.cat-based versions. The
    # slice assign gets lowered to aten::index_put_ -> ONNX Where(bool, int64,
    # float) which onnxruntime rejects with "Type parameter (T) bound to
    # different types" at load time. torch.cat avoids index_put entirely.
    try:
        from lerobot.policies.smolvla import modeling_smolvla as _smv

        def _safe_pad_tensor(tensor, max_len, pad_value=0):
            b, d = tensor.shape[:2]
            if d >= max_len:
                return tensor[:, :max_len]
            pad_shape = (b, max_len - d, *tensor.shape[2:])
            pad = torch.full(
                pad_shape, float(pad_value),
                dtype=tensor.dtype, device=tensor.device,
            )
            return torch.cat([tensor, pad], dim=1)
        _smv.pad_tensor = _safe_pad_tensor

        def _safe_pad_vector(vector, new_dim):
            current_dim = vector.shape[-1]
            if current_dim >= new_dim:
                return vector[..., :new_dim]
            pad_shape = (*vector.shape[:-1], new_dim - current_dim)
            pad = torch.zeros(pad_shape, dtype=vector.dtype, device=vector.device)
            return torch.cat([vector, pad], dim=-1)
        _smv.pad_vector = _safe_pad_vector
    except ImportError:
        pass

    # torch.where in transformers 5.3 smolvlm.forward (line 119) is called with
    # torch.full(..., fill_value=0) (int64 default) branched against a float32
    # tensor. Eager PyTorch silently promotes; torch.export captures it as
    # aten.where.self with mismatched dtypes -> ONNX Where(bool, int64, float)
    # which onnxruntime rejects at load. Wrap torch.where to explicit-cast.
    _orig_where = torch.where

    def _safe_where(condition, x=None, y=None, *args, **kwargs):
        if x is not None and y is not None \
                and hasattr(x, "dtype") and hasattr(y, "dtype") \
                and x.dtype != y.dtype:
            common = torch.promote_types(x.dtype, y.dtype)
            if x.dtype != common:
                x = x.to(common)
            if y.dtype != common:
                y = y.to(common)
        return _orig_where(condition, x, y, *args, **kwargs)
    torch.where = _safe_where


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def export_smolvla_monolithic_modal(num_steps: int = 1):
    """Export SmolVLA as monolithic ONNX via onnx-diagnostic path."""
    import time
    from pathlib import Path
    import torch
    import torch.nn as nn

    _apply_patches()

    print("[smolvla] Loading SmolVLAPolicy...")
    t0 = time.time()
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.eval().to("cpu").to(torch.float32)
    policy.model.config.num_steps = num_steps

    # Force eager attention everywhere — avoids onnx-diagnostic's sdpa_mask
    # patch crashing with None cache_position in transformers 5.x.
    def _force_eager(m):
        for mod in m.modules():
            if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
                mod.config._attn_implementation = "eager"
        # Also set on the top-level config
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"
    _force_eager(policy.model)
    print(f"[smolvla] Loaded in {time.time() - t0:.1f}s; num_steps={num_steps}, attn=eager")

    class SmolVLAMonolithicWrapper(nn.Module):
        def __init__(self, smolvla_model):
            super().__init__()
            self.model = smolvla_model

        def forward(
            self,
            img_cam1, img_cam2, img_cam3,
            mask_cam1, mask_cam2, mask_cam3,
            lang_tokens, lang_masks,
            state, noise,
        ):
            images = [img_cam1, img_cam2, img_cam3]
            img_masks = [mask_cam1, mask_cam2, mask_cam3]
            return self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state,
                noise=noise,
            )

    wrapper = SmolVLAMonolithicWrapper(policy.model).eval()

    cfg = policy.config
    B = 1
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim
    state_dim = getattr(cfg, "max_state_dim", 32)

    # SmolVLA uses 512x512 images
    dummy = dict(
        img_cam1=torch.randn(B, 3, 512, 512, dtype=torch.float32),
        img_cam2=torch.randn(B, 3, 512, 512, dtype=torch.float32),
        img_cam3=torch.randn(B, 3, 512, 512, dtype=torch.float32),
        mask_cam1=torch.ones(B, dtype=torch.bool),
        mask_cam2=torch.ones(B, dtype=torch.bool),
        mask_cam3=torch.ones(B, dtype=torch.bool),
        lang_tokens=torch.randint(0, 49152, (B, 16), dtype=torch.long),  # SmolLM2 vocab
        lang_masks=torch.ones(B, 16, dtype=torch.bool),
        state=torch.randn(B, state_dim, dtype=torch.float32),
        noise=torch.randn(B, chunk, action_dim, dtype=torch.float32),
    )

    output_dir = Path(ONNX_OUTPUT_PATH) / "smolvla_monolithic"
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    # Back to onnx-diagnostic path — with transformers==5.3.0 pinned, its
    # native patches should now match the mask-interface signatures.
    print("[smolvla] torch_export_patches + torch.export.export...")
    t0 = time.time()
    from onnx_diagnostic.torch_export_patches import torch_export_patches

    with torch_export_patches(patch_transformers=True):
        ep = torch.export.export(
            wrapper,
            tuple(dummy.values()),
            dynamic_shapes=None,
            strict=False,
        )
    print(f"[smolvla] torch.export: {time.time() - t0:.1f}s")

    # Diagnostic: find all Where-producing ops with mismatched arg dtypes
    print("[smolvla] scanning for Where-producing ops with dtype mismatch...")
    WHERE_OPS = ("index_put", "scatter", "where", "masked_fill", "masked_scatter")
    for node in ep.graph.nodes:
        tgt_str = str(node.target)
        if not any(op in tgt_str for op in WHERE_OPS):
            continue
        # Val dtypes
        vals = []
        dtypes_seen = set()
        for a in node.args:
            if hasattr(a, "meta") and "val" in a.meta:
                v = a.meta["val"]
                dt = getattr(v, "dtype", None)
                if dt is not None and dt != torch.bool:
                    dtypes_seen.add(str(dt))
                vals.append(f"{dt}/{getattr(v, 'shape', '?')}")
            elif isinstance(a, (list, tuple)):
                for sub in a:
                    if hasattr(sub, "meta") and "val" in sub.meta:
                        v = sub.meta["val"]
                        dt = getattr(v, "dtype", None)
                        if dt is not None and dt != torch.bool:
                            dtypes_seen.add(str(dt))
                vals.append(str(a)[:60])
            else:
                vals.append(str(a)[:40])
        if len(dtypes_seen) > 1:
            stack = node.meta.get("stack_trace", "")
            short = "\n".join(stack.splitlines()[-8:]) if stack else "(no stack)"
            print(f"[smolvla] MISMATCH {tgt_str}: dtypes={dtypes_seen} args={vals}")
            print(f"[smolvla]   stack:\n{short}")
            print("---")

    t0 = time.time()
    input_names = list(dummy.keys())
    torch.onnx.export(
        ep,
        tuple(dummy.values()),
        str(onnx_path),
        input_names=input_names,
        output_names=["actions"],
        opset_version=19,
    )
    print(f"[smolvla] ONNX conversion: {time.time() - t0:.1f}s")

    # Post-export: scan ONNX for Where nodes with mismatched-dtype branches
    # (torch.onnx sometimes lowers index_put to Where(bool, int64, float) even
    # when the aten graph was clean). Insert Cast nodes to coerce the int
    # branch to the float branch's dtype so onnxruntime will load the model.
    print("[smolvla] post-export ONNX Where-type fix...")
    import onnx
    from onnx import helper, TensorProto

    model = onnx.load(str(onnx_path), load_external_data=False)
    shape_info = onnx.shape_inference.infer_shapes(model, check_type=False, strict_mode=False)

    # Build a name -> elem_type map from value_info + inputs + initializers
    name_dtype: dict[str, int] = {}
    for vi in list(shape_info.graph.value_info) + list(shape_info.graph.input) + list(shape_info.graph.output):
        if vi.type.tensor_type.elem_type:
            name_dtype[vi.name] = vi.type.tensor_type.elem_type
    for init in shape_info.graph.initializer:
        name_dtype[init.name] = init.data_type

    FLOAT_TYPES = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE}
    INT_TYPES = {TensorProto.INT64, TensorProto.INT32, TensorProto.INT16, TensorProto.INT8}

    fixes = 0
    new_nodes: list = []
    for node in model.graph.node:
        if node.op_type != "Where":
            new_nodes.append(node)
            continue
        # Where inputs: [condition(bool), X, Y]; output: node.output[0]
        x_name, y_name = node.input[1], node.input[2]
        out_name = node.output[0]
        x_dt = name_dtype.get(x_name)
        y_dt = name_dtype.get(y_name)
        out_dt = name_dtype.get(out_name)
        if not x_dt or not y_dt or x_dt == y_dt:
            new_nodes.append(node)
            continue
        # Mismatch between X and Y. Prefer the OUTPUT's declared dtype (that's
        # what downstream consumers expect). If no output value_info, fall back
        # to matching the non-float side (index_put preserves self dtype).
        if out_dt and out_dt in (x_dt, y_dt):
            target_dt = out_dt
        else:
            # Prefer int when one side is int (index_put semantics)
            target_dt = x_dt if x_dt in INT_TYPES else y_dt
        fixes_this = []
        if x_dt != target_dt:
            cast_out = f"{node.name}__cast_x"
            cast_node = helper.make_node("Cast", [x_name], [cast_out], name=f"{node.name}__cast_x_node", to=target_dt)
            fixes_this.append(cast_node)
            node.input[1] = cast_out
        if y_dt != target_dt:
            cast_out = f"{node.name}__cast_y"
            cast_node = helper.make_node("Cast", [y_name], [cast_out], name=f"{node.name}__cast_y_node", to=target_dt)
            fixes_this.append(cast_node)
            node.input[2] = cast_out
        new_nodes.extend(fixes_this)
        new_nodes.append(node)
        fixes += 1
        print(f"[smolvla]   fixed Where node: {node.name} (X={x_dt}, Y={y_dt}, out_decl={out_dt} -> target={target_dt})")

    if fixes > 0:
        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)
        # Write the protobuf in-place; we didn't touch any tensor weights, so
        # external data references stay valid (pointing to model.onnx.data).
        with open(onnx_path, "wb") as _f:
            _f.write(model.SerializeToString())
        print(f"[smolvla] inserted {fixes} Cast nodes to fix Where dtype mismatches")
    else:
        print("[smolvla] no Where dtype mismatches found")

    onnx_output.commit()

    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / 1e6
        data_files = list(output_dir.glob("*.data"))
        total_mb = sum(f.stat().st_size for f in data_files) / 1e6 + size_mb
        print(f"[smolvla] SUCCESS: {onnx_path} ({total_mb:.1f}MB total, {len(data_files)} data files)")
        return {"status": "ok", "onnx_path": str(onnx_path), "size_mb": total_mb}
    return {"status": "fail", "reason": "onnx file not created"}


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache, ONNX_OUTPUT_PATH: onnx_output},
    secrets=[_hf_secret()],
)
def parity_test_smolvla(num_steps: int = 1):
    """Parity: PyTorch SmolVLA sample_actions(num_steps=N) vs monolithic ONNX(N)."""
    _apply_patches()

    import numpy as np
    import torch
    import onnxruntime as ort
    from pathlib import Path

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print("[parity] Loading SmolVLA...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").eval().to(torch.float32).to("cpu")
    policy.model.config.num_steps = num_steps

    cfg = policy.config
    B = 1
    chunk = cfg.chunk_size
    action_dim = cfg.max_action_dim
    state_dim = getattr(cfg, "max_state_dim", 32)

    torch.manual_seed(42)
    img = torch.randn(B, 3, 512, 512, dtype=torch.float32)
    mask = torch.ones(B, dtype=torch.bool)
    lang_tokens = torch.randint(0, 49152, (B, 16), dtype=torch.long)
    lang_masks = torch.ones(B, 16, dtype=torch.bool)
    state = torch.randn(B, state_dim, dtype=torch.float32)
    noise = torch.randn(B, chunk, action_dim, dtype=torch.float32)

    images = [img, img, img]
    img_masks = [mask, mask, mask]
    print(f"[parity] Running PyTorch ref (num_steps={num_steps})...")
    with torch.no_grad():
        pt_actions = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise,
        )
    pt_np = pt_actions.cpu().numpy()
    print(f"[parity] pt: {pt_np.shape}, first: {pt_np[0, 0, :5]}")

    onnx_path = Path(ONNX_OUTPUT_PATH) / "smolvla_monolithic" / "model.onnx"
    print(f"[parity] Running ONNX ({onnx_path})...")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "img_cam1": img.numpy(),
        "img_cam2": img.numpy(),
        "img_cam3": img.numpy(),
        "mask_cam1": mask.numpy(),
        "mask_cam2": mask.numpy(),
        "mask_cam3": mask.numpy(),
        "lang_tokens": lang_tokens.numpy().astype(np.int64),
        "lang_masks": lang_masks.numpy(),
        "state": state.numpy(),
        "noise": noise.numpy(),
    }
    ort_out = sess.run(None, ort_inputs)[0]
    print(f"[parity] onnx: {ort_out.shape}, first: {ort_out[0, 0, :5]}")

    pt0 = pt_np[0, 0]
    on0 = ort_out[0, 0]
    max_abs = float(np.abs(pt0 - on0).max())
    cos = float(np.dot(pt0, on0) / (np.linalg.norm(pt0) * np.linalg.norm(on0) + 1e-8))
    full_cos = float(
        np.dot(pt_np.flatten(), ort_out.flatten())
        / (np.linalg.norm(pt_np) * np.linalg.norm(ort_out) + 1e-8)
    )
    full_max = float(np.abs(pt_np - ort_out).max())

    print(f"\n====== SMOLVLA PARITY (num_steps={num_steps}) ======")
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


@app.local_entrypoint()
def main(num_steps: int = 1, parity: bool = False):
    if parity:
        result = parity_test_smolvla.remote(num_steps=num_steps)
    else:
        result = export_smolvla_monolithic_modal.remote(num_steps=num_steps)
    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
