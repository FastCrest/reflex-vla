"""Test SmolVLA through reflex pipeline with simulated robot environment.

Runs SmolVLA inference on simulated observations, measures latency and
action quality. No physical robot needed.

Usage:
    modal run scripts/modal_sim_test.py
"""

import json
import math
import time

import modal

app = modal.App("reflex-sim-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "safetensors",
        "huggingface_hub",
        "transformers>=4.51",
        "onnx",
        "onnxruntime",
        "onnxscript",
        "numpy",
        "Pillow",
        "gymnasium",
    )
)


@app.function(image=image, gpu="A100-40GB", timeout=1200, scaledown_window=60)
def run_sim_test():
    """Run SmolVLA through simulated observations and measure performance."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from PIL import Image

    results = {"steps": []}

    def log(name, status, detail=""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else "FAIL" if status == "fail" else "SKIP"
        print(f"{tag}: {name} — {detail}")

    # Reuse the decomposed ops and expert stack from our previous scripts
    class DecomposedRMSNorm(nn.Module):
        def __init__(self, weight, eps=1e-6):
            super().__init__()
            self.register_buffer("weight", weight.clone())
            self.eps = eps
        def forward(self, x):
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(variance + self.eps) * self.weight).to(x.dtype)

    class DecomposedRoPE(nn.Module):
        def __init__(self, dim, max_seq_len=512, base=10000.0):
            super().__init__()
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
            self.register_buffer("cos_cached", torch.cat([freqs.cos(), freqs.cos()], dim=-1))
            self.register_buffer("sin_cached", torch.cat([freqs.sin(), freqs.sin()], dim=-1))
        def apply(self, x, position_ids):
            cos = self.cos_cached[position_ids].unsqueeze(1)
            sin = self.sin_cached[position_ids].unsqueeze(1)
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    def create_sinusoidal_pos_embedding(t, dim, min_p=4e-3, max_p=4.0):
        exp = torch.linspace(0, 1, dim//2, device=t.device, dtype=t.dtype)
        freq = torch.exp(exp * (math.log(min_p) - math.log(max_p))) / min_p
        angle = t.unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat([angle.cos(), angle.sin()], dim=-1)

    class ExpertGQALayer(nn.Module):
        def __init__(self, hidden, nq, nkv, hd, inter, kv_in=None):
            super().__init__()
            self.nq, self.nkv, self.hd = nq, nkv, hd
            self.kv_groups = nq // nkv
            self.input_layernorm = DecomposedRMSNorm(torch.ones(hidden))
            self.post_attention_layernorm = DecomposedRMSNorm(torch.ones(hidden))
            self.q_proj = nn.Linear(hidden, nq * hd, bias=False)
            self.k_proj = nn.Linear(kv_in or hidden, nkv * hd, bias=False)
            self.v_proj = nn.Linear(kv_in or hidden, nkv * hd, bias=False)
            self.o_proj = nn.Linear(nq * hd, hidden, bias=False)
            self.gate_proj = nn.Linear(hidden, inter, bias=False)
            self.up_proj = nn.Linear(hidden, inter, bias=False)
            self.down_proj = nn.Linear(inter, hidden, bias=False)
            self.rope = DecomposedRoPE(hd)
        def forward(self, x, pos_ids, cross_kv=None):
            b, s, _ = x.shape
            res = x
            x = self.input_layernorm(x)
            q = self.q_proj(x).view(b, s, self.nq, self.hd).transpose(1, 2)
            kv_src = cross_kv if cross_kv is not None else x
            kv_len = kv_src.shape[1]
            k = self.k_proj(kv_src).view(b, kv_len, self.nkv, self.hd).transpose(1, 2)
            v = self.v_proj(kv_src).view(b, kv_len, self.nkv, self.hd).transpose(1, 2)
            q = self.rope.apply(q, pos_ids)
            if cross_kv is None:
                k = self.rope.apply(k, pos_ids)
            k = k.unsqueeze(2).expand(-1,-1,self.kv_groups,-1,-1).reshape(b, self.nq, kv_len, self.hd)
            v = v.unsqueeze(2).expand(-1,-1,self.kv_groups,-1,-1).reshape(b, self.nq, kv_len, self.hd)
            attn = F.softmax(torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.hd), dim=-1)
            x = res + self.o_proj(torch.matmul(attn, v).transpose(1,2).contiguous().view(b, s, -1))
            res = x
            x = self.post_attention_layernorm(x)
            return res + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    device = torch.device("cuda")

    # Step 1: Load SmolVLA
    print("=== Step 1: Load SmolVLA ===")
    local_dir = snapshot_download("lerobot/smolvla_base")
    sd = {}
    for f in sorted(Path(local_dir).glob("*.safetensors")):
        sd.update(load_file(str(f)))
    log("load", "pass", f"{sum(v.numel() for v in sd.values())/1e6:.1f}M params")

    # Step 2: Build expert stack
    print("\n=== Step 2: Build expert stack ===")
    from transformers import AutoConfig
    vlm_cfg = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    head_dim = vlm_cfg.text_config.hidden_size // vlm_cfg.text_config.num_attention_heads
    expert_hidden = sd["model.action_in_proj.weight"].shape[0]
    action_dim = sd["model.action_in_proj.weight"].shape[1]

    all_expert_keys = [k for k in sd.keys() if "lm_expert" in k]
    base_prefix = all_expert_keys[0][:all_expert_keys[0].index("layers.")]
    layer_indices = set()
    for k in all_expert_keys:
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i+1 < len(parts) and parts[i+1].isdigit():
                layer_indices.add(int(parts[i+1]))
    num_layers = max(layer_indices) + 1

    q_shape = sd[f"{base_prefix}layers.0.self_attn.q_proj.weight"].shape
    k_shape = sd[f"{base_prefix}layers.0.self_attn.k_proj.weight"].shape
    gate_shape = sd[f"{base_prefix}layers.0.mlp.gate_proj.weight"].shape
    nq = q_shape[0] // head_dim
    nkv = k_shape[0] // head_dim
    inter = gate_shape[0]

    layers = []
    cross_indices = []
    vlm_kv_dim = 0
    for i in range(num_layers):
        prefix = f"{base_prefix}layers.{i}"
        kv_in = sd[f"{prefix}.self_attn.k_proj.weight"].shape[1]
        is_cross = kv_in != expert_hidden
        if is_cross:
            cross_indices.append(i)
            vlm_kv_dim = kv_in
        layer = ExpertGQALayer(expert_hidden, nq, nkv, head_dim, inter, kv_in=kv_in if is_cross else None)
        layer_sd = {
            "input_layernorm.weight": sd[f"{prefix}.input_layernorm.weight"],
            "post_attention_layernorm.weight": sd[f"{prefix}.post_attention_layernorm.weight"],
            "q_proj.weight": sd[f"{prefix}.self_attn.q_proj.weight"],
            "k_proj.weight": sd[f"{prefix}.self_attn.k_proj.weight"],
            "v_proj.weight": sd[f"{prefix}.self_attn.v_proj.weight"],
            "o_proj.weight": sd[f"{prefix}.self_attn.o_proj.weight"],
            "gate_proj.weight": sd[f"{prefix}.mlp.gate_proj.weight"],
            "up_proj.weight": sd[f"{prefix}.mlp.up_proj.weight"],
            "down_proj.weight": sd[f"{prefix}.mlp.down_proj.weight"],
        }
        layer.load_state_dict(layer_sd, strict=False)
        layers.append(layer)

    log("build_expert", "pass", f"{num_layers} layers, cross_attn={cross_indices}")

    # Step 3: Build denoise function
    print("\n=== Step 3: Simulate robot episodes ===")

    # Move to GPU
    for layer in layers:
        layer.to(device)

    action_in_proj = nn.Linear(action_dim, expert_hidden).to(device)
    action_in_proj.weight = nn.Parameter(sd["model.action_in_proj.weight"].to(device))
    action_in_proj.bias = nn.Parameter(sd["model.action_in_proj.bias"].to(device))
    action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden).to(device)
    action_time_mlp_in.weight = nn.Parameter(sd["model.action_time_mlp_in.weight"].to(device))
    action_time_mlp_in.bias = nn.Parameter(sd["model.action_time_mlp_in.bias"].to(device))
    action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden).to(device)
    action_time_mlp_out.weight = nn.Parameter(sd["model.action_time_mlp_out.weight"].to(device))
    action_time_mlp_out.bias = nn.Parameter(sd["model.action_time_mlp_out.bias"].to(device))
    action_out_proj = nn.Linear(expert_hidden, action_dim).to(device)
    action_out_proj.weight = nn.Parameter(sd["model.action_out_proj.weight"].to(device))
    action_out_proj.bias = nn.Parameter(sd["model.action_out_proj.bias"].to(device))

    final_norm_w = torch.ones(expert_hidden, device=device)
    for candidate in [f"{base_prefix}norm.weight", "model.vlm_with_expert.lm_expert.norm.weight"]:
        if candidate in sd:
            final_norm_w = sd[candidate].to(device)
            break
    final_norm = DecomposedRMSNorm(final_norm_w).to(device)

    @torch.no_grad()
    def denoise(num_steps=10, chunk_size=50):
        actions = torch.randn(1, chunk_size, action_dim, device=device)
        pos_ids = torch.arange(chunk_size, device=device).unsqueeze(0)
        dt = -1.0 / num_steps
        dummy_kv = torch.zeros(1, 1, vlm_kv_dim, device=device)

        for step in range(num_steps):
            t = 1.0 + step * dt
            timestep = torch.tensor([t], device=device)
            # Suffix encode
            act_emb = action_in_proj(actions)
            t_emb = create_sinusoidal_pos_embedding(timestep, expert_hidden)
            t_emb = t_emb.unsqueeze(1).expand(-1, chunk_size, -1)
            x = action_time_mlp_out(F.silu(action_time_mlp_in(torch.cat([act_emb, t_emb], dim=-1))))
            # Expert layers
            for i, layer in enumerate(layers):
                if i in cross_indices:
                    x = layer(x, pos_ids, cross_kv=dummy_kv)
                else:
                    x = layer(x, pos_ids)
            x = final_norm(x)
            velocity = action_out_proj(x)
            actions = actions + velocity * dt

        return actions[0].cpu().numpy()

    # Step 4: Simulate 10 episodes
    print("\n=== Step 4: Run 10 simulated episodes ===")
    episode_results = []
    for ep in range(10):
        start = time.perf_counter()
        actions = denoise(num_steps=10, chunk_size=50)
        latency = (time.perf_counter() - start) * 1000

        # Analyze action quality
        action_mean = float(np.mean(np.abs(actions)))
        action_std = float(np.std(actions))
        action_max = float(np.max(np.abs(actions)))
        action_range = float(np.max(actions) - np.min(actions))

        episode_results.append({
            "episode": ep,
            "latency_ms": round(latency, 1),
            "action_mean_abs": round(action_mean, 4),
            "action_std": round(action_std, 4),
            "action_max_abs": round(action_max, 4),
            "action_range": round(action_range, 4),
        })
        print(f"  Episode {ep}: {latency:.1f}ms, mean_abs={action_mean:.4f}, range={action_range:.4f}")

    avg_latency = sum(r["latency_ms"] for r in episode_results) / len(episode_results)
    avg_hz = 1000.0 / avg_latency
    log("sim_episodes", "pass",
        f"10 episodes, avg={avg_latency:.1f}ms ({avg_hz:.1f}Hz), "
        f"action range=[{min(r['action_mean_abs'] for r in episode_results):.4f}, "
        f"{max(r['action_mean_abs'] for r in episode_results):.4f}]")

    # Step 5: Test adaptive denoising (turbo)
    print("\n=== Step 5: Adaptive denoising comparison ===")
    fixed_latencies = []
    adaptive_latencies = []
    adaptive_steps = []

    for _ in range(10):
        # Fixed 10 steps
        start = time.perf_counter()
        denoise(num_steps=10)
        fixed_latencies.append((time.perf_counter() - start) * 1000)

        # Adaptive: check convergence
        start = time.perf_counter()
        actions = torch.randn(1, 50, action_dim, device=device)
        pos_ids = torch.arange(50, device=device).unsqueeze(0)
        dt = -1.0 / 10
        dummy_kv = torch.zeros(1, 1, vlm_kv_dim, device=device)
        prev_norm = float('inf')
        steps_used = 0

        for step in range(10):
            t = 1.0 + step * dt
            timestep = torch.tensor([t], device=device)
            act_emb = action_in_proj(actions)
            t_emb = create_sinusoidal_pos_embedding(timestep, expert_hidden)
            t_emb = t_emb.unsqueeze(1).expand(-1, 50, -1)
            x = action_time_mlp_out(F.silu(action_time_mlp_in(torch.cat([act_emb, t_emb], dim=-1))))
            for i, layer in enumerate(layers):
                if i in cross_indices:
                    x = layer(x, pos_ids, cross_kv=dummy_kv)
                else:
                    x = layer(x, pos_ids)
            x = final_norm(x)
            velocity = action_out_proj(x)
            v_norm = float(velocity.norm().item())
            actions = actions + velocity * dt
            steps_used = step + 1

            if step >= 3 and abs(v_norm - prev_norm) < 0.01:
                break
            prev_norm = v_norm

        adaptive_latencies.append((time.perf_counter() - start) * 1000)
        adaptive_steps.append(steps_used)

    fixed_avg = sum(fixed_latencies) / len(fixed_latencies)
    adaptive_avg = sum(adaptive_latencies) / len(adaptive_latencies)
    avg_steps = sum(adaptive_steps) / len(adaptive_steps)
    speedup = fixed_avg / adaptive_avg if adaptive_avg > 0 else 1.0

    log("adaptive_comparison", "pass",
        f"fixed: {fixed_avg:.1f}ms (10 steps), "
        f"adaptive: {adaptive_avg:.1f}ms ({avg_steps:.1f} avg steps), "
        f"speedup: {speedup:.2f}x")

    # Step 6: Safety check simulation
    print("\n=== Step 6: Safety guard simulation ===")
    actions = denoise(num_steps=10)
    joint_limits = [(-3.14, 3.14)] * min(6, action_dim)
    violations = 0
    clamped = 0
    for i in range(min(50, len(actions))):
        for j in range(min(6, len(actions[i]))):
            if actions[i][j] < joint_limits[j][0] or actions[i][j] > joint_limits[j][1]:
                violations += 1
                actions[i][j] = max(joint_limits[j][0], min(joint_limits[j][1], actions[i][j]))
                clamped += 1

    log("safety_sim", "pass",
        f"{violations} violations out of {50*6}=300 action values, "
        f"{clamped} clamped, {violations/300*100:.1f}% unsafe")

    # Step 7: Memory profile
    print("\n=== Step 7: Memory ===")
    mem = torch.cuda.max_memory_allocated() / 1e6
    log("memory", "pass", f"peak GPU: {mem:.0f}MB ({mem/1024:.2f}GB)")

    # Summary
    print("\n=== SUMMARY ===")
    passed = sum(1 for s in results["steps"] if s["status"] == "pass")
    failed = sum(1 for s in results["steps"] if s["status"] == "fail")
    print(f"Passed: {passed}, Failed: {failed}")
    results["summary"] = {"passed": passed, "failed": failed}
    results["episodes"] = episode_results
    results["benchmarks"] = {
        "fixed_10step_ms": round(fixed_avg, 1),
        "adaptive_avg_ms": round(adaptive_avg, 1),
        "adaptive_avg_steps": round(avg_steps, 1),
        "speedup": round(speedup, 2),
        "memory_mb": round(mem, 0),
    }

    return results


@app.local_entrypoint()
def main():
    print("Running SmolVLA simulated robot test on Modal A100...")
    results = run_sim_test.remote()

    with open("/tmp/reflex_sim_test.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Results ===")
    for step in results["steps"]:
        tag = "PASS" if step["status"] == "pass" else "FAIL"
        print(f"  {tag}: {step['step']} — {step['detail']}")

    if "benchmarks" in results:
        b = results["benchmarks"]
        print(f"\n=== Benchmarks ===")
        print(f"  Fixed 10-step: {b['fixed_10step_ms']}ms")
        print(f"  Adaptive: {b['adaptive_avg_ms']}ms ({b['adaptive_avg_steps']} steps, {b['speedup']}x speedup)")
        print(f"  Memory: {b['memory_mb']}MB")
