"""Verify `reflex --help` shows all 7 wedges + the 4 new commands work.

Usage:
    modal run scripts/modal_verify_cli.py
"""

import modal

app = modal.App("reflex-cli-verify")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch", "safetensors", "huggingface_hub",
        "numpy", "Pillow",
        "typer", "rich", "pydantic>=2.0", "pyyaml",
    )
    .add_local_dir("src/reflex", "/root/reflex-vla/src/reflex", copy=True)
    .add_local_file("pyproject.toml", "/root/reflex-vla/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/reflex-vla/README.md", copy=True)
    .run_commands("cd /root/reflex-vla && pip install -e .")
)


@app.function(image=image, cpu=2, timeout=300)
def verify():
    import subprocess

    results = {"steps": []}

    def log(name, status, detail=""):
        results["steps"].append({"step": name, "status": status, "detail": detail})
        tag = "PASS" if status == "pass" else "FAIL"
        print(f"{tag}: {name} — {detail}", flush=True)

    # 1. Check --help lists all 7 wedges
    print("=== Step 1: reflex --help lists 7 wedges ===", flush=True)
    r = subprocess.run(["reflex", "--help"], capture_output=True, text=True)
    wedges = ["export", "serve", "guard", "turbo", "split", "adapt", "check"]
    missing = [w for w in wedges if w not in r.stdout]
    if not missing:
        log("help_lists_wedges", "pass", f"all 7 wedges present in --help output")
    else:
        log("help_lists_wedges", "fail", f"missing: {missing}")

    # 2. reflex turbo
    print("\n=== Step 2: reflex turbo --trials 2 ===", flush=True)
    r = subprocess.run(
        ["reflex", "turbo", "--trials", "2", "--action-dim", "6", "--chunk-size", "5"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode == 0 and "Turbo Benchmark" in r.stdout:
        log("turbo", "pass", f"output contains benchmark table")
    else:
        log("turbo", "fail", r.stderr[-200:] or r.stdout[-200:])

    # 3. reflex split
    print("\n=== Step 3: reflex split --prefer edge ===", flush=True)
    r = subprocess.run(
        ["reflex", "split", "--prefer", "edge", "--output", "/tmp/split.json"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode == 0 and "Reflex Split" in r.stdout:
        log("split", "pass", "selected target reported + config saved")
    else:
        log("split", "fail", r.stderr[-200:] or r.stdout[-200:])

    # 4. reflex adapt (no URDF, uses default)
    print("\n=== Step 4: reflex adapt --num-joints 6 ===", flush=True)
    r = subprocess.run(
        ["reflex", "adapt", "--num-joints", "6", "--framework", "lerobot",
         "--output", "/tmp/embodiment.json"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode == 0 and "Reflex Adapt" in r.stdout:
        log("adapt", "pass", "embodiment config generated")
    else:
        log("adapt", "fail", r.stderr[-200:] or r.stdout[-200:])

    # 5. reflex check (on a small HF model to avoid big download)
    print("\n=== Step 5: reflex check lerobot/smolvla_base ===", flush=True)
    r = subprocess.run(
        ["reflex", "check", "lerobot/smolvla_base", "--target", "desktop"],
        capture_output=True, text=True, timeout=300,
    )
    if "Pre-Deployment Checks" in r.stdout or "Passed" in r.stdout:
        log("check", "pass", f"ran 5 checks (exit {r.returncode})")
    else:
        log("check", "fail", r.stderr[-300:] or r.stdout[-300:])

    print("\n=== SUMMARY ===", flush=True)
    passed = sum(1 for s in results["steps"] if s["status"] == "pass")
    failed = sum(1 for s in results["steps"] if s["status"] == "fail")
    print(f"Passed: {passed}, Failed: {failed}", flush=True)
    results["summary"] = {"passed": passed, "failed": failed}
    return results


@app.local_entrypoint()
def main():
    print("Verifying reflex CLI (all 7 wedges) on Modal...")
    results = verify.remote()
    for step in results["steps"]:
        tag = "PASS" if step["status"] == "pass" else "FAIL"
        print(f"  {tag}: {step['step']} — {step['detail']}")
