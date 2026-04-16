"""Reflex CLI — deploy VLA models to edge hardware."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from reflex import __version__
from reflex.config import ExportConfig, get_hardware_profile, HARDWARE_PROFILES

app = typer.Typer(
    name="reflex",
    help="Deploy any VLA model to any edge hardware. One command.",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"reflex {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", help="Show version and exit",
        callback=_version_callback, is_eager=True,
    ),
):
    pass


@app.command()
def export(
    model: str = typer.Argument(help="HuggingFace model ID or local checkpoint path"),
    target: str = typer.Option("desktop", help="Target hardware: orin-nano, orin, orin-64, thor, desktop"),
    output: str = typer.Option("./reflex_export", help="Output directory"),
    precision: str = typer.Option("fp16", help="Precision: fp16, fp8, int8"),
    opset: int = typer.Option(19, help="ONNX opset version"),
    chunk_size: int = typer.Option(50, help="Action chunk size"),
    no_validate: bool = typer.Option(False, help="Skip ONNX validation"),
    dry_run: bool = typer.Option(False, help="Check exportability without building engines"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Export a VLA model to ONNX + TensorRT for edge deployment."""
    _setup_logging(verbose)
    hardware = get_hardware_profile(target)

    console.print(f"\n[bold]Reflex Export[/bold]")
    console.print(f"  Model:     {model}")
    console.print(f"  Target:    {hardware.name} ({hardware.memory_gb}GB, {hardware.trt_precision})")
    console.print(f"  Precision: {precision}")
    console.print(f"  Output:    {output}")
    console.print()

    if dry_run:
        console.print("[dim]Checking exportability...[/dim]")
        from reflex.checkpoint import load_checkpoint, detect_model_type, validate_checkpoint

        state_dict, config = load_checkpoint(model)
        model_type = detect_model_type(state_dict)
        console.print(f"  Detected: {model_type or 'unknown'}")
        total_params = sum(v.numel() for v in state_dict.values())
        console.print(f"  Params:   {total_params / 1e6:.1f}M")

        warnings = validate_checkpoint(state_dict, model_type or "unknown")
        for w in warnings:
            console.print(f"  [yellow]Warning: {w}[/yellow]")

        # Check memory fit
        weight_gb = total_params * 2 / 1e9  # FP16
        if weight_gb > hardware.memory_gb * 0.7:
            console.print(f"  [red]Model ({weight_gb:.1f}GB) may not fit on {hardware.name} ({hardware.memory_gb}GB)[/red]")
        else:
            console.print(f"  [green]Model ({weight_gb:.1f}GB) fits on {hardware.name} ({hardware.memory_gb}GB)[/green]")

        if model_type is None:
            console.print("\n[yellow]Unknown model type — export may fail. Supported: smolvla, pi0, pi05.[/yellow]")
        else:
            console.print("\n[green]Dry run complete. Export should work.[/green]")
        raise typer.Exit()

    # Full export — auto-dispatch to the right exporter based on model type
    from reflex.checkpoint import load_checkpoint, detect_model_type
    from reflex.exporters.smolvla_exporter import export_smolvla
    from reflex.exporters.pi0_exporter import export_pi0, export_pi05
    from reflex.exporters.gr00t_exporter import export_gr00t

    # Load once, detect, then pass state_dict to the exporter (avoids double-load)
    console.print("[dim]Loading checkpoint...[/dim]")
    state_dict, _ = load_checkpoint(model)
    model_type = detect_model_type(state_dict) or "smolvla"
    console.print(f"  Detected: [bold]{model_type}[/bold]")

    export_config = ExportConfig(
        model_id=model,
        target=target,
        output_dir=output,
        precision=precision,
        opset=opset,
        action_chunk_size=chunk_size,
        validate=not no_validate,
    )

    import time
    start = time.perf_counter()
    if model_type == "gr00t":
        # Use the full-stack exporter (wraps action_encoder + DiT + action_decoder)
        # so `reflex serve` can run the standard denoising loop.
        from reflex.exporters.gr00t_exporter import export_gr00t_full
        result = export_gr00t_full(export_config, state_dict=state_dict)
    elif model_type == "openvla":
        from reflex.exporters.openvla_exporter import export_openvla
        result = export_openvla(export_config, state_dict=state_dict)
    elif model_type == "pi05":
        result = export_pi05(export_config, state_dict=state_dict)
    elif model_type == "pi0":
        result = export_pi0(export_config, state_dict=state_dict)
    else:
        result = export_smolvla(export_config, state_dict=state_dict)
    elapsed_expert = time.perf_counter() - start

    # Print expert results
    console.print(f"\n[bold green]Expert export complete in {elapsed_expert:.1f}s[/bold green]")

    if "files" in result:
        for name, path in result["files"].items():
            size = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
            console.print(f"  {name}: {path} ({size:.1f}MB)")

    if "metadata" in result and "onnx_validation" in result["metadata"]:
        val = result["metadata"]["onnx_validation"]
        status = "[green]PASS[/green]" if val["passed"] else "[red]FAIL[/red]"
        console.print(f"  Validation: {status} (max_diff={val['max_diff']:.2e})")

    if "metadata" in result and "expert" in result["metadata"]:
        meta = result["metadata"]["expert"]
        console.print(f"  Expert: {meta['num_layers']} layers, {meta['total_params_m']:.1f}M params")

    # For SmolVLA: also export the VLM pipeline (vision_encoder + text_embedder + decoder_prefill)
    # so `reflex serve` can run with real task-conditioned actions instead of noise.
    # Note: VLM weights come from the base SmolVLM2-500M (not the SmolVLA checkpoint's
    # fine-tuned VLM). Fine-tuned VLM weight transfer is tracked as a v0.3 item.
    if model_type == "smolvla":
        console.print("\n[dim]Exporting VLM pipeline (vision + text + decoder)...[/dim]")
        from reflex.exporters.vlm_prefix_exporter import export_vlm_prefix
        vlm_start = time.perf_counter()
        try:
            vlm_path = export_vlm_prefix(output_dir=output, opset=opset)
            elapsed_vlm = time.perf_counter() - vlm_start
            console.print(f"[bold green]VLM export complete in {elapsed_vlm:.1f}s[/bold green]")
            # Show VLM output files
            for fname in ("vision_encoder.onnx", "text_embedder.onnx", "decoder_prefill.onnx"):
                fpath = Path(output) / fname
                if fpath.exists():
                    data_path = fpath.with_suffix(".onnx.data")
                    size = fpath.stat().st_size / 1e6
                    if data_path.exists():
                        size += data_path.stat().st_size / 1e6
                    console.print(f"  {fname}: {size:.1f}MB")
            console.print(
                "  [dim]Note: VLM uses base SmolVLM2-500M weights. "
                "Fine-tuned SmolVLA VLM layers not yet preserved (v0.3 item).[/dim]"
            )
        except Exception as exc:
            console.print(f"[yellow]VLM export skipped: {exc}[/yellow]")
            console.print(
                "[yellow]Server will use dummy VLM conditioning (v0.1 fallback).[/yellow]"
            )

    total_elapsed = time.perf_counter() - start
    console.print(f"\n[bold]Total export: {total_elapsed:.1f}s[/bold]")
    console.print(f"  Output: {output}")
    console.print(f"\n  [dim]Run on target hardware:[/dim]")
    console.print(f"  [cyan]reflex bench {output}[/cyan]")


@app.command()
def validate(
    export_dir: str = typer.Argument("", help="Path to exported model directory"),
    model: str = typer.Option("", help="HuggingFace model ID for PyTorch reference (auto-detect from reflex_config.json if empty)"),
    threshold: float = typer.Option(
        1e-4,
        help="Max acceptable L2 abs diff per action dim. Default 1e-4 (changed from v0.1 stub default of 0.02).",
    ),
    num_cases: int = typer.Option(5, help="Number of seeded fixtures"),
    seed: int = typer.Option(0, help="RNG seed for fixtures + initial noise"),
    device: str = typer.Option("cpu", help="Device for PyTorch reference: cpu or cuda"),
    output_json: bool = typer.Option(False, "--output-json", help="Emit pure JSON instead of Rich tables"),
    init_ci: bool = typer.Option(False, "--init-ci", help="Emit .github/workflows/reflex-validate.yml and exit"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Validate exported ONNX vs PyTorch reference (round-trip parity)."""
    _setup_logging(verbose)

    if init_ci:
        from reflex.ci_template import emit_ci_template
        out = Path(".github/workflows/reflex-validate.yml")
        try:
            emit_ci_template(out, reflex_version=__version__)
        except FileExistsError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2)
        except Exception as exc:
            console.print(f"[red]Failed to emit CI template: {exc}[/red]")
            raise typer.Exit(2)
        console.print(f"[green]Wrote CI template:[/green] {out}")
        raise typer.Exit(0)

    if not export_dir:
        console.print("[red]export_dir is required (unless --init-ci is passed).[/red]")
        raise typer.Exit(2)

    if device not in ("cpu", "cuda"):
        console.print(f"[red]--device must be 'cpu' or 'cuda', got: {device}[/red]")
        raise typer.Exit(2)

    from reflex.validate_roundtrip import ValidateRoundTrip

    try:
        runner = ValidateRoundTrip(
            export_dir=Path(export_dir),
            model_id=model or None,
            threshold=threshold,
            num_test_cases=num_cases,
            seed=seed,
            device=device,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    try:
        result = runner.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Validation interrupted by user.[/yellow]")
        raise typer.Exit(130)
    except FileNotFoundError as exc:
        console.print(f"[red]Missing required file: {exc}[/red]")
        raise typer.Exit(2)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)
    except Exception as exc:
        if verbose:
            import traceback
            traceback.print_exc()
        console.print(f"[red]Validation failed with unexpected error: {exc}[/red]")
        console.print("[yellow]Re-run with --verbose for the full traceback.[/yellow]")
        raise typer.Exit(2)

    summary = result.get("summary", {})
    passed = bool(summary.get("passed", False))

    if output_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        console.print("\n[bold]Reflex Validate[/bold]")
        console.print(f"  Export: {export_dir}")
        console.print(f"  Model type: {result.get('model_type')}")
        console.print(f"  Threshold: {result.get('threshold')}")

        per_table = Table(title="Per-fixture results", show_header=True, header_style="bold")
        per_table.add_column("fixture_idx", justify="right")
        per_table.add_column("max_abs_diff", justify="right")
        per_table.add_column("mean_abs_diff", justify="right")
        per_table.add_column("passed", justify="center")
        for r in result.get("results", []):
            ok = bool(r.get("passed"))
            per_table.add_row(
                str(r.get("fixture_idx", "")),
                f"{float(r.get('max_abs_diff', 0)):.2e}",
                f"{float(r.get('mean_abs_diff', 0)):.2e}",
                "[green]PASS[/green]" if ok else "[red]FAIL[/red]",
            )
        console.print(per_table)

        sum_table = Table(title="Summary", show_header=True, header_style="bold")
        sum_table.add_column("metric")
        sum_table.add_column("value")
        sum_table.add_row("max_abs_diff_across_all", f"{float(summary.get('max_abs_diff_across_all', 0)):.2e}")
        sum_table.add_row("passed", "[green]PASS[/green]" if passed else "[red]FAIL[/red]")
        sum_table.add_row("num_cases", str(result.get("num_test_cases")))
        sum_table.add_row("seed", str(result.get("seed")))
        sum_table.add_row("threshold", str(result.get("threshold")))
        console.print(sum_table)

    raise typer.Exit(0 if passed else 1)


@app.command(name="bench")
def benchmark_cmd(
    export_dir: str = typer.Argument(help="Path to exported model directory"),
    iterations: int = typer.Option(100, help="Number of benchmark iterations"),
    warmup: int = typer.Option(20, help="Warmup iterations (excluded from stats)"),
    device: str = typer.Option("cuda", help="Device: cuda or cpu"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Benchmark exported model latency end-to-end (denoising loop on actual hardware).

    Loads the export, warms up, runs N iterations of the full denoising loop,
    reports mean/p50/p95/p99 latency. Use this to verify your install is fast
    before pointing a real robot at the server.
    """
    _setup_logging(verbose)
    import time as _t
    import numpy as np

    export_path = Path(export_dir)
    if not export_path.exists():
        console.print(f"[red]Export directory not found: {export_dir}[/red]")
        raise typer.Exit(1)

    onnx_files = list(export_path.glob("*.onnx"))
    if not onnx_files:
        console.print(f"[red]No ONNX file in {export_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Reflex Benchmark[/bold]")
    console.print(f"  Export:    {export_dir}")
    console.print(f"  Device:    {device}")
    console.print(f"  Warmup:    {warmup}")
    console.print(f"  Iterations: {iterations}")

    from reflex.runtime.server import ReflexServer
    server = ReflexServer(export_dir, device=device, strict_providers=False)
    console.print("[dim]Loading model...[/dim]")
    t0 = _t.perf_counter()
    server.load()
    load_s = _t.perf_counter() - t0
    if not server.ready:
        console.print("[red]Model failed to load.[/red]")
        raise typer.Exit(1)
    console.print(
        f"  Loaded:    {load_s:.1f}s  (mode={server._inference_mode})"
    )

    # Warmup
    console.print(f"[dim]Warming up ({warmup} iterations)...[/dim]")
    for _ in range(warmup):
        server.predict()

    # Bench
    console.print(f"[dim]Benchmarking ({iterations} iterations)...[/dim]")
    latencies: list[float] = []
    for _ in range(iterations):
        t0 = _t.perf_counter()
        server.predict()
        latencies.append((_t.perf_counter() - t0) * 1000)
    latencies.sort()

    mean = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    minv = latencies[0]
    maxv = latencies[-1]

    console.print(f"\n[bold]Per-chunk latency (10-step denoise loop):[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column(justify="right")
    table.add_row("min",  f"{minv:7.2f} ms")
    table.add_row("mean", f"{mean:7.2f} ms")
    table.add_row("p50",  f"{p50:7.2f} ms")
    table.add_row("p95",  f"{p95:7.2f} ms")
    table.add_row("p99",  f"{p99:7.2f} ms")
    table.add_row("max",  f"{maxv:7.2f} ms")
    table.add_row("hz",   f"{1000.0/mean:7.1f}")
    console.print(table)

    console.print(
        f"\n  [dim]Inference mode:[/dim] [bold]{server._inference_mode}[/bold]"
    )
    if server._inference_mode == "onnx_cpu" and device == "cuda":
        console.print(
            "  [yellow]Note: requested device=cuda but ended up on CPU. "
            "Install onnxruntime-gpu and CUDA 12 + cuDNN 9 for GPU performance.[/yellow]"
        )


@app.command()
def guard(
    action: str = typer.Argument(help="Action to check: 'init' to create config, 'check' to validate"),
    urdf: str = typer.Option("", help="URDF file path to extract joint limits"),
    config: str = typer.Option("", help="Safety config JSON file path"),
    output: str = typer.Option("./safety_config.json", help="Output path for safety config"),
    num_joints: int = typer.Option(6, help="Number of joints (when no URDF)"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Configure and test safety guardrails for VLA actions."""
    _setup_logging(verbose)

    from reflex.safety import ActionGuard, SafetyLimits

    if action == "init":
        if urdf:
            limits = SafetyLimits.from_urdf(urdf)
            console.print(f"[green]Extracted limits from URDF: {urdf}[/green]")
        else:
            limits = SafetyLimits.default(num_joints)
            console.print(f"[yellow]Using default limits for {num_joints} joints[/yellow]")

        console.print(f"  Joints: {len(limits.joint_names)}")
        for i, name in enumerate(limits.joint_names):
            console.print(
                f"    {name}: pos=[{limits.position_min[i]:.2f}, {limits.position_max[i]:.2f}], "
                f"vel_max={limits.velocity_max[i]:.2f}"
            )

        limits.save(output)
        console.print(f"\n[bold green]Safety config saved: {output}[/bold green]")
        console.print(f"[dim]Use with: reflex serve --safety-config {output}[/dim]")

    elif action == "check":
        if config:
            limits = SafetyLimits.from_json(config)
        elif urdf:
            limits = SafetyLimits.from_urdf(urdf)
        else:
            limits = SafetyLimits.default(num_joints)

        guard_instance = ActionGuard(limits=limits, mode="clamp")
        import numpy as np

        test_actions = np.random.randn(5, num_joints).astype(np.float32) * 5
        safe_actions, results = guard_instance.check(test_actions)

        console.print(f"\n[bold]Safety Check (5 random actions, range [-5, 5]):[/bold]")
        for i, r in enumerate(results):
            status = "[green]SAFE[/green]" if r.safe else "[red]CLAMPED[/red]" if r.clamped else "[red]REJECTED[/red]"
            console.print(f"  Action {i}: {status} ({len(r.violations)} violations, {r.check_time_ms:.3f}ms)")
            for v in r.violations[:3]:
                console.print(f"    {v}")

    else:
        console.print(f"[red]Unknown action: {action}. Use 'init' or 'check'.[/red]")
        raise typer.Exit(1)


@app.command()
def serve(
    export_dir: str = typer.Argument(help="Path to exported model directory"),
    port: int = typer.Option(8000, help="Server port"),
    host: str = typer.Option("0.0.0.0", help="Server host"),
    device: str = typer.Option("cuda", help="Device: cuda or cpu"),
    providers: str = typer.Option(
        "",
        help="Comma-separated ORT execution providers (e.g. "
             "'CUDAExecutionProvider,CPUExecutionProvider'). Overrides --device "
             "for provider selection when set.",
    ),
    no_strict_providers: bool = typer.Option(
        False,
        "--no-strict-providers",
        help="Allow silent fallback to CPU if the requested GPU provider fails "
             "to load. OFF by default — by default the server raises a loud "
             "error instead of silently falling back. Set this only if you "
             "explicitly want best-effort fallback.",
    ),
    safety_config: str = typer.Option(
        "",
        help="Path to a SafetyLimits JSON (from `reflex guard init`). When set, "
             "every returned action is clamped to the configured joint limits "
             "and violation counts are logged.",
    ),
    adaptive_steps: bool = typer.Option(
        False,
        "--adaptive-steps",
        help="Use reflex turbo adaptive denoising — stops the denoise loop "
             "early when velocity norm converges. Saves latency on easy tasks.",
    ),
    cloud_fallback: str = typer.Option(
        "",
        help="URL of a remote reflex serve (e.g. http://cloud-host:8000). When "
             "set, a reflex split orchestrator is configured for cloud-edge "
             "routing. v0.1 stores config only; full dispatch lands in Phase VI.",
    ),
    deadline_ms: float = typer.Option(
        0.0,
        help="Per-request deadline in ms. 0 = disabled. When set, predict() "
             "returns the last-known-good action instead if inference exceeds "
             "the deadline. Deadline misses are logged and counted.",
    ),
    max_batch: int = typer.Option(
        1,
        help="Multi-robot batching: serve up to N concurrent /act requests in "
             "one batched ONNX inference. Default 1 (no batching). "
             "Throughput-per-GPU scales sublinearly with batch size — typical "
             "wins are 2-3x at batch=4-8 for transformer-style VLAs.",
    ),
    batch_timeout_ms: float = typer.Option(
        5.0,
        help="With --max-batch > 1, wait up to this many ms after the first "
             "request before flushing the batch. Lower = lower per-request "
             "latency; higher = better batching efficiency under bursty load.",
    ),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Start a VLA inference server. POST /act with image + instruction → actions.

    Composable wedges: --safety-config (guard), --adaptive-steps (turbo),
    --cloud-fallback (split), --deadline-ms (WCET).
    """
    _setup_logging(verbose)

    export_path = Path(export_dir)
    if not export_path.exists():
        console.print(f"[red]Export directory not found: {export_dir}[/red]")
        console.print(f"[dim]Run 'reflex export' first to create an export.[/dim]")
        raise typer.Exit(1)

    onnx_files = list(export_path.glob("*.onnx"))
    if not onnx_files:
        console.print(f"[red]No ONNX files found in {export_dir}[/red]")
        raise typer.Exit(1)

    # Parse providers
    provider_list: list[str] | None = None
    if providers:
        provider_list = [p.strip() for p in providers.split(",") if p.strip()]

    # Detect the common "I pip installed onnxruntime instead of onnxruntime-gpu"
    # footgun before we spin up the server.
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        console.print(
            "[red]onnxruntime is not installed.[/red]\n"
            "For GPU: [cyan]pip install onnxruntime-gpu[/cyan]\n"
            "For CPU: [cyan]pip install onnxruntime[/cyan]"
        )
        raise typer.Exit(1)

    cuda_requested = (
        device == "cuda"
        or (provider_list and "CUDAExecutionProvider" in provider_list)
    )
    cuda_available_in_ort = "CUDAExecutionProvider" in available

    console.print(f"\n[bold]Reflex Serve[/bold]")
    console.print(f"  Export:  {export_dir}")
    console.print(f"  Device:  {device}")
    if provider_list:
        console.print(f"  Providers: {provider_list}")
    console.print(f"  Strict:  {not no_strict_providers}")
    console.print(f"  Server:  http://{host}:{port}")
    console.print(f"  [dim]ORT available providers: {available}[/dim]")

    # Composed wedges summary
    composed = []
    if safety_config:
        composed.append(f"[cyan]safety[/cyan]={safety_config}")
    if adaptive_steps:
        composed.append("[cyan]adaptive-steps[/cyan]")
    if cloud_fallback:
        composed.append(f"[cyan]cloud-fallback[/cyan]={cloud_fallback}")
    if deadline_ms > 0:
        composed.append(f"[cyan]deadline[/cyan]={deadline_ms:.0f}ms")
    if max_batch > 1:
        composed.append(f"[cyan]batch[/cyan]={max_batch}@{batch_timeout_ms:.0f}ms")
    if composed:
        console.print(f"  Wedges:  {' · '.join(composed)}")

    if cuda_requested and not cuda_available_in_ort:
        console.print(
            "\n[red]⚠ CUDAExecutionProvider not available in this ORT install.[/red]\n"
            "  Likely cause: you installed `onnxruntime` (CPU-only).\n"
            "  Fix:   [cyan]pip uninstall onnxruntime && pip install onnxruntime-gpu[/cyan]\n"
            "  Also:  ORT 1.20+ requires CUDA 12.x + cuDNN 9.x on the library path.\n"
            "  Or:    pass [cyan]--device cpu[/cyan] to explicitly use CPU.\n"
            "  Or:    pass [cyan]--no-strict-providers[/cyan] to allow CPU fallback anyway.\n"
        )
        if not no_strict_providers:
            raise typer.Exit(1)

    console.print()
    console.print(f"  [dim]Endpoints:[/dim]")
    console.print(f"  [cyan]POST /act[/cyan]     — send image + instruction, get actions")
    console.print(f"  [cyan]GET  /health[/cyan]  — check server status")
    console.print(f"  [cyan]GET  /config[/cyan]  — view model config")
    console.print()

    try:
        from reflex.runtime.server import create_app
        import uvicorn
    except ImportError:
        console.print("[red]Install serve dependencies: pip install 'reflex-vla[serve]'[/red]")
        raise typer.Exit(1)

    app_instance = create_app(
        export_dir,
        device=device,
        providers=provider_list,
        strict_providers=not no_strict_providers,
        safety_config=safety_config or None,
        adaptive_steps=adaptive_steps,
        cloud_fallback_url=cloud_fallback,
        deadline_ms=deadline_ms if deadline_ms > 0 else None,
        max_batch=max_batch,
        batch_timeout_ms=batch_timeout_ms,
    )
    console.print("[bold green]Starting server...[/bold green]")
    uvicorn.run(app_instance, host=host, port=port, log_level="info" if verbose else "warning")


@app.command()
def targets():
    """List supported hardware targets."""
    table = Table(title="Supported Hardware Targets")
    table.add_column("Target", style="cyan")
    table.add_column("Name")
    table.add_column("Memory")
    table.add_column("FP8")
    table.add_column("Precision")

    for key, hw in HARDWARE_PROFILES.items():
        table.add_row(
            key,
            hw.name,
            f"{hw.memory_gb} GB",
            "yes" if hw.fp8_support else "no",
            hw.trt_precision,
        )

    console.print(table)


@app.command()
def models():
    """List supported VLA models and their export status."""
    from reflex.checkpoint import SUPPORTED_MODELS

    table = Table(title="Supported VLA Models")
    table.add_column("Type", style="cyan")
    table.add_column("HF ID")
    table.add_column("Params")
    table.add_column("Action head")
    table.add_column("Export")

    status_map = {
        "smolvla": "[green]✓ ONNX + validated[/green]",
        "pi0": "[green]✓ ONNX + validated[/green]",
        "pi05": "[green]✓ ONNX + AdaRMSNorm[/green]",
        "gr00t": "[green]✓ DiT + AdaLN + validated[/green]",
        "openvla": "[yellow]use optimum-onnx; Reflex only ships postprocess helpers[/yellow]",
    }

    for key, info in SUPPORTED_MODELS.items():
        table.add_row(
            key,
            info["hf_id"],
            f"{info['params_m']}M",
            info["action_head"],
            status_map.get(key, "[yellow]planned[/yellow]"),
        )

    console.print(table)
    console.print("\n[dim]Usage:[/dim] [cyan]reflex export <hf_id>[/cyan] — auto-detects model type.")


@app.command()
def turbo(
    strategy: str = typer.Option("adaptive", help="Strategy: fixed, adaptive, cuda_graph"),
    action_dim: int = typer.Option(6, help="Action dimension to benchmark"),
    chunk_size: int = typer.Option(10, help="Action chunk size"),
    trials: int = typer.Option(5, help="Number of benchmark trials"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Benchmark action-head denoising optimizations (Wedge 4).

    Compares fixed-step vs adaptive denoising on a synthetic model to show
    the latency headroom that flow-matching VLAs have.
    """
    _setup_logging(verbose)
    from reflex.kernels.turbo import TurboOptimizer, TurboConfig
    import torch
    import torch.nn as nn

    class _DummyModel(nn.Module):
        def __init__(self, action_dim=6, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(action_dim + 1 + hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim),
            )
            self.pos_embed = nn.Embedding(100, hidden)

        def forward(self, noisy, timestep, pos_ids):
            b, c, _ = noisy.shape
            pos = self.pos_embed(pos_ids)
            t = timestep.unsqueeze(1).unsqueeze(2).expand(b, c, 1)
            return self.net(torch.cat([noisy, t, pos], dim=-1))

    console.print(f"\n[bold]Reflex Turbo[/bold]")
    console.print(f"  Strategy: {strategy}")
    console.print(f"  action_dim={action_dim}, chunk_size={chunk_size}, trials={trials}")

    model = _DummyModel(action_dim=action_dim, hidden=16)
    optimizer = TurboOptimizer(TurboConfig(strategy=strategy))
    results = optimizer.benchmark_strategies(
        model, action_dim=action_dim, chunk_size=chunk_size, device="cpu", n_trials=trials,
    )

    table = Table(title="Turbo Benchmark")
    table.add_column("Strategy", style="cyan")
    table.add_column("Mean ms")
    table.add_column("Steps used")
    table.add_column("Converged early")
    for name, runs in results.items():
        mean_ms = sum(r.latency_ms for r in runs) / len(runs)
        mean_steps = sum(r.steps_used for r in runs) / len(runs)
        converged = sum(1 for r in runs if r.converged_early)
        table.add_row(name, f"{mean_ms:.1f}", f"{mean_steps:.1f}", f"{converged}/{len(runs)}")
    console.print(table)


@app.command()
def split(
    cloud_url: str = typer.Option("", help="Cloud inference URL (optional)"),
    prefer: str = typer.Option("edge", help="Prefer: edge, cloud, auto"),
    fallback_mode: str = typer.Option("last_action", help="Fallback: last_action, zero, edge_small"),
    action_dim: int = typer.Option(6, help="Action dimension"),
    chunk_size: int = typer.Option(50, help="Action chunk size"),
    output: str = typer.Option("./split_config.json", help="Output path for config"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Configure cloud-edge split inference orchestration (Wedge 5)."""
    _setup_logging(verbose)
    from reflex.runtime.split import SplitOrchestrator, SplitConfig, InferenceTarget

    cfg = SplitConfig(
        cloud_url=cloud_url,
        prefer=prefer,
        fallback_mode=fallback_mode,
    )
    orch = SplitOrchestrator(cfg)
    target = orch._select_target()

    console.print(f"\n[bold]Reflex Split[/bold]")
    console.print(f"  Cloud URL:       {cloud_url or '(none)'}")
    console.print(f"  Prefer:          {prefer}")
    console.print(f"  Fallback mode:   {fallback_mode}")
    console.print(f"  Selected target: [cyan]{target.value}[/cyan]")

    if target == InferenceTarget.FALLBACK:
        fb = orch._get_fallback_actions(action_dim=action_dim, chunk_size=chunk_size)
        console.print(f"  Fallback shape:  {fb.shape}")

    import json
    Path(output).write_text(json.dumps({
        "cloud_url": cloud_url,
        "prefer": prefer,
        "fallback_mode": fallback_mode,
        "selected_target": target.value,
        "action_dim": action_dim,
        "chunk_size": chunk_size,
    }, indent=2))
    console.print(f"\n[green]Split config saved: {output}[/green]")


@app.command()
def adapt(
    urdf: str = typer.Option("", help="URDF file path"),
    name: str = typer.Option("generic_arm", help="Robot name (when no URDF)"),
    num_joints: int = typer.Option(6, help="Number of joints (when no URDF)"),
    source_dim: int = typer.Option(0, help="Source VLA action dim (0 = same as robot)"),
    framework: str = typer.Option("lerobot", help="Target framework: lerobot, openpi, gr00t"),
    output: str = typer.Option("./embodiment_config.json", help="Output path"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Adapt a VLA to a new robot embodiment (Wedge 6)."""
    _setup_logging(verbose)
    from reflex.models.adapt import EmbodimentAdapter

    if urdf:
        adapter = EmbodimentAdapter.from_urdf(urdf)
        console.print(f"[green]Loaded URDF: {urdf}[/green]")
    else:
        adapter = EmbodimentAdapter.default(name, num_joints)
        console.print(f"[yellow]Using default {num_joints}-joint embodiment[/yellow]")

    console.print(f"\n[bold]Reflex Adapt[/bold]")
    console.print(f"  Robot:        {adapter.config.name}")
    console.print(f"  Joints:       {adapter.config.num_joints}")
    console.print(f"  Action dim:   {adapter.config.action_dim}")
    console.print(f"  Framework:    {framework}")

    # Action mapping if source dim specified
    src = source_dim or adapter.config.action_dim
    mapping = adapter.create_mapping(source_dim=src)
    console.print(f"\n  Action mapping: [cyan]{mapping.mapping_type}[/cyan] "
                  f"({src}-dim source → {adapter.config.action_dim}-dim target)")

    # Framework config
    fw_config = adapter.generate_framework_config(framework)
    adapter.config.save(output)
    console.print(f"\n[green]Embodiment config saved: {output}[/green]")
    console.print(f"[dim]Framework config summary:[/dim]")
    for k, v in list(fw_config.items())[:5]:
        console.print(f"  {k}: {v}")


@app.command()
def check(
    checkpoint: str = typer.Argument(help="HuggingFace ID or local path"),
    target: str = typer.Option("desktop", help="Target hardware: orin-nano, orin, orin-64, thor, desktop"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Pre-deployment validation — 5 checks before shipping (Wedge 7)."""
    _setup_logging(verbose)
    from reflex.validate_training import run_all_checks

    console.print(f"\n[bold]Reflex Check[/bold]")
    console.print(f"  Checkpoint: {checkpoint}")
    console.print(f"  Target:     {target}")
    console.print()

    results = run_all_checks(checkpoint, target=target)

    table = Table(title="Pre-Deployment Checks")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail")
    n_pass = 0
    for r in results:
        status = "[green]PASS[/green]" if r.passed else \
                 f"[yellow]WARN[/yellow]" if r.severity == "warning" else "[red]FAIL[/red]"
        if r.passed:
            n_pass += 1
        table.add_row(r.name, status, r.detail[:80])

    console.print(table)
    console.print(f"\n  Passed: [bold]{n_pass}/{len(results)}[/bold]")
    if n_pass < len(results):
        raise typer.Exit(1)


@app.command()
def doctor():
    """Diagnose common Reflex install + GPU issues. Run this BEFORE opening a bug.

    Checks Python version, torch + CUDA availability, ONNX Runtime install
    + execution providers, TensorRT (trtexec), HuggingFace Hub auth, and
    common version mismatches that cause the silent CPU fallback footgun.
    """
    import platform
    import shutil
    import sys

    table = Table(title="Reflex Doctor")
    table.add_column("Check", style="cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail")

    def add(name: str, ok: bool, detail: str):
        symbol = "[green]✓[/green]" if ok else "[yellow]⚠[/yellow]"
        table.add_row(name, symbol, detail)

    # Python
    py = sys.version_info
    add(
        "Python version",
        py >= (3, 10),
        f"{py.major}.{py.minor}.{py.micro} (need ≥3.10)",
    )

    # OS / architecture
    add("Platform", True, f"{platform.system()} {platform.machine()}")

    # torch + CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        cuda_detail = (
            f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
            f"available={cuda_ok}"
        )
        if cuda_ok:
            cuda_detail += f", devices={torch.cuda.device_count()}, "
            cuda_detail += f"name={torch.cuda.get_device_name(0)}"
        add("torch + CUDA", cuda_ok, cuda_detail)
    except ImportError as e:
        add("torch + CUDA", False, f"torch not installed: {e}")

    # ONNX Runtime + execution providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_trt = "TensorrtExecutionProvider" in providers
        has_cuda = "CUDAExecutionProvider" in providers
        ort_detail = f"ort {ort.__version__}, providers={providers}"
        add(
            "ONNX Runtime",
            True,
            ort_detail,
        )
        add(
            "  → CUDAExecutionProvider",
            has_cuda,
            "available" if has_cuda else (
                "NOT available — install onnxruntime-gpu or check CUDA 12 + cuDNN 9 system libs"
            ),
        )
        add(
            "  → TensorrtExecutionProvider",
            has_trt,
            "available — reflex serve will auto-prefer this" if has_trt else
            "NOT available — TRT FP16 disabled, will use CUDA EP",
        )
    except ImportError:
        add(
            "ONNX Runtime",
            False,
            "not installed — run `pip install onnxruntime-gpu` (or [onnx] for CPU)",
        )

    # ONNX (the format library)
    try:
        import onnx
        add("onnx (graph format)", True, f"version {onnx.__version__}")
    except ImportError:
        add("onnx (graph format)", False, "not installed — included in core deps now")

    # onnxscript (needed for torch.onnx.export new path)
    try:
        import onnxscript
        add("onnxscript", True, f"version {onnxscript.__version__}")
    except ImportError:
        add("onnxscript", False, "not installed — needed by torch.onnx.export")

    # transformers + huggingface_hub
    try:
        import transformers
        add("transformers", True, f"version {transformers.__version__}")
    except ImportError:
        add("transformers", False, "not installed — needed for some exporters")
    try:
        import huggingface_hub
        add("huggingface_hub", True, f"version {huggingface_hub.__version__}")
    except ImportError:
        add("huggingface_hub", False, "not installed — needed to download checkpoints")

    # FastAPI + uvicorn (for serve)
    try:
        import fastapi
        import uvicorn
        add("fastapi + uvicorn", True, f"fastapi {fastapi.__version__} / uvicorn {uvicorn.__version__}")
    except ImportError:
        add(
            "fastapi + uvicorn",
            False,
            "not installed — run `pip install reflex-vla[serve,gpu]` for the server",
        )

    # safetensors
    try:
        import safetensors
        add("safetensors", True, f"version {safetensors.__version__}")
    except ImportError:
        add("safetensors", False, "not installed — needed to load checkpoints")

    # trtexec (for building .trt engines via reflex export)
    trtexec_path = shutil.which("trtexec")
    add(
        "trtexec (TensorRT)",
        bool(trtexec_path),
        trtexec_path or "not on PATH — TRT engine build skipped during reflex export "
                         "(install Jetpack on Jetson, or use nvcr.io/nvidia/tensorrt container)",
    )

    # Disk space at /tmp (where exports default)
    try:
        usage = shutil.disk_usage("/tmp")
        free_gb = usage.free / 1e9
        add(
            "Free disk in /tmp",
            free_gb > 10,
            f"{free_gb:.1f} GB free (need ~10 GB for largest model export)",
        )
    except Exception as e:
        add("Free disk in /tmp", False, str(e))

    # HuggingFace cache
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    if os.path.exists(hf_home):
        try:
            usage = shutil.disk_usage(hf_home)
            add("HF cache disk", usage.free > 10e9, f"{hf_home} ({usage.free / 1e9:.1f} GB free)")
        except Exception:
            pass

    # Reflex itself
    try:
        from reflex import __version__ as reflex_version
        add("reflex-vla", True, f"version {reflex_version}")
    except Exception as e:
        add("reflex-vla", False, str(e))

    console.print(table)
    console.print(
        "\n[dim]If something here is unexpected, see "
        "[cyan]docs/getting_started.md → Troubleshooting[/cyan] before "
        "opening an issue.[/dim]"
    )


if __name__ == "__main__":
    app()
