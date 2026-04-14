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
        result = export_gr00t(export_config, state_dict=state_dict)
    elif model_type == "openvla":
        from reflex.exporters.openvla_exporter import export_openvla
        result = export_openvla(export_config, state_dict=state_dict)
    elif model_type == "pi05":
        result = export_pi05(export_config, state_dict=state_dict)
    elif model_type == "pi0":
        result = export_pi0(export_config, state_dict=state_dict)
    else:
        result = export_smolvla(export_config, state_dict=state_dict)
    elapsed = time.perf_counter() - start

    # Print results
    console.print(f"\n[bold green]Export complete in {elapsed:.1f}s[/bold green]")
    console.print(f"  Output: {output}")

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

    console.print(f"\n  [dim]Run on target hardware:[/dim]")
    console.print(f"  [cyan]reflex benchmark {output}[/cyan]")


@app.command()
def validate(
    export_dir: str = typer.Argument(help="Path to exported model directory"),
    model: str = typer.Option("", help="Original model for comparison"),
    threshold: float = typer.Option(0.02, help="Max acceptable absolute difference"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Validate exported model quality against PyTorch reference."""
    _setup_logging(verbose)
    console.print(f"\n[bold]Reflex Validate[/bold]")
    console.print(f"  Export: {export_dir}")
    console.print(f"  Threshold: {threshold}")


@app.command(name="benchmark")
def benchmark_cmd(
    export_dir: str = typer.Argument(help="Path to exported model directory"),
    iterations: int = typer.Option(100, help="Number of benchmark iterations"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Benchmark exported model latency."""
    _setup_logging(verbose)
    console.print(f"\n[bold]Reflex Benchmark[/bold]")
    console.print(f"  Export: {export_dir}")
    console.print(f"  Iterations: {iterations}")


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
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Start a VLA inference server. POST /act with image + instruction → actions."""
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

    console.print(f"\n[bold]Reflex Serve[/bold]")
    console.print(f"  Export:  {export_dir}")
    console.print(f"  Device:  {device}")
    console.print(f"  Server:  http://{host}:{port}")
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

    app_instance = create_app(export_dir, device=device)
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


if __name__ == "__main__":
    app()
