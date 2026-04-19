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
    monolithic: bool = typer.Option(
        True,
        "--monolithic/--decomposed",
        help="Export path selector. Default: --monolithic (the cos=+1.000000 verified "
             "path, one ONNX file). Opt into --decomposed only if you specifically need "
             "the 5-stage export for debugging; --decomposed is the older path with "
             "known correctness gaps. Monolithic requires `pip install "
             "'reflex-vla[monolithic]'` (pins transformers==5.3.0).",
    ),
    num_steps: int = typer.Option(
        10,
        help="Denoise steps baked into the monolithic ONNX. "
             "Canonical flow-matching = 10 (SmolVLA, pi0, pi0.5); use 1 for exact "
             "one-shot Euler. GR00T (DDPM) uses 4 as its runtime default. "
             "Only used when --monolithic is set.",
    ),
):
    """Export a VLA model to ONNX + TensorRT for edge deployment."""
    _setup_logging(verbose)
    hardware = get_hardware_profile(target)

    if monolithic:
        console.print(f"\n[bold]Reflex Export (monolithic, cos=1.0 verified path)[/bold]")
        console.print(f"  Model:      {model}")
        console.print(f"  Output:     {output}")
        console.print(f"  num_steps:  {num_steps}")
        console.print()

        if dry_run:
            console.print("[yellow]--dry-run not supported with --monolithic yet (v0.3 item). "
                          "Re-run without --dry-run to export.[/yellow]")
            raise typer.Exit()

        try:
            from reflex.exporters.monolithic import export_monolithic
        except ImportError as exc:
            console.print(f"[red]{exc}[/red]")
            console.print("\n[cyan]Fix: pip install 'reflex-vla[monolithic]' "
                          "(pins transformers==5.3.0; use a clean venv to avoid "
                          "the base transformers<5.0 conflict)[/cyan]")
            raise typer.Exit(2)

        import time
        start = time.perf_counter()
        try:
            result = export_monolithic(model, output, num_steps=num_steps, target=target)
        except ImportError as exc:
            console.print(f"[red]Missing monolithic dep: {exc}[/red]")
            raise typer.Exit(2)
        elapsed = time.perf_counter() - start
        console.print(f"\n[bold green]Monolithic export complete in {elapsed:.1f}s[/bold green]")
        console.print(f"  ONNX: {result['onnx_path']}")
        console.print(f"  Size: {result['size_mb']:.1f} MB")

        try:
            from reflex.verification_report import write_verification_report
            report_path = write_verification_report(output, parity=None)
            console.print(f"  Verification manifest: {report_path}")
        except Exception as exc:
            console.print(f"[yellow]Verification manifest skipped: {exc}[/yellow]")

        console.print(f"\n  [dim]Next:[/dim] [cyan]reflex serve {output}[/cyan]")
        raise typer.Exit(0)

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
            # Pass the loaded state_dict so the VLM exporter can overlay the
            # fine-tuned vision/text weights instead of using BASE SmolVLM2.
            vlm_path = export_vlm_prefix(
                output_dir=output, opset=opset, state_dict=state_dict
            )
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

        # Save state_proj weights from checkpoint so the VLM orchestrator can
        # project robot state through the REAL trained matrix instead of the
        # random init we were falling back to (that was silently destroying
        # state information in every prefix — the ONE bug hiding behind all
        # the others, found by the PyTorch-vs-ONNX diff on 2026-04-17).
        try:
            import numpy as np
            sp_w_keys = [k for k in state_dict if k.endswith("state_proj.weight")]
            sp_b_keys = [k for k in state_dict if k.endswith("state_proj.bias")]
            if sp_w_keys:
                sp_w = state_dict[sp_w_keys[0]].detach().cpu().numpy().astype(
                    np.float32
                )
                np.save(Path(output) / "state_proj_weight.npy", sp_w)
                console.print(
                    f"  state_proj weight: {sp_w.shape} → state_proj_weight.npy"
                )
            if sp_b_keys:
                sp_b = state_dict[sp_b_keys[0]].detach().cpu().numpy().astype(
                    np.float32
                )
                np.save(Path(output) / "state_proj_bias.npy", sp_b)
                console.print(
                    f"  state_proj bias: {sp_b.shape} → state_proj_bias.npy"
                )
            if not sp_w_keys:
                console.print(
                    "  [yellow]WARNING: no state_proj weight in checkpoint — "
                    "orchestrator will fall back to random init and state "
                    "conditioning will be garbage.[/yellow]"
                )
        except Exception as exc:
            console.print(f"[yellow]state_proj save failed: {exc}[/yellow]")

        # Copy LeRobot policy normalizer/unnormalizer from the HF repo into the
        # export dir. Without these, the model receives un-normalized state and
        # returns actions in normalized space — producing garbage trajectories
        # in sim. Critical for LIBERO / real-robot eval success.
        if "/" in model and not Path(model).exists():
            try:
                from huggingface_hub import hf_hub_download

                console.print(
                    "\n[dim]Copying policy preprocessor/postprocessor stats...[/dim]"
                )
                import shutil

                stats_files = [
                    "policy_preprocessor.json",
                    "policy_postprocessor.json",
                    "policy_preprocessor_step_5_normalizer_processor.safetensors",
                    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                ]
                copied = 0
                for fname in stats_files:
                    try:
                        src = hf_hub_download(repo_id=model, filename=fname)
                        shutil.copy(src, Path(output) / fname)
                        copied += 1
                    except Exception:
                        # Not all SmolVLA checkpoints ship these (e.g. base)
                        pass
                if copied:
                    console.print(
                        f"  Copied {copied}/{len(stats_files)} normalizer files "
                        f"→ {output}"
                    )
                else:
                    console.print(
                        "  [dim]No normalizer files found in checkpoint "
                        "(base model or older format) — adapter will skip "
                        "normalization.[/dim]"
                    )
            except Exception as exc:
                console.print(
                    f"[yellow]Normalizer copy skipped: {exc}[/yellow]"
                )

    total_elapsed = time.perf_counter() - start
    console.print(f"\n[bold]Total export: {total_elapsed:.1f}s[/bold]")
    console.print(f"  Output: {output}")

    try:
        from reflex.verification_report import write_verification_report
        report_path = write_verification_report(output, parity=None)
        console.print(f"  [dim]Verification manifest: {report_path}[/dim]")
    except Exception as exc:
        console.print(f"[yellow]Verification manifest skipped: {exc}[/yellow]")

    console.print(f"\n  [dim]Run on target hardware:[/dim]")
    console.print(f"  [cyan]reflex bench {output}[/cyan]")


@app.command()
def validate(
    target: str = typer.Argument("", help="Export directory OR HuggingFace model ID (with --pre-export)"),
    model: str = typer.Option("", help="HuggingFace model ID for PyTorch reference (auto-detect from reflex_config.json if empty)"),
    threshold: float = typer.Option(
        1e-4,
        help="Max acceptable L2 abs diff per action dim. Default 1e-4.",
    ),
    num_cases: int = typer.Option(5, help="Number of seeded fixtures"),
    seed: int = typer.Option(0, help="RNG seed for fixtures + initial noise"),
    device: str = typer.Option("cpu", help="Device for PyTorch reference: cpu or cuda"),
    output_json: bool = typer.Option(False, "--output-json", help="Emit pure JSON instead of Rich tables"),
    init_ci: bool = typer.Option(False, "--init-ci", help="Emit .github/workflows/reflex-validate.yml and exit"),
    quick: bool = typer.Option(
        False, "--quick",
        help="Fast static checks only (file exists, ONNX loadable, no NaN). Skip parity harness.",
    ),
    pre_export: bool = typer.Option(
        False, "--pre-export",
        help="Check a raw checkpoint before exporting. Takes model ID, not export dir.",
    ),
    hardware: str = typer.Option("desktop", help="Hardware target for --pre-export memory check"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Validate an export: full parity (default), static checks (--quick), or pre-export checkpoint health (--pre-export)."""
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

    if not target:
        console.print("[red]Export directory or model ID is required (unless --init-ci).[/red]")
        raise typer.Exit(2)

    # --pre-export: check a raw checkpoint (replaces old `reflex check`)
    if pre_export:
        from reflex.validate_training import run_all_checks
        console.print(f"\n[bold]Reflex Validate (pre-export)[/bold]")
        console.print(f"  Checkpoint: {target}")
        console.print(f"  Target:     {hardware}\n")

        results = run_all_checks(target, target=hardware)
        table = Table(title="Pre-export checks")
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Detail")
        n_pass = 0
        for r in results:
            status = "[green]PASS[/green]" if r.passed else (
                "[yellow]WARN[/yellow]" if r.severity == "warning" else "[red]FAIL[/red]"
            )
            if r.passed:
                n_pass += 1
            table.add_row(r.name, status, r.detail[:80])
        console.print(table)
        console.print(f"\n  Passed: [bold]{n_pass}/{len(results)}[/bold]")
        raise typer.Exit(0 if n_pass == len(results) else 1)

    # --quick: static checks on an export directory (faster than full parity)
    if quick:
        export_path = Path(target)
        console.print(f"\n[bold]Reflex Validate (--quick)[/bold]")
        console.print(f"  Export: {export_path}\n")

        table = Table(title="Static export checks")
        table.add_column("Check", style="cyan")
        table.add_column("Status")
        table.add_column("Detail")
        n_pass = n_total = 0

        def _check(name: str, ok: bool, detail: str) -> None:
            nonlocal n_pass, n_total
            n_total += 1
            if ok:
                n_pass += 1
            status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
            table.add_row(name, status, detail[:80])

        _check("export_dir exists", export_path.exists(), str(export_path))
        config_path = export_path / "reflex_config.json"
        _check("reflex_config.json", config_path.exists(), str(config_path))

        config: dict = {}
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                _check("config parses", True, f"{len(config)} keys")
            except Exception as e:
                _check("config parses", False, str(e))

        # Check each expected ONNX file
        import onnxruntime as ort
        import numpy as np
        for fname in ("expert_stack.onnx", "vision_encoder.onnx", "text_embedder.onnx", "decoder_prefill.onnx"):
            fpath = export_path / fname
            if fpath.exists():
                try:
                    sess = ort.InferenceSession(str(fpath), providers=["CPUExecutionProvider"])
                    inputs = [inp.name for inp in sess.get_inputs()]
                    _check(f"{fname} loads", True, f"inputs={inputs}")
                except Exception as e:
                    _check(f"{fname} loads", False, str(e)[:80])
            else:
                # Only the expert_stack is required; VLM files are optional for non-SmolVLA
                if fname == "expert_stack.onnx":
                    _check(f"{fname} present", False, "missing (required)")
                else:
                    table.add_row(fname, "[dim]skipped[/dim]", "not present")

        console.print(table)
        console.print(f"\n  Passed: [bold]{n_pass}/{n_total}[/bold]")
        raise typer.Exit(0 if n_pass == n_total else 1)

    # Default: full ONNX-vs-PyTorch parity harness
    export_dir = target  # rename for legacy code paths below

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

    try:
        from reflex.verification_report import write_verification_report
        report_path = write_verification_report(export_dir, parity=result)
        if not output_json:
            console.print(f"  [dim]Updated verification receipt: {report_path}[/dim]")
    except Exception as exc:
        if not output_json:
            console.print(f"[yellow]Verification receipt update skipped: {exc}[/yellow]")

    raise typer.Exit(0 if passed else 1)


@app.command(name="bench")
def benchmark_cmd(
    export_dir: str = typer.Argument(help="Path to exported model directory"),
    iterations: int = typer.Option(100, help="Number of benchmark iterations"),
    warmup: int = typer.Option(20, help="Warmup iterations (excluded from stats)"),
    device: str = typer.Option("cuda", help="Device: cuda or cpu"),
    benchmark: str = typer.Option(
        "",
        "--benchmark",
        help="Also run task-success eval: simpler, maniskill (requires pip install 'reflex-vla[eval]'). LIBERO archived 2026-04-17 — see archive/scripts/.",
    ),
    episodes_per_task: int = typer.Option(
        10, help="Episodes per task for --benchmark (full suites use 50)"
    ),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Benchmark exported model — latency (default) and optional task success.

    Default: loads the export, warms up, runs N iterations of the denoising loop,
    reports mean/p50/p95/p99 latency.

    With --benchmark <suite>: also runs task-success evaluation on the named
    simulation benchmark (SimplerEnv, ManiSkill). Requires the [eval] extra —
    sim dependencies are not in the base install.

    LIBERO was archived on 2026-04-17 — reflex's product wedge is deployment
    parity + latency, not sim benchmarking. Archived scripts live at
    archive/scripts/ if you want to resurrect them.
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

    # If --benchmark was requested, gate on the eval extra being installed
    if benchmark:
        try:
            import vla_eval  # noqa: F401
        except ImportError:
            console.print(
                f"[red]--benchmark {benchmark} requires the eval extra.[/red]\n"
                f"  Install with: [cyan]pip install 'reflex-vla[eval]'[/cyan]\n"
                f"  Or run without --benchmark for latency-only.",
            )
            raise typer.Exit(2)
        valid = ("simpler", "maniskill")
        if benchmark not in valid:
            console.print(f"[red]Unknown benchmark '{benchmark}'. Try one of: {', '.join(valid)}[/red]")
            raise typer.Exit(2)

    console.print(f"\n[bold]Reflex Benchmark[/bold]")
    console.print(f"  Export:    {export_dir}")
    console.print(f"  Device:    {device}")
    console.print(f"  Warmup:    {warmup}")
    console.print(f"  Iterations: {iterations}")
    if benchmark:
        console.print(f"  Benchmark: [cyan]{benchmark}[/cyan] ({episodes_per_task} eps/task)")

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

    # Task-success evaluation (optional, gated on --benchmark flag + [eval] extra)
    if benchmark:
        console.print(f"\n[bold]Task-success eval: {benchmark}[/bold]")
        try:
            from reflex.eval import run_task_benchmark
        except ImportError as exc:
            console.print(
                f"[red]reflex.eval module missing: {exc}[/red]\n"
                f"  The benchmark-plugin framework ships in v0.2 — see GOALS.yaml."
            )
            raise typer.Exit(2)

        eval_result = run_task_benchmark(
            benchmark,
            export_dir=export_dir,
            episodes_per_task=episodes_per_task,
            device=device,
        )
        success_rate = eval_result.get("success_rate", 0.0)
        console.print(f"\n  Task success: [bold]{success_rate * 100:.1f}%[/bold] "
                      f"({eval_result.get('episodes_completed', 0)} episodes)")


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


@app.command(name="ros2-serve")
def ros2_serve(
    export_dir: str = typer.Argument(help="Path to exported model directory"),
    device: str = typer.Option("cuda", help="ORT execution device: cuda or cpu"),
    image_topic: str = typer.Option(
        "/camera/image_raw",
        help="sensor_msgs/Image topic for observation frames",
    ),
    state_topic: str = typer.Option(
        "/joint_states",
        help="sensor_msgs/JointState topic — .position field becomes the state vector",
    ),
    task_topic: str = typer.Option(
        "/reflex/task",
        help="std_msgs/String topic for the text instruction",
    ),
    action_topic: str = typer.Option(
        "/reflex/actions",
        help="std_msgs/Float32MultiArray topic — published chunk, flattened",
    ),
    rate_hz: float = typer.Option(20.0, help="Inference rate (Hz)"),
    safety_config: str = typer.Option("", help="Path to SafetyLimits JSON"),
    node_name: str = typer.Option("reflex_vla", help="ROS2 node name"),
):
    """Run a ROS2 node wrapping reflex inference.

    Requires ROS2 installed via apt or robostack (rclpy is NOT pip-installable).
    Source your ROS2 environment before running:

        source /opt/ros/humble/setup.bash   # or iron / jazzy
        reflex ros2-serve ./my_export/
    """
    try:
        from reflex.runtime.ros2_bridge import run_ros2_bridge
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    console.print(f"[bold green]Starting reflex ros2 bridge[/bold green]")
    console.print(f"  export_dir: {export_dir}")
    console.print(f"  node_name: {node_name}")
    console.print(f"  rate_hz: {rate_hz}")
    console.print(f"  subs: {image_topic}, {state_topic}, {task_topic}")
    console.print(f"  pub:  {action_topic}")
    try:
        run_ros2_bridge(
            export_dir,
            device=device,
            safety_config=safety_config or None,
            image_topic=image_topic,
            state_topic=state_topic,
            task_topic=task_topic,
            action_topic=action_topic,
            rate_hz=rate_hz,
            node_name=node_name,
        )
    except ImportError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)


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
def distill(
    teacher: str = typer.Argument(help="Path to exported teacher model directory"),
    output: str = typer.Option("./distilled_student", help="Output directory for distilled student"),
    recipe: str = typer.Option("dmpo", help="Distillation recipe: dmpo (default), pi_flow"),
    dataset: str = typer.Option("libero_10", help="Training dataset: libero_10, libero_long, droid, open_x"),
    episodes: int = typer.Option(1000, help="Training episodes"),
    steps: int = typer.Option(10000, help="Training steps"),
    batch_size: int = typer.Option(32, help="Batch size"),
    lr: float = typer.Option(1e-4, help="Learning rate"),
    device: str = typer.Option("cuda", help="Device: cuda or cpu"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """Distill a flow-matching VLA into a one-step student (5-10x faster).

    DMPO recipe (arXiv 2601.20701) eliminates the teacher dependency and targets
    1000+ Hz on consumer GPUs. pi-Flow (arXiv 2510.14974) is also available but
    DMPO is the preferred default.

    v0.2 status: command scaffold. Full training loop ships in v0.2.1 —
    track progress at GOALS.yaml: `distill-dmpo`.
    """
    _setup_logging(verbose)
    from reflex.distill import get_recipe

    console.print(f"\n[bold]Reflex Distill[/bold]")
    console.print(f"  Recipe:  [cyan]{recipe}[/cyan]")
    console.print(f"  Teacher: {teacher}")
    console.print(f"  Student: {output}")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Steps:   {steps}")
    console.print(f"  Device:  {device}\n")

    try:
        trainer_cls = get_recipe(recipe)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2)

    # Load config for the chosen recipe
    if recipe == "dmpo":
        from reflex.distill.dmpo import DMPOConfig
        config = DMPOConfig(
            teacher_export_dir=teacher,
            student_output_dir=output,
            dataset=dataset,
            num_episodes=episodes,
            num_training_steps=steps,
            batch_size=batch_size,
            learning_rate=lr,
            device=device,
        )
    else:
        from reflex.distill.pi_flow import PiFlowConfig
        config = PiFlowConfig(
            teacher_export_dir=teacher,
            student_output_dir=output,
            dataset=dataset,
        )

    trainer = trainer_cls(config)
    try:
        result = trainer.train()
        console.print("\n[bold green]Distillation complete[/bold green]")
        for key, val in result.items():
            console.print(f"  {key}: {val}")
    except NotImplementedError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(1)


@app.command(hidden=True)
def turbo(
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """[DEPRECATED] Adaptive denoising now lives on `reflex serve --adaptive-steps`."""
    console.print(
        "[yellow]`reflex turbo` is deprecated and will be removed in v0.3.[/yellow]\n"
        "[yellow]Adaptive denoising is now a flag on serve:[/yellow]\n"
        "  [cyan]reflex serve <export> --adaptive-steps[/cyan]\n\n"
        "[dim]Note: adaptive denoising only produces safe results on pi0.\n"
        "For pi0.5/SmolVLA/GR00T, use `reflex distill` instead (v0.2+).[/dim]"
    )
    raise typer.Exit(0)


@app.command(hidden=True)
def split(
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """[DEPRECATED] Cloud-edge orchestration is now a flag on `reflex serve`."""
    console.print(
        "[yellow]`reflex split` is deprecated and will be removed in v0.3.[/yellow]\n"
        "[yellow]Cloud-edge fallback is now a flag on serve:[/yellow]\n"
        "  [cyan]reflex serve <export> --cloud-fallback <url>[/cyan]\n\n"
        "[dim]Fewer than 10% of production deployments use cloud-edge split,\n"
        "so a dedicated command was removed in favor of a flag.[/dim]"
    )
    raise typer.Exit(0)


@app.command(hidden=True)
def adapt(
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """[DEPRECATED] Velocity clamping folded into `reflex guard`. Cross-embodiment archived."""
    console.print(
        "[yellow]`reflex adapt` is deprecated and will be removed in v0.3.[/yellow]\n"
        "[yellow]Velocity/torque limits are now part of `reflex guard`:[/yellow]\n"
        "  [cyan]reflex guard init --urdf <file> --output ./safety.json[/cyan]\n\n"
        "[dim]Cross-embodiment action remapping had no users; archived in\n"
        "reflex_context/06_archive/. Open an issue if you need it back.[/dim]"
    )
    raise typer.Exit(0)


@app.command(hidden=True)
def check(
    checkpoint: str = typer.Argument(help="HuggingFace ID or local path"),
    target: str = typer.Option("desktop", help="Target hardware: orin-nano, orin, orin-64, thor, desktop"),
    verbose: bool = typer.Option(False, help="Verbose logging"),
):
    """[DEPRECATED] Replaced by `reflex validate --pre-export`. Forwards for compat."""
    console.print(
        "[yellow]`reflex check` is deprecated and will be removed in v0.3.[/yellow]\n"
        "[yellow]Use:[/yellow] [cyan]reflex validate "
        f"{checkpoint} --pre-export --hardware {target}[/cyan]\n"
    )
    _setup_logging(verbose)
    from reflex.validate_training import run_all_checks

    results = run_all_checks(checkpoint, target=target)
    table = Table(title="Pre-Deployment Checks")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Detail")
    n_pass = 0
    for r in results:
        status = "[green]PASS[/green]" if r.passed else (
            "[yellow]WARN[/yellow]" if r.severity == "warning" else "[red]FAIL[/red]"
        )
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
