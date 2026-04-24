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
    from_distilled: bool = typer.Option(
        False,
        "--from-distilled",
        help="Treat MODEL as a reflex-saved SnapFlow student checkpoint dir "
             "(contains model.safetensors with target_time_embed_mlp.* keys). "
             "Auto-detects pi0 vs pi0.5 from config.json, exports at 1-NFE "
             "with target_time=1 baked in. Output ONNX has the same I/O "
             "signature as the matching teacher family's monolithic export, "
             "so `reflex serve` loads it through the standard path.",
    ),
):
    """Export a VLA model to ONNX + TensorRT for edge deployment."""
    _setup_logging(verbose)
    hardware = get_hardware_profile(target)

    if monolithic:
        label = "SnapFlow student (1-NFE)" if from_distilled else "monolithic, cos=1.0 verified path"
        console.print(f"\n[bold]Reflex Export ({label})[/bold]")
        console.print(f"  Model:      {model}")
        console.print(f"  Output:     {output}")
        if not from_distilled:
            console.print(f"  num_steps:  {num_steps}")
        console.print()

        if dry_run:
            console.print("[yellow]--dry-run not supported with --monolithic yet (v0.3 item). "
                          "Re-run without --dry-run to export.[/yellow]")
            raise typer.Exit()

        try:
            if from_distilled:
                from reflex.exporters.monolithic import export_snapflow_student_monolithic
            else:
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
            if from_distilled:
                result = export_snapflow_student_monolithic(model, output, target=target)
            else:
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
    api_key: str = typer.Option(
        "",
        help="If set, every /act and /config request must include a matching "
             "X-Reflex-Key header or it's rejected 401. /health stays "
             "unauthenticated so load balancers can probe readiness. For "
             "production use, pass via env var (e.g. --api-key $REFLEX_API_KEY) "
             "rather than hardcoding.",
    ),
    replan_hz: float = typer.Option(
        0.0,
        help="If >0, enable async replan-while-execute action buffering "
             "(the Physical Intelligence sliding_window pattern). Set with "
             "--execute-hz. Example: --execute-hz 100 --replan-hz 20 means "
             "the robot pops an action 100 times/sec while fresh chunks are "
             "generated 20 times/sec. Buffer capacity is auto-sized from "
             "the ratio. 0 = disabled (return full chunks, current default).",
    ),
    execute_hz: float = typer.Option(
        0.0,
        help="Execute frequency in Hz — the rate at which the robot pops "
             "an action from the buffer. Only used when --replan-hz > 0.",
    ),
    rtc: bool = typer.Option(
        False,
        "--rtc",
        help="Enable Real-Time Chunking (RTC) — wraps inference with "
             "lerobot's RTCProcessor so the robot keeps executing the tail "
             "of one chunk while the next chunk is being computed. 2-3× "
             "effective throughput on Jetson-class latency. Requires "
             "`pip install reflex-vla[rtc]` (pulls lerobot==0.5.1).",
    ),
    rtc_execution_horizon: int = typer.Option(
        10,
        "--rtc-execution-horizon",
        help="With --rtc: number of actions locked to the previous chunk "
             "while the next is computed. Higher = more guidance, smoother "
             "transitions; lower = more freedom for the new chunk. Default 10.",
    ),
    rtc_schedule: str = typer.Option(
        "LINEAR",
        "--rtc-schedule",
        help="With --rtc: prefix attention schedule. ZEROS | ONES | LINEAR | EXP. "
             "Default LINEAR (matches lerobot's RTCConfig default).",
    ),
    rtc_max_guidance_weight: float = typer.Option(
        10.0,
        "--rtc-max-guidance-weight",
        help="With --rtc: max guidance weight clamp. Higher = stronger pull "
             "toward previous chunk's prefix; lower = looser. Default 10.0.",
    ),
    rtc_debug: bool = typer.Option(
        False,
        "--rtc-debug",
        help="With --rtc: enable lerobot's debug Tracker for per-step state "
             "capture. Useful for replay forensics; small per-call overhead.",
    ),
    record: str = typer.Option(
        "",
        help="If set, write every /act request+response to a JSONL trace in "
             "this directory. One file per server session, named "
             "<YYYYMMDD-HHMMSS>-<model_hash>-<session_id>.jsonl[.gz]. "
             "Replay with `reflex replay <file> --model <export>`. See "
             "TECHNICAL_PLAN §D.1 for the schema.",
    ),
    record_images: str = typer.Option(
        "hash_only",
        "--record-images",
        help="Image redaction policy when --record is set: "
             "'full' (~40MB/1k calls gzipped, base64 JPEG kept) | "
             "'hash_only' (~0.9MB/1k calls, image_sha256 only — default; "
             "sufficient for replay against a fixed image corpus) | "
             "'none' (drop image entirely; minimal size).",
    ),
    record_no_gzip: bool = typer.Option(
        False,
        "--record-no-gzip",
        help="When --record is set, write plain .jsonl instead of .jsonl.gz. "
             "Useful for quick grep during dev; production should keep gzip on.",
    ),
    embodiment: str = typer.Option(
        "",
        help="Per-embodiment config preset name (franka, so100, ur5, etc.). "
             "Loads configs/embodiments/<name>.json. Empty = no embodiment "
             "config (current default behavior). See "
             "docs/embodiment_schema.md for the schema and adding new presets.",
    ),
    custom_embodiment_config: str = typer.Option(
        "",
        "--custom-embodiment-config",
        help="Path to a custom embodiment config JSON. Overrides --embodiment "
             "if both are set. Use this for robots not covered by the shipped "
             "presets.",
    ),
    inject_latency_ms: float = typer.Option(
        0.0,
        "--inject-latency-ms",
        help="Synthetic deployment-latency injection (B.4 A2C2 transfer-validation "
             "gate). Adds asyncio.sleep AFTER inference + JSONL recording so "
             "recorded latency_ms is true compute cost while client observes "
             "inference + injected delay. Range [0, 1000]. 0 = off (default). "
             "Used to simulate Jetson-class deployment latency on Modal A10G "
             "for the A2C2 transfer gate; see arxiv 2509.23224 §4 for "
             "matching paper methodology.",
    ),
    no_prewarm: bool = typer.Option(
        False,
        "--no-prewarm",
        help="Skip the synthetic warmup forward at lifespan startup. Default "
             "behavior: warmup runs, /health returns 503 until warmup succeeds, "
             "then 200. With --no-prewarm: /health returns 200 the moment "
             "server.load() completes; first /act bears the 30-90s engine-build "
             "cost. Use only for fast-start dev workflows; production behind a "
             "load balancer should leave prewarm ON.",
    ),
    max_consecutive_crashes: int = typer.Option(
        5,
        "--max-consecutive-crashes",
        help="Circuit breaker: after this many consecutive /act predict "
             "exceptions or error-result responses, server.health_state flips "
             "to 'degraded' — /health returns 503, /act returns 503 with "
             "Retry-After: 60. Successful /act resets the counter. Default 5. "
             "Set to 0 to disable.",
    ),
    ros2: bool = typer.Option(
        False,
        "--ros2",
        help="Run a ROS2 bridge instead of the HTTP server. Subscribes to "
             "image/state/task topics and publishes action chunks to an "
             "action topic. Requires rclpy (apt-installed, not pip) — see "
             "reflex ros2-serve for the standalone equivalent. Mutually "
             "exclusive with the HTTP flags above (port, host, api-key, "
             "max-batch, etc. are ignored in ROS2 mode).",
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

    # Resolve --embodiment / --custom-embodiment-config (B.1). Validate
    # early — before any compute or runtime checks — so a bad config fails
    # loud at the CLI layer, not at first /act.
    embodiment_cfg = None
    if custom_embodiment_config or embodiment:
        from reflex.embodiments import EmbodimentConfig, list_presets
        from reflex.embodiments.validate import (
            format_errors,
            validate_embodiment_config,
        )
        try:
            if custom_embodiment_config:
                if embodiment:
                    console.print(
                        f"[yellow]--custom-embodiment-config overrides "
                        f"--embodiment {embodiment}[/yellow]"
                    )
                embodiment_cfg = EmbodimentConfig.load_custom(custom_embodiment_config)
            else:
                embodiment_cfg = EmbodimentConfig.load_preset(embodiment)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]Failed to load embodiment config: {exc}[/red]")
            console.print(
                f"[dim]Available presets: {list_presets() or '(none)'}[/dim]"
            )
            raise typer.Exit(1)

        ok, errs = validate_embodiment_config(embodiment_cfg)
        if not ok:
            console.print(
                f"[red]Embodiment config '{embodiment_cfg.embodiment}' failed "
                f"validation:[/red]"
            )
            console.print(format_errors(errs))
            raise typer.Exit(1)
        warnings = [e for e in errs if e["severity"] == "warn"]
        if warnings:
            console.print(
                f"[yellow]Embodiment config '{embodiment_cfg.embodiment}' "
                f"has warnings:[/yellow]"
            )
            console.print(format_errors(warnings))

    # Build RtcAdapterConfig if --rtc was passed (B.3 Day 1). Validates at
    # the CLI layer — fail loud before runtime imports (same pattern as
    # embodiment validation above).
    rtc_cfg = None
    if rtc:
        from reflex.runtime.rtc_adapter import RtcAdapterConfig
        try:
            rtc_cfg = RtcAdapterConfig(
                enabled=True,
                replan_hz=replan_hz if replan_hz > 0 else 20.0,
                execute_hz=execute_hz if execute_hz > 0 else 100.0,
                rtc_execution_horizon=rtc_execution_horizon,
                prefix_attention_schedule=rtc_schedule,
                max_guidance_weight=rtc_max_guidance_weight,
                debug=rtc_debug,
            )
        except ValueError as exc:
            console.print(f"[red]Invalid RTC config: {exc}[/red]")
            raise typer.Exit(1)

    # ROS2 mode short-circuits the HTTP path — hand off to the bridge.
    if ros2:
        try:
            from reflex.runtime.ros2_bridge import run_ros2_bridge
        except ImportError as exc:
            console.print(f"[red]ros2 bridge unavailable: {exc}[/red]")
            raise typer.Exit(2)
        console.print(f"[bold green]reflex serve --ros2[/bold green]")
        console.print(f"  export:   {export_dir}")
        console.print(f"  device:   {device}")
        console.print(
            f"  [dim]HTTP flags ignored in ROS2 mode. Use `reflex ros2-serve` "
            f"for full topic/rate customization.[/dim]"
        )
        try:
            run_ros2_bridge(
                export_dir,
                device=device,
                safety_config=safety_config or None,
            )
        except KeyboardInterrupt:
            console.print("[yellow]ros2 bridge stopped.[/yellow]")
        return

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
    if embodiment_cfg is not None:
        composed.append(f"[cyan]embodiment[/cyan]={embodiment_cfg.embodiment}")
    if record:
        composed.append(
            f"[cyan]record[/cyan]={record} ({record_images}"
            f"{', no-gzip' if record_no_gzip else ''})"
        )
    if rtc:
        composed.append(
            f"[cyan]rtc[/cyan]=horizon{rtc_execution_horizon}/{rtc_schedule}"
        )
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

    if replan_hz > 0 and execute_hz <= 0:
        console.print(
            "[red]--replan-hz requires --execute-hz > 0 (the robot's pop rate).[/red]"
        )
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
        api_key=api_key or None,
        replan_hz=replan_hz if replan_hz > 0 else None,
        execute_hz=execute_hz if execute_hz > 0 else None,
        embodiment_config=embodiment_cfg,
        record_dir=record or None,
        record_image_redaction=record_images,
        record_gzip=not record_no_gzip,
        rtc_config=rtc_cfg,
        inject_latency_ms=inject_latency_ms,
        prewarm=not no_prewarm,
        max_consecutive_crashes=max_consecutive_crashes,
    )
    if api_key:
        composed.append("[cyan]api-key-auth[/cyan]")
    if replan_hz > 0:
        composed.append(
            f"[cyan]replan[/cyan]={replan_hz:g}Hz/execute={execute_hz:g}Hz"
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
def replay(
    trace_file: str = typer.Argument(help="Path to recorded JSONL trace (.jsonl or .jsonl.gz)"),
    model: str = typer.Option(
        "",
        "--model",
        help="Path to target export dir for replay. Required unless --no-replay.",
    ),
    diff: str = typer.Option(
        "actions",
        "--diff",
        help="Diff mode (Day 2 ships actions only; latency/cache/all in Day 3).",
    ),
    n: int = typer.Option(
        0,
        "--n",
        help="Replay first N records only. 0 = all.",
    ),
    output: str = typer.Option(
        "",
        "--output",
        help="Write machine-readable diff report to this JSON path.",
    ),
    fail_on: str = typer.Option(
        "",
        "--fail-on",
        help="Exit non-zero if any diff of this type fails (e.g. --fail-on actions).",
    ),
    no_replay: bool = typer.Option(
        False,
        "--no-replay",
        help="Parse the trace + print header/counts without loading the model. "
             "Useful for inspecting traces and validating their schema.",
    ),
):
    """Replay a recorded /act trace against a target model.

    Day 2 scope: load JSONL, replay each request, compute per-record actions
    diff (cosine + max_abs). Latency / cache / guard diff modes land Day 3.

    Trace format: TECHNICAL_PLAN §D.1 (schema v1).
    """
    from reflex.replay.cli import run_replay

    if not no_replay and not model:
        console.print(
            "[red]--model is required (or pass --no-replay to inspect the trace "
            "without loading a model).[/red]"
        )
        raise typer.Exit(1)
    code = run_replay(
        trace_file,
        model or None,
        diff_mode=diff,
        n=n,
        output_json=output,
        fail_on=fail_on,
        no_replay=no_replay,
    )
    if code != 0:
        raise typer.Exit(code)


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


# `reflex distill` registered below via `app.command(name="distill")` on the
# finetune package. Kept out of this file so test collection doesn't pull
# lerobot + torch + SnapFlow deps just to load the CLI module.


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
        "[dim]Cross-embodiment action remapping had no users; archived.\n"
        "Open an issue if you need it back.[/dim]"
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
def doctor(
    model: str = typer.Option(
        "",
        "--model",
        help="Optional path to an exported model directory. When passed, runs "
             "deploy diagnostics (5 falsifiable checks for known LeRobot async "
             "issues + systemic VLA deploy failures) AFTER the system probe. "
             "Without --model, runs system probe only.",
    ),
    embodiment: str = typer.Option(
        "custom",
        "--embodiment",
        help="Embodiment preset (franka/so100/ur5) for deploy-diagnostic cross-checks. "
             "Only used when --model is also passed.",
    ),
    rtc: bool = typer.Option(
        False,
        "--rtc",
        help="Validate RTC chunk-boundary alignment in deploy diagnostics. "
             "Only used when --model is also passed.",
    ),
    output_format: str = typer.Option(
        "human",
        "--format",
        help="Output format for deploy diagnostics: 'human' (table) or 'json' "
             "(machine-readable, schema_version=1). System probe is always human-readable.",
    ),
    skip: list[str] = typer.Option(
        [],
        "--skip",
        help="Deploy-diagnostic check IDs to skip. Repeatable.",
    ),
):
    """Diagnose Reflex install + GPU issues + (optionally) per-deploy issues.

    Two modes:
      reflex doctor                              # system probe (Python, CUDA,
                                                 # ORT providers, fastapi, etc.)
      reflex doctor --model ./export/pi05 \\     # system probe + 5 deploy checks
                    --embodiment franka          # against your specific export

    Exit codes: 0 all pass, 1 at least one deploy-check fail, 2 invocation error,
    3 environment error.

    Plan: features/01_serve/subfeatures/_dx_gaps/reflex-doctor_plan.md
    """
    import platform
    import shutil
    import sys

    if output_format not in ("human", "json"):
        console.print(
            f"[red]--format must be 'human' or 'json', got {output_format!r}[/red]"
        )
        raise typer.Exit(2)

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

    # Deploy diagnostics — only when --model is passed (B.4 Day 1 + future)
    if model:
        from reflex.diagnostics import (
            exit_code as _exit_code,
            format_human,
            format_json,
            run_all_checks,
        )

        console.print()
        console.print("[bold]Deploy diagnostics:[/bold]")
        console.print()

        results = run_all_checks(
            model_path=model,
            embodiment_name=embodiment,
            rtc=rtc,
            skip=skip,
        )
        if output_format == "json":
            console.print(format_json(
                results,
                model_path=model,
                embodiment_name=embodiment,
            ))
        else:
            console.print(format_human(results))

        code = _exit_code(results)
        if code != 0:
            raise typer.Exit(code)


@app.command(name="validate-dataset")
def validate_dataset(
    path: str = typer.Argument(help="Path to LeRobot v3.0 dataset root (contains meta/info.json)"),
    embodiment: str = typer.Option(
        "",
        "--embodiment",
        help="Embodiment preset (franka/so100/ur5) for cross-checking action_dim. "
             "When set, loads configs/embodiments/<name>.json and compares its "
             "action_dim against the dataset's declared action shape. Optional.",
    ),
    custom_embodiment_config: str = typer.Option(
        "",
        "--custom-embodiment-config",
        help="Path to a custom embodiment config JSON. Overrides --embodiment.",
    ),
    output_format: str = typer.Option(
        "human",
        "--format",
        help="Report format: 'human' (default, plain-text) or 'json' (machine-readable).",
    ),
    output: str = typer.Option(
        "",
        "--output",
        help="Write the report to this file instead of stdout. Useful in CI.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Treat WARN findings as BLOCKERs. Use in CI when any deviation should fail.",
    ),
):
    """Validate a LeRobot training dataset against the model + embodiment expectations.

    Pre-flight check before spending Modal credits on a training/distillation run that
    would crash mid-way due to action-dim mismatch, NaN actions, or schema drift. Pairs
    with `reflex doctor` (which validates model + runtime).

    Exit codes: 0 ok, 1 warnings, 2 blockers (or warnings under --strict).

    Examples:
      reflex validate-dataset ~/datasets/aloha_sim
      reflex validate-dataset ~/datasets/aloha_sim --embodiment franka
      reflex validate-dataset ~/datasets/aloha_sim --format json --output report.json
      reflex validate-dataset ~/datasets/aloha_sim --strict   # CI gating
    """
    from reflex.validation import (
        Decision,
        format_human,
        format_json,
        overall_decision,
        run_all_checks,
    )

    if output_format not in ("human", "json"):
        console.print(f"[red]--format must be 'human' or 'json', got {output_format!r}[/red]")
        raise typer.Exit(2)

    dataset_path = Path(path)
    if not dataset_path.exists():
        console.print(f"[red]Dataset path does not exist: {dataset_path}[/red]")
        raise typer.Exit(2)

    embodiment_cfg = None
    if custom_embodiment_config or embodiment:
        try:
            from reflex.embodiments import EmbodimentConfig
            if custom_embodiment_config:
                embodiment_cfg = EmbodimentConfig.load_custom(custom_embodiment_config)
            else:
                embodiment_cfg = EmbodimentConfig.load_preset(embodiment)
        except Exception as e:
            console.print(f"[yellow]Could not load embodiment config: {e}[/yellow]")

    results = run_all_checks(dataset_path, embodiment_config=embodiment_cfg, strict=strict)
    decision = overall_decision(results, strict=strict)

    if output_format == "json":
        report = format_json(results, dataset_root=str(dataset_path), decision=decision)
    else:
        report = format_human(results, dataset_root=str(dataset_path))

    if output:
        Path(output).write_text(report)
        console.print(f"Report written to {output} — decision: [bold]{decision.value.upper()}[/bold]")
    else:
        if output_format == "json":
            console.print(report)
        else:
            console.print(report)
            console.print(f"\nOverall decision: [bold]{decision.value.upper()}[/bold]")

    exit_code = {Decision.OK: 0, Decision.WARN: 1, Decision.BLOCKER: 2, Decision.SKIPPED: 0}[decision]
    if exit_code != 0:
        raise typer.Exit(exit_code)


# `reflex models {list, pull, info}` — curated VLA registry browser/downloader.
# Defined inline so the typer subgroup wiring stays visible at the CLI surface.
models_app = typer.Typer(help="Browse + download Reflex-compatible VLA models from HuggingFace.")


@models_app.command("list")
def models_list(
    family: str = typer.Option("", "--family", help="Filter by family: pi0/pi05/smolvla/openvla/groot"),
    device: str = typer.Option("", "--device",
                                help="Filter by supported device (orin_nano, agx_orin, thor, a10g, a100, h100, h200)"),
    embodiment: str = typer.Option("", "--embodiment", help="Filter by supported embodiment (franka, so100, ur5)"),
    output_format: str = typer.Option("human", "--format", help="'human' (table) or 'json'"),
):
    """List Reflex-compatible models from the curated registry.

    Examples:
      reflex models list
      reflex models list --family pi05
      reflex models list --device orin_nano
      reflex models list --device a10g --embodiment franka
    """
    from reflex.registry import REGISTRY, filter_models

    if output_format not in ("human", "json"):
        console.print(f"[red]--format must be 'human' or 'json', got {output_format!r}[/red]")
        raise typer.Exit(2)

    entries = filter_models(
        family=family or None, device=device or None, embodiment=embodiment or None,
    )

    if output_format == "json":
        import json
        rows = [
            {
                "model_id": e.model_id,
                "hf_repo": e.hf_repo,
                "family": e.family,
                "action_dim": e.action_dim,
                "size_mb": e.size_mb,
                "supported_embodiments": list(e.supported_embodiments),
                "supported_devices": list(e.supported_devices),
                "requires_export": e.requires_export,
                "license": e.license,
                "description": e.description,
            }
            for e in entries
        ]
        typer.echo(json.dumps({"n": len(rows), "models": rows}, indent=2))
        return

    if not entries:
        console.print(
            f"[yellow]No models match filters (family={family or 'any'}, "
            f"device={device or 'any'}, embodiment={embodiment or 'any'}).[/yellow]"
        )
        console.print(f"Registry has {len(REGISTRY)} entries total — drop filters to see all.")
        return

    table = Table(title=f"Reflex Model Registry ({len(entries)} of {len(REGISTRY)})")
    table.add_column("model_id", style="cyan", no_wrap=True)
    table.add_column("family", no_wrap=True)
    table.add_column("a_dim", justify="right")
    table.add_column("size", justify="right")
    table.add_column("embodiments")
    table.add_column("devices")
    table.add_column("description")
    for e in entries:
        size_str = f"{e.size_mb / 1000:.1f}GB" if e.size_mb >= 1000 else f"{e.size_mb}MB"
        table.add_row(
            e.model_id, e.family, str(e.action_dim), size_str,
            ", ".join(e.supported_embodiments), ", ".join(e.supported_devices),
            e.description[:60] + ("..." if len(e.description) > 60 else ""),
        )
    console.print(table)
    console.print(
        "\n[dim]reflex models pull <model_id>   # download to ~/.cache/reflex/models/<id>[/dim]"
        "\n[dim]reflex models info <model_id>   # see benchmarks + per-device support[/dim]"
    )


@models_app.command("pull")
def models_pull(
    model_id: str = typer.Argument(help="Registry id from `reflex models list`"),
    target_dir: str = typer.Option("", "--target-dir",
                                    help="Where to write weights. Default: ~/.cache/reflex/models/<model_id>/"),
    no_verify: bool = typer.Option(False, "--no-verify",
                                    help="Skip the post-download structure check"),
    revision: str = typer.Option("", "--revision",
                                  help="Override the registry's pinned hf_revision (advanced)"),
):
    """Download a model's weights from HuggingFace into the local cache.

    Example:
      reflex models pull pi05-libero
      reflex models pull smolvla-base --target-dir /data/models/smolvla
    """
    from reflex.registry import by_id

    entry = by_id(model_id)
    if entry is None:
        from reflex.registry import REGISTRY
        available = sorted(e.model_id for e in REGISTRY)
        console.print(f"[red]Unknown model_id: {model_id!r}[/red]")
        console.print(f"Available: {', '.join(available)}")
        raise typer.Exit(2)

    target = Path(target_dir) if target_dir else (Path.home() / ".cache" / "reflex" / "models" / model_id)
    target.mkdir(parents=True, exist_ok=True)

    rev = revision or entry.hf_revision
    rev_str = rev if rev else "HEAD (unpinned — consider --revision for reproducibility)"

    console.print(f"Pulling [cyan]{entry.model_id}[/cyan]")
    console.print(f"  hf_repo:  {entry.hf_repo}")
    console.print(f"  revision: {rev_str}")
    console.print(f"  size:     ~{entry.size_mb}MB")
    console.print(f"  target:   {target}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        console.print("[red]huggingface_hub not installed. pip install reflex-vla[/red]")
        raise typer.Exit(2)

    try:
        snapshot_download(
            repo_id=entry.hf_repo,
            revision=rev or None,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        console.print(f"[red]Download failed: {type(e).__name__}: {e}[/red]")
        raise typer.Exit(1)

    if not no_verify:
        contents = sorted(p.name for p in target.iterdir())
        console.print(f"[green]Pulled.[/green] {len(contents)} top-level entries: {contents[:10]}")
        if entry.requires_export:
            console.print(
                f"\n[yellow]Next: this model ships as raw weights. Run "
                f"[cyan]reflex export {target}[/cyan] to produce ONNX, then "
                f"[cyan]reflex serve <export-dir>[/cyan].[/yellow]"
            )
        else:
            console.print(
                f"\n[green]Ready to serve:[/green] [cyan]reflex serve {target}[/cyan]"
            )


@models_app.command("info")
def models_info(
    model_id: str = typer.Argument(help="Registry id from `reflex models list`"),
    output_format: str = typer.Option("human", "--format", help="'human' or 'json'"),
):
    """Show benchmarks + per-device support for a single model.

    Example:
      reflex models info pi05-libero
    """
    from reflex.registry import by_id

    entry = by_id(model_id)
    if entry is None:
        console.print(f"[red]Unknown model_id: {model_id!r}[/red]")
        raise typer.Exit(2)

    if output_format == "json":
        import json
        body = {
            "model_id": entry.model_id,
            "hf_repo": entry.hf_repo,
            "hf_revision": entry.hf_revision,
            "family": entry.family,
            "action_dim": entry.action_dim,
            "size_mb": entry.size_mb,
            "supported_embodiments": list(entry.supported_embodiments),
            "supported_devices": list(entry.supported_devices),
            "requires_export": entry.requires_export,
            "license": entry.license,
            "description": entry.description,
            "benchmarks": [
                {"device": b.device, "p50_ms": b.p50_ms, "p99_ms": b.p99_ms,
                 "vram_mb": b.vram_mb, "measured_at": b.measured_at}
                for b in entry.benchmarks
            ],
        }
        typer.echo(json.dumps(body, indent=2))
        return

    console.print(f"[bold cyan]{entry.model_id}[/bold cyan] ([dim]{entry.hf_repo}[/dim])")
    console.print(f"  family:        {entry.family}")
    console.print(f"  action_dim:    {entry.action_dim}")
    console.print(f"  size:          {entry.size_mb}MB")
    console.print(f"  license:       {entry.license}")
    console.print(f"  embodiments:   {', '.join(entry.supported_embodiments) or '(none)'}")
    console.print(f"  devices:       {', '.join(entry.supported_devices) or '(none)'}")
    console.print(f"  needs export:  {'YES — run reflex export after pull' if entry.requires_export else 'NO — Reflex-ready'}")
    console.print(f"\n{entry.description}")

    if entry.benchmarks:
        bt = Table(title="Benchmarks")
        bt.add_column("device")
        bt.add_column("p50 (ms)", justify="right")
        bt.add_column("p99 (ms)", justify="right")
        bt.add_column("VRAM (MB)", justify="right")
        bt.add_column("measured")
        for b in entry.benchmarks:
            bt.add_row(b.device, f"{b.p50_ms:.1f}", f"{b.p99_ms:.1f}",
                       str(b.vram_mb), b.measured_at)
        console.print()
        console.print(bt)
    else:
        console.print("\n[dim]No benchmarks yet. Run [cyan]reflex bench <export>[/cyan] after pull.[/dim]")


app.add_typer(models_app, name="models")


# Register `reflex finetune` + `reflex distill` subcommands. Lazy-import
# protects users who don't have training deps installed — they only break
# if they run the commands themselves.
try:
    from reflex.finetune.cli import finetune_command
    app.command(name="finetune")(finetune_command)
except Exception as _finetune_import_exc:  # pragma: no cover - defensive
    pass

try:
    from reflex.finetune.cli_distill import distill_command
    app.command(name="distill")(distill_command)
except Exception as _distill_import_exc:  # pragma: no cover - defensive
    pass


if __name__ == "__main__":
    app()
