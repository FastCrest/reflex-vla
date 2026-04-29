# Changelog

## v0.7.1 — 2026-04-29

Surfaces the previously-silent `--cuda-graphs` flag with measured A100 + A10G perf numbers. The flag has been in main since v0.5.0 (commits `49110d2` → `7dfcab6`, April 24-25) but no CHANGELOG entry told users it existed. v0.7.1 fixes that.

### Documented (no new code, surfacing existing capability)
- **`reflex serve --cuda-graphs`** — opt-in CUDA graph capture for the decomposed pi0.5 path (`vlm_prefix.onnx` + `expert_denoise.onnx`). Captures both ONNX sessions at startup using ORT's `enable_cuda_graph=1` + replays for every subsequent request. Skips per-op kernel-launch overhead.
- **Measured speedup on A100-80GB (pi0.5 num_steps=10, n=200 paired iterations, 5 warmup discarded):**
  - Per-chunk latency: 270.85 ms → 207.74 ms = **1.30× speedup, p99 304 → 212 ms (-30%)**
  - `vlm_prefix`: 93.45 → 87.34 ms (1.07× mean, jitter 1.4× tighter)
  - `expert_denoise` (runs 10× per chunk): 17.74 → 12.04 ms (**1.47× mean, p99 -40%, jitter 4.1× tighter**)
- **Measured speedup on A10G (n=100 paired):**
  - `expert_denoise`: 12.34 → 10.47 ms (**1.18× mean, p99 -35%, jitter 14× tighter**)
  - `vlm_prefix`: capacity-bounded — capture buffer doesn't fit alongside working memory on A10G's 24 GB envelope. Wrapper gracefully degrades to eager per ADR `2026-04-24-cuda-graphs-architecture` decision #5 (tier-aware semantics). Same numerics, no perf change for that session, expert still wins.
- **`reflex doctor` cuda-graphs diagnostic** — `tests/test_check_cuda_graphs.py` (already in v0.5.0+) checks `vlm_prefix` + `expert_denoise` capture status. Surface with `reflex doctor` after running `reflex serve --cuda-graphs` to verify capture on your hardware.
- **Customer-facing docs** at `docs/cuda_graphs.md` (already in v0.5.0+) — when to enable, hardware tier matrix, what metrics to watch (`reflex_cuda_graph_captured_total` / `reflex_cuda_graph_replayed_total` / `reflex_cuda_graph_eager_fallback_total` / `reflex_cuda_graph_capture_failed_at_init_total` / `reflex_cuda_graph_capture_seconds`).

### Added (this release)
- **`tests/test_cuda_graphs_integration.py`** (Day 7 of cuda-graphs plan) — 7 mock-based + 1 CUDA-gated integration tests covering create_app flag propagation, legacy ReflexServer no-op log, Pi05DecomposedInference provider wiring, capture-failed-at-init eager fallback, /metrics endpoint counter exposure, and label cardinality bound. The CUDA-gated test runs on Modal A10G/A100; mock-based tests run anywhere.
- **`scripts/modal_cuda_graphs_ab.py`** — Day 8-9 A/B benchmark script. Loads a decomposed export from the `pi0-onnx-outputs` Modal volume, runs N iterations OFF (eager) + N iterations ON (cuda-graphs) for each session (`vlm_prefix` + `expert_denoise`), computes ISB-1 stats with 5 warmup discarded + 95% CI on means, writes JSON output to volume. 30s progress logging so degenerate runs are catchable mid-flight.

### Notes
- **The `--cuda-graphs` flag was never deprecated, hidden, or feature-flagged.** It was always live in `reflex serve --help` since v0.5.0. v0.7.1 simply documents it with measured numbers.
- **Hardware tier guidance:**
  - A100 / H100: capture both sessions (1.30× per-chunk)
  - A10G: capture expert_denoise only (1.18× expert mean, vlm_prefix gracefully degrades to eager)
  - Orin Nano / smaller: not yet measured. Try with `reflex doctor` to check if your GPU has headroom.
- **Why the win shape is "predictability > raw speed":** robotics control loops care about p99 + jitter more than mean. CUDA graph replay eliminates per-op kernel-launch jitter; the means are smaller wins because ORT already uses fused cuBLAS kernels for the heavy ops.
- **Validation experiments:** Modal app URLs in the Reflex context vault at `reflex_context/03_experiments/2026-04-29-cuda-graphs-ab-modal-a10g.md` + `2026-04-29-cuda-graphs-ab-modal-a100.md`. Total Modal cost: $3.50.

## v0.7.0 — 2026-04-29

ORT-TensorRT execution provider first-class support. Most users now get the **5.55× speedup** measured on Modal A10G (108.11 ms → 19.49 ms on SmolVLA monolithic, 2026-04-29) automatically — without manual `LD_LIBRARY_PATH` setup.

### Added
- **`tensorrt>=10.0,<11`** is now in the default `[gpu]` extras. Provides `libnvinfer.so.10` so ORT-TRT EP loads at runtime instead of silently falling back to ORT-CUDA EP. Adds ~2 GB to install footprint — that's the cost of the 5.55× win. Linux only (the `; sys_platform == 'linux'` marker keeps Mac installs untouched).
- **`[gpu-min]` extras** as escape hatch for users who don't want the `tensorrt` install — gets you ORT-CUDA EP only (~5× slower on transformer workloads). Use when bandwidth/storage matter more than perf.
- **Auto-`LD_LIBRARY_PATH` patch in `reflex/__init__.py`.** Prepends pip-installed `tensorrt_libs/`, `nvidia/cudnn/lib/`, `nvidia/cublas/lib/` paths so ORT can find the shared objects without manual env config. Idempotent. Opt-out via `REFLEX_NO_LD_LIBRARY_PATH_PATCH=1`. No-op on macOS, Windows, or when paths don't exist.
- **`reflex doctor` validates the full ORT-TRT EP load chain.** Four new checks:
  1. `libnvinfer.so.10` loadable via `ctypes.CDLL`
  2. `libcublas.so.12` loadable
  3. `libcudnn.so.9` loadable
  4. Empty `ort.InferenceSession` with `TensorrtExecutionProvider` succeeds AND active providers includes TRT EP (gold-standard end-to-end check)
  Each failure has a `pip install` remediation hint inline.
- **README "Performance" section** documents the measured 5.55× claim with the Modal hardware (A10G), the workload (SmolVLA monolithic), the method (5+20 forward passes), the date, and the reproducer command. Plus a "How to verify" section pointing at `reflex doctor`.
- **14 new unit tests** (8 LD_LIBRARY_PATH patch + 6 doctor TRT EP) covering Linux/macOS/Windows + opt-out + idempotency + lib loadable / not / fallback / session-create-throws branches.

### Changed
- `[gpu]` extras now Linux-marked for `tensorrt`. Mac users on `[gpu]` get cuDNN + cuBLAS pulls but no `tensorrt` (which has no Mac wheel anyway).
- `__version__` bumped to `0.7.0` in `src/reflex/__init__.py`.

### Notes
- **No Blackwell support changes.** v0.5.5 documents Blackwell as not yet supported (ORT bundled cuBLAS/cuDNN lacks sm_120 kernels). v0.7's TRT EP work doesn't change this — the upstream gap is at the kernel level, not the install level.
- **The 5.55× was measured on SmolVLA monolithic + A10G only.** Other model architectures (pi0.5 decomposed, GR00T DiT) and other hardware tiers (Orin Nano, T4, L4, H100) may show different ratios. Broader matrix tracked for v0.7.x.
- **The original v0.7 plan was different.** ADR `2026-04-28-tensorrt-llm-runtime-path.md` proposed adding a `--runtime tensorrt-llm` flag with a new RuntimeSession abstraction (~3-4 weeks). Three independent research subagents (Lenses 1+2+3 of `tensorrt-llm-runtime_research.md`) plus a Modal A10G spike invalidated that direction: TRT-LLM is LLM-shaped (not VLA-shaped), doesn't actually unblock Blackwell, and the perf win it was trying to deliver was already in v0.6.0 via ORT-TRT EP — users just weren't getting it because their installs were missing libs. The real v0.7 fix turned out to be 5 days of install/docs/doctor hardening, not 3-4 weeks of new architecture. Full reasoning in the ADRs at `reflex_context/01_decisions/2026-04-{28,29}-*.md`.

## v0.6.0 — 2026-04-29

Hardware-aware decomposed export + C-level crash visibility. Two real-fix surfaces from the v0.5.x first-tester debug session.

### Added
- **`--export-mode {auto, parallel, sequential}` flag** on `reflex export` for pi0.5 decomposed exports. `auto` (default) probes free GPU VRAM and picks parallel iff `2 × estimated_model_vram + 1 GB buffer < free_vram`. `parallel` forces parallel and **fails loudly with `InsufficientVRAMError` before any model load** if the GPU can't fit (no silent fallback). `sequential` always works (the safe baseline). Per CLAUDE.md "no silent degradation."
- **Multiprocessing wrapper for decomposed export** — when parallel mode is selected, `vlm_prefix` and `expert_denoise` exports run as two `multiprocessing.get_context("spawn")` worker processes that each load the policy independently from disk and produce their respective ONNX artifact in parallel. Subprocess crashes propagate to the parent with full traceback (no silent SubprocessError swallowing).
- **`faulthandler.enable()` in FastAPI lifespan** — Python's built-in fault handler now installs at server startup so any C-level crash (SIGSEGV / SIGABRT / SIGFPE) during model load prints a Python traceback to stderr **before** the process dies. Without this, signal-based deaths were silent (caught 2026-04-28 by Rob's RTX 5090 segfault). Disable via `REFLEX_NO_FAULTHANDLER=1` env var.
- **18 unit tests** for export-mode auto-detection across the hardware matrix (Mac CPU, Orin Nano 8 GB, A10G 24 GB, A100 80 GB, RTX 5090 32 GB, T4 16 GB borderline) — verify mode selection + InsufficientVRAMError behavior.
- **Modal A100-80GB validation** of the full pipeline (`scripts/modal_export_pi05_decomposed.py --export-mode auto/parallel`). Auto run produced clean 12,995 MB decomposed export. Forced-parallel raised `InsufficientVRAMError` before model load. Full reproducer + cost (~$5) in `reflex_context/03_experiments/2026-04-28-pi05-decomposed-export-mode-modal.md`.

### Notes
- **Pi0.5 parallel actual-success not yet recorded** — would need a GPU with at least the conservative free-VRAM threshold (~112.7 GB free for pi0.5, only realistic on H200 / multi-GPU setups with workload that frees memory first). Documented honestly in the experiment note. SmolVLA decomposed export (which has much smaller per-pass VRAM) is the realistic parallel-win path; will land in a v0.6.x follow-up if/when SmolVLA decomposed export ships.
- **No Blackwell support changes** in v0.6.0 — Blackwell remains documented as not yet supported (per v0.5.5 README). Real Blackwell fix is the v0.7 TensorRT-LLM runtime path (ADR `2026-04-28-tensorrt-llm-runtime-path.md` in the vault).
- Repo gitignore extended to prevent agents/IDEs from accidentally creating a sibling `reflex_context/` inside reflex-vla (caught + cleaned up 2026-04-29).

## v0.5.5 — 2026-04-28

Blackwell GPU support — TensorRT EP auto-disabled to prevent segfault.

### Fixed
- **TensorRT EP no longer segfaults on Blackwell GPUs (RTX 50-series, B200, GB200).** Caught 2026-04-28 by first-tester Rob (RTX 5090, exit code 139 + `ip 0x0` per dmesg). Root cause: ONNX Runtime's bundled TensorRT runtime predates Blackwell (sm_100), so TRT can't register kernels for the unknown architecture and leaves a NULL function pointer that gets called during model load. Auto-detects Blackwell via `nvidia-smi` and excludes `TensorrtExecutionProvider` from the providers list before session creation.
- Falls back to `CUDAExecutionProvider` on Blackwell — same numerics (cos=+1.0 vs PyTorch), ~3-5× slower per inference. Suitable for chat / dev / prototyping / low-Hz robot control. Marginal for real-time control above 20 Hz.
- **Loud, multi-line warning at every server startup on Blackwell** explaining the perf trade-off, the upstream tracking issue, and what workloads are/aren't impacted. Per CLAUDE.md "no silent fallbacks that paper over errors" — this is documented degradation, not silent. The warning auto-clears when ORT ships a Blackwell-aware TensorRT bundle.

### Notes
- Detection patterns: `rtx 50`, `rtx pro 60`, `blackwell`, `b200`, `gb200`. Add new patterns to `_BLACKWELL_GPU_PATTERNS` in `runtime/server.py` if NVIDIA ships another Blackwell SKU under a different name.
- No code changes affect non-Blackwell GPUs — TRT EP path is identical to v0.5.4 there.
- Subprocess wrapper for catching C-level crashes generically (so the next "TRT segfaults on a future architecture we don't know about" is also visible) is deferred to v0.6.0 — needs careful lifespan handling design + test matrix.

## v0.5.4 — 2026-04-28

Real fixes for the four UX issues surfaced by the first-tester debugging session.

### Added
- **Pre-startup model load logging + loud failure on lifespan crash.** Before v0.5.4, if `server.load()` raised an exception during FastAPI's lifespan startup, uvicorn swallowed the exception and the user saw "Waiting for application startup." followed by silent process death. Now we wrap `server.load()`, print a full traceback to stderr with a clear "FATAL: model load failed" banner, list the three most common causes (stale cache / ORT-TRT mismatch / GPU OOM) with concrete fixes, then re-raise so uvicorn exits non-zero. Added "Loading model from ..." phase log so users can see what's happening during the 10-60s cold start.
- **Export cache version pinning + auto-invalidation.** `reflex go` now writes `_reflex_meta.json` alongside `VERIFICATION.md` with the reflex version, model_id, export_target, export_mode, and timestamp. On cache hit, validates: (1) version matches current package, (2) target matches what we'd build for current hardware. Mismatches → loud `⚠ Cache stale` message + automatic `rm -rf` of the cache dir + rebuild. Legacy caches (no `_reflex_meta.json`, built by ≤0.5.3) are also rebuilt automatically. No more silent stale-cache server crashes.

### Fixed
- **`__version__` in `reflex/__init__.py` was stale.** Pinned to `0.5.0` since launch despite shipping v0.5.0 → v0.5.3. Now matches `pyproject.toml`. The cache versioning fix above relies on `__version__` being correct.
- **Bundled embodiment presets used fractional `rtc_execution_horizon`** (`0.5` for franka/ur5, `0.4` for so100), triggering a deprecation warning at every `reflex go` startup since v0.5.2. Converted to integer counts: franka/ur5 → `25` (0.5 × chunk_size 50), so100 → `12` (0.4 × chunk_size 30). Schema v2 will reject fractionals.

### Documentation
- **README "Upgrading" section.** Calls out `pip install --upgrade reflex-vla` and `uv add --refresh reflex-vla` (uv caches the package index aggressively and won't see new releases without `--refresh` — caught when the first tester couldn't pull v0.5.3 right after release). Documents the manual cache-clear escape hatch for users still on ≤0.5.3 caches.

### Notes
- All four bugs caught from the same first-tester (Rob, RTX 5090, Arch Linux) debugging session that spanned roughly 90 minutes from public install URL going live → v0.5.4 ship.
- Total v0.5.x velocity: 5 patch releases in 6 hours, all driven by real first-user feedback.

## v0.5.3 — 2026-04-28

Quickstart polish — README + embodiment error message.

### Changed
- **README quickstart now leads with the no-embodiment smoke test.** The headline example used to be `reflex go --model smolvla-base --embodiment franka`, which crashes the server hard if the preset isn't loadable. The first command is now `reflex go --model smolvla-base` (always works, returns raw actions). Embodiment is the second example, framed as "now make it real."
- **Embodiment-not-found error is much more actionable.** Lists bundled presets, then three workarounds in priority order: (1) drop `--embodiment`, (2) use a bundled preset, (3) drop your own JSON at `~/.cache/reflex/embodiments/<name>.json` + use `--custom-embodiment-config`. Prior message just said "run scripts/emit_embodiment_presets.py" which isn't shipped to pip-installed users.

### Notes
- No behavioral changes to embodiment loading semantics — preset-not-found still fails hard at startup. The hard-vs-soft fail decision is parked for a separate v0.5.4 discussion.

## v0.5.2 — 2026-04-28

Embodiment presets ship in the package now.

### Fixed
- **Embodiment preset JSONs (`franka`, `so100`, `ur5`) now ship inside the package** at `reflex/embodiments/presets/`. Before v0.5.2 these lived only in `<repo>/configs/embodiments/` outside the package — so `pip install`ed users running `reflex go --embodiment franka` (which is the example in the README) hit `Unknown embodiment preset 'franka'. Available: (none)`. Caught within the first hour of public install (Rob, RTX 5090 testing run).
- **`pyproject.toml` now explicitly force-includes** `src/reflex/embodiments/presets/` in the wheel via `[tool.hatch.build.targets.wheel.force-include]` so the JSONs are guaranteed to ship.
- **Dev-mode fallback**: when running from a source checkout (editable install) and the bundled presets dir is missing for any reason, falls back to `<repo>/configs/embodiments/`. Keeps the dev workflow working.

### Notes
- No behavioral changes to embodiment loading or normalization — just package bundling.
- Embodiment is still optional: `reflex go --model X` (no `--embodiment`) works for testing without normalization.

## v0.5.1 — 2026-04-28

First-tester polish — bugs caught within the first hour of public install.

### Added
- **Bootstrap installer** at `https://fastcrest.com/install`. One-liner: `curl -fsSL https://fastcrest.com/install | sh`. Detects platform (Mac / Jetson Orin / NVIDIA GPU / CPU) and picks the right `[serve,*]` extras automatically. Bails fast on unsupported hardware (original Maxwell-era Jetson Nano: JetPack 4.6 + 4 GB memory + no Tensor Cores can't run modern VLAs — redirected to Mac / Orin / cloud paths). Bootstraps `pip` via `ensurepip` when missing (caught on Arch with system Python 3.13 lacking pip module). Source: `install.sh` in the repo root.

### Fixed
- **`reflex doctor` /tmp check is no longer misleading on tmpfs systems.** Many Linux distros mount `/tmp` as `tmpfs` (RAM-backed), where the previous "Free disk in /tmp" check was actually measuring free RAM. Doctor now detects tmpfs via `/proc/mounts` and labels the check accordingly: `(tmpfs/RAM-backed — model exports use ~/.cache/reflex/exports instead)`. Also clarifies that `/tmp` only holds transient ONNX/TRT scratch — the real export cache lives at `~/.cache/reflex/exports` (or `$REFLEX_HOME/exports`). Reduced the threshold from 10 GB → 2 GB since /tmp doesn't need to hold the full model artifact.

### Notes
- No functional code changes vs v0.5.0 in the model export, serve, or chat paths.
- The bootstrap installer is independent of the PyPI package — it just calls `pip install` after pre-flight checks. Existing `pip install reflex-vla` continues to work.

## v0.5.0 — 2026-04-28

License + repo visibility milestone.

### Changed
- **License: Apache 2.0 → Business Source License 1.1** (auto-converts to Apache 2.0 in 4 years). Same source-available model HashiCorp, MongoDB, Sentry, Cockroach, and Couchbase use. Free for any non-competitive use (personal, commercial, internal); restricts only competing hosted/embedded offerings. Older releases (v0.3.x, v0.4.x) remain Apache-licensed forever — that grant cannot be retracted.
- **GitHub repo flipped from private to public.** Source now visible at https://github.com/rylinjames/reflex-vla. Earlier hidden by accident.

### Security
- **Scrubbed leaked HuggingFace token** (`hf_rfnFx...`) from git history via `git filter-repo`. Also scrubbed accidentally-committed `.agents/` editor-agent session logs (135 files across 4 commits). The token has been revoked at huggingface.co.
- Added `.agents/` to `.gitignore` to prevent recurrence.

### Notes
- This is a license/visibility release — no functional code changes vs v0.4.1.
- The closed-source-binary architecture explored mid-development was reversed: BSL 1.1 provides legal protection against commercial cloning without losing the open-source adoption story.

## v0.4.1 — 2026-04-27

UX onboarding pass — discoverability + persistence + the missing one-command tool.

### Added
- **First-run welcome card in `reflex chat`** — shown once per machine (cached in `$REFLEX_HOME/.welcomed`), explains what the assistant can do, lists slash commands, suggests starter prompts. Blank-prompt-paralysis is dead.
- **Slash commands**: `/help`, `/tools`, `/history`, `/clear`, `/reset`, `/tour`. `/tools` lists all 17 tools grouped by category (Deploy / Models / Train / Inspect / Status). `/history` shows the conversation so far. `/tour` shows 5 example prompts to copy-paste.
- **Conversation persistence** — every chat session auto-saves to `$REFLEX_HOME/chat_history/session-YYYYMMDD-HHMMSS.jsonl` after each turn. New CLI flag: `reflex chat --resume` loads the most recent session so Ctrl+C never loses context.
- **`deploy_one_command` chat tool** wrapping `reflex go`. The chat agent can now do "deploy smolvla to my mac" as a single tool call instead of manually chaining 4 tools (probe → pull → export → serve). Closes the audit gap where the headline command wasn't in chat.

### Changed
- **`reflex` (no args) shows a curated action-first summary** instead of typer's alphabetical command dump. Leads with `chat` and `go` — the two commands 90% of users want — followed by `doctor` and `models list`. Full alphabetical list still available via `reflex --help`.

### Notes
- Total chat tools is now **17** (was 16). Regression test updated.
- 47/47 tests pass.

## v0.4.0 — 2026-04-27

Polish release — bundles five small wins + one new optional surface.

### Added
- **Textual TUI for `reflex chat`** (opt-in via `pip install 'reflex-vla[tui]'`, then `reflex chat --tui`). Multi-panel layout: scrollable transcript, dedicated tool-calls panel with live status, persistent input box, status bar with token/tool count. Mouse + scroll-back + keyboard shortcuts (Ctrl+L clear, Ctrl+R reset). Falls back to the Rich REPL automatically if textual isn't installed. New module: `src/reflex/chat/tui.py`.
- **Examples directory** — `examples/01-chat-quickstart.md`, `02-deploy-smolvla-jetson.md`, `03-distill-pi05.md`, `04-record-and-replay.md`. Self-contained walkthroughs for each major workflow.
- **Once-per-day upgrade nag** — `reflex --version` (or any subcommand) now prints a one-line nag to stderr if PyPI has a newer release. Cached 24h in `$REFLEX_HOME/.upgrade_check`. Disable via `REFLEX_NO_UPGRADE_CHECK=1`. Skipped on editable installs. New module: `src/reflex/upgrade_check.py`.
- **PyPI install digest script** — `python scripts/install_digest.py` pulls download counts via pypistats and prints a Markdown summary suitable for sharing.

### Changed
- **Chat tool result truncation now keeps the tail** (`executor.py:_smart_truncate`). Long stack traces and compile errors put the actionable info at the end; the old head-only truncation lost that. New default: 1/3 head + 2/3 tail with a marker.

### Notes
- `[tui]` extra adds ~5 MB (textual). Base install footprint unchanged for users who don't want the TUI.

## v0.3.5 — 2026-04-27

### Added
- **`reflex chat` now streams tokens live.** Reply renders to the terminal as it arrives instead of showing "thinking…" then a full block. Multi-tool queries still surface tool calls between turns. Use `--no-stream` for scripts that pipe output.
- New backend method: `ChatBackend.chat_stream()` — yields parsed OpenAI delta chunks (Server-Sent Events through the existing Cloudflare Worker proxy).
- New helper: `assemble_stream(chunks, on_token, on_tool_call_progress)` — assembles streaming chunks into a final assistant message dict, with optional callbacks for live UI.
- New event: `LoopState` emits `token` events (per content fragment) and `turn_start` events (per LLM round-trip).
- New flag: `LoopState.streaming: bool = True` — set False for tests that need deterministic single-shot replies.

### Changed
- **System prompt tightened against hallucination.** Added a CRITICAL rule: "Copy verbatim values (versions, paths, IDs, sizes, error messages) exactly from tool output. Do not paraphrase, round, or 'fix' them. If you didn't run a tool that returned the value, say 'I don't have that information' instead of guessing." Closes the v0.3.0 case where chat cited "torch 2.10.0" when the actual was 2.11.0.

### Fixed
- (No regressions — 46/46 tests pass including all 16 chat-tool routes.)

## v0.3.4 — 2026-04-27

### Changed
- **`reflex doctor` now suggests the right install extras for your platform.** On Apple Silicon (no NVIDIA), recommends `'reflex-vla[serve,onnx]'` (CPU runtime). On NVIDIA boxes, still recommends `'reflex-vla[serve,gpu]'`.
- **`reflex models pull` now accepts HuggingFace repo IDs** in addition to registry aliases. `reflex models pull lerobot/smolvla_base` works just like `reflex models pull smolvla-base` — automatically resolved to the registry entry.

### Fixed
- Doctor's install hint no longer eats `[serve,gpu]` due to Rich markup interpretation (escaped properly with raw string + escaped bracket).

## v0.3.3 — 2026-04-27

### Added
- **`reflex go` now actually deploys.** When a model has `requires_export=True`, `reflex go` runs the export inline (chains pull → export → serve) instead of printing manual instructions. Closes the biggest README→reality gap.
- New CLI command: **`reflex status`** — list running `reflex serve` processes via `ps` regex (PID, uptime, port, command).
- New CLI command: **`reflex config show`** — dump effective config (paths, defaults, env vars).
- New CLI command: **`reflex inspect traces`** — scan `~/.cache/reflex/traces` and `/tmp/traces` for JSONL files written by `reflex serve --record`. Filters: `--since`, `--task`, `--status`, `--limit`.
- New env var: **`REFLEX_HOME`** — overrides `~/.cache/reflex` for export cache root + config defaults.
- New regression test: `tests/test_chat_tools_executable.py` parametrized over all 16 chat tools — asserts each routes to a real CLI command.

### Changed
- `reflex chat` proxy default URL is now `https://chat.fastcrest.com` (was `fastcrest-proxy.fastcrest.workers.dev`).
- Export cache for `reflex go` lands at `~/.cache/reflex/exports/<model_id>/` (or `$REFLEX_HOME/exports/<model_id>/`). Cache-skip on `VERIFICATION.md` marker.

### Fixed
- **Rich markup ate `[monolithic]`** in install hints. Users saw `pip install 'reflex-vla'` instead of `pip install 'reflex-vla[monolithic]'`. Fixed by escaping brackets or using `markup=False` in 5 places.
- 4 `reflex chat` tools routed to non-existent CLI commands. All four now route correctly thanks to the new `status` / `config show` / `inspect traces` commands.

## v0.3.2 — 2026-04-26

### Changed
- **24× CLI speedup.** `reflex/__init__.py` now lazy-loads `validate_roundtrip` + `fixtures` via PEP 562 `__getattr__` instead of eager-importing torch. `reflex --version` 2.4s → **0.10s**. Every fast-path command (`--help`, `chat`, `models list`, `inspect targets`, `inspect traces`, `config show`) is now sub-second on a warm cache. `reflex doctor` still imports torch on-demand (correct — it's the diagnostic).

## v0.3.0 — 2026-04-26

### Added
- **`reflex chat` ships.** Natural-language CLI agent: GPT-5 Mini routes user prompts to 16 reflex tools (export, serve, bench, eval, distill, finetune, traces, doctor, etc.) and runs them as subprocess. Hosted Cloudflare Worker proxy at `chat.fastcrest.com` — free tier 100 calls/day per machine, no signup, no API key.
- New module: `src/reflex/chat/{schema,backends,executor,loop,console}.py`
- New CLI: `reflex chat [--proxy-url URL] [--dry-run]`

### Changed
- **PyPI launch.** `pip install reflex-vla` now works without a git URL. Bumped from internal `0.1.0` to public `0.3.0`.
- README rebranded with "Reflex by [FastCrest](https://fastcrest.com)" header + chat quickstart at top.

## v0.2.x (pre-PyPI internal milestone tags — never published)

## Unreleased

### Added
- `reflex validate` command now runs a real ONNX/TRT-vs-PyTorch round-trip parity check.
  - Seeded fixtures for SmolVLA, pi0, GR00T (pi0.5 and OpenVLA defer to v2).
  - Per-fixture max/mean L2 abs-diff + summary, JSON and Rich table output.
  - `--init-ci` emits a GitHub Actions workflow template at `.github/workflows/reflex-validate.yml`.
  - Exit codes: 0 pass, 1 fail, 2 error.
- Public exports added to `reflex`: `ValidateRoundTrip`, `load_fixtures`, `SUPPORTED_MODEL_TYPES`.

### Changed
- **BREAKING (from stub):** `reflex validate` default `--threshold` changed from `0.02` (the v0.1 placeholder) to `1e-4`. The stub never performed real validation so no existing deployments depended on the old default. Pass `--threshold 0.02` explicitly to match the previous behavior.
- `reflex validate` now requires a valid `reflex_config.json` inside the export directory — the stub accepted any path.

### Fixed
- `_pytorch_backend` SmolVLA path no longer swallows `AutoConfig` fetch errors silently — now logs a warning with the exception and continues with the fallback head_dim.
- CLI handler now catches `KeyboardInterrupt` explicitly (exits 130) instead of emitting a raw traceback.

## v0.1.0 (previous)

Initial release — see README for the seven-wedge scope at that time.
