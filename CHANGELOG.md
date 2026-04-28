# Changelog

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
