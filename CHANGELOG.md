# Changelog

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
