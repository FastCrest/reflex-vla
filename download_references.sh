#!/usr/bin/env bash
# Clone reference repos for serve design work.
# See reflex_context/01_decisions/2026-04-22-reference-repos-folder.md.
#
# Usage:  bash download_references.sh
#
# Idempotent — skips repos that already exist. To refresh, delete the
# subdir and re-run.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p reference
cd reference

clone_if_missing() {
  local name="$1"; local url="$2"
  if [[ -d "$name" ]]; then
    echo "  ✓ $name already cloned (skip)"
  else
    echo "  → cloning $name from $url"
    git clone --depth 1 "$url" "$name"
  fi
}

echo "[reference] cloning competitor OSS repos (shallow, --depth 1)…"

# Inference servers
clone_if_missing triton                  https://github.com/triton-inference-server/server.git
clone_if_missing vllm                    https://github.com/vllm-project/vllm.git
clone_if_missing tgi                     https://github.com/huggingface/text-generation-inference.git
clone_if_missing ray                     https://github.com/ray-project/ray.git
clone_if_missing trtllm                  https://github.com/NVIDIA/TensorRT-LLM.git

# VLA reference impls
clone_if_missing lerobot                 https://github.com/huggingface/lerobot.git
clone_if_missing openpi                  https://github.com/Physical-Intelligence/openpi.git

echo ""
echo "[reference] done. Total size:"
du -sh . 2>/dev/null || true

if [[ ! -f NOTES.md ]]; then
  cat > NOTES.md <<'EOF'
# Reference Notes

Lookup-first log. When you grep for a pattern in `reference/` and find
something useful, jot a one-liner here so future-you doesn't have to
re-grep. Format:

- `<repo>/<path>:<line>` — what it does, when we'd copy it

## Examples

- `vllm/vllm/core/block_manager_v2.py` — paged-attention KV cache. Read
  before designing episode-aware prefix cache.
- `lerobot/lerobot/policies/pi05/processing_pi05.py` — pi0.5
  preprocessor. State-in-lang behavior is in PI05PrepareTokenizerStep.
- `openpi/src/openpi/policies/pi0_pytorch.py` — canonical pi0 forward.
- `triton/src/core/dynamic_batch_scheduler.cc` — Triton's batching
  scheduler. Reference for our continuous-batching work.

(Add yours below)

EOF
fi

echo ""
echo "[reference] NOTES.md ready at reference/NOTES.md"
echo "[reference] grep workflow:  grep -r '<concept>' reference/ | head"
