#!/usr/bin/env bash
# Reflex VLA bootstrap installer
# Runs hardware + Python checks BEFORE pip install so users on
# unsupported configs don't waste time debugging the wrong thing.
#
# Usage:
#   curl -sSf https://fastcrest.com/install | sh
#   curl -sSf https://fastcrest.com/install | sh -s -- --extras serve,gpu
#
# Source: https://github.com/rylinjames/reflex-vla
set -eu

# -- ANSI colors --------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  BOLD="\033[1m"; DIM="\033[2m"; RESET="\033[0m"
  RED="\033[31m"; GREEN="\033[32m"; YELLOW="\033[33m"; BLUE="\033[34m"; CYAN="\033[36m"
else
  BOLD=""; DIM=""; RESET=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; CYAN=""
fi

info()  { printf "%b%s%b\n" "${BLUE}" "$*" "${RESET}"; }
ok()    { printf "%b✓%b %s\n" "${GREEN}" "${RESET}" "$*"; }
warn()  { printf "%b⚠%b %s\n" "${YELLOW}" "${RESET}" "$*"; }
fail()  { printf "%b✗%b %s\n" "${RED}" "${RESET}" "$*"; }
note()  { printf "%b%s%b\n" "${DIM}" "$*" "${RESET}"; }

printf "\n%b%s%b\n" "${BOLD}${CYAN}" "Reflex VLA installer" "${RESET}"
note   "  Bootstrap that checks your environment before installing."
echo

# -- Parse args ---------------------------------------------------------------
EXTRAS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --extras) EXTRAS="$2"; shift 2 ;;
    --extras=*) EXTRAS="${1#--extras=}"; shift ;;
    *) shift ;;
  esac
done

# -- Detect OS ----------------------------------------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"
info "Environment"
echo "  OS:   $OS"
echo "  Arch: $ARCH"

# -- Detect Python ------------------------------------------------------------
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$candidate"; break
  fi
done

if [ -z "$PYTHON" ]; then
  echo
  fail "No Python interpreter found."
  note "  Install Python 3.10+ first:"
  note "    macOS:  brew install python@3.11"
  note "    Linux:  pyenv install 3.11.9 && pyenv global 3.11.9"
  exit 1
fi

PY_VER="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
PY_MAJOR="$($PYTHON -c 'import sys; print(sys.version_info.major)')"
PY_MINOR="$($PYTHON -c 'import sys; print(sys.version_info.minor)')"

echo "  Python: $PY_VER  ($PYTHON)"
echo

# -- Hardware detection -------------------------------------------------------
JETSON_MODEL=""
if [ -r /proc/device-tree/model ]; then
  # Strip null bytes that Jetson DT model files often have
  JETSON_MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || true)"
fi

IS_OLD_NANO=0
IS_ORIN=0
if [ -n "$JETSON_MODEL" ]; then
  case "$JETSON_MODEL" in
    *"Jetson Nano"*|*"jetson-nano"*|*"Tegra X1"*|*"p3450"*|*"p3448"*)
      IS_OLD_NANO=1 ;;
    *"Orin"*|*"orin"*)
      IS_ORIN=1 ;;
  esac
fi

# -- Branch on hardware -------------------------------------------------------
if [ "$IS_OLD_NANO" -eq 1 ]; then
  echo
  fail "Detected: $JETSON_MODEL"
  echo
  warn "The original Jetson Nano (Maxwell, 2019) is too old for Reflex."
  note "  • NVIDIA EOL'd it at JetPack 4.6 (Ubuntu 18 / Python 3.6) in 2022."
  note "  • Maxwell GPU has no Tensor Cores; modern VLA models can't run usefully."
  note "  • 4 GB shared memory will OOM loading pi0 / SmolVLA in FP32."
  echo
  info "Recommended paths:"
  echo "  ${BOLD}1. Run on a Mac${RESET} (CPU path, works for chat + dev)"
  echo "       pip install 'reflex-vla[serve,onnx]'"
  echo "  ${BOLD}2. Run on Jetson Orin${RESET} (Orin Nano / NX / AGX) — full GPU support"
  echo "  ${BOLD}3. Run in the cloud${RESET} (Modal, Lambda, RunPod with NVIDIA T4+)"
  echo
  exit 1
fi

# -- Python version check -----------------------------------------------------
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  fail "Python $PY_VER is too old. Reflex requires Python 3.10+."
  echo
  info "Install a newer Python:"
  case "$OS" in
    Darwin)
      note "  brew install python@3.11"
      note "  # or: pyenv install 3.11.9 && pyenv global 3.11.9" ;;
    Linux)
      note "  pyenv install 3.11.9 && pyenv global 3.11.9"
      note "  # or via your distro: apt install python3.11   (Ubuntu 22+)" ;;
    *)
      note "  Use pyenv or your platform's package manager to install Python 3.10+." ;;
  esac
  echo
  exit 1
fi
ok "Python $PY_VER (>=3.10 required)"

# -- Pick install extras based on detected platform ---------------------------
if [ -z "$EXTRAS" ]; then
  if [ "$IS_ORIN" -eq 1 ]; then
    EXTRAS="serve,gpu"
    ok "Detected Jetson Orin → installing with [serve,gpu]"
  elif [ "$OS" = "Darwin" ]; then
    EXTRAS="serve,onnx"
    ok "Detected macOS → installing with [serve,onnx] (CPU runtime)"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    EXTRAS="serve,gpu"
    ok "Detected NVIDIA GPU → installing with [serve,gpu]"
  else
    EXTRAS="serve,onnx"
    ok "No GPU detected → installing with [serve,onnx] (CPU runtime)"
  fi
fi
echo

# -- Ensure pip is available --------------------------------------------------
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
  warn "pip not available for $PYTHON — trying to bootstrap via ensurepip"
  if "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1; then
    ok "Installed pip via ensurepip"
  else
    fail "pip is missing and ensurepip can't bootstrap it on this system."
    echo
    info "Install pip for your distro, then re-run this installer:"
    note "  Arch:    sudo pacman -S python-pip"
    note "  Debian:  sudo apt install python3-pip"
    note "  Fedora:  sudo dnf install python3-pip"
    note "  Alpine:  sudo apk add py3-pip"
    note "  macOS:   pip ships with the Python.org installer / Homebrew Python"
    echo
    exit 1
  fi
fi

# -- Run pip install ----------------------------------------------------------
PIP_TARGET="reflex-vla[$EXTRAS]"
info "Installing: $PIP_TARGET"
echo
"$PYTHON" -m pip install --upgrade "$PIP_TARGET"

echo
ok "Installed."
echo
info "Next:"
echo "  reflex chat        # natural-language CLI"
echo "  reflex doctor      # check your install"
echo "  reflex --help      # see all commands"
echo
note "Source: https://github.com/rylinjames/reflex-vla"
