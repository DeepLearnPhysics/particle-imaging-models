#!/usr/bin/env bash
set -euo pipefail

LAUNCHER_ONLY=0

# clone target and branch, overridable for forks and CI
REPO="${PIMM_REPO:-DeepLearnPhysics/particle-imaging-models}"
BRANCH="${PIMM_BRANCH:-main}"
# set SKIP_CLONE=1 to install from the current checkout instead of cloning
SKIP_CLONE="${SKIP_CLONE:-}"

usage() {
  cat <<'EOF'
Usage: ./install.sh [--launcher-only]

  --launcher-only  install only the pimm launch/submit command dependencies

Run in a checkout, or bootstrap from scratch:

  curl -sSL https://raw.githubusercontent.com/DeepLearnPhysics/particle-imaging-models/main/install.sh | bash

Environment: PIMM_REPO, PIMM_BRANCH, SKIP_CLONE=1.
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  command -v curl >/dev/null 2>&1 || fail "curl is required to install uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null 2>&1 || fail "uv was installed but is not on PATH"
}

# a pimm checkout is any directory whose pyproject.toml declares name = "particle-imaging-models"
in_pimm_checkout() {
  [ -f pyproject.toml ] && grep -qE '^name = "particle-imaging-models"' pyproject.toml
}

ensure_checkout() {
  if [ "$SKIP_CLONE" = "1" ] || in_pimm_checkout; then
    return
  fi
  command -v git >/dev/null 2>&1 || fail "git is required to clone pimm"
  local dir
  dir="$(basename "$REPO")"
  if [ ! -d "$dir/.git" ]; then
    echo "cloning $REPO (branch $BRANCH) into $dir"
    git clone --branch "$BRANCH" "https://github.com/${REPO}.git" "$dir"
  fi
  cd "$dir"
}

# the prebuilt training wheels exist only for linux x86_64
require_supported_platform() {
  [ "$(uname -s)" = "Linux" ] || fail "the training environment supports Linux only"
  [ "$(uname -m)" = "x86_64" ] || fail "the training environment supports x86_64 only"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --launcher-only)
      LAUNCHER_ONLY=1
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
  shift
done

ensure_uv
ensure_checkout

if [ "$LAUNCHER_ONLY" -eq 1 ]; then
  uv sync --locked --only-group launcher
  uv pip install --no-deps -e .
  echo "launcher environment ready. run commands with: uv run --no-sync pimm"
  exit 0
fi

require_supported_platform

uv sync --locked

uv run python - <<PY
import importlib

modules = ["pimm", "pimm.models", "spconv", "torch"]
for module in modules:
    importlib.import_module(module)
print("validated:", ", ".join(modules))
PY

echo "training environment ready. run commands with: uv run pimm"
