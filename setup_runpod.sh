#!/usr/bin/env bash
# Bootstrap a fresh RunPod (or any Linux+CUDA box) for the autoresearch project.
# Idempotent: safe to re-run.
#
# Usage:
#   bash setup_runpod.sh
#
# What it does:
#   1. Installs `uv` (fast Python package manager) if missing
#   2. `uv sync` — installs torch (CUDA 12.4 wheel) + matplotlib + pyarrow + ...
#   3. Verifies CUDA is visible from PyTorch
#   4. `uv run prepare.py` — downloads the validation/training shards and
#      trains the BPE tokenizer (~2 min on a fast disk + GPU)
#   5. Sanity-checks `uv run train.py` parses (without actually training)
#
# After this script finishes you can:
#   bash run_overnight.sh    # to launch Claude Code in autonomous mode
#   uv run train.py          # to run a single experiment manually
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; NC='\033[0m'
say() { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[setup]${NC} $*"; }
die()  { echo -e "${RED}[setup ERROR]${NC} $*" >&2; exit 1; }

cd "$(dirname "$0")"

# --- 1. uv -------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1090
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
uv --version || die "uv install failed"

# --- 2. deps -----------------------------------------------------------------
say "Syncing Python dependencies (this installs PyTorch ~ 2 GB)..."
uv sync

# --- 3. CUDA visibility ------------------------------------------------------
say "Checking CUDA via PyTorch..."
uv run python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    print("CUDA NOT VISIBLE TO PYTORCH", file=sys.stderr)
    sys.exit(2)
print(f"  torch:        {torch.__version__}")
print(f"  cuda:         {torch.version.cuda}")
print(f"  device:       {torch.cuda.get_device_name(0)}")
print(f"  vram (GB):    {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}")
print(f"  capability:   {torch.cuda.get_device_capability(0)}")
PY
say "CUDA OK."

# --- 4. data + tokenizer -----------------------------------------------------
if [ ! -f "$HOME/.cache/autoresearch/tokenizer/tokenizer.pkl" ]; then
    say "Running prepare.py (downloading FineWeb-Edu shards + training tokenizer)..."
    uv run prepare.py
else
    say "Tokenizer + data already cached at ~/.cache/autoresearch/ — skipping prepare.py."
fi

# --- 5. train.py syntax sanity ----------------------------------------------
say "Parsing train.py..."
uv run python -c "import ast; ast.parse(open('train.py').read()); print('  train.py parses')"

# --- 6. (optional) install Claude Code CLI if not present --------------------
if ! command -v claude >/dev/null 2>&1; then
    warn "Claude Code CLI ('claude') is not installed."
    warn "If you want to drive this autoresearch loop autonomously, install it via:"
    warn "    npm install -g @anthropic-ai/claude-code"
    warn "    # then: claude login   (or set ANTHROPIC_API_KEY)"
    warn "Or you can SSH from your laptop and run claude there with this folder mounted."
fi

say "Setup complete. Next steps:"
echo "    1. Smoke-test:    uv run train.py     # ~5 min training + ~30 s eval"
echo "    2. Overnight:     bash run_overnight.sh"
echo ""
