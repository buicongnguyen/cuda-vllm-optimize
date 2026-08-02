#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$SOURCE_ROOT"
SOURCE_GIT_SHA="$(git -c safe.directory="$SOURCE_ROOT" -C "$SOURCE_ROOT" rev-parse HEAD 2>/dev/null || true)"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/lfm-racebench-rtx4080}"
WORK_ROOT="${WORK_ROOT:-$HOME/src/cuda-vllm-optimize}"

if [[ "$(uname -s)" != "Linux" ]] || ! grep -qi microsoft /proc/version; then
  echo "ERROR: run this script inside Ubuntu WSL2." >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not visible inside WSL2. Update the Windows NVIDIA driver first." >&2
  exit 2
fi

GPU_LINE="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | head -n 1)"
echo "GPU: $GPU_LINE"
if [[ "${GPU_LINE^^}" != *"RTX 4080 SUPER"* ]]; then
  echo "WARNING: this setup was designed for RTX 4080 Super; detected: $GPU_LINE" >&2
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required. Install it with: sudo apt-get update && sudo apt-get install -y curl" >&2
  exit 2
fi

if [[ "$SOURCE_ROOT" == /mnt/* ]]; then
  case "$WORK_ROOT" in
    "$HOME"/src/*) ;;
    *)
      echo "ERROR: WORK_ROOT must stay under $HOME/src when syncing from /mnt." >&2
      exit 2
      ;;
  esac
  if ! command -v rsync >/dev/null 2>&1; then
    echo "ERROR: rsync is required to copy the repo into the WSL filesystem." >&2
    exit 2
  fi
  echo "Mirroring Windows checkout into WSL ext4: $WORK_ROOT"
  mkdir -p "$WORK_ROOT"
  rsync -a --delete \
    --exclude '.git/' \
    --exclude 'results/' \
    --exclude '__pycache__/' \
    --exclude '*.egg-info/' \
    "$SOURCE_ROOT/" "$WORK_ROOT/"
  if [[ -n "$SOURCE_GIT_SHA" ]]; then
    printf '%s\n' "$SOURCE_GIT_SHA" > "$WORK_ROOT/.source-git-sha"
  fi
  ROOT_DIR="$WORK_ROOT"
fi

if ! command -v uv >/dev/null 2>&1; then
  INSTALLER="$(mktemp)"
  curl -LsSf https://astral.sh/uv/install.sh -o "$INSTALLER"
  sh "$INSTALLER"
  rm -f "$INSTALLER"
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating $VENV_DIR with Python $PYTHON_VERSION"
uv python install "$PYTHON_VERSION"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
else
  echo "Reusing existing virtual environment: $VENV_DIR"
  if ! "$VENV_DIR/bin/python" -c 'import sys; raise SystemExit(0 if f"{sys.version_info.major}.{sys.version_info.minor}" == sys.argv[1] else 1)' "$PYTHON_VERSION"; then
    echo "ERROR: $VENV_DIR does not use Python $PYTHON_VERSION. Choose a new VENV_DIR or remove that dedicated venv." >&2
    exit 2
  fi
fi
source "$VENV_DIR/bin/activate"

echo "Installing vLLM $VLLM_VERSION and the replay harness"
if ! python -c 'import importlib.metadata as m, sys; raise SystemExit(0 if m.version("vllm") == sys.argv[1] else 1)' "$VLLM_VERSION" >/dev/null 2>&1; then
  uv pip install --torch-backend=auto "vllm==$VLLM_VERSION"
fi
uv pip install -e "$ROOT_DIR[rtx4080]"

if ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
  echo "No system C compiler found; installing portable Zig compiler for Triton JIT"
  uv pip install "ziglang==0.16.0"
  CC_WRAPPER="$VENV_DIR/bin/triton-cc"
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    'args=()' \
    'for arg in "$@"; do' \
    '  if [[ "$arg" == "-l:libcuda.so.1" && -f /usr/lib/wsl/lib/libcuda.so.1 ]]; then' \
    '    args+=(/usr/lib/wsl/lib/libcuda.so.1)' \
    '  else' \
    '    args+=("$arg")' \
    '  fi' \
    'done' \
    'exec "$(dirname "$0")/python" -m ziglang cc "${args[@]}"' \
    > "$CC_WRAPPER"
  chmod +x "$CC_WRAPPER"
fi

mkdir -p "$ROOT_DIR/results/rtx4080"
uv pip freeze > "$ROOT_DIR/results/rtx4080/environment.freeze.txt"

python "$ROOT_DIR/scripts/rtx4080_lab.py" doctor

cat <<EOF

Setup complete.

Activate:
  source "$VENV_DIR/bin/activate"

Smoke run (downloads the pinned model on first use):
  cd "$ROOT_DIR"
  python scripts/rtx4080_lab.py run --mode smoke

Full R0/B/R0-prime experiment:
  python scripts/rtx4080_lab.py run --mode aba
EOF
