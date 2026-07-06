#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-jlab}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
INSTALL_KERNEL="${INSTALL_KERNEL:-1}"
KERNEL_NAME="${KERNEL_NAME:-aimo-eval-probes}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-AIMO Eval Probes}"

echo "Repo root: $ROOT_DIR"
echo "Virtualenv: $VENV_DIR"
echo "Python: $PYTHON_BIN"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

BASE_PACKAGES=(
  pandas
  numpy
  pyarrow
  scipy
  scikit-learn
  matplotlib
  datasets
  huggingface_hub
  ipykernel
  jupyterlab
  pytorch-lightning
  redis
  retry
  transformers
)

python -m pip install "${BASE_PACKAGES[@]}"

python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

if [[ "$INSTALL_KERNEL" == "1" ]]; then
  python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"
fi

if [[ -f "$ROOT_DIR/.gitmodules" ]] && git -C "$ROOT_DIR" config --file .gitmodules --get-regexp '^submodule\\.holmes-evaluation\\.' >/dev/null; then
  git -C "$ROOT_DIR" submodule update --init --recursive holmes-evaluation
elif [[ ! -d "$ROOT_DIR/holmes-evaluation/.git" ]]; then
  git clone --branch probe_only https://github.com/Holmes-Benchmark/holmes-evaluation.git "$ROOT_DIR/holmes-evaluation"
fi

cat <<EOF

Setup complete.

Activate:
  source "$VENV_DIR/bin/activate"

PyTorch wheel index:
  $TORCH_INDEX_URL

If you need a different build, rerun with for example:
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 bash scripts/setup_server.sh

EOF
