#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-jlab}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
HOLMES_REPO="${HOLMES_REPO:-https://github.com/Holmes-Benchmark/holmes-evaluation.git}"
HOLMES_BRANCH="${HOLMES_BRANCH:-probe_only}"
INSTALL_KERNEL="${INSTALL_KERNEL:-0}"
KERNEL_NAME="${KERNEL_NAME:-aimo-probe-artifacts}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-AIMO Probe Artifacts}"

echo "Repo root: $ROOT_DIR"
echo "Virtualenv: $VENV_DIR"
echo "Python: $PYTHON_BIN"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements.txt"
python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

if [[ -f "$ROOT_DIR/.gitmodules" ]] && git -C "$ROOT_DIR" config --file .gitmodules --get-regexp '^submodule\\.holmes-evaluation\\.' >/dev/null; then
  git -C "$ROOT_DIR" submodule update --init --recursive holmes-evaluation
else
  if [[ ! -d "$ROOT_DIR/holmes-evaluation/.git" ]]; then
    git clone --branch "$HOLMES_BRANCH" "$HOLMES_REPO" "$ROOT_DIR/holmes-evaluation"
  else
    echo "Holmes checkout already exists: $ROOT_DIR/holmes-evaluation"
  fi
fi

if [[ "$INSTALL_KERNEL" == "1" ]]; then
  python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"
fi

cat <<EOF

Setup complete.

Activate:
  source "$VENV_DIR/bin/activate"

Encode representations:
  PYTHON_BIN="$VENV_DIR/bin/python" \\
  MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \\
  OUTPUT_DIR=data/eval_adoption_internals_qwen0_5b_math \\
  bash run_encoding.bash

Train probes and export layer/seed pickle artifacts:
  "$VENV_DIR/bin/python" scripts/probe.py \\
    --internals-dir data/eval_adoption_internals_qwen0_5b_math \\
    --results-dir results/eval_adoption_model_is_robust_qwen0_5b_probe_artifacts \\
    --model-name qwen2.5-0.5b-instruct \\
    --target-col model_is_robust \\
    --method probe \\
    --seeds 42,43,44,45,46 \\
    --num-folds 4 \\
    --dump-probe-artifacts \\
    --artifact-model-id Qwen/Qwen2.5-0.5B-Instruct

Final pickle artifacts will be written to:
  results/eval_adoption_model_is_robust_qwen0_5b_probe_artifacts/probe_artifacts/

EOF
