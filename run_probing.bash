#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv-jlab/bin/python}"
INTERNALS_ROOT="${INTERNALS_ROOT:-data/eval_adoption_internals_table_filtered}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
MODEL_NAME="${MODEL_NAME:-deepseek-r1-0528-qwen3-8b}"
TARGET_COL="${TARGET_COL:-absolute_accuracy_decay}"
REDUCED_DIM="${REDUCED_DIM:-0}"
SEEDS="${SEEDS:-42,43,44,45,46}"
NUM_FOLDS="${NUM_FOLDS:-4}"
DEV_FRACTION="${DEV_FRACTION:-0.2}"
NUM_WORKERS="${NUM_WORKERS:-}"
BINARY_EVAL_COL="${BINARY_EVAL_COL:-model_is_robust}"
THRESHOLD="${THRESHOLD:-auto}"
DUMP_PROBE_ARTIFACTS="${DUMP_PROBE_ARTIFACTS:-0}"
ARTIFACT_MODEL_ID="${ARTIFACT_MODEL_ID:-}"
LAYERS="${LAYERS:-}"
PERMUTATION_TYPES="${PERMUTATION_TYPES:-}"
CONTROL_TASKS="${CONTROL_TASKS:-}"

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr '/:' '__' | tr -cs 'a-z0-9._-' '_'
}

MODEL_SLUG="$(slugify "$MODEL_NAME")"
TARGET_SLUG="$(slugify "$TARGET_COL")"

discover_views() {
  if [[ -f "$INTERNALS_ROOT/metadata.csv" ]]; then
    printf '%s\n' "$INTERNALS_ROOT"
    return
  fi

  local found=0
  local candidate
  for candidate in "$INTERNALS_ROOT"/*; do
    if [[ -d "$candidate" && -f "$candidate/metadata.csv" ]]; then
      printf '%s\n' "$candidate"
      found=1
    fi
  done

  if [[ "$found" -eq 0 ]]; then
    echo "No internals directories found under $INTERNALS_ROOT" >&2
    exit 1
  fi
}

while IFS= read -r INTERNALS_DIR; do
  VIEW_NAME="$(basename "$INTERNALS_DIR")"

  RESULT_DIR="$RESULTS_ROOT/eval_adoption_${TARGET_SLUG}_${MODEL_SLUG}_${VIEW_NAME}_probe"
  if [[ "$REDUCED_DIM" -gt 0 ]]; then
    RESULT_DIR="${RESULT_DIR}_pca${REDUCED_DIM}"
  fi
  RESULT_DIR="${RESULT_DIR}_cv_v1"

  ARGS=(
    scripts/probe.py
    --internals-dir "$INTERNALS_DIR"
    --results-dir "$RESULT_DIR"
    --model-name "$MODEL_NAME"
    --target-col "$TARGET_COL"
    --method probe
    --seeds "$SEEDS"
    --num-folds "$NUM_FOLDS"
    --dev-fraction "$DEV_FRACTION"
    --reduced-dim "$REDUCED_DIM"
    --binary-eval-col "$BINARY_EVAL_COL"
    --threshold "$THRESHOLD"
  )

  if [[ -n "$NUM_WORKERS" ]]; then
    ARGS+=(--num-workers "$NUM_WORKERS")
  fi
  if [[ "$DUMP_PROBE_ARTIFACTS" == "1" ]]; then
    ARGS+=(--dump-probe-artifacts)
  fi
  if [[ -n "$ARTIFACT_MODEL_ID" ]]; then
    ARGS+=(--artifact-model-id "$ARTIFACT_MODEL_ID")
  fi
  if [[ -n "$LAYERS" ]]; then
    ARGS+=(--layers "$LAYERS")
  fi
  if [[ -n "$PERMUTATION_TYPES" ]]; then
    ARGS+=(--permutation-types "$PERMUTATION_TYPES")
  fi
  if [[ -n "$CONTROL_TASKS" ]]; then
    ARGS+=(--control-tasks "$CONTROL_TASKS")
  fi

  echo "Running probe for $VIEW_NAME:"
  printf '  %q' "$PYTHON_BIN" "${ARGS[@]}"
  printf '\n'

  "$PYTHON_BIN" "${ARGS[@]}"
done < <(discover_views)
