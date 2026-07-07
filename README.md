# AIMO Baseline: Encode, Probe, Export

This repository contains the representation-probe baseline for the AIMO
Interpretability Challenge. The intended workflow is deliberately small:

1. Set up the Python environment and Holmes submodule.
2. Encode final input-token hidden states for each dataset row.
3. Train linear probes across layers and seeds.
4. Optionally export a submission artifact with the trained probes.

## Setup

Clone with submodules, or initialize the submodule after cloning:

```bash
git submodule update --init --recursive holmes-evaluation
```

Create the environment:

```bash
bash scripts/setup_probe_artifact_env.sh
```

By default this creates `.venv-jlab` and installs CUDA 12.1 PyTorch wheels. For
CPU-only PyTorch:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
bash scripts/setup_probe_artifact_env.sh
```

Activate the environment:

```bash
source .venv-jlab/bin/activate
```

## Encode Representations

The encoder runs the model once per unique `original_problem` and saves every
layer's final input-token hidden state.

```bash
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
DATASET_CSV=data/math-robust-final.csv \
OUTPUT_DIR=data/eval_adoption_internals_qwen0_5b_math \
DEVICE=auto \
bash run_encoding.bash
```

The output directory contains:

- `metadata.csv`
- `layer_000.npy`
- `layer_001.npy`
- ...

## Train Probes

Run linear probes on the encoded representations:

```bash
INTERNALS_ROOT=data/eval_adoption_internals_qwen0_5b_math \
MODEL_NAME=qwen2.5-0.5b-instruct \
TARGET_COL=model_is_robust \
NUM_WORKERS=1 \
bash run_probing.bash
```

Useful optional limits for quick tests:

```bash
LAYERS=12 \
PERMUTATION_TYPES=domain \
CONTROL_TASKS=NONE \
SEEDS=42,43,44,45,46 \
NUM_FOLDS=2 \
bash run_probing.bash
```

## Export Probe Artifact

To export a pickle artifact for the submission bundle, enable artifact dumping
and provide the Hugging Face model id used during encoding:

```bash
INTERNALS_ROOT=data/eval_adoption_internals_qwen0_5b_math \
MODEL_NAME=qwen2.5-0.5b-instruct \
TARGET_COL=model_is_robust \
DUMP_PROBE_ARTIFACTS=1 \
ARTIFACT_MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct \
NUM_WORKERS=1 \
bash run_probing.bash
```

Final artifacts are written to:

```text
results/<run-name>/probe_artifacts/*.pkl
```

Each pickle contains:

- `best_layer_index`: layer selected from the best validation probe
- `best_probe`: layer, seed, metric, and score
- `probes[layer]["weights"]`: shape `(5, hidden_dim)` for the default five seeds
- `probes[layer]["bias"]`: shape `(5,)`
- `probes[layer]["threshold"]`: shape `(5,)`
- `aggregation`: `mean_margin`

The submission should load `best_layer_index`, encode that layer's final
input-token hidden state, compute all five probe scores, and return
`mean(scores - threshold) >= 0`.

## Direct Commands

The wrappers call these scripts:

```bash
python scripts/encode.py --help
python scripts/probe.py --help
```

Use the direct scripts when you need fine-grained options beyond the wrapper
environment variables.
