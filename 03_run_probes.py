"""
Run layer-wise probing to predict model robustness from internal representations.

For each transformer layer, trains a linear probe on the hidden state of the
last input token to predict model_is_robust. Runs three conditions:
  - NONE:          standard probe
  - RANDOMIZATION: labels are randomly shuffled (control — tests label memorisation)
  - PERMUTATION:   hidden-state vectors shuffled (control — breaks repr→label correspondence)

Results are written to results/<probe_name>/ as CSVs via pytorch-lightning CSVLogger.

Requires:
    git clone --branch probe_only https://github.com/Holmes-Benchmark/holmes-evaluation.git
    pip install pytorch-lightning retry

Run 02_extract_internals.py first to populate data/internals/.
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Holmes expects its own core/ directory on sys.path (relative imports inside)
HOLMES_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holmes-evaluation", "core")
if HOLMES_CORE not in sys.path:
    sys.path.insert(0, HOLMES_CORE)

from probing_worker import GeneralProbeWorker          # noqa: E402
from utilities.data_loading import ProbingDataset      # noqa: E402

INTERNALS_DIR = "data/internals"
RESULTS_DIR = "results"
SEED = 42
# Qwen2.5-0.5B hidden size
HIDDEN_DIM = 896

CONTROL_TASKS = ["NONE", "RANDOMIZATION", "PERMUTATION"]


def _to_inputs(ids: list[int]) -> list:
    """Wrap problem indices into the Holmes token-tuple format: [('id_str', sent, start, end)]."""
    return [[(f"problem_{i}", 0, 0, len(f"problem_{i}"))] for i in ids]


def make_split(
    hidden_states: np.ndarray,
    labels: list[int],
    problem_ids: list[int],
    seed: int,
    control_task: str,
) -> tuple[ProbingDataset, ProbingDataset, ProbingDataset]:
    """70 / 15 / 15 stratified split with optional control task transformation."""
    indices = list(range(len(labels)))
    train_idx, rest_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=labels
    )
    dev_idx, test_idx = train_test_split(rest_idx, test_size=0.50, random_state=seed)

    rng = random.Random(seed)

    def _apply_control(idx: list[int], vecs: np.ndarray, lbls: list[int]):
        if control_task == "RANDOMIZATION":
            lbls = list(lbls)
            rng.shuffle(lbls)
        elif control_task == "PERMUTATION":
            # Shuffle hidden-state vectors to break repr→label correspondence
            shuffled = list(range(len(idx)))
            rng.shuffle(shuffled)
            vecs = vecs[shuffled]
        return vecs, lbls

    def _make_ds(idx: list[int]) -> ProbingDataset:
        vecs = hidden_states[idx]
        lbls = [labels[i] for i in idx]
        vecs, lbls = _apply_control(idx, vecs, lbls)
        inputs = _to_inputs([problem_ids[i] for i in idx])
        encoded = list(vecs)  # list of 1-D np.ndarray, shape (hidden_dim,)
        return ProbingDataset(inputs, encoded, lbls)

    train_ds = _make_ds(train_idx)
    dev_ds = _make_ds(dev_idx)
    test_ds = _make_ds(test_idx)

    dev_ds.update_seen(train_ds.unique_inputs)
    test_ds.update_seen(train_ds.unique_inputs)

    return train_ds, dev_ds, test_ds


def run_layer(
    layer_idx: int,
    hidden_states: np.ndarray,
    labels: list[int],
    problem_ids: list[int],
    n_total_layers: int,
):
    for control_task in CONTROL_TASKS:
        print(f"  Layer {layer_idx:03d} | {control_task}")
        train_ds, dev_ds, test_ds = make_split(
            hidden_states, labels, problem_ids, SEED, control_task
        )

        probe_name = f"robustness_L{layer_idx:03d}_{control_task.lower()}"
        worker = GeneralProbeWorker(
            hyperparameter={
                "seed": SEED,
                "encoding": "full",
                "batch_size": 8,
                "num_labels": 2,
                "num_hidden_layers": 0,        # linear probe
                "input_dim": HIDDEN_DIM,
                "output_dim": HIDDEN_DIM,
                "hidden_dim": HIDDEN_DIM,
                "learning_rate": 1e-3,
                "dropout": 0.1,
                "warmup_rate": 0.1,
                "optimizer": torch.optim.Adam,
                "probe_task_type": "SENTENCE",
                "model_name": "qwen-0.5b-instruct",
                "control_task_type": control_task,
                "sample_size": 0,
            },
            train_dataset=train_ds,
            dev_dataset=dev_ds,
            test_dataset=test_ds,
            n_layers=n_total_layers,
            probe_name=probe_name,
            project_prefix="aimo",
            dump_preds=True,
            force=True,
            result_folder=RESULTS_DIR,
            logging="local",
        )
        worker.run_fold()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metadata = pd.read_csv(os.path.join(INTERNALS_DIR, "metadata.csv"))
    labels = metadata["model_is_robust"].astype(int).tolist()
    problem_ids = list(range(len(labels)))

    layer_files = sorted(
        f for f in os.listdir(INTERNALS_DIR)
        if f.startswith("layer_") and f.endswith(".npy")
    )
    n_layers = len(layer_files)
    print(f"Probing {n_layers} layers  |  n_problems={len(labels)}")

    for layer_file in layer_files:
        layer_idx = int(layer_file.replace("layer_", "").replace(".npy", ""))
        hidden_states = np.load(os.path.join(INTERNALS_DIR, layer_file))
        run_layer(layer_idx, hidden_states, labels, problem_ids, n_layers)

    print(f"\nDone. Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
