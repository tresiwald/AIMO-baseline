"""
Run regression probes on eval-adoption internals.

For each `permutation_type`, this script:
1. creates an outer 80:20 train/test split,
2. carves a validation split out of the training pool for early stopping,
3. trains one linear regression probe per transformer layer,
4. writes results via the Holmes CSV logger.

The regression target is `absolute_accuracy_decay`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

HOLMES_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holmes-evaluation", "core")
if HOLMES_CORE not in sys.path:
    sys.path.insert(0, HOLMES_CORE)

from probing_worker import GeneralProbeWorker  # noqa: E402
from utilities.data_loading import ProbingDataset  # noqa: E402

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--internals-dir",
        default="data/eval_adoption_internals",
        help="Directory containing metadata.csv and layer_XXX.npy files",
    )
    parser.add_argument(
        "--results-dir",
        default="results/eval_adoption_absolute_accuracy_decay",
        help="Directory where probe outputs are written",
    )
    parser.add_argument(
        "--model-name",
        default="eval-adoption-probe",
        help="Name recorded in Holmes run metadata",
    )
    return parser.parse_args()


def to_inputs(row_ids: list[int], permutation_type: str) -> list[list[tuple[str, int, int, int]]]:
    return [[(f"{permutation_type}__row_{row_id}", 0, 0, len(f"{permutation_type}__row_{row_id}"))] for row_id in row_ids]


def make_split(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    row_ids: list[int],
    permutation_type: str,
) -> tuple[ProbingDataset, ProbingDataset, ProbingDataset]:
    indices = np.arange(len(labels))

    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=SEED)

    if len(train_idx) < 5:
        raise ValueError(
            f"Not enough rows in training pool for permutation_type={permutation_type!r}: {len(train_idx)}"
        )

    dev_fraction_of_train = max(1, round(len(train_idx) * 0.20)) / len(train_idx)
    train_idx, dev_idx = train_test_split(
        train_idx,
        test_size=dev_fraction_of_train,
        random_state=SEED,
    )

    def make_dataset(idx: np.ndarray) -> ProbingDataset:
        idx = np.asarray(idx)
        vecs = hidden_states[idx]
        lbls = labels[idx].astype(float).tolist()
        inputs = to_inputs([row_ids[i] for i in idx.tolist()], permutation_type)
        encoded = list(vecs)
        return ProbingDataset(inputs, encoded, lbls)

    train_ds = make_dataset(train_idx)
    dev_ds = make_dataset(dev_idx)
    test_ds = make_dataset(test_idx)

    dev_ds.update_seen(train_ds.unique_inputs)
    test_ds.update_seen(train_ds.unique_inputs)

    return train_ds, dev_ds, test_ds


def run_layer(
    layer_idx: int,
    hidden_states: np.ndarray,
    labels: np.ndarray,
    row_ids: list[int],
    permutation_type: str,
    n_total_layers: int,
    results_dir: str,
    model_name: str,
) -> None:
    train_ds, dev_ds, test_ds = make_split(hidden_states, labels, row_ids, permutation_type)
    hidden_dim = int(hidden_states.shape[1])
    probe_name = f"absolute_accuracy_decay__{permutation_type}__L{layer_idx:03d}"

    print(f"  {permutation_type:>9} | layer {layer_idx:03d} | n={len(labels)}")
    worker = GeneralProbeWorker(
        hyperparameter={
            "seed": SEED,
            "encoding": "full",
            "batch_size": 8,
            "num_labels": 1,
            "num_hidden_layers": 0,
            "input_dim": hidden_dim,
            "output_dim": hidden_dim,
            "hidden_dim": hidden_dim,
            "learning_rate": 1e-3,
            "dropout": 0.1,
            "warmup_rate": 0.1,
            "optimizer": torch.optim.Adam,
            "probe_task_type": "SENTENCE",
            "model_name": model_name,
            "control_task_type": permutation_type.upper(),
            "sample_size": 0,
        },
        train_dataset=train_ds,
        dev_dataset=dev_ds,
        test_dataset=test_ds,
        n_layers=n_total_layers,
        probe_name=probe_name,
        project_prefix="eval-adoption",
        dump_preds=True,
        force=True,
        result_folder=results_dir,
        logging="local",
    )
    worker.run_fold()


def main() -> None:
    args = parse_args()
    internals_dir = Path(args.internals_dir)
    os.makedirs(args.results_dir, exist_ok=True)

    metadata = pd.read_csv(internals_dir / "metadata.csv").sort_values("row_id")
    metadata["absolute_accuracy_decay"] = metadata["absolute_accuracy_decay"].astype(float)

    layer_files = sorted(
        f.name
        for f in internals_dir.iterdir()
        if f.name.startswith("layer_") and f.suffix == ".npy"
    )
    n_layers = len(layer_files)
    print(f"Probing {n_layers} layers across {metadata['permutation_type'].nunique()} permutation types")

    for permutation_type, subset in metadata.groupby("permutation_type", sort=True):
        subset = subset.reset_index(drop=True)
        row_ids = subset["row_id"].astype(int).tolist()
        labels = subset["absolute_accuracy_decay"].to_numpy(dtype=np.float32)
        subset_indices = subset["row_id"].to_numpy(dtype=int)

        print(f"\nPermutation type: {permutation_type} | rows={len(subset)}")
        for layer_file in layer_files:
            layer_idx = int(layer_file.replace("layer_", "").replace(".npy", ""))
            layer_states = np.load(internals_dir / layer_file)
            subset_states = layer_states[subset_indices]
            run_layer(
                layer_idx=layer_idx,
                hidden_states=subset_states,
                labels=labels,
                row_ids=row_ids,
                permutation_type=permutation_type,
                n_total_layers=n_layers,
                results_dir=args.results_dir,
                model_name=args.model_name,
            )

    print(f"\nDone. Results saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
