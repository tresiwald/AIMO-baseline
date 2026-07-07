"""
Run eval-adoption probes with automatic task-type inference from the target column.

For each `permutation_type`, this script:
1. runs a repeated K-fold evaluation across configurable seeds and folds,
2. carves a validation split out of each fold's training pool for early stopping,
3. trains one linear probe per transformer layer and control setup,
4. writes results via the Holmes CSV logger.

The default target is `absolute_accuracy_decay`, but the script can also run
classification probes, e.g. against `model_is_robust`. Task type is inferred
from the target column values.

`permutation_type` is an eval-adoption perturbation label, not a Holmes control
task. We therefore keep it in the probe name and dataset row identifiers, while
running Holmes control-task variants (`NONE`, `RANDOMIZATION`) explicitly as a
separate sweep dimension.

Dimensionality reduction is optional. When enabled, each split is projected into
a lower-dimensional PCA space fit on the training vectors only. This reduces
hidden states while preserving as much geometry/variance as possible.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
import pickle
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge

REPO_ROOT = Path(__file__).resolve().parents[1]
HOLMES_CORE = REPO_ROOT / "holmes-evaluation" / "core"
if str(HOLMES_CORE) not in sys.path:
    sys.path.insert(0, str(HOLMES_CORE))

from probing_worker import GeneralProbeWorker  # noqa: E402
from utilities.data_loading import ProbingDataset  # noqa: E402

CONTROL_TASKS = ["NONE", "RANDOMIZATION"]
DEFAULT_REDUCED_DIM = 0
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_NUM_FOLDS = 4
DEFAULT_DEV_FRACTION = 0.20
DEFAULT_TARGET_COL = "absolute_accuracy_decay"
DEFAULT_NUM_WORKERS = os.cpu_count() or 1
DEFAULT_METHOD = "probe"
DEFAULT_KERNEL = "rbf"
DEFAULT_KERNEL_ALPHAS = [0.01, 0.1, 1.0, 10.0]
DEFAULT_KERNEL_CS = [0.1, 1.0, 10.0, 100.0]
DEFAULT_KERNEL_GAMMAS = ["scale", 0.001, 0.01, 0.1, 1.0]
DEFAULT_BINARY_EVAL_COL = "model_is_robust"


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
    parser.add_argument(
        "--reduced-dim",
        type=int,
        default=DEFAULT_REDUCED_DIM,
        help="Project hidden states to this many PCA dimensions before probing. Use 0 to disable.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated random seeds used for repeated K-fold evaluation.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=DEFAULT_NUM_FOLDS,
        help="Number of cross-validation folds per seed.",
    )
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=DEFAULT_DEV_FRACTION,
        help="Fraction of each fold's training pool reserved for validation.",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_COL,
        help="Metadata column used as the probe target, e.g. absolute_accuracy_decay or model_is_robust. Task type is inferred from its values.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of parallel worker processes. Each worker handles one permutation/layer/seed/fold/control task.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=["probe", "kernel"],
        help="Evaluation method: linear probe or kernel baseline.",
    )
    parser.add_argument(
        "--kernel",
        default=DEFAULT_KERNEL,
        choices=["rbf", "linear"],
        help="Kernel type used when --method kernel.",
    )
    parser.add_argument(
        "--kernel-alphas",
        default=",".join(str(value) for value in DEFAULT_KERNEL_ALPHAS),
        help="Comma-separated regularization strengths for kernel ridge regression.",
    )
    parser.add_argument(
        "--kernel-c-values",
        default=",".join(str(value) for value in DEFAULT_KERNEL_CS),
        help="Comma-separated C values for kernel SVC classification.",
    )
    parser.add_argument(
        "--kernel-gammas",
        default=",".join(str(value) for value in DEFAULT_KERNEL_GAMMAS),
        help="Comma-separated gamma values for RBF kernels. Use 'scale' to derive gamma from train variance.",
    )
    parser.add_argument(
        "--binary-eval-col",
        default=DEFAULT_BINARY_EVAL_COL,
        help="Optional binary label column used to compute threshold-based accuracy from regression predictions. Set to empty to disable.",
    )
    parser.add_argument(
        "--threshold",
        type=parse_threshold_arg,
        default=None,
        help="Threshold for converting regression predictions to binary labels. Use 'auto' to infer on the train split.",
    )
    parser.add_argument(
        "--dump-probe-artifacts",
        action="store_true",
        help="Write per-run probe_artifact.npz files and assemble layer/seed probe pickle artifacts.",
    )
    parser.add_argument(
        "--artifact-model-id",
        default=None,
        help="HF model id stored in exported probe artifacts. Defaults to --model-name.",
    )
    parser.add_argument(
        "--artifact-system-prompt-path",
        default=str(REPO_ROOT / "prompts" / "solve.txt"),
        help="System prompt text file stored in exported probe artifacts.",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Optional comma-separated layer indices to run, e.g. '12' or '8,12,16'. Defaults to all layers.",
    )
    parser.add_argument(
        "--permutation-types",
        default=None,
        help="Optional comma-separated permutation_type values to run. Defaults to all values in metadata.",
    )
    parser.add_argument(
        "--control-tasks",
        default=None,
        help="Optional comma-separated Holmes control tasks to run from NONE,RANDOMIZATION. Defaults to both.",
    )
    return parser.parse_args()


def parse_seeds(seed_arg: str) -> list[int]:
    seeds = [int(chunk.strip()) for chunk in seed_arg.split(",") if chunk.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided")
    return seeds


def parse_float_grid(raw: str) -> list[float]:
    values = [float(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def parse_gamma_grid(raw: str) -> list[str | float]:
    values: list[str | float] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        if item.lower() == "scale":
            values.append("scale")
        else:
            values.append(float(item))
    if not values:
        raise ValueError("Expected at least one gamma value.")
    return values


def parse_optional_str_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    return values or None


def parse_optional_layers(raw: str | None) -> list[int] | None:
    values = parse_optional_str_list(raw)
    if values is None:
        return None
    return [int(value) for value in values]


def parse_threshold_arg(value: str) -> float | None:
    if value.strip().lower() == "auto":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold must be a float or 'auto'") from exc


def infer_task_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        raise ValueError("Target column contains no non-null values.")

    if pd.api.types.is_bool_dtype(non_null):
        return "classification"

    if pd.api.types.is_integer_dtype(non_null):
        unique_vals = set(non_null.astype(int).unique().tolist())
        if unique_vals.issubset({0, 1}):
            return "classification"

    if pd.api.types.is_numeric_dtype(non_null):
        unique_vals = set(pd.Series(non_null).astype(float).unique().tolist())
        if unique_vals.issubset({0.0, 1.0}):
            return "classification"
        return "regression"

    return "classification"


def to_inputs(
    row_ids: list[int],
    permutation_type: str,
    prefix: str = "id",
) -> list[list[tuple[str, int, int, int]]]:
    return [
        [(f"{prefix}__{permutation_type}__row_{row_id}", 0, 0, len(f"{prefix}__{permutation_type}__row_{row_id}"))]
        for row_id in row_ids
    ]


def load_internals(internals_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    metadata = pd.read_csv(internals_dir / "metadata.csv").sort_values("row_id").reset_index(drop=True)
    layer_files = sorted(
        f.name
        for f in internals_dir.iterdir()
        if f.name.startswith("layer_") and f.suffix == ".npy"
    )
    return metadata, layer_files


def apply_control(
    vecs: np.ndarray,
    lbls: np.ndarray,
    control_task: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    vecs = np.asarray(vecs)
    lbls = np.asarray(lbls)
    if control_task == "RANDOMIZATION":
        shuffled_labels = lbls.tolist()
        rng.shuffle(shuffled_labels)
        return vecs, np.asarray(shuffled_labels, dtype=lbls.dtype)
    return vecs, lbls


def maybe_reduce(
    train_vecs: np.ndarray,
    dev_vecs: np.ndarray,
    test_vecs: np.ndarray,
    reduced_dim: int,
    seed: int,
    permutation_type: str,
 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if reduced_dim <= 0:
        return train_vecs, dev_vecs, test_vecs

    n_components = min(reduced_dim, train_vecs.shape[0], train_vecs.shape[1])
    if n_components < 1:
        raise ValueError(
            f"Cannot fit PCA for permutation_type={permutation_type!r}: "
            f"train shape={train_vecs.shape}"
        )

    pca = PCA(n_components=n_components, svd_solver="full", random_state=seed)
    train_reduced = pca.fit_transform(train_vecs).astype(np.float32)
    dev_reduced = pca.transform(dev_vecs).astype(np.float32)
    test_reduced = pca.transform(test_vecs).astype(np.float32)
    return train_reduced, dev_reduced, test_reduced


def build_dataset(
    row_ids: list[int],
    permutation_type: str,
    vecs: np.ndarray,
    lbls: np.ndarray,
    prefix: str,
) -> ProbingDataset:
    inputs = to_inputs(row_ids, permutation_type, prefix=prefix)
    encoded = list(np.asarray(vecs, dtype=np.float32))
    return ProbingDataset(inputs, encoded, np.asarray(lbls).tolist())


def make_fold_datasets(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    row_ids: list[int],
    permutation_type: str,
    control_task: str,
    reduced_dim: int,
    task_type: str,
    train_pool_idx: np.ndarray,
    test_idx: np.ndarray,
    dev_fraction: float,
    seed: int,
) -> tuple[ProbingDataset, ProbingDataset, ProbingDataset]:
    train_pool_idx = np.asarray(train_pool_idx)
    test_idx = np.asarray(test_idx)

    if len(train_pool_idx) < 2:
        raise ValueError(
            f"Not enough rows in training pool for permutation_type={permutation_type!r}: {len(train_pool_idx)}"
        )

    dev_count = max(1, round(len(train_pool_idx) * dev_fraction))
    dev_count = min(dev_count, len(train_pool_idx) - 1)
    dev_fraction_of_train = dev_count / len(train_pool_idx)
    train_pool_labels = labels[train_pool_idx]
    dev_stratify = None
    if task_type == "classification":
        _, counts = np.unique(train_pool_labels, return_counts=True)
        if np.all(counts >= 2):
            dev_stratify = train_pool_labels

    train_idx, dev_idx = train_test_split(
        train_pool_idx,
        test_size=dev_fraction_of_train,
        random_state=seed,
        stratify=dev_stratify,
    )

    train_vecs, train_lbls = apply_control(hidden_states[train_idx], labels[train_idx], control_task, seed)
    dev_vecs, dev_lbls = apply_control(hidden_states[dev_idx], labels[dev_idx], control_task, seed)
    test_vecs, test_lbls = apply_control(hidden_states[test_idx], labels[test_idx], control_task, seed)

    train_vecs, dev_vecs, test_vecs = maybe_reduce(
        train_vecs,
        dev_vecs,
        test_vecs,
        reduced_dim,
        seed,
        permutation_type,
    )

    train_ds = build_dataset([row_ids[i] for i in train_idx.tolist()], permutation_type, train_vecs, train_lbls, prefix="train")
    dev_ds = build_dataset([row_ids[i] for i in dev_idx.tolist()], permutation_type, dev_vecs, dev_lbls, prefix="dev")
    test_ds = build_dataset([row_ids[i] for i in test_idx.tolist()], permutation_type, test_vecs, test_lbls, prefix="test")

    dev_ds.update_seen(train_ds.unique_inputs)
    test_ds.update_seen(train_ds.unique_inputs)

    return train_ds, dev_ds, test_ds


def make_fold_arrays(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    row_ids: list[int],
    permutation_type: str,
    control_task: str,
    reduced_dim: int,
    task_type: str,
    train_pool_idx: np.ndarray,
    test_idx: np.ndarray,
    dev_fraction: float,
    seed: int,
) -> dict[str, np.ndarray | list[int] | None]:
    train_ds, dev_ds, test_ds = make_fold_datasets(
        hidden_states=hidden_states,
        labels=labels,
        row_ids=row_ids,
        permutation_type=permutation_type,
        control_task=control_task,
        reduced_dim=reduced_dim,
        task_type=task_type,
        train_pool_idx=train_pool_idx,
        test_idx=test_idx,
        dev_fraction=dev_fraction,
        seed=seed,
    )

    result: dict[str, np.ndarray | list[int] | None] = {
        "train_vecs": np.asarray(train_ds.inputs_encoded, dtype=np.float32),
        "train_lbls": np.asarray(train_ds.labels),
        "train_row_ids": [int(item[0][0].split("_")[-1]) for item in train_ds.inputs],
        "dev_vecs": np.asarray(dev_ds.inputs_encoded, dtype=np.float32),
        "dev_lbls": np.asarray(dev_ds.labels),
        "dev_row_ids": [int(item[0][0].split("_")[-1]) for item in dev_ds.inputs],
        "test_vecs": np.asarray(test_ds.inputs_encoded, dtype=np.float32),
        "test_lbls": np.asarray(test_ds.labels),
        "test_row_ids": [int(item[0][0].split("_")[-1]) for item in test_ds.inputs],
    }
    return result


def run_worker(worker: GeneralProbeWorker) -> tuple[str, pd.DataFrame, torch.nn.Module]:
    logger = worker.get_logger()

    if worker.logging == "local":
        log_dir = logger.log_dir
        result_log_dir = log_dir
        if os.path.exists(f"{logger.root_dir}/done") and not worker.force:
            print(f"Already done at {logger.root_dir}/done")
            return f"{logger.root_dir}/done", pd.DataFrame(), None
    else:
        raise NotImplementedError("This script currently supports only local logging")

    os.makedirs(log_dir, exist_ok=True)
    worker.hyperparameter["dump_id"] = log_dir
    worker.hyperparameter["cache_folder"] = worker.cache_folder
    worker.hyperparameter["result_folder"] = worker.result_folder

    prediction_frame, probing_model = worker.train_run(log_dir=result_log_dir, logger=logger)
    if worker.dump_preds:
        prediction_frame.to_csv(result_log_dir + "/preds.csv")
    worker.mark_run_as_done(logger=logger)
    return f"{logger.root_dir}/done", prediction_frame, probing_model


def read_optional_text(path: str | None) -> str:
    if not path:
        return ""
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Artifact system prompt file not found: {prompt_path}")
    return prompt_path.read_text().strip()


def final_linear_layer(probing_model: torch.nn.Module) -> torch.nn.Linear:
    for layer in reversed(list(probing_model.layers)):
        if isinstance(layer, torch.nn.Linear):
            return layer
    raise ValueError("Could not find a final torch.nn.Linear layer in the probing model.")


def read_metrics_row(done_dir: Path) -> dict[str, float | int | str]:
    metrics_path = done_dir / "metrics.csv"
    if not metrics_path.exists():
        return {}
    metrics_df = pd.read_csv(metrics_path)
    if metrics_df.empty:
        return {}
    return metrics_df.iloc[-1].to_dict()


def safe_artifact_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "artifact"


def export_probe_artifact(
    probing_model: torch.nn.Module,
    done_dir: Path,
    layer_idx: int,
    seed: int,
    fold_idx: int,
    permutation_type: str,
    control_task: str,
    model_id: str,
    system_prompt: str,
    target_col: str,
    task_type: str,
    threshold: float | None,
) -> Path:
    linear = final_linear_layer(probing_model)
    weight = linear.weight.detach().cpu().float().numpy()
    bias = linear.bias.detach().cpu().float().numpy() if linear.bias is not None else np.zeros(weight.shape[0], dtype=np.float32)
    num_labels = int(probing_model.hyperparameter["num_labels"])

    if num_labels == 2:
        export_weights = (weight[1] - weight[0]).astype(np.float32)
        export_bias = float(bias[1] - bias[0])
        export_threshold = 0.0
    elif num_labels == 1:
        metrics_row = read_metrics_row(done_dir)
        resolved_threshold = threshold
        if resolved_threshold is None and "threshold" in metrics_row and pd.notna(metrics_row["threshold"]):
            resolved_threshold = float(metrics_row["threshold"])
        if resolved_threshold is None:
            resolved_threshold = 0.0

        # Regression threshold metrics define robust as prediction < threshold.
        # The submission wrapper uses score >= threshold, so export the negated score.
        export_weights = (-weight.reshape(-1)).astype(np.float32)
        export_bias = float(-bias.reshape(-1)[0])
        export_threshold = float(-resolved_threshold)
    else:
        raise ValueError("Portable probe export supports binary classification or scalar regression probes only.")

    artifact_path = done_dir / "probe_artifact.npz"
    np.savez(
        artifact_path,
        model_id=np.asarray(model_id),
        layer_index=np.asarray(layer_idx, dtype=np.int64),
        weights=export_weights,
        bias=np.asarray(export_bias, dtype=np.float32),
        threshold=np.asarray(export_threshold, dtype=np.float32),
        seeds=np.asarray([seed], dtype=np.int64),
        fold_index=np.asarray(fold_idx, dtype=np.int64),
        permutation_type=np.asarray(permutation_type),
        control_task=np.asarray(control_task),
        ensemble_size=np.asarray(1, dtype=np.int64),
        aggregation=np.asarray("mean_margin"),
        system_prompt=np.asarray(system_prompt),
        target_col=np.asarray(target_col),
        task_type=np.asarray(task_type),
    )
    return artifact_path


def npz_scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    value = data[key]
    if hasattr(value, "item"):
        return str(value.item())
    return str(value)


def pickle_safe_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {pickle_safe_value(key): pickle_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [pickle_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(pickle_safe_value(item) for item in value)
    return value


def select_metric_from_row(metrics_row: dict[str, float | int | str], task_type: str) -> tuple[str, float] | None:
    if task_type == "classification":
        candidates = [
            ("val balanced_acc", 1.0),
            ("val f1", 1.0),
            ("val acc", 1.0),
            ("full test balanced_acc", 1.0),
            ("full test f1", 1.0),
            ("full test acc", 1.0),
        ]
    else:
        candidates = [
            ("val threshold_balanced_accuracy", 1.0),
            ("val threshold_accuracy", 1.0),
            ("val pearson", 1.0),
            ("val error", -1.0),
            ("val loss", -1.0),
        ]

    for column, multiplier in candidates:
        if column not in metrics_row:
            continue
        value = pd.to_numeric(pd.Series([metrics_row[column]]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        return column, float(value) * multiplier
    return None


def assemble_probe_pickle_artifacts(
    results_dir: str,
    target_col: str,
    permutation_types: list[str],
    control_tasks: list[str],
    layer_indices: list[int],
    model_name: str,
    seeds: list[int],
    num_folds: int,
) -> None:
    output_dir = Path(results_dir) / "probe_artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped_folds = 0
    skipped_layers = 0
    groups = []
    all_probe_scores: list[dict[str, float | int | str]] = []
    all_layer_metric_values: dict[int, list[float]] = {}
    model_id = None
    system_prompt = None
    task_type = None

    for permutation_type in permutation_types:
        for control_task in control_tasks:
            for fold_idx in range(num_folds):
                probes_by_layer: dict[int, dict[str, np.ndarray | list[int]]] = {}
                layer_scores: dict[int, float] = {}
                layer_metric_names: dict[int, str] = {}
                probe_scores: list[dict[str, float | int | str]] = []
                for layer_idx in layer_indices:
                    artifacts: list[tuple[int, Path]] = []
                    for seed in seeds:
                        artifact_path = (
                            expected_done_dir(
                                results_dir=results_dir,
                                target_col=target_col,
                                permutation_type=permutation_type,
                                control_task=control_task,
                                layer_idx=layer_idx,
                                model_name=model_name,
                                fold_idx=fold_idx,
                                seed=seed,
                            )
                            / "probe_artifact.npz"
                        )
                        if artifact_path.exists():
                            artifacts.append((seed, artifact_path))

                    if len(artifacts) != len(seeds):
                        skipped_layers += 1
                        continue

                    weights = []
                    biases = []
                    thresholds = []
                    metric_values = []
                    metric_name = None
                    for seed, artifact_path in artifacts:
                        with np.load(artifact_path, allow_pickle=False) as data:
                            artifact_model_id = npz_scalar(data, "model_id")
                            artifact_system_prompt = npz_scalar(data, "system_prompt")
                            artifact_task_type = npz_scalar(data, "task_type")
                            if model_id is None:
                                model_id = artifact_model_id
                                system_prompt = artifact_system_prompt
                                task_type = artifact_task_type
                            elif (
                                model_id != artifact_model_id
                                or system_prompt != artifact_system_prompt
                                or task_type != artifact_task_type
                            ):
                                raise ValueError(f"Inconsistent metadata while assembling {artifact_path}")
                            weights.append(np.asarray(data["weights"], dtype=np.float32).reshape(-1))
                            biases.append(float(np.asarray(data["bias"]).reshape(-1)[0]))
                            thresholds.append(float(np.asarray(data["threshold"]).reshape(-1)[0]))

                        metrics = read_metrics_row(artifact_path.parent)
                        selected_metric = select_metric_from_row(metrics, artifact_task_type)
                        if selected_metric is not None:
                            metric_name, metric_value = selected_metric
                            metric_values.append(metric_value)
                            probe_scores.append(
                                {
                                    "layer_index": int(layer_idx),
                                    "seed": int(seed),
                                    "metric": metric_name,
                                    "score": float(metric_value),
                                }
                            )
                            all_probe_scores.append(
                                {
                                    "permutation_type": permutation_type,
                                    "control_task": control_task,
                                    "fold_index": int(fold_idx),
                                    "layer_index": int(layer_idx),
                                    "seed": int(seed),
                                    "metric": metric_name,
                                    "score": float(metric_value),
                                }
                            )
                            all_layer_metric_values.setdefault(layer_idx, []).append(float(metric_value))

                    probes_by_layer[layer_idx] = {
                        "weights": np.stack(weights).astype(np.float32),
                        "bias": np.asarray(biases, dtype=np.float32),
                        "threshold": np.asarray(thresholds, dtype=np.float32),
                        "seeds": list(seeds),
                    }
                    if metric_values:
                        layer_scores[layer_idx] = float(np.mean(metric_values))
                        if metric_name is not None:
                            layer_metric_names[layer_idx] = metric_name

                if not probes_by_layer:
                    skipped_folds += 1
                    continue

                if probe_scores:
                    best_probe = max(probe_scores, key=lambda item: float(item["score"]))
                    best_layer_idx = int(best_probe["layer_index"])
                    selection_metric = str(best_probe["metric"])
                else:
                    best_probe = None
                    best_layer_idx = sorted(probes_by_layer)[0]
                    selection_metric = "unavailable"

                groups.append(
                    {
                        "permutation_type": permutation_type,
                        "control_task": control_task,
                        "fold_index": fold_idx,
                        "seeds": list(seeds),
                        "layer_indices": sorted(probes_by_layer),
                        "best_layer_index": int(best_layer_idx),
                        "best_probe": best_probe,
                        "selection_metric": selection_metric,
                        "layer_scores": layer_scores,
                        "probe_scores": probe_scores,
                        "aggregation": "mean_margin",
                        "probes": probes_by_layer,
                    }
                )

    if not groups:
        print(
            f"Assembled 0 layer/seed probe pickle artifact(s) in {output_dir}/"
            + (
                f" ({skipped_layers} incomplete layer group(s), {skipped_folds} empty fold group(s) skipped)"
                if skipped_layers or skipped_folds
                else ""
            )
        )
        return

    layer_scores = {
        int(layer_idx): float(np.mean(metric_values))
        for layer_idx, metric_values in sorted(all_layer_metric_values.items())
        if metric_values
    }
    if layer_scores:
        best_layer_idx = max(layer_scores, key=layer_scores.get)
        best_probe = max(all_probe_scores, key=lambda item: float(item["score"])) if all_probe_scores else None
        selection_metric = str(best_probe["metric"]) if best_probe is not None else "mean_validation_metric"
    else:
        best_layer_idx = sorted({layer for group in groups for layer in group["layer_indices"]})[0]
        best_probe = None
        selection_metric = "unavailable"

    artifact = {
        "schema_version": 2,
        "artifact_type": "all_folds_layers_seed_probe_ensemble",
        "model_id": model_id,
        "system_prompt": system_prompt,
        "target_col": target_col,
        "task_type": task_type,
        "seeds": list(seeds),
        "fold_indices": sorted({int(group["fold_index"]) for group in groups}),
        "layer_indices": sorted({int(layer) for group in groups for layer in group["layer_indices"]}),
        "permutation_types": sorted({str(group["permutation_type"]) for group in groups}),
        "control_tasks": sorted({str(group["control_task"]) for group in groups}),
        "best_layer_index": int(best_layer_idx),
        "best_probe": best_probe,
        "selection_metric": selection_metric,
        "layer_scores": layer_scores,
        "probe_scores": all_probe_scores,
        "recommended_strategy": {
            "name": "best_layer_mean_margin",
            "layer_index": int(best_layer_idx),
            "description": (
                "At inference, encode best_layer_index and average score-threshold "
                "margins across every stored seed/fold probe for that layer."
            ),
        },
        "alternative_strategy": {
            "name": "all_layers_mean_margin",
            "description": (
                "Average margins across every stored seed/fold/layer probe. This is "
                "available from the same groups data but is not the default."
            ),
        },
        "aggregation": "mean_margin",
        "groups": groups,
    }
    output_path = output_dir / "probe_artifact.pkl"
    with output_path.open("wb") as handle:
        pickle.dump(pickle_safe_value(artifact), handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Assembled 1 layer/seed/fold probe pickle artifact: {output_path}"
        + (
            f" ({skipped_layers} incomplete layer group(s), {skipped_folds} empty fold group(s) skipped)"
            if skipped_layers or skipped_folds
            else ""
        )
    )


def evaluate_regression_dataset(
    probing_model: torch.nn.Module,
    dataset: ProbingDataset,
) -> pd.DataFrame:
    dataloader = probing_model.get_test_dataloader(dataset, 300, shuffle=False)
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_losses: list[np.ndarray] = []
    all_seen: list[np.ndarray] = []

    probing_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, y, seen_indices = batch
            else:
                x, y = batch
                seen_indices = torch.zeros(len(y), dtype=torch.bool)
            x = x.to(probing_model.device)
            y = y.to(probing_model.device)
            pred = probing_model(x).squeeze(1)
            losses = probing_model.loss(pred, y).detach().cpu().numpy()
            all_preds.append(pred.detach().cpu().double().numpy())
            all_labels.append(y.detach().cpu().numpy())
            all_losses.append(losses)
            all_seen.append(seen_indices.detach().cpu().numpy())

    preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.float64)
    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float64)
    losses = np.concatenate(all_losses) if all_losses else np.array([], dtype=np.float64)
    seen = np.concatenate(all_seen) if all_seen else np.array([], dtype=bool)

    instances = [instance_input for instance_input in dataset.inputs]
    prediction_frame = pd.DataFrame(
        {
            "instance": instances,
            "pred": preds,
            "label": labels,
            "loss": losses,
            "seen": np.where(seen, "seen", "ood"),
        }
    )

    return prediction_frame


def evaluate_classification_dataset(
    probing_model: torch.nn.Module,
    dataset: ProbingDataset,
) -> pd.DataFrame:
    dataloader = probing_model.get_test_dataloader(dataset, 300, shuffle=False)
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_losses: list[np.ndarray] = []
    all_seen: list[np.ndarray] = []

    probing_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, y, seen_indices = batch
            else:
                x, y = batch
                seen_indices = torch.zeros(len(y), dtype=torch.bool)
            x = x.to(probing_model.device)
            y = y.to(probing_model.device)
            logits = probing_model(x)
            pred = logits.argmax(dim=1)
            losses = (pred != y).detach().cpu().numpy().astype(np.float64)
            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
            all_losses.append(losses)
            all_seen.append(seen_indices.detach().cpu().numpy())

    preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    losses = np.concatenate(all_losses) if all_losses else np.array([], dtype=np.float64)
    seen = np.concatenate(all_seen) if all_seen else np.array([], dtype=bool)

    instances = [instance_input for instance_input in dataset.inputs]
    prediction_frame = pd.DataFrame(
        {
            "instance": instances,
            "pred": preds,
            "label": labels,
            "loss": losses,
            "seen": np.where(seen, "seen", "ood"),
        }
    )
    return prediction_frame


def apply_classification_metrics(
    metrics_row: dict[str, float | int | str],
    val_preds: np.ndarray,
    val_labels: np.ndarray,
    test_preds: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, float | int | str]:
    val_balanced_acc = float(balanced_accuracy_score(val_labels, val_preds))
    test_balanced_acc = float(balanced_accuracy_score(test_labels, test_preds))

    metrics_row["val balanced_acc"] = val_balanced_acc
    metrics_row["full test balanced_acc"] = test_balanced_acc
    metrics_row["unseen test balanced_acc"] = test_balanced_acc
    if "middle test balanced_acc" not in metrics_row:
        metrics_row["middle test balanced_acc"] = np.nan
    if "upper test balanced_acc" not in metrics_row:
        metrics_row["upper test balanced_acc"] = np.nan
    return metrics_row


def expected_done_dirs(
    results_dir: str,
    target_col: str,
    permutation_type: str,
    control_task: str,
    layer_idx: int,
    model_name: str,
    fold_idx: int,
    seed: int,
) -> list[Path]:
    probe_name = f"{target_col}__{permutation_type}__{control_task.lower()}__L{layer_idx:03d}"
    run_root = (
        Path(results_dir)
        / f"eval-adoption-{probe_name}"
        / model_name.replace("/", "__")
        / "full"
        / control_task
        / str(fold_idx)
        / str(seed)
        / "0"
    )
    return [
        run_root / str(fold_idx) / "done",
        run_root / "done",
    ]


def expected_done_dir(
    results_dir: str,
    target_col: str,
    permutation_type: str,
    control_task: str,
    layer_idx: int,
    model_name: str,
    fold_idx: int,
    seed: int,
) -> Path:
    for done_dir in expected_done_dirs(
        results_dir=results_dir,
        target_col=target_col,
        permutation_type=permutation_type,
        control_task=control_task,
        layer_idx=layer_idx,
        model_name=model_name,
        fold_idx=fold_idx,
        seed=seed,
    ):
        if (done_dir / "metrics.csv").exists() or (done_dir / "probe_artifact.npz").exists():
            return done_dir
    return expected_done_dirs(
        results_dir=results_dir,
        target_col=target_col,
        permutation_type=permutation_type,
        control_task=control_task,
        layer_idx=layer_idx,
        model_name=model_name,
        fold_idx=fold_idx,
        seed=seed,
    )[0]


def task_is_done(
    results_dir: str,
    target_col: str,
    permutation_type: str,
    control_task: str,
    layer_idx: int,
    model_name: str,
    fold_idx: int,
    seed: int,
) -> bool:
    return any(
        (done_dir / "metrics.csv").exists()
        for done_dir in expected_done_dirs(
            results_dir=results_dir,
            target_col=target_col,
            permutation_type=permutation_type,
            control_task=control_task,
            layer_idx=layer_idx,
            model_name=model_name,
            fold_idx=fold_idx,
            seed=seed,
        )
    )


def standardize_splits(
    train_vecs: np.ndarray,
    dev_vecs: np.ndarray,
    test_vecs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_vecs = np.asarray(train_vecs, dtype=np.float64)
    dev_vecs = np.asarray(dev_vecs, dtype=np.float64)
    test_vecs = np.asarray(test_vecs, dtype=np.float64)
    mean = train_vecs.mean(axis=0, keepdims=True)
    std = train_vecs.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    train_scaled = (train_vecs - mean) / std
    dev_scaled = (dev_vecs - mean) / std
    test_scaled = (test_vecs - mean) / std
    return train_scaled, dev_scaled, test_scaled


def resolve_gamma(train_vecs: np.ndarray, gamma: str | float) -> float:
    if gamma != "scale":
        return float(gamma)
    variance = float(np.var(train_vecs))
    n_features = max(1, int(train_vecs.shape[1]))
    if variance <= 0:
        return 1.0 / n_features
    return 1.0 / (n_features * variance)


def compute_kernel_matrix(
    train_vecs: np.ndarray,
    other_vecs: np.ndarray,
    kernel_kind: str,
    gamma: str | float,
) -> np.ndarray:
    if kernel_kind == "linear":
        return linear_kernel(other_vecs, train_vecs)
    return rbf_kernel(other_vecs, train_vecs, gamma=resolve_gamma(train_vecs, gamma))


def safe_pearson(preds: np.ndarray, labels: np.ndarray) -> float:
    if len(preds) < 2 or np.isclose(np.std(preds), 0.0) or np.isclose(np.std(labels), 0.0):
        return float("nan")
    return float(np.corrcoef(preds, labels)[0, 1])


def threshold_predictions(y_pred: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(y_pred, dtype=np.float64) < float(threshold)


def compute_threshold_metrics(
    y_true_binary: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true_binary = np.asarray(y_true_binary).astype(bool)
    y_pred_binary = threshold_predictions(y_pred, threshold)
    return {
        "threshold_accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
        "threshold_balanced_accuracy": float(balanced_accuracy_score(y_true_binary, y_pred_binary)),
    }


def find_optimal_threshold(
    y_true_binary: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    eval_df = pd.DataFrame(
        {
            "prediction": pd.Series(y_pred).astype(float),
            "label": pd.Series(y_true_binary).astype(bool),
        }
    ).sort_values("prediction", kind="stable")

    grouped = (
        eval_df.groupby("prediction", sort=True)["label"]
        .agg(["sum", "count"])
        .reset_index()
    )
    grouped["neg_count"] = grouped["count"] - grouped["sum"]

    total_neg = int(grouped["neg_count"].sum())
    best_correct = total_neg
    best_threshold = float(grouped.iloc[0]["prediction"])

    pos_prefix = 0
    neg_prefix = 0
    for idx in range(len(grouped)):
        row = grouped.iloc[idx]
        pos_prefix += int(row["sum"])
        neg_prefix += int(row["neg_count"])
        correct = pos_prefix + (total_neg - neg_prefix)
        if correct <= best_correct:
            continue
        if idx == len(grouped) - 1:
            threshold = float("inf")
        else:
            left = float(row["prediction"])
            right = float(grouped.iloc[idx + 1]["prediction"])
            threshold = left + (right - left) / 2.0
        best_correct = correct
        best_threshold = threshold
    return best_threshold


def row_id_label_lookup(row_ids: list[int], labels: np.ndarray | list[int] | list[bool]) -> dict[int, bool]:
    return {int(row_id): bool(label) for row_id, label in zip(row_ids, labels)}


def dataset_row_ids(dataset: ProbingDataset) -> list[int]:
    return [int(item[0][0].split("_")[-1]) for item in dataset.inputs]


def lookup_binary_labels(row_ids: list[int], label_lookup: dict[int, bool] | None) -> np.ndarray | None:
    if label_lookup is None:
        return None
    return np.asarray([label_lookup[int(row_id)] for row_id in row_ids], dtype=bool)


def apply_threshold_metrics(
    metrics_row: dict[str, float | int | str],
    train_preds: np.ndarray,
    train_binary_labels: np.ndarray | None,
    val_preds: np.ndarray,
    val_binary_labels: np.ndarray | None,
    test_preds: np.ndarray,
    test_binary_labels: np.ndarray | None,
    threshold: float | None,
) -> dict[str, float | int | str]:
    if train_binary_labels is None or val_binary_labels is None or test_binary_labels is None:
        return metrics_row

    resolved_threshold = threshold
    if resolved_threshold is None:
        resolved_threshold = find_optimal_threshold(train_binary_labels, train_preds)

    metrics_row["threshold"] = float(resolved_threshold)
    train_metrics = compute_threshold_metrics(train_binary_labels, train_preds, resolved_threshold)
    val_metrics = compute_threshold_metrics(val_binary_labels, val_preds, resolved_threshold)
    test_metrics = compute_threshold_metrics(test_binary_labels, test_preds, resolved_threshold)
    metrics_row["train threshold_accuracy"] = train_metrics["threshold_accuracy"]
    metrics_row["train threshold_balanced_accuracy"] = train_metrics["threshold_balanced_accuracy"]
    metrics_row["val threshold_accuracy"] = val_metrics["threshold_accuracy"]
    metrics_row["val threshold_balanced_accuracy"] = val_metrics["threshold_balanced_accuracy"]
    metrics_row["full test threshold_accuracy"] = test_metrics["threshold_accuracy"]
    metrics_row["full test threshold_balanced_accuracy"] = test_metrics["threshold_balanced_accuracy"]
    metrics_row["unseen test threshold_accuracy"] = test_metrics["threshold_accuracy"]
    metrics_row["unseen test threshold_balanced_accuracy"] = test_metrics["threshold_balanced_accuracy"]

    return metrics_row


def build_prediction_frame(
    row_ids: list[int],
    permutation_type: str,
    preds: np.ndarray,
    labels: np.ndarray,
    losses: np.ndarray,
    prefix: str,
    seen_label: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instance": [
                str([(f"{prefix}__{permutation_type}__row_{row_id}", 0, 0, len(f"{prefix}__{permutation_type}__row_{row_id}"))])
                for row_id in row_ids
            ],
            "pred": preds,
            "label": labels,
            "loss": losses,
            "seen": seen_label,
        }
    )


def select_best_kernel_regression(
    train_vecs: np.ndarray,
    train_lbls: np.ndarray,
    dev_vecs: np.ndarray,
    dev_lbls: np.ndarray,
    kernel_kind: str,
    alphas: list[float],
    gammas: list[str | float],
) -> tuple[KernelRidge, dict[str, float | str], np.ndarray]:
    best_model = None
    best_params = None
    best_dev_preds = None
    best_dev_error = None

    gamma_grid = gammas if kernel_kind == "rbf" else ["scale"]
    for alpha in alphas:
        for gamma in gamma_grid:
            train_kernel = compute_kernel_matrix(train_vecs, train_vecs, kernel_kind, gamma)
            dev_kernel = compute_kernel_matrix(train_vecs, dev_vecs, kernel_kind, gamma)
            model = KernelRidge(alpha=alpha, kernel="precomputed")
            model.fit(train_kernel, train_lbls)
            dev_preds = model.predict(dev_kernel).astype(np.float64)
            dev_error = mean_squared_error(dev_lbls, dev_preds)
            if best_dev_error is None or dev_error < best_dev_error:
                best_model = model
                best_params = {"alpha": alpha, "gamma": gamma, "dev_error": float(dev_error)}
                best_dev_preds = dev_preds
                best_dev_error = dev_error

    assert best_model is not None and best_params is not None and best_dev_preds is not None
    return best_model, best_params, best_dev_preds


def select_best_kernel_classifier(
    train_vecs: np.ndarray,
    train_lbls: np.ndarray,
    dev_vecs: np.ndarray,
    dev_lbls: np.ndarray,
    kernel_kind: str,
    c_values: list[float],
    gammas: list[str | float],
) -> tuple[SVC, dict[str, float | str], np.ndarray]:
    best_model = None
    best_params = None
    best_dev_preds = None
    best_dev_f1 = None

    gamma_grid = gammas if kernel_kind == "rbf" else ["scale"]
    for c_value in c_values:
        for gamma in gamma_grid:
            train_kernel = compute_kernel_matrix(train_vecs, train_vecs, kernel_kind, gamma)
            dev_kernel = compute_kernel_matrix(train_vecs, dev_vecs, kernel_kind, gamma)
            model = SVC(C=c_value, kernel="precomputed")
            model.fit(train_kernel, train_lbls)
            dev_preds = model.predict(dev_kernel)
            dev_f1 = f1_score(dev_lbls, dev_preds, average="macro")
            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_model = model
                best_params = {"C": c_value, "gamma": gamma, "dev_f1": float(dev_f1)}
                best_dev_preds = dev_preds
                best_dev_f1 = dev_f1

    assert best_model is not None and best_params is not None and best_dev_preds is not None
    return best_model, best_params, best_dev_preds


def write_kernel_outputs(
    done_dir: Path,
    metrics_row: dict[str, float | int | str],
    preds_df: pd.DataFrame,
) -> None:
    done_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics_row]).to_csv(done_dir / "metrics.csv", index=False)
    preds_df.to_csv(done_dir / "preds.csv")


def run_kernel_layer(
    layer_idx: int,
    hidden_states: np.ndarray,
    labels: np.ndarray,
    row_ids: list[int],
    permutation_type: str,
    control_task: str,
    reduced_dim: int,
    target_col: str,
    task_type: str,
    seed: int,
    fold_idx: int,
    train_pool_idx: np.ndarray,
    test_idx: np.ndarray,
    dev_fraction: float,
    results_dir: str,
    model_name: str,
    kernel_kind: str,
    kernel_alphas: list[float],
    kernel_c_values: list[float],
    kernel_gammas: list[str | float],
    binary_eval_labels: np.ndarray | None = None,
    threshold: float | None = None,
) -> None:
    split = make_fold_arrays(
        hidden_states=hidden_states,
        labels=labels,
        row_ids=row_ids,
        permutation_type=permutation_type,
        control_task=control_task,
        reduced_dim=reduced_dim,
        task_type=task_type,
        train_pool_idx=train_pool_idx,
        test_idx=test_idx,
        dev_fraction=dev_fraction,
        seed=seed,
    )
    train_vecs, dev_vecs, test_vecs = standardize_splits(
        split["train_vecs"],
        split["dev_vecs"],
        split["test_vecs"],
    )
    train_lbls = np.asarray(split["train_lbls"])
    dev_lbls = np.asarray(split["dev_lbls"])
    test_lbls = np.asarray(split["test_lbls"])
    binary_eval_lookup = row_id_label_lookup(row_ids, binary_eval_labels) if binary_eval_labels is not None else None
    train_binary_labels = lookup_binary_labels(split["train_row_ids"], binary_eval_lookup)
    dev_binary_labels = lookup_binary_labels(split["dev_row_ids"], binary_eval_lookup)
    test_binary_labels = lookup_binary_labels(split["test_row_ids"], binary_eval_lookup)
    hidden_dim = int(train_vecs.shape[1])
    done_dir = expected_done_dir(
        results_dir=results_dir,
        target_col=target_col,
        permutation_type=permutation_type,
        control_task=control_task,
        layer_idx=layer_idx,
        model_name=model_name,
        fold_idx=fold_idx,
        seed=seed,
    )

    print(
        f"  {permutation_type:>9} | {control_task:>13} | "
        f"seed={seed} | fold={fold_idx} | layer {layer_idx:03d} | n={len(labels)} | d={hidden_dim} | method=kernel"
    )

    if task_type == "regression":
        model, best_params, dev_preds = select_best_kernel_regression(
            train_vecs=train_vecs,
            train_lbls=train_lbls.astype(np.float64),
            dev_vecs=dev_vecs,
            dev_lbls=dev_lbls.astype(np.float64),
            kernel_kind=kernel_kind,
            alphas=kernel_alphas,
            gammas=kernel_gammas,
        )
        test_kernel = compute_kernel_matrix(train_vecs, test_vecs, kernel_kind, best_params["gamma"])
        test_preds = model.predict(test_kernel).astype(np.float64)
        train_kernel = compute_kernel_matrix(train_vecs, train_vecs, kernel_kind, best_params["gamma"])
        train_preds = model.predict(train_kernel).astype(np.float64)
        test_losses = (test_preds - test_lbls.astype(np.float64)) ** 2
        preds_df = build_prediction_frame(
            row_ids=split["test_row_ids"],
            permutation_type=permutation_type,
            preds=test_preds,
            labels=test_lbls,
            losses=test_losses,
            prefix="test",
            seen_label="unseen",
        )
        metrics_row = {
            "epoch": 0,
            "step": 0,
            "full test error": float(mean_squared_error(test_lbls, test_preds)),
            "full test pearson": safe_pearson(test_preds, test_lbls.astype(np.float64)),
            "middle test error": np.nan,
            "middle test pearson": np.nan,
            "unseen test error": float(mean_squared_error(test_lbls, test_preds)),
            "unseen test pearson": safe_pearson(test_preds, test_lbls.astype(np.float64)),
            "upper test error": np.nan,
            "upper test pearson": np.nan,
            "val error": float(mean_squared_error(dev_lbls, dev_preds)),
            "val loss": float(mean_squared_error(dev_lbls, dev_preds)),
            "val loss sum": float(np.sum((dev_preds - dev_lbls.astype(np.float64)) ** 2)),
            "val pearson": safe_pearson(dev_preds, dev_lbls.astype(np.float64)),
            "val_ref": float(-mean_squared_error(dev_lbls, dev_preds)),
            "kernel_type": kernel_kind,
            "kernel_alpha": float(best_params["alpha"]),
            "kernel_gamma": best_params["gamma"],
            "method": "kernel",
        }
        metrics_row = apply_threshold_metrics(
            metrics_row,
            train_preds=train_preds,
            train_binary_labels=train_binary_labels,
            val_preds=dev_preds,
            val_binary_labels=dev_binary_labels,
            test_preds=test_preds,
            test_binary_labels=test_binary_labels,
            threshold=threshold,
        )
    else:
        model, best_params, dev_preds = select_best_kernel_classifier(
            train_vecs=train_vecs,
            train_lbls=train_lbls.astype(np.int64),
            dev_vecs=dev_vecs,
            dev_lbls=dev_lbls.astype(np.int64),
            kernel_kind=kernel_kind,
            c_values=kernel_c_values,
            gammas=kernel_gammas,
        )
        test_kernel = compute_kernel_matrix(train_vecs, test_vecs, kernel_kind, best_params["gamma"])
        test_preds = model.predict(test_kernel).astype(np.int64)
        test_losses = (test_preds != test_lbls.astype(np.int64)).astype(np.float64)
        preds_df = build_prediction_frame(
            row_ids=split["test_row_ids"],
            permutation_type=permutation_type,
            preds=test_preds,
            labels=test_lbls,
            losses=test_losses,
            prefix="test",
            seen_label="unseen",
        )
        metrics_row = {
            "epoch": 0,
            "step": 0,
            "full test acc": float(accuracy_score(test_lbls, test_preds)),
            "full test balanced_acc": float(balanced_accuracy_score(test_lbls, test_preds)),
            "full test f1": float(f1_score(test_lbls, test_preds, average="macro")),
            "middle test acc": np.nan,
            "middle test balanced_acc": np.nan,
            "middle test f1": np.nan,
            "unseen test acc": float(accuracy_score(test_lbls, test_preds)),
            "unseen test balanced_acc": float(balanced_accuracy_score(test_lbls, test_preds)),
            "unseen test f1": float(f1_score(test_lbls, test_preds, average="macro")),
            "upper test acc": np.nan,
            "upper test balanced_acc": np.nan,
            "upper test f1": np.nan,
            "val acc": float(accuracy_score(dev_lbls, dev_preds)),
            "val balanced_acc": float(balanced_accuracy_score(dev_lbls, dev_preds)),
            "val f1": float(f1_score(dev_lbls, dev_preds, average="macro")),
            "val_ref": float(f1_score(dev_lbls, dev_preds, average="macro")),
            "kernel_type": kernel_kind,
            "kernel_c": float(best_params["C"]),
            "kernel_gamma": best_params["gamma"],
            "method": "kernel",
        }
    write_kernel_outputs(done_dir, metrics_row, preds_df)


def run_layer(
    layer_idx: int,
    hidden_states: np.ndarray,
    labels: np.ndarray,
    row_ids: list[int],
    permutation_type: str,
    control_task: str,
    reduced_dim: int,
    target_col: str,
    task_type: str,
    num_labels: int,
    seed: int,
    fold_idx: int,
    train_pool_idx: np.ndarray,
    test_idx: np.ndarray,
    dev_fraction: float,
    n_total_layers: int,
    results_dir: str,
    model_name: str,
    binary_eval_labels: np.ndarray | None = None,
    threshold: float | None = None,
    dump_probe_artifacts: bool = False,
    artifact_model_id: str | None = None,
    artifact_system_prompt: str = "",
) -> None:
    train_ds, dev_ds, test_ds = make_fold_datasets(
        hidden_states,
        labels,
        row_ids,
        permutation_type,
        control_task,
        reduced_dim,
        task_type,
        train_pool_idx=train_pool_idx,
        test_idx=test_idx,
        dev_fraction=dev_fraction,
        seed=seed,
    )
    hidden_dim = int(np.asarray(train_ds.inputs_encoded[0]).shape[0])
    probe_name = (
        f"{target_col}__{permutation_type}__"
        f"{control_task.lower()}__L{layer_idx:03d}"
    )

    print(
        f"  {permutation_type:>9} | {control_task:>13} | "
        f"seed={seed} | fold={fold_idx} | layer {layer_idx:03d} | n={len(labels)} | d={hidden_dim}"
    )
    worker = GeneralProbeWorker(
        hyperparameter={
            "seed": seed,
            "encoding": "full",
            "batch_size": 8,
            "num_labels": num_labels,
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
            "control_task_type": control_task,
            "sample_size": fold_idx,
            "fold": fold_idx,
        },
        train_dataset=train_ds,
        dev_dataset=dev_ds,
        test_dataset=test_ds,
        n_layers=n_total_layers,
        probe_name=probe_name,
        project_prefix="eval-adoption",
        dump_preds=True,
        force=False,
        result_folder=results_dir,
        logging="local",
    )
    result_log_dir, _, probing_model = run_worker(worker)

    if task_type == "regression" and probing_model is not None:
        binary_eval_lookup = row_id_label_lookup(row_ids, binary_eval_labels) if binary_eval_labels is not None else None
        train_preds_df = evaluate_regression_dataset(probing_model, train_ds)
        dev_preds_df = evaluate_regression_dataset(probing_model, dev_ds)
        test_preds_df = evaluate_regression_dataset(probing_model, test_ds)
        train_binary_labels = lookup_binary_labels(dataset_row_ids(train_ds), binary_eval_lookup)
        dev_binary_labels = lookup_binary_labels(dataset_row_ids(dev_ds), binary_eval_lookup)
        test_binary_labels = lookup_binary_labels(dataset_row_ids(test_ds), binary_eval_lookup)
        metrics_path = Path(result_log_dir) / "metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            if not metrics_df.empty:
                updated_row = metrics_df.iloc[-1].to_dict()
                updated_row = apply_threshold_metrics(
                    updated_row,
                    train_preds=train_preds_df["pred"].to_numpy(dtype=np.float64),
                    train_binary_labels=train_binary_labels,
                    val_preds=dev_preds_df["pred"].to_numpy(dtype=np.float64),
                    val_binary_labels=dev_binary_labels,
                    test_preds=test_preds_df["pred"].to_numpy(dtype=np.float64),
                    test_binary_labels=test_binary_labels,
                    threshold=threshold,
                )
                for column in updated_row:
                    if column not in metrics_df.columns:
                        metrics_df[column] = np.nan
                metrics_df.iloc[-1] = pd.Series(updated_row)
                metrics_df.to_csv(metrics_path, index=False)
    elif task_type == "classification" and probing_model is not None:
        dev_preds_df = evaluate_classification_dataset(probing_model, dev_ds)
        test_preds_df = evaluate_classification_dataset(probing_model, test_ds)
        metrics_path = Path(result_log_dir) / "metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            if not metrics_df.empty:
                updated_row = metrics_df.iloc[-1].to_dict()
                updated_row = apply_classification_metrics(
                    updated_row,
                    val_preds=dev_preds_df["pred"].to_numpy(dtype=np.int64),
                    val_labels=dev_preds_df["label"].to_numpy(dtype=np.int64),
                    test_preds=test_preds_df["pred"].to_numpy(dtype=np.int64),
                    test_labels=test_preds_df["label"].to_numpy(dtype=np.int64),
                )
                for column in updated_row:
                    if column not in metrics_df.columns:
                        metrics_df[column] = np.nan
                metrics_df.iloc[-1] = pd.Series(updated_row)
                metrics_df.to_csv(metrics_path, index=False)

    if dump_probe_artifacts and probing_model is not None:
        artifact_path = export_probe_artifact(
            probing_model=probing_model,
            done_dir=Path(result_log_dir),
            layer_idx=layer_idx,
            seed=seed,
            fold_idx=fold_idx,
            permutation_type=permutation_type,
            control_task=control_task,
            model_id=artifact_model_id or model_name,
            system_prompt=artifact_system_prompt,
            target_col=target_col,
            task_type=task_type,
            threshold=threshold,
        )
        print(f"    wrote probe artifact: {artifact_path}")


def run_probe_task(task: dict) -> dict:
    layer_states = np.load(task["internals_dir"] / task["layer_file"])
    subset_states = layer_states[task["subset_indices"]]

    if task["method"] == "kernel":
        run_kernel_layer(
            layer_idx=task["layer_idx"],
            hidden_states=subset_states,
            labels=task["labels"],
            row_ids=task["row_ids"],
            permutation_type=task["permutation_type"],
            control_task=task["control_task"],
            reduced_dim=task["reduced_dim"],
            target_col=task["target_col"],
            task_type=task["task_type"],
            seed=task["seed"],
            fold_idx=task["fold_idx"],
            train_pool_idx=task["train_pool_idx"],
            test_idx=task["test_idx"],
            dev_fraction=task["dev_fraction"],
            results_dir=task["results_dir"],
            model_name=task["model_name"],
            kernel_kind=task["kernel"],
            kernel_alphas=task["kernel_alphas"],
            kernel_c_values=task["kernel_c_values"],
            kernel_gammas=task["kernel_gammas"],
            binary_eval_labels=task["binary_eval_labels"],
            threshold=task["threshold"],
        )
    else:
        run_layer(
            layer_idx=task["layer_idx"],
            hidden_states=subset_states,
            labels=task["labels"],
            row_ids=task["row_ids"],
            permutation_type=task["permutation_type"],
            control_task=task["control_task"],
            reduced_dim=task["reduced_dim"],
            target_col=task["target_col"],
            task_type=task["task_type"],
            num_labels=task["num_labels"],
            seed=task["seed"],
            fold_idx=task["fold_idx"],
            train_pool_idx=task["train_pool_idx"],
            test_idx=task["test_idx"],
            dev_fraction=task["dev_fraction"],
            n_total_layers=task["n_total_layers"],
            results_dir=task["results_dir"],
            model_name=task["model_name"],
            binary_eval_labels=task["binary_eval_labels"],
            threshold=task["threshold"],
            dump_probe_artifacts=task["dump_probe_artifacts"],
            artifact_model_id=task["artifact_model_id"],
            artifact_system_prompt=task["artifact_system_prompt"],
        )
    return {
        "permutation_type": task["permutation_type"],
        "layer_idx": task["layer_idx"],
        "seed": task["seed"],
        "fold_idx": task["fold_idx"],
        "control_task": task["control_task"],
    }


def main() -> None:
    args = parse_args()
    internals_dir = Path(args.internals_dir)
    os.makedirs(args.results_dir, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    kernel_alphas = parse_float_grid(args.kernel_alphas)
    kernel_c_values = parse_float_grid(args.kernel_c_values)
    kernel_gammas = parse_gamma_grid(args.kernel_gammas)
    artifact_model_id = args.artifact_model_id or args.model_name
    artifact_system_prompt = read_optional_text(args.artifact_system_prompt_path) if args.dump_probe_artifacts else ""
    selected_layers = parse_optional_layers(args.layers)
    selected_permutation_types = parse_optional_str_list(args.permutation_types)
    selected_control_tasks = parse_optional_str_list(args.control_tasks) or CONTROL_TASKS
    invalid_control_tasks = sorted(set(selected_control_tasks).difference(CONTROL_TASKS))
    if invalid_control_tasks:
        raise ValueError(f"Unsupported control task(s): {invalid_control_tasks}. Expected values from {CONTROL_TASKS}.")
    metadata, layer_files = load_internals(internals_dir)
    if args.target_col not in metadata.columns:
        raise ValueError(
            f"Target column {args.target_col!r} not found in {internals_dir / 'metadata.csv'}"
        )
    if selected_permutation_types is not None:
        metadata = metadata[metadata["permutation_type"].isin(selected_permutation_types)].copy()
        if metadata.empty:
            raise ValueError(f"No rows matched --permutation-types {selected_permutation_types}.")
    if selected_layers is not None:
        selected_layer_set = set(selected_layers)
        layer_files = [
            layer_file
            for layer_file in layer_files
            if int(layer_file.replace("layer_", "").replace(".npy", "")) in selected_layer_set
        ]
        missing_layers = sorted(selected_layer_set.difference(
            int(layer_file.replace("layer_", "").replace(".npy", "")) for layer_file in layer_files
        ))
        if missing_layers:
            raise ValueError(f"Layer file(s) not found for --layers {missing_layers}.")
    n_layers = len(layer_files)
    if n_layers == 0:
        raise ValueError("No layer files selected.")
    task_type = infer_task_type(metadata[args.target_col])
    binary_eval_col = args.binary_eval_col.strip()
    use_binary_eval = False
    if task_type == "regression" and binary_eval_col:
        if binary_eval_col in metadata.columns:
            use_binary_eval = True
            metadata = metadata[metadata[binary_eval_col].notna()].copy()
            metadata[binary_eval_col] = metadata[binary_eval_col].astype(bool)
        else:
            print(f"Binary eval column {binary_eval_col!r} not found; threshold metrics disabled.")

    if task_type == "regression":
        metadata[args.target_col] = metadata[args.target_col].astype(float)
        num_labels = 1
    else:
        metadata = metadata[metadata[args.target_col].notna()].copy()
        metadata[args.target_col] = metadata[args.target_col].astype(int)
        unique_labels = sorted(metadata[args.target_col].unique().tolist())
        if unique_labels != list(range(len(unique_labels))):
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            metadata[args.target_col] = metadata[args.target_col].map(label_map).astype(int)
        num_labels = int(metadata[args.target_col].nunique())
        if num_labels < 2:
            raise ValueError(f"Classification target {args.target_col!r} has fewer than 2 classes.")

    print(
        f"Running method={args.method} over {n_layers} layers across "
        f"{metadata['permutation_type'].nunique()} permutation types, "
        f"{len(selected_control_tasks)} control settings, target={args.target_col}, task={task_type}, "
        f"{len(seeds)} seeds, {args.num_folds} folds, and {args.num_workers} worker(s)"
    )
    if use_binary_eval:
        print(f"Threshold metrics enabled via binary label column {binary_eval_col!r} with threshold={args.threshold if args.threshold is not None else 'auto'}")

    layer_indices = [int(layer_file.replace("layer_", "").replace(".npy", "")) for layer_file in layer_files]
    permutation_types = sorted(str(value) for value in metadata["permutation_type"].dropna().unique().tolist())

    tasks: list[dict] = []
    skipped_done = 0
    for permutation_type, subset in metadata.groupby("permutation_type", sort=True):
        subset = subset.reset_index(drop=True)
        row_ids = subset["row_id"].astype(int).tolist()
        labels = subset[args.target_col].to_numpy(dtype=np.float32 if task_type == "regression" else np.int64)
        binary_eval_labels = subset[binary_eval_col].to_numpy(dtype=bool) if use_binary_eval else None
        subset_indices = subset["row_id"].to_numpy(dtype=int)

        print(f"\nPermutation type: {permutation_type} | rows={len(subset)}")
        if len(subset) < args.num_folds:
            print(
                f"  Skipping {permutation_type}: requires at least {args.num_folds} rows "
                f"for {args.num_folds}-fold CV, found {len(subset)}"
            )
            continue

        if task_type == "classification":
            class_counts = subset[args.target_col].value_counts()
            if class_counts.min() < args.num_folds:
                print(
                    f"  Skipping {permutation_type}: smallest class count is {class_counts.min()}, "
                    f"which is insufficient for {args.num_folds}-fold stratified CV"
                )
                continue

        for layer_file in layer_files:
            layer_idx = int(layer_file.replace("layer_", "").replace(".npy", ""))

            for seed in seeds:
                if task_type == "classification":
                    splitter = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=seed)
                    split_iter = splitter.split(subset_indices, labels)
                else:
                    splitter = KFold(n_splits=args.num_folds, shuffle=True, random_state=seed)
                    split_iter = splitter.split(subset_indices)
                for fold_idx, (train_pool_idx, test_idx) in enumerate(split_iter):
                    for control_task in selected_control_tasks:
                        if task_is_done(
                            results_dir=args.results_dir,
                            target_col=args.target_col,
                            permutation_type=permutation_type,
                            control_task=control_task,
                            layer_idx=layer_idx,
                            model_name=args.model_name,
                            fold_idx=fold_idx,
                            seed=seed,
                        ):
                            skipped_done += 1
                            continue
                        tasks.append(
                            {
                                "internals_dir": internals_dir,
                                "layer_file": layer_file,
                                "layer_idx": layer_idx,
                                "subset_indices": subset_indices,
                                "labels": labels,
                                "row_ids": row_ids,
                                "permutation_type": permutation_type,
                                "control_task": control_task,
                                "reduced_dim": args.reduced_dim,
                                "target_col": args.target_col,
                                "task_type": task_type,
                                "num_labels": num_labels,
                                "method": args.method,
                                "kernel": args.kernel,
                                "kernel_alphas": kernel_alphas,
                                "kernel_c_values": kernel_c_values,
                                "kernel_gammas": kernel_gammas,
                                "binary_eval_labels": binary_eval_labels,
                                "threshold": args.threshold,
                                "seed": seed,
                                "fold_idx": fold_idx,
                                "train_pool_idx": train_pool_idx,
                                "test_idx": test_idx,
                                "dev_fraction": args.dev_fraction,
                                "n_total_layers": n_layers,
                                "results_dir": args.results_dir,
                                "model_name": args.model_name,
                                "dump_probe_artifacts": bool(args.dump_probe_artifacts and args.method == "probe"),
                                "artifact_model_id": artifact_model_id,
                                "artifact_system_prompt": artifact_system_prompt,
                            }
                        )

    print(f"Built {len(tasks)} probe task(s)")
    if skipped_done:
        print(f"Skipped {skipped_done} already-completed task(s)")

    if args.num_workers <= 1:
        for task in tasks:
            run_probe_task(task)
    else:
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=ctx,
        ) as executor:
            for _ in executor.map(run_probe_task, tasks):
                pass

    if args.dump_probe_artifacts and args.method == "probe":
        assemble_probe_pickle_artifacts(
            results_dir=args.results_dir,
            target_col=args.target_col,
            permutation_types=permutation_types,
            control_tasks=selected_control_tasks,
            layer_indices=layer_indices,
            model_name=args.model_name,
            seeds=seeds,
            num_folds=args.num_folds,
        )

    print(f"\nDone. Results saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
