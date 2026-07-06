#!/usr/bin/env python3
"""Patch Holmes probe classification metrics for binary decoded predictions.

The upstream probe_only branch can flatten classification predictions before
calling argmax(dim=1). Binary outputs may already be decoded as a 1D class
tensor, which raises IndexError during smoke tests and export runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PATCHES = [
    (
        """        for set_name, preds, labels in metric_inputs:
            preds = preds.reshape(-1)
            labels = labels.reshape(-1)

            if preds.numel() == 0:
                continue

            for metric, func in self.metrics.items():
                #if metric == "pearson":
                #    pred_labels = pred_labels.squeeze(dim=1)
                if self.hyperparameter["num_labels"] > 1:
                    metric_result = func(preds.argmax(1), labels)
                else:
                    metric_result = func(preds, labels)
                metric_results[set_name + " test " + metric] = float(metric_result)
""",
        """        for set_name, preds, labels in metric_inputs:
            labels = labels.reshape(-1)

            if preds.numel() == 0:
                continue

            for metric, func in self.metrics.items():
                #if metric == "pearson":
                #    pred_labels = pred_labels.squeeze(dim=1)
                if self.hyperparameter["num_labels"] > 1:
                    pred_classes = preds.argmax(1) if preds.ndim > 1 else preds.long()
                    metric_result = func(pred_classes, labels)
                else:
                    preds = preds.reshape(-1)
                    metric_result = func(preds, labels)
                metric_results[set_name + " test " + metric] = float(metric_result)
""",
    ),
    (
        """        if self.hyperparameter["num_labels"] >= 2:
            self.test_raw_preds = pred_labels.detach().cpu().double()
            self.test_preds = pred_labels.argmax(dim=1).detach().cpu().double().numpy()
            self.test_labels = truth_labels.detach().cpu()
""",
        """        if self.hyperparameter["num_labels"] >= 2:
            self.test_raw_preds = pred_labels.detach().cpu().double()
            if pred_labels.ndim > 1:
                self.test_preds = pred_labels.argmax(dim=1).detach().cpu().double().numpy()
            else:
                self.test_preds = pred_labels.detach().cpu().double().numpy()
            self.test_labels = truth_labels.detach().cpu()
""",
    ),
    (
        """        for metric, func in self.metrics.items():

            if self.hyperparameter["num_labels"] > 1:
                metric_result = func(pred_labels.argmax(1), truth_labels)
            else:
                metric_result = func(pred_labels, truth_labels)
""",
        """        for metric, func in self.metrics.items():

            if self.hyperparameter["num_labels"] > 1:
                pred_classes = pred_labels.argmax(1) if pred_labels.ndim > 1 else pred_labels.long()
                metric_result = func(pred_classes, truth_labels)
            else:
                metric_result = func(pred_labels, truth_labels)
""",
    ),
    (
        """        if self.hyperparameter["num_labels"] >= 2:
            self.dev_preds = pred_labels.argmax(dim=1).detach().cpu().int().numpy()
            self.dev_losses = losses.detach().cpu().double().numpy()
""",
        """        if self.hyperparameter["num_labels"] >= 2:
            if pred_labels.ndim > 1:
                self.dev_preds = pred_labels.argmax(dim=1).detach().cpu().int().numpy()
            else:
                self.dev_preds = pred_labels.detach().cpu().int().numpy()
            self.dev_losses = losses.detach().cpu().double().numpy()
""",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    source = args.path.read_text()
    patched = source
    changed = False
    for old, new in PATCHES:
        if new in patched:
            continue
        if old not in patched:
            raise RuntimeError(f"Expected Holmes source block not found in {args.path}")
        patched = patched.replace(old, new)
        changed = True

    if changed:
        args.path.write_text(patched)
        print(f"Patched Holmes classification handling: {args.path}")
    else:
        print(f"Holmes classification patch already present: {args.path}")


if __name__ == "__main__":
    main()
