"""
Create mock-up robustness data for Qwen/Qwen2.5-0.5B-Instruct.

Loads unique problems from the AIMO interpretability challenge dataset and
assigns random model_is_robust labels to simulate model evaluation output.
Saves to data/mockup_data.parquet and data/mockup_data.csv.
"""
import os
import random
import pandas as pd
from datasets import load_dataset

SEED = 42
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "michal-stefanik/aimo-interp-challenge-sample-v2"


def main():
    random.seed(SEED)
    os.makedirs("data", exist_ok=True)

    ds = load_dataset(DATASET_ID, split="validation")
    df = ds.to_pandas()

    unique_problems = (
        df[["problem_id", "original_problem"]]
        .drop_duplicates("problem_id")
        .reset_index(drop=True)
    )

    unique_problems["model_id"] = MODEL_ID
    unique_problems["model_is_robust"] = [
        random.choice([True, False]) for _ in range(len(unique_problems))
    ]

    unique_problems.to_parquet("data/mockup_data.parquet", index=False)
    unique_problems.to_csv("data/mockup_data.csv", index=False)

    n_robust = unique_problems["model_is_robust"].sum()
    n_total = len(unique_problems)
    print(f"Saved {n_total} problems  (robust={n_robust}, non-robust={n_total - n_robust})")
    print("Output: data/mockup_data.parquet, data/mockup_data.csv")


if __name__ == "__main__":
    main()
