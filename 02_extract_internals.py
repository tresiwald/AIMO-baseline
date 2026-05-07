"""
Extract layer-wise hidden states from the last input token for each problem.

Uses lamina to capture internal representations of Qwen/Qwen2.5-0.5B-Instruct
while it processes each math problem with the AIMO solve system prompt.
Saves one .npy file per layer (shape: n_problems x hidden_dim) plus metadata.

Requires:
    pip install git+https://github.com/tresiwald/lamina.git[hf]

Run 01_mockup_data.py first to generate data/mockup_data.parquet.
"""
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lamina
from lamina import InternalsDataset, InternalsInstance

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT_PATH = "prompts/solve.txt"
DATA_PATH = "data/mockup_data.parquet"
OUTPUT_DIR = "data/internals"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    system_prompt = open(SYSTEM_PROMPT_PATH).read().strip()
    df = pd.read_parquet(DATA_PATH)

    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    instances = [
        InternalsInstance(
            text=row["original_problem"],
            properties={
                "problem_id": row["problem_id"],
                "model_is_robust": int(row["model_is_robust"]),
            },
            system_prompt=system_prompt,
        )
        for _, row in df.iterrows()
    ]
    dataset = InternalsDataset(instances)

    print(f"Extracting internals for {len(instances)} problems ...")
    records = dataset.run(model, tokenizer, generate_kwargs={"max_new_tokens": 1})

    n_layers = records[0].run.num_layers
    hidden_dim = records[0].run.input_hidden_states[0].shape[-1]
    print(f"Layers: {n_layers}  |  hidden_dim: {hidden_dim}")

    # Per-layer array of shape (n_problems, hidden_dim) — last input token
    for layer_idx in range(n_layers):
        layer_vecs = np.stack([
            rec.run.input_hidden_states[layer_idx][0, -1, :]
            for rec in records
        ])
        np.save(os.path.join(OUTPUT_DIR, f"layer_{layer_idx:03d}.npy"), layer_vecs)

    metadata = pd.DataFrame([
        {
            "problem_id": rec.properties["problem_id"],
            "model_is_robust": rec.properties["model_is_robust"],
            "original_problem": rec.instance.text,
        }
        for rec in records
    ])
    metadata.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

    print(f"Saved {n_layers} layer files + metadata to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
