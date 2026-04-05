from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "data" / "raw" / "train.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TRAIN_OUTPUT_PATH = PROCESSED_DIR / "preprcessed_train.parquet"
EVAL_OUTPUT_PATH = PROCESSED_DIR / "preprcessed_eval.parquet"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

df = pd.read_csv(TRAIN_PATH)

label_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
df["label"] = np.argmax(df[label_cols].to_numpy(), axis=1)


def build_input_text(prompt: str, response_a: str, response_b: str) -> str:
    return f"""You are a judge that predicts which response a human would prefer.

[Prompt]
{prompt}

[Response A]
{response_a}

[Response B]
{response_b}
"""


df["text"] = df.apply(
    lambda row: build_input_text(
        row["prompt"],
        row["response_a"],
        row["response_b"],
    ),
    axis=1,
)

processed_df = (
    df[["text", "label"]]
    .sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    )
    .reset_index(drop=True)
)
train_size = int(len(processed_df) * TRAIN_RATIO)
train_df = processed_df.iloc[:train_size].copy()
eval_df = processed_df.iloc[train_size:].copy()

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_parquet(TRAIN_OUTPUT_PATH, index=False)
eval_df.to_parquet(EVAL_OUTPUT_PATH, index=False)

print(f"saved train to: {TRAIN_OUTPUT_PATH} ({len(train_df)} rows)")
print(f"saved eval to: {EVAL_OUTPUT_PATH} ({len(eval_df)} rows)")
