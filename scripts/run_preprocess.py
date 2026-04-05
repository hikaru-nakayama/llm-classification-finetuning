from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "data" / "raw" / "train.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TRAIN_OUTPUT_PATH = PROCESSED_DIR / "preprcessed_train.parquet"
EVAL_OUTPUT_PATH = PROCESSED_DIR / "preprcessed_eval.parquet"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_TRAIN_TOKENS = 2048

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

[Decision]
Choose exactly one label:
0 = A is preferred
1 = B is preferred
2 = Tie

Label:
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

train_df["token_length"] = train_df["text"].apply(
    lambda text: len(
        tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    )
)
train_df = train_df.loc[
    train_df["token_length"] <= MAX_TRAIN_TOKENS, ["text", "label"]
].reset_index(drop=True)

print(train_df["label"].dtype)
print(train_df["label"].value_counts())
print(sorted(train_df["label"].unique()))

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_parquet(TRAIN_OUTPUT_PATH, index=False)
eval_df.to_parquet(EVAL_OUTPUT_PATH, index=False)

print(f"saved train to: {TRAIN_OUTPUT_PATH} ({len(train_df)} rows)")
print(f"saved eval to: {EVAL_OUTPUT_PATH} ({len(eval_df)} rows)")
print(f"filtered train to <= {MAX_TRAIN_TOKENS} tokens")
