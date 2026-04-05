from pathlib import Path
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocess import build_input_text, randomize_ab_order

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


processed_df = (
    df[["prompt", "response_a", "response_b", "label"]]
    .sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    )
    .reset_index(drop=True)
)
train_size = int(len(processed_df) * TRAIN_RATIO)
train_df = processed_df.iloc[:train_size].copy()
eval_df = processed_df.iloc[train_size:].copy()

train_df = randomize_ab_order(train_df, random_seed=RANDOM_SEED)

train_df["text"] = train_df.apply(
    lambda row: build_input_text(
        row["prompt"],
        row["response_a"],
        row["response_b"],
    ),
    axis=1,
)
eval_df["text"] = eval_df.apply(
    lambda row: build_input_text(
        row["prompt"],
        row["response_a"],
        row["response_b"],
    ),
    axis=1,
)

train_df["token_length"] = train_df["text"].apply(
    lambda text: len(
        tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    )
)
train_df = train_df.loc[
    train_df["token_length"] <= MAX_TRAIN_TOKENS, ["text", "label"]
].reset_index(drop=True)
eval_df = eval_df[["text", "label"]].reset_index(drop=True)

print(train_df["label"].dtype)
print(train_df["label"].value_counts())
print(sorted(train_df["label"].unique()))

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_parquet(TRAIN_OUTPUT_PATH, index=False)
eval_df.to_parquet(EVAL_OUTPUT_PATH, index=False)

print(f"saved train to: {TRAIN_OUTPUT_PATH} ({len(train_df)} rows)")
print(f"saved eval to: {EVAL_OUTPUT_PATH} ({len(eval_df)} rows)")
print(f"filtered train to <= {MAX_TRAIN_TOKENS} tokens")
