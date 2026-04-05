from __future__ import annotations

import numpy as np
import pandas as pd


LABEL_A = 0
LABEL_B = 1
LABEL_TIE = 2


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


def swap_ab_rows(df: pd.DataFrame, swap_mask: pd.Series) -> pd.DataFrame:
    swapped_df = df.copy()
    mask = swap_mask.astype(bool)

    swapped_df.loc[mask, ["response_a", "response_b"]] = swapped_df.loc[
        mask, ["response_b", "response_a"]
    ].to_numpy()

    swapped_df.loc[mask, "label"] = swapped_df.loc[mask, "label"].replace(
        {
            LABEL_A: LABEL_B,
            LABEL_B: LABEL_A,
        }
    )
    return swapped_df


def randomize_ab_order(
    df: pd.DataFrame,
    random_seed: int,
    swap_probability: float = 0.5,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    swap_mask = pd.Series(
        rng.random(len(df)) < swap_probability,
        index=df.index,
    )
    return swap_ab_rows(df, swap_mask)
