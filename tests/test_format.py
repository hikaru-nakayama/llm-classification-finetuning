from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocess import LABEL_A, LABEL_B, LABEL_TIE, randomize_ab_order, swap_ab_rows


def test_swap_ab_rows_flips_label_and_keeps_tie():
    df = pd.DataFrame(
        {
            "response_a": ["a0", "a1", "a2"],
            "response_b": ["b0", "b1", "b2"],
            "label": [LABEL_A, LABEL_B, LABEL_TIE],
        }
    )

    swapped = swap_ab_rows(df, pd.Series([True, True, True]))

    assert swapped["response_a"].tolist() == ["b0", "b1", "b2"]
    assert swapped["response_b"].tolist() == ["a0", "a1", "a2"]
    assert swapped["label"].tolist() == [LABEL_B, LABEL_A, LABEL_TIE]


def test_randomize_ab_order_is_reproducible_for_seed():
    df = pd.DataFrame(
        {
            "response_a": ["a0", "a1", "a2", "a3"],
            "response_b": ["b0", "b1", "b2", "b3"],
            "label": [LABEL_A, LABEL_B, LABEL_TIE, LABEL_A],
        }
    )

    randomized_1 = randomize_ab_order(df, random_seed=42)
    randomized_2 = randomize_ab_order(df, random_seed=42)

    pd.testing.assert_frame_equal(randomized_1, randomized_2)
