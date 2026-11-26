"""Convert each row of model logits directly to softmax probabilities."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax


CLASS_ORDER = [
    "Cholangiocarcinoma",
    "Ampullary",
    "Duodenal",
    "Pancreatic",
]
LOGIT_COLS = [f"{c}_logit" for c in CLASS_ORDER]
PROB_COLS = [f"{c}_prob" for c in CLASS_ORDER]


def with_softmax_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute probabilities from logits, ignoring any supplied probability columns.

    Keep fold-level rows so downstream loaders average probabilities only after
    applying softmax to each prediction. Labels are not needed for this step.
    """
    required = ["file_id", "fold", "dataset_type"] + LOGIT_COLS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")
    logits = df[LOGIT_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError('Logits contain NaN or infinite values.')
    out = df[required + (["true_label"] if "true_label" in df.columns else [])].copy()
    out["dataset_type"] = out["dataset_type"].astype(str).str.strip().str.upper()
    out[PROB_COLS] = softmax(logits, axis=1)
    return out


def run_softmax(input_csv: str, output_csv: str) -> None:
    out = with_softmax_probs(pd.read_csv(input_csv))
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compute softmax directly from logits.')
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    run_softmax(args.input_csv, args.output_csv)
