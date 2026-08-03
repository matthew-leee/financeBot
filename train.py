"""
Training entrypoint.

Pipeline:
  fetch history -> build (X, y) -> walk-forward validate -> fit final -> save JSON

Run:  python train.py
"""

from __future__ import annotations

import pandas as pd

import config
from src.data import fetch_bars, make_dataset
from src.model_io import save_model
from src.validation import fit_final_model, walk_forward_validate


def build_training_frame() -> tuple[pd.DataFrame, pd.Series]:
    """Fetch every symbol, build per-symbol datasets, and stack them."""
    symbols = config.EQUITY_SYMBOLS + config.CRYPTO_SYMBOLS
    x_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []

    for symbol in symbols:
        print(f"[train] Fetching {symbol} ...")
        bars = fetch_bars(symbol)
        if bars.empty or len(bars) < 60:
            print(f"[train] Skipping {symbol}: not enough data.")
            continue
        x, y = make_dataset(bars)
        x_parts.append(x)
        y_parts.append(y)

    if not x_parts:
        raise RuntimeError("No training data assembled. Check symbols/credentials.")

    x_all = pd.concat(x_parts, ignore_index=True)
    y_all = pd.concat(y_parts, ignore_index=True)
    print(f"[train] Assembled dataset: {x_all.shape[0]} rows, {x_all.shape[1]} features.")
    return x_all, y_all


def main() -> None:
    x, y = build_training_frame()

    print("[train] Running walk-forward validation ...")
    report = walk_forward_validate(x, y, n_splits=5)
    summary = report.summary()
    print(f"[train] Walk-forward summary: {summary}")

    print("[train] Fitting final model on all history ...")
    model = fit_final_model(x, y)

    save_model(model, summary)
    print("[train] Done.")


if __name__ == "__main__":
    main()
