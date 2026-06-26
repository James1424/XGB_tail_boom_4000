import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .model_config import ID_COLUMNS, LEAKAGE_COLUMNS, MIN_TRAINING_PANEL_ROWS, PANEL_FALLBACKS, PANEL_FILE


def find_panel_file() -> Path:
    candidates = [PANEL_FILE] + PANEL_FALLBACKS
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError(
        "No full training panel found. The model requires outputs/clean_monthly_panel.csv "
        "from the integrated build step, or data/clean_monthly_panel.csv for local/manual runs. "
        "outputs/panel_head_20000.csv is only a display sample and is intentionally not accepted for training."
    )


def load_panel() -> pd.DataFrame:
    path = find_panel_file()
    df = pd.read_csv(path)
    if len(df) < MIN_TRAINING_PANEL_ROWS:
        raise ValueError(
            f"Training panel at {path} has only {len(df):,} rows. "
            f"Expected at least {MIN_TRAINING_PANEL_ROWS:,}. "
            "This usually means a sample file such as panel_head_20000.csv was used accidentally."
        )
    if "month" not in df.columns or "ticker" not in df.columns:
        raise ValueError("Panel must contain month and ticker columns.")
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["month", "ticker"]).reset_index(drop=True)
    return df


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    bad = set(ID_COLUMNS + LEAKAGE_COLUMNS)
    features = []
    for c in df.columns:
        if c in bad:
            continue
        if c.startswith("future_") or c.startswith("label_") or c.startswith("monthly_"):
            continue
        if df[c].dtype.kind in "biufc":
            features.append(c)
    return features


def columns_matching(features: Iterable[str], patterns: Iterable[str]) -> list[str]:
    out = []
    for col in features:
        if any(p in col for p in patterns):
            out.append(col)
    return out
