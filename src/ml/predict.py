from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "round_win_lgbm.txt"

FEATURE_COLUMNS = [
    "alive_team",
    "alive_enemy",
    "seconds_remaining",
    "bomb_planted",
    "bomb_time_since_plant",
    "bomb_time_remaining",
    "opening_kill_for_team",
    "side",
    "snapshot_type",
    "map_name",
]
CATEGORICAL_COLUMNS = ["side", "snapshot_type", "map_name"]


def _load_libraries() -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "lightgbm is required to load the round win model. Install it in the active environment first."
        ) from exc

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to convert Polars frames into LightGBM input. Install it in the active environment first."
        ) from exc

    return lgb


def _ensure_columns(dataset: pl.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Snapshots are missing required columns: {missing_columns}")


def _prepare_feature_frame(snapshots: pl.DataFrame) -> Any:
    _ensure_columns(snapshots, FEATURE_COLUMNS)

    prepared = snapshots.with_columns(
        [
            pl.col("alive_team").cast(pl.Int64),
            pl.col("alive_enemy").cast(pl.Int64),
            pl.col("seconds_remaining").cast(pl.Float64),
            pl.col("bomb_planted").cast(pl.Boolean),
            pl.col("bomb_time_since_plant").cast(pl.Float64),
            pl.col("bomb_time_remaining").cast(pl.Float64),
            pl.col("opening_kill_for_team").cast(pl.Boolean),
            pl.col("side").cast(pl.Utf8),
            pl.col("snapshot_type").cast(pl.Utf8),
            pl.col("map_name").fill_null("unknown").cast(pl.Utf8),
        ]
    )

    feature_frame = prepared.select(FEATURE_COLUMNS).to_pandas()
    for column in CATEGORICAL_COLUMNS:
        feature_frame[column] = feature_frame[column].astype("category")
    return feature_frame


def load_round_win_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    lgb = _load_libraries()
    resolved_path = Path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model not found: {resolved_path}")
    return lgb.Booster(model_file=str(resolved_path))


def predict_probabilities(model: Any, snapshots: pl.DataFrame) -> pl.DataFrame:
    if snapshots.is_empty():
        return snapshots.with_columns(pl.lit(None, dtype=pl.Float64).alias("win_probability"))

    feature_frame = _prepare_feature_frame(snapshots)
    probabilities = model.predict(feature_frame)

    if len(probabilities) != snapshots.height:
        raise RuntimeError(
            f"Prediction row count mismatch: got {len(probabilities)} predictions for {snapshots.height} rows."
        )

    return snapshots.with_columns(
        pl.Series("win_probability", probabilities, dtype=pl.Float64)
    )
