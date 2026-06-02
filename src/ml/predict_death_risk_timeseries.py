from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "ml" / "death_risk_timeseries_5s.parquet"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "death_risk_timeseries_5s_lgbm.txt"
DEFAULT_METRICS_PATH = REPO_ROOT / "data" / "ml" / "death_risk_timeseries_5s_metrics.json"
DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "data" / "ml" / "death_risk_timeseries_5s_thresholds.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "ml" / "death_risk_timeseries_5s_predictions.parquet"

TARGET_COLUMN = "death_within_5s"
MATCH_ID_COLUMN = "match_id"
OUTPUT_COLUMNS = [
    "match_id",
    "map_name",
    "round_num",
    "tick",
    "steamid",
    "player_name",
    "side",
    "death_risk_5s",
    "death_risk_rank_global",
    "death_risk_percentile_global",
    "death_risk_bucket_global",
    "risk_label",
    "death_within_5s",
    "kill_within_5s",
    "damage_dealt_next_5s",
    "damage_taken_next_5s",
    "alive_team_at_snapshot",
    "alive_enemy_at_snapshot",
    "seconds_remaining_at_snapshot",
    "bomb_planted_at_snapshot",
    "player_hp",
    "weapon",
    "has_armor",
    "equipment_value",
    "nearest_teammate_distance",
    "nearest_enemy_distance",
    "prior_round_phase",
]
BUCKET_CUTOFFS = [
    ("top_1_percent", 0.01),
    ("top_5_percent", 0.05),
    ("top_10_percent", 0.10),
    ("top_20_percent", 0.20),
]
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict time-sampled death risk and export ranked probabilities.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the time-sampled dataset (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained LightGBM model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Path to the saved training metrics JSON (default: {DEFAULT_METRICS_PATH})",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=DEFAULT_THRESHOLDS_PATH,
        help=f"Path to the saved threshold analysis JSON (default: {DEFAULT_THRESHOLDS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to save the prediction parquet (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed used during training.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def load_lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit(
            "lightgbm is required to run this script. Install it in the active environment first."
        ) from exc
    return lgb


def load_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required to build model inputs for prediction export. Install it in the active environment first."
        ) from exc
    return pd


def ensure_columns(dataset: pl.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise SystemExit(f"Dataset is missing required columns: {missing_columns}")


def split_match_ids(match_ids: list[str], seed: int) -> dict[str, list[str]]:
    if len(match_ids) < 3:
        raise SystemExit(
            f"Need at least 3 unique matches for train/validation/test splitting, found {len(match_ids)}."
        )

    shuffled = list(match_ids)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * 0.70)
    validation_end = train_end + int(total * 0.15)
    split_ids = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    if any(not split_ids[name] for name in split_ids):
        raise SystemExit("Match split produced an empty partition.")

    train_ids = set(split_ids["train"])
    validation_ids = set(split_ids["validation"])
    test_ids = set(split_ids["test"])
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Train/validation/test match sets overlap.")
    return split_ids


def filter_by_match_ids(dataset: pl.DataFrame, match_ids: list[str]) -> pl.DataFrame:
    return dataset.filter(pl.col(MATCH_ID_COLUMN).is_in(match_ids))


def build_numeric_fill_values(
    train_df: pl.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_fill_strategy: dict[str, str],
) -> dict[str, float]:
    fill_values: dict[str, float] = {}
    for column in feature_columns:
        if column in categorical_columns:
            continue
        strategy = numeric_fill_strategy.get(column)
        if strategy is None:
            raise SystemExit(f"Missing numeric fill strategy for column: {column}")
        if strategy.startswith("zero"):
            fill_values[column] = 0.0
            continue

        median_value = train_df.select(pl.col(column).median()).item()
        fill_values[column] = float(median_value) if median_value is not None else 0.0
    return fill_values


def apply_numeric_fill_values(
    dataset: pl.DataFrame,
    *,
    feature_columns: list[str],
    categorical_columns: list[str],
    fill_values: dict[str, float],
) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for column in feature_columns:
        if column in categorical_columns:
            expressions.append(pl.col(column).fill_null("unknown").cast(pl.Utf8))
        elif column in {"bomb_planted_at_snapshot", "has_armor"}:
            expressions.append(pl.col(column).fill_null(False).cast(pl.Float64))
        else:
            expressions.append(pl.col(column).fill_null(fill_values[column]).cast(pl.Float64))
    return dataset.with_columns(expressions)


def build_category_values(train_df: pl.DataFrame, categorical_columns: list[str]) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for column in categorical_columns:
        values = (
            train_df.select(pl.col(column).fill_null("unknown").cast(pl.Utf8).unique().sort())
            .to_series()
            .to_list()
        )
        if "unknown" not in values:
            values.append("unknown")
        categories[column] = [str(value) for value in values]
    return categories


def prepare_frame(
    dataset: pl.DataFrame,
    *,
    feature_columns: list[str],
    categorical_columns: list[str],
    category_values: dict[str, list[str]],
) -> tuple[Any, np.ndarray]:
    feature_frame = dataset.select(feature_columns).to_pandas()
    for column in categorical_columns:
        known_values = set(category_values[column])
        feature_frame[column] = (
            feature_frame[column]
            .fillna("unknown")
            .astype(str)
            .where(feature_frame[column].fillna("unknown").astype(str).isin(known_values), "unknown")
            .astype("category")
            .cat.set_categories(category_values[column])
        )

    target = (
        dataset.select(pl.col(TARGET_COLUMN).cast(pl.Int8))
        .to_numpy()
        .ravel()
        .astype(np.int32, copy=False)
    )
    return feature_frame, target


def predict_probability(model: Any, frame: Any) -> np.ndarray:
    probabilities = model.predict(frame)
    return np.asarray(probabilities, dtype=float)


def top_bucket_labels(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = len(probs)
    order = np.argsort(-probs, kind="mergesort")
    ranks = np.empty(total, dtype=np.int64)
    ranks[order] = np.arange(1, total + 1, dtype=np.int64)

    if total > 1:
        percentiles = (1.0 - ((ranks - 1) / (total - 1))) * 100.0
    else:
        percentiles = np.full(total, 100.0, dtype=float)

    buckets = np.full(total, "normal", dtype=object)
    for bucket_name, fraction in reversed(BUCKET_CUTOFFS):
        cutoff = max(1, int(math.ceil(total * fraction)))
        buckets[ranks <= cutoff] = bucket_name
    return ranks, percentiles, buckets


def add_risk_labels(
    probs: np.ndarray,
    *,
    threshold_precision_50: float,
    best_f1_threshold: float,
    threshold_recall_50: float,
) -> np.ndarray:
    labels = np.full(len(probs), "low", dtype=object)
    labels[probs >= threshold_recall_50] = "medium"
    labels[probs >= best_f1_threshold] = "high"
    labels[probs >= threshold_precision_50] = "critical"
    return labels


def summarize_and_log(dataset: pl.DataFrame, output_path: Path) -> None:
    probabilities = dataset["death_risk_5s"].to_numpy()
    LOGGER.info(
        "rows loaded=%d predictions generated=%d probability_min=%.6f probability_mean=%.6f probability_max=%.6f",
        dataset.height,
        dataset.height,
        float(np.min(probabilities)) if len(probabilities) else float("nan"),
        float(np.mean(probabilities)) if len(probabilities) else float("nan"),
        float(np.max(probabilities)) if len(probabilities) else float("nan"),
    )

    for column in ["risk_label", "death_risk_bucket_global"]:
        if column in dataset.columns:
            grouped = dataset.group_by(column).agg(pl.len().alias("rows")).sort(column)
            for row in grouped.to_dicts():
                LOGGER.info("rows per %s=%s rows=%d", column, row[column], row["rows"])

    for column in ["risk_label", "death_risk_bucket_global"]:
        if column in dataset.columns and "death_within_5s" in dataset.columns:
            grouped = (
                dataset.group_by(column)
                .agg(
                    [
                        pl.len().alias("rows"),
                        pl.col("death_within_5s").cast(pl.Boolean).mean().alias("death_rate"),
                    ]
                )
                .sort(column)
            )
            for row in grouped.to_dicts():
                LOGGER.info("death rate per %s=%s death_rate=%.6f", column, row[column], float(row["death_rate"]))

    LOGGER.info("output path=%s", output_path)


def main() -> None:
    configure_logging()
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.metrics.exists():
        raise SystemExit(f"Metrics file not found: {args.metrics}")
    if not args.thresholds.exists():
        raise SystemExit(f"Thresholds file not found: {args.thresholds}")

    pd = load_pandas()
    lgb = load_lightgbm()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    feature_columns = metrics.get("feature_columns")
    categorical_columns = metrics.get("categorical_columns")
    numeric_fill_strategy = metrics.get("numeric_fill_strategy")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise SystemExit("Metrics JSON is missing feature_columns.")
    if not isinstance(categorical_columns, list):
        raise SystemExit("Metrics JSON is missing categorical_columns.")
    if not isinstance(numeric_fill_strategy, dict):
        raise SystemExit("Metrics JSON is missing numeric_fill_strategy.")

    validation_thresholds = thresholds.get("validation", {}).get("recommended_thresholds", {})
    best_f1_threshold = validation_thresholds.get("best_f1_threshold")
    threshold_precision_50 = validation_thresholds.get("threshold_precision_50")
    threshold_recall_50 = validation_thresholds.get("threshold_recall_50")
    if not all(isinstance(item, dict) and "threshold" in item for item in [best_f1_threshold, threshold_precision_50, threshold_recall_50]):
        raise SystemExit("Thresholds JSON is missing required recommended thresholds.")

    dataset = pl.read_parquet(args.dataset)
    LOGGER.info("rows loaded=%d", dataset.height)
    ensure_columns(
        dataset,
        [
            MATCH_ID_COLUMN,
            TARGET_COLUMN,
            "round_num",
            "tick",
            "steamid",
            "player_name",
            "side",
            *feature_columns,
        ],
    )
    if dataset.is_empty():
        raise SystemExit(f"Dataset is empty: {args.dataset}")
    if dataset.select(pl.col(TARGET_COLUMN).is_null().any()).item():
        raise SystemExit(f"Target column contains null values: {TARGET_COLUMN}")

    unique_match_ids = dataset.select(pl.col(MATCH_ID_COLUMN).unique().sort()).to_series().to_list()
    split_ids = split_match_ids([str(match_id) for match_id in unique_match_ids], seed=args.seed)

    train_df = filter_by_match_ids(dataset, split_ids["train"])
    validation_df = filter_by_match_ids(dataset, split_ids["validation"])
    test_df = filter_by_match_ids(dataset, split_ids["test"])

    fill_values = build_numeric_fill_values(
        train_df,
        feature_columns,
        categorical_columns,
        numeric_fill_strategy,
    )
    train_df = apply_numeric_fill_values(
        train_df,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        fill_values=fill_values,
    )
    validation_df = apply_numeric_fill_values(
        validation_df,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        fill_values=fill_values,
    )
    test_df = apply_numeric_fill_values(
        test_df,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        fill_values=fill_values,
    )
    category_values = build_category_values(train_df, categorical_columns)

    load_pandas()
    processed_dataset = apply_numeric_fill_values(
        dataset,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        fill_values=fill_values,
    )
    X_all, _ = prepare_frame(
        processed_dataset,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        category_values=category_values,
    )

    booster = lgb.Booster(model_file=str(args.model))
    probabilities = predict_probability(booster, X_all)
    ranks, percentiles, buckets = top_bucket_labels(probabilities)
    risk_labels = add_risk_labels(
        probabilities,
        threshold_precision_50=float(threshold_precision_50["threshold"]),
        best_f1_threshold=float(best_f1_threshold["threshold"]),
        threshold_recall_50=float(threshold_recall_50["threshold"]),
    )

    output_df = processed_dataset.with_columns(
        [
            pl.Series("death_risk_5s", probabilities),
            pl.Series("death_risk_rank_global", ranks),
            pl.Series("death_risk_percentile_global", percentiles),
            pl.Series("death_risk_bucket_global", buckets),
            pl.Series("risk_label", risk_labels),
        ]
    ).select(OUTPUT_COLUMNS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_parquet(args.output)

    summarize_and_log(output_df, args.output)


if __name__ == "__main__":
    main()
