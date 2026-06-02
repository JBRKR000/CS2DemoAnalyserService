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
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "ml" / "death_risk_timeseries_5s_thresholds.json"

TARGET_COLUMN = "death_within_5s"
MATCH_ID_COLUMN = "match_id"
THRESHOLDS = [round(value, 2) for value in np.arange(0.01, 1.0, 0.01)]
TOP_K_BUCKETS = [1, 2, 5, 10, 20]
BOOLEAN_NUMERIC_COLUMNS = {"bomb_planted_at_snapshot", "has_armor"}
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze thresholds for the time-sampled death risk model.",
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to save threshold analysis JSON (default: {DEFAULT_OUTPUT_PATH})",
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


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_for_json(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


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
            "pandas is required to build model inputs for threshold analysis. Install it in the active environment first."
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

    assert_disjoint_match_sets(split_ids)
    return split_ids


def assert_disjoint_match_sets(split_ids: dict[str, list[str]]) -> None:
    train_ids = set(split_ids["train"])
    validation_ids = set(split_ids["validation"])
    test_ids = set(split_ids["test"])
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Train/validation/test match sets overlap.")


def filter_by_match_ids(dataset: pl.DataFrame, match_ids: list[str]) -> pl.DataFrame:
    return dataset.filter(pl.col(MATCH_ID_COLUMN).is_in(match_ids))


def positive_rate(y_true: np.ndarray) -> float:
    return float(np.mean(y_true))


def log_loss_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    probabilities = np.clip(y_prob.astype(float), 1e-15, 1.0 - 1e-15)
    losses = -(y_true * np.log(probabilities) + (1 - y_true) * np.log(1.0 - probabilities))
    return float(np.mean(losses))


def roc_auc_score_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(y_prob, kind="mergesort")
    sorted_scores = y_prob[order]
    sorted_true = y_true[order]

    rank_sum = 0.0
    index = 0
    total = len(sorted_scores)
    while index < total:
        tie_end = index + 1
        while tie_end < total and sorted_scores[tie_end] == sorted_scores[index]:
            tie_end += 1

        avg_rank = (index + 1 + tie_end) / 2.0
        positives_in_tie = int(np.sum(sorted_true[index:tie_end]))
        rank_sum += positives_in_tie * avg_rank
        index = tie_end

    auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def precision_recall_f1_accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (y_prob >= threshold).astype(np.int32)
    true_positives = int(np.sum((predictions == 1) & (y_true == 1)))
    false_positives = int(np.sum((predictions == 1) & (y_true == 0)))
    false_negatives = int(np.sum((predictions == 0) & (y_true == 1)))
    true_negatives = int(np.sum((predictions == 0) & (y_true == 0)))
    positive_predictions = int(np.sum(predictions == 1))
    total = len(y_true)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    accuracy = (true_positives + true_negatives) / total if total else float("nan")
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "positive_predictions": positive_predictions,
        "positive_prediction_rate": float(positive_predictions / total) if total else float("nan"),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }


def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict[str, Any]]:
    return [precision_recall_f1_accuracy(y_true, y_prob, threshold) for threshold in THRESHOLDS]


def _best_by_precision_or_recall(
    thresholds: list[dict[str, Any]],
    *,
    minimum_recall: float | None = None,
    minimum_precision: float | None = None,
    prefer: str,
) -> dict[str, Any] | None:
    candidates = thresholds
    if minimum_recall is not None:
        candidates = [row for row in candidates if row["recall"] >= minimum_recall]
    if minimum_precision is not None:
        candidates = [row for row in candidates if row["precision"] >= minimum_precision]
    if not candidates:
        return None

    if prefer == "precision":
        return max(candidates, key=lambda row: (row["precision"], row["recall"], row["threshold"]))
    if prefer == "recall":
        return max(candidates, key=lambda row: (row["recall"], row["precision"], row["threshold"]))
    raise ValueError(f"Unknown preference: {prefer}")


def recommend_thresholds(thresholds: list[dict[str, Any]]) -> dict[str, Any]:
    best_f1 = max(thresholds, key=lambda row: (row["f1"], row["precision"], row["recall"], row["threshold"]))
    return {
        "best_f1_threshold": best_f1,
        "threshold_recall_50": _best_by_precision_or_recall(
            thresholds, minimum_recall=0.50, prefer="precision"
        ),
        "threshold_recall_70": _best_by_precision_or_recall(
            thresholds, minimum_recall=0.70, prefer="precision"
        ),
        "threshold_precision_30": _best_by_precision_or_recall(
            thresholds, minimum_precision=0.30, prefer="recall"
        ),
        "threshold_precision_50": _best_by_precision_or_recall(
            thresholds, minimum_precision=0.50, prefer="recall"
        ),
    }


def analyze_top_k(y_true: np.ndarray, y_prob: np.ndarray, baseline_death_rate: float) -> list[dict[str, Any]]:
    if len(y_true) == 0:
        return []

    order = np.argsort(-y_prob, kind="mergesort")
    ranked_true = y_true[order]
    total_deaths = int(np.sum(ranked_true == 1))
    total_rows = len(ranked_true)

    buckets: list[dict[str, Any]] = []
    for percent in TOP_K_BUCKETS:
        bucket_rows = max(1, int(math.ceil(total_rows * (percent / 100.0))))
        bucket_true = ranked_true[:bucket_rows]
        captured_deaths = int(np.sum(bucket_true == 1))
        death_rate = float(np.mean(bucket_true)) if bucket_rows else float("nan")
        lift = death_rate / baseline_death_rate if baseline_death_rate > 0 else None
        buckets.append(
            {
                "bucket": f"top_{percent}_percent",
                "bucket_percent": percent,
                "rows": bucket_rows,
                "death_rate": death_rate,
                "baseline_death_rate": baseline_death_rate,
                "lift": lift,
                "precision_equivalent": death_rate,
                "captured_deaths": captured_deaths,
                "captured_death_rate": float(captured_deaths / total_deaths) if total_deaths > 0 else 0.0,
            }
        )
    return buckets


def build_numeric_fill_values(train_df: pl.DataFrame, numeric_fill_strategy: dict[str, str], feature_columns: list[str], categorical_columns: list[str]) -> dict[str, float]:
    fill_values: dict[str, float] = {}
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    for column in numeric_columns:
        strategy = numeric_fill_strategy.get(column)
        if strategy is None:
            raise SystemExit(f"Missing numeric fill strategy for column: {column}")
        if strategy.startswith("zero"):
            fill_values[column] = 0.0
        else:
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
        elif column in BOOLEAN_NUMERIC_COLUMNS or dataset.schema.get(column) == pl.Boolean:
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
    pd: Any,
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


def assert_binary_target(split_name: str, target: np.ndarray) -> None:
    if len(np.unique(target)) != 2:
        raise SystemExit(f"{split_name} split target must contain both classes.")


def top_k_lookup(bucket_rows: list[dict[str, Any]], percent: int) -> dict[str, Any] | None:
    return next((row for row in bucket_rows if row["bucket_percent"] == percent), None)


def main() -> None:
    configure_logging()
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.metrics.exists():
        raise SystemExit(f"Metrics file not found: {args.metrics}")

    pd = load_pandas()
    lgb = load_lightgbm()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    feature_columns = metrics.get("feature_columns")
    categorical_columns = metrics.get("categorical_columns")
    numeric_fill_strategy = metrics.get("numeric_fill_strategy")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise SystemExit("Metrics JSON is missing feature_columns.")
    if not isinstance(categorical_columns, list):
        raise SystemExit("Metrics JSON is missing categorical_columns.")
    if not isinstance(numeric_fill_strategy, dict):
        raise SystemExit("Metrics JSON is missing numeric_fill_strategy.")

    dataset = pl.read_parquet(args.dataset)
    LOGGER.info("dataset_loaded=%s rows=%s columns=%s", args.dataset, dataset.height, len(dataset.columns))
    ensure_columns(dataset, [MATCH_ID_COLUMN, TARGET_COLUMN, *feature_columns])
    if dataset.is_empty():
        raise SystemExit(f"Dataset is empty: {args.dataset}")
    if dataset.select(pl.col(TARGET_COLUMN).is_null().any()).item():
        raise SystemExit(f"Target column contains null values: {TARGET_COLUMN}")

    unique_match_ids = dataset.select(pl.col(MATCH_ID_COLUMN).unique().sort()).to_series().to_list()
    split_ids = split_match_ids([str(match_id) for match_id in unique_match_ids], seed=args.seed)

    train_df = filter_by_match_ids(dataset, split_ids["train"])
    validation_df = filter_by_match_ids(dataset, split_ids["validation"])
    test_df = filter_by_match_ids(dataset, split_ids["test"])
    assert_disjoint_match_sets(split_ids)

    fill_values = build_numeric_fill_values(train_df, numeric_fill_strategy, feature_columns, categorical_columns)
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

    X_validation, y_validation = prepare_frame(
        validation_df,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        category_values=category_values,
        pd=pd,
    )
    X_test, y_test = prepare_frame(
        test_df,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        category_values=category_values,
        pd=pd,
    )

    assert_binary_target("validation", y_validation)
    assert_binary_target("test", y_test)

    booster = lgb.Booster(model_file=str(args.model))
    validation_probabilities = predict_probability(booster, X_validation)
    test_probabilities = predict_probability(booster, X_test)

    validation_thresholds = sweep_thresholds(y_validation, validation_probabilities)
    test_thresholds = sweep_thresholds(y_test, test_probabilities)
    recommended = recommend_thresholds(validation_thresholds)

    validation_baseline = positive_rate(y_validation)
    test_baseline = positive_rate(y_test)
    validation_top_k = analyze_top_k(y_validation, validation_probabilities, validation_baseline)
    test_top_k = analyze_top_k(y_test, test_probabilities, test_baseline)

    test_thresholds_at_recommended: dict[str, Any] = {}
    for name, recommendation in recommended.items():
        if recommendation is None:
            test_thresholds_at_recommended[name] = None
            continue
        threshold = float(recommendation["threshold"])
        matching = next((row for row in test_thresholds if row["threshold"] == threshold), None)
        test_thresholds_at_recommended[name] = matching

    output_payload = {
        "validation": {
            "baseline_death_rate": validation_baseline,
            "thresholds": validation_thresholds,
            "recommended_thresholds": recommended,
            "top_k": validation_top_k,
        },
        "test": {
            "baseline_death_rate": test_baseline,
            "thresholds_at_recommended_validation_thresholds": test_thresholds_at_recommended,
            "top_k": test_top_k,
        },
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, output_payload)

    best_f1 = recommended.get("best_f1_threshold")
    test_at_best_f1 = test_thresholds_at_recommended.get("best_f1_threshold")
    top_5_test = top_k_lookup(test_top_k, 5)
    top_10_test = top_k_lookup(test_top_k, 10)

    LOGGER.info("validation baseline death rate=%.6f", validation_baseline)
    LOGGER.info("test baseline death rate=%.6f", test_baseline)
    if best_f1 is not None and test_at_best_f1 is not None:
        LOGGER.info(
            "best_f1_threshold=%.2f test_precision=%.6f test_recall=%.6f test_f1=%.6f test_accuracy=%.6f",
            best_f1["threshold"],
            test_at_best_f1["precision"],
            test_at_best_f1["recall"],
            test_at_best_f1["f1"],
            test_at_best_f1["accuracy"],
        )
    if top_5_test is not None:
        LOGGER.info(
            "top_5_percent_test_death_rate=%.6f lift=%.6f",
            top_5_test["death_rate"],
            top_5_test["lift"] if top_5_test["lift"] is not None else float("nan"),
        )
    if top_10_test is not None:
        LOGGER.info(
            "top_10_percent_test_death_rate=%.6f lift=%.6f",
            top_10_test["death_rate"],
            top_10_test["lift"] if top_10_test["lift"] is not None else float("nan"),
        )
    LOGGER.info("saved_output=%s", args.output)


if __name__ == "__main__":
    main()
