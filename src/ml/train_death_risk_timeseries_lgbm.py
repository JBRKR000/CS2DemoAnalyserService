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
DEFAULT_FEATURE_IMPORTANCE_PATH = (
    REPO_ROOT / "data" / "ml" / "death_risk_timeseries_5s_feature_importance.json"
)

TARGET_COLUMN = "death_within_5s"
MATCH_ID_COLUMN = "match_id"
FEATURE_COLUMNS = [
    "alive_team_at_snapshot",
    "alive_enemy_at_snapshot",
    "seconds_remaining_at_snapshot",
    "bomb_planted_at_snapshot",
    "player_hp",
    "side",
    "map_name",
    "weapon",
    "has_armor",
    "equipment_value",
    "nearest_teammate_distance",
    "nearest_enemy_distance",
    "prior_round_phase",
]
CATEGORICAL_COLUMNS = ["side", "map_name", "weapon", "prior_round_phase"]
FORBIDDEN_FEATURE_COLUMNS = {
    "death_within_5s",
    "kill_within_5s",
    "damage_dealt_next_5s",
    "damage_taken_next_5s",
    "match_id",
    "round_num",
    "tick",
    "steamid",
    "player_name",
    "player_alive",
    "sample_time_seconds",
    "build_version",
    "has_helmet",
    "money",
}
NULL_FILL_ZERO_COLUMNS = {
    "bomb_planted_at_snapshot",
    "has_armor",
}
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LightGBM model for time-sampled death risk within 5 seconds.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to time-sampled death risk parquet (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Where to save the trained LightGBM model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Where to save metrics JSON (default: {DEFAULT_METRICS_PATH})",
    )
    parser.add_argument(
        "--feature-importance-out",
        type=Path,
        default=DEFAULT_FEATURE_IMPORTANCE_PATH,
        help=(
            f"Where to save feature importance JSON "
            f"(default: {DEFAULT_FEATURE_IMPORTANCE_PATH})"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split/training seed.")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=2000,
        help="Maximum number of boosting rounds before early stopping.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    return parser.parse_args()


def load_lightgbm() -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit(
            "lightgbm is required to run this script. Install it in the active environment first."
        ) from exc
    return lgb


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


def ensure_columns(dataset: pl.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise SystemExit(f"Dataset is missing required columns: {missing_columns}")


def assert_no_forbidden_columns(feature_columns: list[str]) -> list[str]:
    forbidden_selected = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(feature_columns))
    if forbidden_selected:
        raise AssertionError(f"Forbidden feature columns selected: {forbidden_selected}")
    return sorted(FORBIDDEN_FEATURE_COLUMNS)


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


def cast_dataset(dataset: pl.DataFrame) -> pl.DataFrame:
    cast_expressions: list[pl.Expr] = [
        pl.col(MATCH_ID_COLUMN).cast(pl.Utf8),
        pl.col(TARGET_COLUMN).cast(pl.Int8),
    ]
    for column in FEATURE_COLUMNS:
        if column in CATEGORICAL_COLUMNS:
            cast_expressions.append(pl.col(column).fill_null("unknown").cast(pl.Utf8))
        else:
            cast_expressions.append(pl.col(column))
    return dataset.select([MATCH_ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]).with_columns(
        cast_expressions
    )


def build_numeric_fill_values(train_df: pl.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    fill_values: dict[str, float] = {}
    strategies: dict[str, str] = {}
    numeric_columns = [column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]
    for column in numeric_columns:
        null_count = int(train_df.select(pl.col(column).is_null().sum()).item())
        if column in NULL_FILL_ZERO_COLUMNS:
            fill_values[column] = 0.0
            strategies[column] = "zero" if null_count > 0 else "zero_no_nulls"
            continue

        median_value = train_df.select(pl.col(column).median()).item()
        if median_value is None:
            fill_values[column] = 0.0
            strategies[column] = "zero_fallback_all_null"
        else:
            fill_values[column] = float(median_value)
            strategies[column] = "median" if null_count > 0 else "median_no_nulls"

    return fill_values, strategies


def apply_numeric_fill_values(dataset: pl.DataFrame, fill_values: dict[str, float]) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for column in FEATURE_COLUMNS:
        if column in CATEGORICAL_COLUMNS:
            expressions.append(pl.col(column).fill_null("unknown").cast(pl.Utf8))
        elif column in NULL_FILL_ZERO_COLUMNS:
            expressions.append(pl.col(column).fill_null(False).cast(pl.Float64))
        else:
            expressions.append(pl.col(column).fill_null(fill_values[column]).cast(pl.Float64))
    return dataset.with_columns(expressions)


def build_category_values(dataset: pl.DataFrame) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for column in CATEGORICAL_COLUMNS:
        values = (
            dataset.select(pl.col(column).fill_null("unknown").cast(pl.Utf8).unique().sort())
            .to_series()
            .to_list()
        )
        if "unknown" not in values:
            values.append("unknown")
        categories[column] = [str(value) for value in values]
    return categories


def prepare_frame(
    dataset: pl.DataFrame,
    category_values: dict[str, list[str]],
) -> tuple[Any, np.ndarray]:
    feature_frame = dataset.select(FEATURE_COLUMNS).to_pandas()
    for column in CATEGORICAL_COLUMNS:
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


def positive_rate(y_true: np.ndarray) -> float:
    return float(np.mean(y_true))


def assert_binary_target(split_name: str, target: np.ndarray) -> None:
    if len(np.unique(target)) != 2:
        raise SystemExit(f"{split_name} split target must contain both classes.")


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    positive_count = int(np.sum(y_train == 1))
    negative_count = int(np.sum(y_train == 0))
    if positive_count == 0 or negative_count == 0:
        raise SystemExit("Training split target must contain both classes.")
    return float(negative_count / positive_count)


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


def pr_auc_score_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return float("nan")

    order = np.argsort(-y_prob, kind="mergesort")
    sorted_true = y_true[order]
    true_positives = np.cumsum(sorted_true == 1)
    false_positives = np.cumsum(sorted_true == 0)
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / positives

    precision = np.concatenate(([1.0], precision.astype(float, copy=False)))
    recall = np.concatenate(([0.0], recall.astype(float, copy=False)))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = (y_prob >= threshold).astype(np.int32)
    true_positive = int(np.sum((predictions == 1) & (y_true == 1)))
    true_negative = int(np.sum((predictions == 0) & (y_true == 0)))
    false_positive = int(np.sum((predictions == 1) & (y_true == 0)))
    false_negative = int(np.sum((predictions == 0) & (y_true == 1)))

    total = len(y_true)
    accuracy = (true_positive + true_negative) / total if total else float("nan")
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    if precision + recall == 0.0:
        f1_score = 0.0
    else:
        f1_score = 2.0 * precision * recall / (precision + recall)

    return {
        "accuracy_at_0_5": float(accuracy),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1_score),
    }


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    metrics = {
        "log_loss": log_loss_score(y_true, y_prob),
        "roc_auc": roc_auc_score_binary(y_true, y_prob),
        "pr_auc": pr_auc_score_binary(y_true, y_prob),
    }
    metrics.update(threshold_metrics(y_true, y_prob, threshold=0.5))
    return metrics


def build_params(args: argparse.Namespace, scale_pos_weight: float) -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": args.seed,
        "feature_fraction_seed": args.seed,
        "bagging_seed": args.seed,
        "data_random_seed": args.seed,
        "num_threads": 0,
    }


def train_model(
    lgb: Any,
    X_train: Any,
    y_train: np.ndarray,
    X_validation: Any,
    y_validation: np.ndarray,
    params: dict[str, Any],
    n_estimators: int,
) -> Any:
    model = lgb.LGBMClassifier(**params, n_estimators=n_estimators)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ],
        categorical_feature=CATEGORICAL_COLUMNS,
    )
    return model


def predict_probability(model: Any, X: Any) -> np.ndarray:
    probabilities = model.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise RuntimeError("Expected binary predict_proba output with two columns.")
    return probabilities[:, 1].astype(float, copy=False)


def build_feature_importance_payload(model: Any) -> list[dict[str, Any]]:
    importance_split = model.booster_.feature_importance(importance_type="split")
    importance_gain = model.booster_.feature_importance(importance_type="gain")

    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(FEATURE_COLUMNS):
        rows.append(
            {
                "feature": feature,
                "importance_gain": float(importance_gain[index]),
                "importance_split": int(importance_split[index]),
            }
        )
    rows.sort(key=lambda item: (item["importance_gain"], item["importance_split"]), reverse=True)
    return rows


def log_split_summary(label: str, dataset: pl.DataFrame, match_ids: list[str], target: np.ndarray) -> None:
    LOGGER.info("%s_rows=%s", label, dataset.height)
    LOGGER.info("%s_matches=%s", label, len(match_ids))
    LOGGER.info("%s_positive_rate=%.6f", label, positive_rate(target))


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.n_estimators <= 0:
        raise SystemExit("--n-estimators must be > 0.")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be > 0.")
    if args.num_leaves <= 1:
        raise SystemExit("--num-leaves must be > 1.")
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    lgb = load_lightgbm()

    dataset = pl.read_parquet(args.dataset)
    LOGGER.info("dataset_loaded=%s rows=%s columns=%s", args.dataset, dataset.height, len(dataset.columns))
    if dataset.is_empty():
        raise SystemExit(f"Dataset is empty: {args.dataset}")

    if TARGET_COLUMN not in dataset.columns:
        raise AssertionError(f"Target column missing: {TARGET_COLUMN}")
    if MATCH_ID_COLUMN not in dataset.columns:
        raise AssertionError(f"Match id column missing: {MATCH_ID_COLUMN}")

    ensure_columns(dataset, [MATCH_ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS])
    forbidden_columns_checked = assert_no_forbidden_columns(FEATURE_COLUMNS)

    if dataset.select(pl.col(TARGET_COLUMN).is_null().any()).item():
        raise SystemExit(f"Target column contains null values: {TARGET_COLUMN}")

    dataset = cast_dataset(dataset)
    unique_match_ids = dataset.select(pl.col(MATCH_ID_COLUMN).unique().sort()).to_series().to_list()
    split_ids = split_match_ids([str(match_id) for match_id in unique_match_ids], seed=args.seed)

    train_df = filter_by_match_ids(dataset, split_ids["train"])
    validation_df = filter_by_match_ids(dataset, split_ids["validation"])
    test_df = filter_by_match_ids(dataset, split_ids["test"])

    fill_values, numeric_fill_strategy = build_numeric_fill_values(train_df)
    LOGGER.info("numeric_fill_strategy=%s", sanitize_for_json(numeric_fill_strategy))
    train_df = apply_numeric_fill_values(train_df, fill_values)
    validation_df = apply_numeric_fill_values(validation_df, fill_values)
    test_df = apply_numeric_fill_values(test_df, fill_values)

    category_values = build_category_values(train_df)
    X_train, y_train = prepare_frame(train_df, category_values)
    X_validation, y_validation = prepare_frame(validation_df, category_values)
    X_test, y_test = prepare_frame(test_df, category_values)

    assert_binary_target("train", y_train)
    assert_binary_target("validation", y_validation)
    assert_binary_target("test", y_test)

    scale_pos_weight = compute_scale_pos_weight(y_train)

    LOGGER.info(
        "split_sizes train_rows=%s validation_rows=%s test_rows=%s train_matches=%s validation_matches=%s test_matches=%s",
        train_df.height,
        validation_df.height,
        test_df.height,
        len(split_ids["train"]),
        len(split_ids["validation"]),
        len(split_ids["test"]),
    )
    log_split_summary("train", train_df, split_ids["train"], y_train)
    log_split_summary("validation", validation_df, split_ids["validation"], y_validation)
    log_split_summary("test", test_df, split_ids["test"], y_test)
    LOGGER.info("scale_pos_weight=%.6f", scale_pos_weight)

    model = train_model(
        lgb=lgb,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        params=build_params(args, scale_pos_weight),
        n_estimators=args.n_estimators,
    )

    validation_probabilities = predict_probability(model, X_validation)
    test_probabilities = predict_probability(model, X_test)
    validation_metrics = evaluate_predictions(y_validation, validation_probabilities)
    test_metrics = evaluate_predictions(y_test, test_probabilities)

    metrics_payload = {
        "train_rows": train_df.height,
        "validation_rows": validation_df.height,
        "test_rows": test_df.height,
        "train_matches": len(split_ids["train"]),
        "validation_matches": len(split_ids["validation"]),
        "test_matches": len(split_ids["test"]),
        "positive_rate_train": positive_rate(y_train),
        "positive_rate_validation": positive_rate(y_validation),
        "positive_rate_test": positive_rate(y_test),
        "scale_pos_weight": scale_pos_weight,
        "validation_log_loss": validation_metrics["log_loss"],
        "validation_roc_auc": validation_metrics["roc_auc"],
        "validation_pr_auc": validation_metrics["pr_auc"],
        "validation_accuracy_at_0_5": validation_metrics["accuracy_at_0_5"],
        "validation_precision_at_0_5": validation_metrics["precision_at_0_5"],
        "validation_recall_at_0_5": validation_metrics["recall_at_0_5"],
        "validation_f1_at_0_5": validation_metrics["f1_at_0_5"],
        "test_log_loss": test_metrics["log_loss"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_accuracy_at_0_5": test_metrics["accuracy_at_0_5"],
        "test_precision_at_0_5": test_metrics["precision_at_0_5"],
        "test_recall_at_0_5": test_metrics["recall_at_0_5"],
        "test_f1_at_0_5": test_metrics["f1_at_0_5"],
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "forbidden_columns_checked": forbidden_columns_checked,
        "numeric_fill_strategy": numeric_fill_strategy,
        "seed": args.seed,
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.model_out))
    save_json(args.metrics_out, metrics_payload)
    save_json(args.feature_importance_out, build_feature_importance_payload(model))

    LOGGER.info("metrics=%s", sanitize_for_json(metrics_payload))
    LOGGER.info("saved_model=%s", args.model_out)
    LOGGER.info("saved_metrics=%s", args.metrics_out)
    LOGGER.info("saved_feature_importance=%s", args.feature_importance_out)


if __name__ == "__main__":
    main()
