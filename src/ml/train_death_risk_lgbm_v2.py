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
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "ml" / "decision_snapshots.parquet"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "death_risk_5s_lgbm_v2.txt"
DEFAULT_METRICS_PATH = REPO_ROOT / "data" / "ml" / "death_risk_5s_metrics_v2.json"
DEFAULT_FEATURE_IMPORTANCE_PATH = (
    REPO_ROOT / "data" / "ml" / "death_risk_5s_feature_importance_v2.json"
)

TARGET_COLUMN = "death_within_5s"
MATCH_ID_COLUMN = "match_id"
REQUIRED_FEATURE_COLUMNS = [
    "alive_team_at_snapshot",
    "alive_enemy_at_snapshot",
    "seconds_remaining_at_snapshot",
    "bomb_planted_at_snapshot",
    "player_side",
    "map_name",
]
OPTIONAL_FEATURE_COLUMNS = ["prior_round_phase"]
CATEGORICAL_BASE_COLUMNS = ["player_side", "map_name"]
CATEGORICAL_OPTIONAL_COLUMNS = ["prior_round_phase"]
FORBIDDEN_FEATURE_COLUMNS = {
    "source_situation_type",
    "seconds_before_event",
    "event_weapon",
    "weapon",
    "is_awp_event",
    "is_rifle_event",
    "is_pistol_event",
    "was_opening_context",
    "ml_impact_at_event",
    "action_value_class",
    "death_within_5s",
    "kill_within_5s",
    "opening_duel_within_5s",
    "opening_duel_won_within_5s",
    "high_cost_death_within_5s",
    "high_impact_kill_within_5s",
    "source_situation_id",
    "event_tick",
    "snapshot_tick",
}
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an anti-leakage v2 LightGBM death risk model.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to decision snapshots parquet (default: {DEFAULT_DATASET_PATH})",
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
        help=f"Where to save feature importance JSON (default: {DEFAULT_FEATURE_IMPORTANCE_PATH})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split/training seed.")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=1000,
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


def ensure_columns(dataset: pl.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise SystemExit(f"Dataset is missing required columns: {missing_columns}")


def select_feature_columns(dataset: pl.DataFrame) -> tuple[list[str], list[str]]:
    feature_columns = list(REQUIRED_FEATURE_COLUMNS)
    feature_columns.extend(
        column for column in OPTIONAL_FEATURE_COLUMNS if column in dataset.columns
    )
    categorical_columns = list(CATEGORICAL_BASE_COLUMNS)
    categorical_columns.extend(
        column for column in CATEGORICAL_OPTIONAL_COLUMNS if column in feature_columns
    )

    forbidden_selected = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(feature_columns))
    if forbidden_selected:
        raise AssertionError(f"Forbidden feature columns selected: {forbidden_selected}")

    ensure_columns(dataset, [MATCH_ID_COLUMN, TARGET_COLUMN, *feature_columns])
    return feature_columns, categorical_columns


def split_match_ids(match_ids: list[str], seed: int) -> dict[str, list[str]]:
    if len(match_ids) < 3:
        raise SystemExit(
            f"Need at least 3 unique matches for train/validation/test splitting, found {len(match_ids)}."
        )

    shuffled = list(match_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * 0.70)
    validation_end = train_end + int(total * 0.15)

    split_ids = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    if any(not ids for ids in split_ids.values()):
        raise SystemExit("Match split produced an empty partition.")

    assert_disjoint_match_sets(split_ids)
    return split_ids


def assert_disjoint_match_sets(split_ids: dict[str, list[str]]) -> None:
    train = set(split_ids["train"])
    validation = set(split_ids["validation"])
    test = set(split_ids["test"])
    if train & validation or train & test or validation & test:
        raise AssertionError("Train/validation/test match sets overlap.")


def assert_binary_target(split_name: str, target: np.ndarray) -> None:
    if len(np.unique(target)) != 2:
        raise SystemExit(f"{split_name} split target must contain both classes.")


def filter_by_match_ids(dataset: pl.DataFrame, match_ids: list[str]) -> pl.DataFrame:
    return dataset.filter(pl.col(MATCH_ID_COLUMN).is_in(match_ids))


def normalize_dataset(
    dataset: pl.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> pl.DataFrame:
    dataset = dataset.select([MATCH_ID_COLUMN, TARGET_COLUMN, *feature_columns])

    if dataset.select(pl.col(TARGET_COLUMN).is_null().any()).item():
        raise SystemExit(f"Target column contains null values: {TARGET_COLUMN}")

    expressions: list[pl.Expr] = [
        pl.col(MATCH_ID_COLUMN).cast(pl.Utf8),
        pl.col(TARGET_COLUMN).cast(pl.Int8),
    ]
    for column in feature_columns:
        if column in categorical_columns:
            expressions.append(pl.col(column).fill_null("unknown").cast(pl.Utf8))
        else:
            expressions.append(pl.col(column))

    return dataset.with_columns(expressions)


def build_category_values(
    dataset: pl.DataFrame,
    categorical_columns: list[str],
) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {}
    for column in categorical_columns:
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
    feature_columns: list[str],
    categorical_columns: list[str],
    category_values: dict[str, list[str]],
) -> tuple[Any, np.ndarray]:
    feature_frame = dataset.select(feature_columns).to_pandas()
    for column in categorical_columns:
        feature_frame[column] = (
            feature_frame[column]
            .fillna("unknown")
            .astype(str)
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


def accuracy_score_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    predictions = (y_prob >= 0.5).astype(np.int32)
    return float(np.mean(predictions == y_true))


def positive_rate(y_true: np.ndarray) -> float:
    return float(np.mean(y_true))


def build_params(args: argparse.Namespace) -> dict[str, Any]:
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
    categorical_columns: list[str],
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
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
        categorical_feature=categorical_columns,
    )
    return model


def predict_probability(model: Any, X: Any) -> np.ndarray:
    probabilities = model.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise RuntimeError("Expected binary predict_proba output with two columns.")
    return probabilities[:, 1].astype(float, copy=False)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": log_loss_score(y_true, y_prob),
        "roc_auc": roc_auc_score_binary(y_true, y_prob),
        "accuracy_at_0_5": accuracy_score_binary(y_true, y_prob),
    }


def build_feature_importance_payload(
    model: Any,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    importance_split = model.booster_.feature_importance(importance_type="split")
    importance_gain = model.booster_.feature_importance(importance_type="gain")

    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(feature_columns):
        rows.append(
            {
                "feature": feature,
                "importance_gain": float(importance_gain[index]),
                "importance_split": int(importance_split[index]),
            }
        )

    rows.sort(key=lambda item: (item["importance_gain"], item["importance_split"]), reverse=True)
    return rows


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


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def log_split_summary(
    label: str,
    dataset: pl.DataFrame,
    match_ids: list[str],
    target: np.ndarray,
) -> None:
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

    feature_columns, categorical_columns = select_feature_columns(dataset)
    dataset = normalize_dataset(dataset, feature_columns, categorical_columns)
    category_values = build_category_values(dataset, categorical_columns)

    unique_match_ids = dataset.select(pl.col(MATCH_ID_COLUMN).unique().sort()).to_series().to_list()
    split_ids = split_match_ids([str(match_id) for match_id in unique_match_ids], seed=args.seed)

    train_df = filter_by_match_ids(dataset, split_ids["train"])
    validation_df = filter_by_match_ids(dataset, split_ids["validation"])
    test_df = filter_by_match_ids(dataset, split_ids["test"])

    X_train, y_train = prepare_frame(
        train_df, feature_columns, categorical_columns, category_values
    )
    X_validation, y_validation = prepare_frame(
        validation_df, feature_columns, categorical_columns, category_values
    )
    X_test, y_test = prepare_frame(test_df, feature_columns, categorical_columns, category_values)

    assert_binary_target("train", y_train)
    assert_binary_target("validation", y_validation)
    assert_binary_target("test", y_test)

    log_split_summary("train", train_df, split_ids["train"], y_train)
    log_split_summary("validation", validation_df, split_ids["validation"], y_validation)
    log_split_summary("test", test_df, split_ids["test"], y_test)
    LOGGER.info("feature_columns=%s", feature_columns)
    LOGGER.info("categorical_columns=%s", categorical_columns)

    model = train_model(
        lgb=lgb,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        categorical_columns=categorical_columns,
        params=build_params(args),
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
        "validation_log_loss": validation_metrics["log_loss"],
        "validation_roc_auc": validation_metrics["roc_auc"],
        "validation_accuracy_at_0_5": validation_metrics["accuracy_at_0_5"],
        "test_log_loss": test_metrics["log_loss"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_accuracy_at_0_5": test_metrics["accuracy_at_0_5"],
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "forbidden_columns_checked": sorted(FORBIDDEN_FEATURE_COLUMNS),
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.model_out))
    save_json(args.metrics_out, metrics_payload)
    save_json(args.feature_importance_out, build_feature_importance_payload(model, feature_columns))

    LOGGER.info("metrics=%s", sanitize_for_json(metrics_payload))
    LOGGER.info("saved_model=%s", args.model_out)
    LOGGER.info("saved_metrics=%s", args.metrics_out)
    LOGGER.info("saved_feature_importance=%s", args.feature_importance_out)


if __name__ == "__main__":
    main()
