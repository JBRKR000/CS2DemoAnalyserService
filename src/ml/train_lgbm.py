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
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "ml" / "round_snapshots.parquet"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "round_win_lgbm.txt"
DEFAULT_METRICS_PATH = REPO_ROOT / "data" / "ml" / "round_win_lgbm_metrics.json"
DEFAULT_FEATURE_IMPORTANCE_PATH = REPO_ROOT / "data" / "ml" / "round_win_lgbm_feature_importance.json"

TARGET_COLUMN = "team_won_round"
MATCH_ID_COLUMN = "match_id"
FEATURE_COLUMNS = [
    "alive_team",
    "alive_enemy",
    "seconds_remaining",
    "bomb_planted",
    "opening_kill_for_team",
    "side",
    "snapshot_type",
    "map_name",
]
CATEGORICAL_COLUMNS = ["side", "snapshot_type", "map_name"]
THRESHOLDS = [round(value, 2) for value in np.arange(0.35, 0.651, 0.05)]
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline LightGBM model for CS2 round win probability snapshots.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the round snapshot parquet dataset (default: {DEFAULT_DATASET_PATH})",
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
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=1000,
        help="Maximum number of boosting rounds before early stopping.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=0,
        help="How many random LightGBM hyperparameter trials to run before final training.",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Optional row-level sampling fraction for the train/validation splits during search only.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for match split, sampling, and parameter search.",
    )
    return parser.parse_args()


def load_libraries() -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit(
            "lightgbm is required to run this script. Install it in the active environment first."
        ) from exc

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required to convert Polars frames into LightGBM input. Install it in the active environment first."
        ) from exc

    return lgb


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
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    train_ids = shuffled[:train_end]
    val_ids = shuffled[train_end:val_end]
    test_ids = shuffled[val_end:]

    if not train_ids or not val_ids or not test_ids:
        raise SystemExit(
            "Match split produced an empty partition. Add more matches or adjust split logic."
        )

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def filter_by_match_ids(dataset: pl.DataFrame, match_ids: list[str]) -> pl.DataFrame:
    return dataset.filter(pl.col(MATCH_ID_COLUMN).is_in(match_ids))


def sample_rows(dataset: pl.DataFrame, fraction: float, seed: int) -> pl.DataFrame:
    if fraction >= 1.0 or dataset.is_empty():
        return dataset
    sampled = dataset.sample(fraction=fraction, with_replacement=False, shuffle=True, seed=seed)
    return sampled if not sampled.is_empty() else dataset


def prepare_frame(dataset: pl.DataFrame) -> tuple[Any, np.ndarray]:
    feature_frame = dataset.select(FEATURE_COLUMNS).to_pandas()
    for column in CATEGORICAL_COLUMNS:
        feature_frame[column] = feature_frame[column].astype("category")

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


def accuracy_score_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    predictions = (y_prob >= threshold).astype(np.int32)
    return float(np.mean(predictions == y_true))


def f1_score_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    predictions = (y_prob >= threshold).astype(np.int32)
    true_positive = int(np.sum((predictions == 1) & (y_true == 1)))
    false_positive = int(np.sum((predictions == 1) & (y_true == 0)))
    false_negative = int(np.sum((predictions == 0) & (y_true == 1)))

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    if precision_denominator == 0 or recall_denominator == 0:
        return 0.0

    precision = true_positive / precision_denominator
    recall = true_positive / recall_denominator
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return float(2 * precision * recall / denominator)


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    return {
        "log_loss": log_loss_score(y_true, y_prob),
        "roc_auc": roc_auc_score_binary(y_true, y_prob),
        "accuracy_at_0_5": accuracy_score_binary(y_true, y_prob, threshold=threshold),
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    results: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        results.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score_binary(y_true, y_prob, threshold),
                "f1": f1_score_binary(y_true, y_prob, threshold),
            }
        )

    best_by_accuracy = max(results, key=lambda item: (item["accuracy"], item["f1"], -abs(item["threshold"] - 0.5)))
    best_by_f1 = max(results, key=lambda item: (item["f1"], item["accuracy"], -abs(item["threshold"] - 0.5)))

    return {
        "thresholds": results,
        "best_accuracy": best_by_accuracy,
        "best_f1": best_by_f1,
    }


def baseline_params(seed: int) -> dict[str, Any]:
    # Log loss and calibration matter more than raw accuracy here because the
    # model is intended for round win probabilities and downstream probability deltas.
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "min_gain_to_split": 0.0,
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "num_threads": 0,
    }


def sample_search_params(rng: random.Random, seed: int) -> dict[str, Any]:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": rng.choice([0.01, 0.02, 0.03, 0.05, 0.08, 0.1]),
        "num_leaves": rng.choice([7, 15, 31, 47, 63, 95, 127]),
        "max_depth": rng.choice([-1, 3, 4, 5, 6, 8]),
        "min_data_in_leaf": rng.choice([20, 30, 50, 75, 100, 150, 200]),
        "feature_fraction": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        "bagging_fraction": rng.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
        "bagging_freq": rng.choice([0, 1, 3, 5]),
        "lambda_l1": rng.choice([0.0, 0.1, 0.5, 1.0, 2.0]),
        "lambda_l2": rng.choice([0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
        "min_gain_to_split": rng.choice([0.0, 0.01, 0.05, 0.1]),
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "num_threads": 0,
    }


def train_model(
    lgb: Any,
    X_train: Any,
    y_train: np.ndarray,
    X_eval: Any,
    y_eval: np.ndarray,
    params: dict[str, Any],
    n_estimators: int,
) -> Any:
    model = lgb.LGBMClassifier(
        **params,
        n_estimators=n_estimators,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_eval, y_eval)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
        categorical_feature=CATEGORICAL_COLUMNS,
    )
    return model


def train_final_model(
    lgb: Any,
    X_train: Any,
    y_train: np.ndarray,
    params: dict[str, Any],
    n_estimators: int,
) -> Any:
    model = lgb.LGBMClassifier(
        **params,
        n_estimators=n_estimators,
    )
    model.fit(
        X_train,
        y_train,
        categorical_feature=CATEGORICAL_COLUMNS,
    )
    return model


def predict_probability(model: Any, X: Any) -> np.ndarray:
    probabilities = model.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise RuntimeError("Expected binary predict_proba output with two columns.")
    return probabilities[:, 1].astype(float, copy=False)


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_for_json(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_feature_importance_payload(model: Any) -> list[dict[str, Any]]:
    split_importance = model.booster_.feature_importance(importance_type="split")
    gain_importance = model.booster_.feature_importance(importance_type="gain")

    rows: list[dict[str, Any]] = []
    for index, feature_name in enumerate(FEATURE_COLUMNS):
        rows.append(
            {
                "feature": feature_name,
                "split_importance": int(split_importance[index]),
                "gain_importance": float(gain_importance[index]),
            }
        )

    rows.sort(key=lambda item: (item["gain_importance"], item["split_importance"]), reverse=True)
    return rows


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )


def log_split_summary(label: str, dataset: pl.DataFrame, match_ids: list[str]) -> None:
    LOGGER.info("%s_rows=%s", label, dataset.height)
    LOGGER.info("%s_matches=%s", label, len(match_ids))


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.n_estimators <= 0:
        raise SystemExit("--n-estimators must be > 0.")
    if args.n_trials < 0:
        raise SystemExit("--n-trials must be >= 0.")
    if not 0 < args.sample_frac <= 1.0:
        raise SystemExit("--sample-frac must be in the range (0, 1].")
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    lgb = load_libraries()

    dataset = pl.read_parquet(args.dataset)
    ensure_columns(dataset, [MATCH_ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS])
    if dataset.is_empty():
        raise SystemExit(f"Dataset is empty: {args.dataset}")

    dataset = dataset.select([MATCH_ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS])
    dataset = dataset.with_columns(
        [
            pl.col("alive_team").cast(pl.Int64),
            pl.col("alive_enemy").cast(pl.Int64),
            pl.col("seconds_remaining").cast(pl.Float64),
            pl.col("bomb_planted").cast(pl.Boolean),
            pl.col("opening_kill_for_team").cast(pl.Boolean),
            pl.col("side").cast(pl.Utf8),
            pl.col("snapshot_type").cast(pl.Utf8),
            pl.col("map_name").fill_null("unknown").cast(pl.Utf8),
            pl.col(TARGET_COLUMN).cast(pl.Boolean),
        ]
    )

    unique_match_ids = dataset.select(pl.col(MATCH_ID_COLUMN).unique().sort()).to_series().to_list()
    split_ids = split_match_ids(unique_match_ids, seed=args.seed)

    train_df = filter_by_match_ids(dataset, split_ids["train"])
    val_df = filter_by_match_ids(dataset, split_ids["val"])
    test_df = filter_by_match_ids(dataset, split_ids["test"])

    log_split_summary("train", train_df, split_ids["train"])
    log_split_summary("val", val_df, split_ids["val"])
    log_split_summary("test", test_df, split_ids["test"])

    X_train, y_train = prepare_frame(train_df)
    X_val, y_val = prepare_frame(val_df)
    X_test, y_test = prepare_frame(test_df)

    search_history: list[dict[str, Any]] = []
    selected_params = baseline_params(seed=args.seed)
    best_trial_result: dict[str, Any] | None = None

    if args.n_trials > 0:
        rng = random.Random(args.seed)
        search_train_df = sample_rows(train_df, fraction=args.sample_frac, seed=args.seed)
        search_val_df = sample_rows(val_df, fraction=args.sample_frac, seed=args.seed + 1)

        X_search_train, y_search_train = prepare_frame(search_train_df)
        X_search_val, y_search_val = prepare_frame(search_val_df)

        best_result: dict[str, Any] | None = None
        for trial_index in range(args.n_trials):
            params = sample_search_params(rng=rng, seed=args.seed + trial_index)
            model = train_model(
                lgb=lgb,
                X_train=X_search_train,
                y_train=y_search_train,
                X_eval=X_search_val,
                y_eval=y_search_val,
                params=params,
                n_estimators=args.n_estimators,
            )
            val_probabilities = predict_probability(model, X_search_val)
            trial_result = {
                "trial": trial_index + 1,
                "params": params,
                "validation_log_loss": log_loss_score(y_search_val, val_probabilities),
                "validation_roc_auc": roc_auc_score_binary(y_search_val, val_probabilities),
                "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            }
            search_history.append(trial_result)
            LOGGER.info(
                "trial=%s val_log_loss=%.6f val_roc_auc=%.6f",
                trial_result["trial"],
                trial_result["validation_log_loss"],
                trial_result["validation_roc_auc"],
            )

            if best_result is None or trial_result["validation_log_loss"] < best_result["validation_log_loss"]:
                best_result = trial_result

        if best_result is not None:
            selected_params = dict(best_result["params"])
            best_trial_result = dict(best_result)

    search_metrics: dict[str, Any] = {
        "n_trials": args.n_trials,
        "n_estimators": args.n_estimators,
        "sample_frac": args.sample_frac,
        "selection_metric": "validation_log_loss",
        "selected_params": selected_params,
        "best_trial": best_trial_result["trial"] if best_trial_result is not None else None,
        "best_validation_log_loss": best_trial_result["validation_log_loss"] if best_trial_result is not None else None,
        "best_validation_roc_auc": best_trial_result["validation_roc_auc"] if best_trial_result is not None else None,
        "best_params": best_trial_result["params"] if best_trial_result is not None else selected_params,
        "search_history": search_history,
    }

    baseline_val_model = train_model(
        lgb=lgb,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_val,
        y_eval=y_val,
        params=selected_params,
        n_estimators=args.n_estimators,
    )
    baseline_val_probabilities = predict_probability(baseline_val_model, X_val)
    validation_metrics = evaluate_predictions(y_val, baseline_val_probabilities)
    validation_thresholds = threshold_sweep(y_val, baseline_val_probabilities)

    train_val_df = pl.concat([train_df, val_df], how="vertical_relaxed")
    X_train_val, y_train_val = prepare_frame(train_val_df)
    final_n_estimators = int(
        getattr(baseline_val_model, "best_iteration_", 0)
        or getattr(baseline_val_model, "n_estimators_", 0)
        or args.n_estimators
    )

    final_model = train_final_model(
        lgb=lgb,
        X_train=X_train_val,
        y_train=y_train_val,
        params=selected_params,
        n_estimators=final_n_estimators,
    )
    test_probabilities = predict_probability(final_model, X_test)
    test_metrics = evaluate_predictions(y_test, test_probabilities)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    final_model.booster_.save_model(str(args.model_out))

    feature_importance = build_feature_importance_payload(final_model)
    save_json(args.feature_importance_out, {"feature_importance": feature_importance})

    metrics_payload = {
        "dataset_path": args.dataset,
        "model_path": args.model_out,
        "feature_importance_path": args.feature_importance_out,
        "seed": args.seed,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "split_summary": {
            "train_rows": train_df.height,
            "val_rows": val_df.height,
            "test_rows": test_df.height,
            "train_match_count": len(split_ids["train"]),
            "val_match_count": len(split_ids["val"]),
            "test_match_count": len(split_ids["test"]),
            "train_match_ids": split_ids["train"],
            "val_match_ids": split_ids["val"],
            "test_match_ids": split_ids["test"],
        },
        "search": search_metrics,
        "validation": {
            **validation_metrics,
            "selected_n_estimators": final_n_estimators,
            "threshold_sweep": validation_thresholds,
        },
        "test": test_metrics,
        "notes": [
            "Model selection is based on validation log_loss.",
            "Log loss and calibration matter more than raw accuracy for win probability deltas.",
            "Test split is evaluated exactly once after training on train+validation matches.",
        ],
    }
    save_json(args.metrics_out, metrics_payload)

    LOGGER.info("validation_log_loss=%.6f", validation_metrics["log_loss"])
    LOGGER.info("validation_roc_auc=%.6f", validation_metrics["roc_auc"])
    LOGGER.info("test_log_loss=%.6f", test_metrics["log_loss"])
    LOGGER.info("test_roc_auc=%.6f", test_metrics["roc_auc"])
    LOGGER.info("test_accuracy_at_0_5=%.6f", test_metrics["accuracy_at_0_5"])
    LOGGER.info("saved_model=%s", args.model_out)
    LOGGER.info("saved_metrics=%s", args.metrics_out)
    LOGGER.info("saved_feature_importance=%s", args.feature_importance_out)


if __name__ == "__main__":
    main()
