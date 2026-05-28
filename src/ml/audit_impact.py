from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.dataset import DEFAULT_OUTPUT_PATH
from ml.predict import DEFAULT_MODEL_PATH, load_round_win_model, predict_probabilities
from sectors.ml_impact import build_ml_impact_from_snapshots


REPO_ROOT = SRC_DIR.parent
DEFAULT_IMPACT_PATH = REPO_ROOT / "data" / "ml" / "ml_event_impact.parquet"
DEFAULT_SUMMARY_OUTPUT_PATH = REPO_ROOT / "data" / "ml" / "impact_audit_summary.json"
DEFAULT_WORST_EVENTS_OUTPUT_PATH = REPO_ROOT / "data" / "ml" / "impact_audit_worst_events.json"
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit ML event impact output for CT/T perspective consistency.",
    )
    parser.add_argument(
        "--impact",
        type=Path,
        default=DEFAULT_IMPACT_PATH,
        help=f"Path to the full impact parquet (default: {DEFAULT_IMPACT_PATH})",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to the round snapshot parquet used if impact must be rebuilt (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the LightGBM model used if impact must be rebuilt (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT_PATH,
        help=f"Where to write the audit summary JSON (default: {DEFAULT_SUMMARY_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--worst-events-out",
        type=Path,
        default=DEFAULT_WORST_EVENTS_OUTPUT_PATH,
        help=f"Where to write worst inconsistent events JSON (default: {DEFAULT_WORST_EVENTS_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--worst-n",
        type=int,
        default=20,
        help="How many worst events to print and save for each inconsistency view (default: 20).",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )


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


def load_or_rebuild_impact(impact_path: Path, dataset_path: Path, model_path: Path) -> pl.DataFrame:
    if impact_path.exists():
        impact = pl.read_parquet(impact_path)
        LOGGER.info("impact_loaded=%s path=%s", impact.height, impact_path)
        return impact

    LOGGER.info("impact_missing=%s rebuilding_from_dataset=%s", impact_path, dataset_path)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    snapshots = pl.read_parquet(dataset_path)
    LOGGER.info("snapshots_loaded=%s path=%s", snapshots.height, dataset_path)
    if snapshots.is_empty():
        raise SystemExit(f"Dataset is empty: {dataset_path}")

    model = load_round_win_model(model_path)
    snapshots_with_probs = predict_probabilities(model, snapshots)
    impact = build_ml_impact_from_snapshots(snapshots_with_probs)["ml_event_impact"]
    LOGGER.info("impact_rebuilt=%s", impact.height)
    return impact


def require_columns(frame: pl.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise SystemExit(f"Impact output is missing required columns: {missing_columns}")


def metric_summary(frame: pl.DataFrame, column: str) -> dict[str, float | None]:
    if frame.is_empty():
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    row = frame.select(
        [
            pl.col(column).mean().alias("mean"),
            pl.col(column).median().alias("median"),
            pl.col(column).quantile(0.95).alias("p95"),
            pl.col(column).max().alias("max"),
        ]
    ).row(0, named=True)
    return {
        "mean": row["mean"],
        "median": row["median"],
        "p95": row["p95"],
        "max": row["max"],
    }


def delta_summary(frame: pl.DataFrame, by: str | None = None) -> list[dict[str, Any]] | dict[str, Any]:
    if frame.is_empty():
        if by is None:
            return {
                "rows": 0,
                "mean": None,
                "median": None,
                "p95": None,
                "min": None,
                "max": None,
            }
        return []

    aggregations = [
        pl.len().alias("rows"),
        pl.col("win_prob_delta").mean().alias("mean"),
        pl.col("win_prob_delta").median().alias("median"),
        pl.col("win_prob_delta").quantile(0.95).alias("p95"),
        pl.col("win_prob_delta").min().alias("min"),
        pl.col("win_prob_delta").max().alias("max"),
    ]
    if by is None:
        return frame.select(aggregations).row(0, named=True)

    return (
        frame.group_by(by)
        .agg(aggregations)
        .sort(by)
        .to_dicts()
    )


def build_consistency(impact: pl.DataFrame) -> tuple[pl.DataFrame, int, int]:
    event_impact = impact.filter(pl.col("event_id").is_not_null())
    null_event_id_rows = impact.height - event_impact.height

    side_counts = (
        event_impact.group_by("event_id")
        .agg(
            [
                pl.col("side").n_unique().alias("side_count"),
                pl.col("side").unique().sort().alias("sides"),
            ]
        )
    )
    missing_side_event_ids = side_counts.filter(
        ~(
            (pl.col("side_count") == 2)
            & pl.col("sides").list.contains("CT")
            & pl.col("sides").list.contains("T")
        )
    ).height

    ct = (
        event_impact.filter(pl.col("side") == "CT")
        .select(
            [
                "event_id",
                "match_id",
                "map_name",
                "round_num",
                "event_type",
                "kill_context_type",
                "tick_before",
                "tick_after",
                pl.col("win_prob_before").alias("win_prob_before_CT"),
                pl.col("win_prob_after").alias("win_prob_after_CT"),
                pl.col("win_prob_delta").alias("win_prob_delta_CT"),
            ]
        )
    )
    t = (
        event_impact.filter(pl.col("side") == "T")
        .select(
            [
                "event_id",
                pl.col("win_prob_before").alias("win_prob_before_T"),
                pl.col("win_prob_after").alias("win_prob_after_T"),
                pl.col("win_prob_delta").alias("win_prob_delta_T"),
            ]
        )
    )
    consistency = (
        ct.join(t, on="event_id", how="inner")
        .with_columns(
            [
                (pl.col("win_prob_before_CT") + pl.col("win_prob_before_T")).alias("prob_sum_before"),
                (pl.col("win_prob_after_CT") + pl.col("win_prob_after_T")).alias("prob_sum_after"),
                (pl.col("win_prob_delta_CT") + pl.col("win_prob_delta_T")).alias("delta_sum"),
            ]
        )
        .with_columns(
            [
                (pl.col("prob_sum_before") - 1.0).abs().alias("prob_sum_before_error"),
                (pl.col("prob_sum_after") - 1.0).abs().alias("prob_sum_after_error"),
                pl.col("delta_sum").abs().alias("delta_sum_abs"),
            ]
        )
    )
    return consistency, missing_side_event_ids, null_event_id_rows


def log_metric_summary(label: str, summary: dict[str, float | None]) -> None:
    LOGGER.info(
        "%s mean=%s median=%s p95=%s max=%s",
        label,
        summary["mean"],
        summary["median"],
        summary["p95"],
        summary["max"],
    )


def log_worst_events(label: str, frame: pl.DataFrame) -> None:
    LOGGER.info("%s_count=%s", label, frame.height)
    for row in frame.to_dicts():
        LOGGER.info("%s %s", label, row)


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.worst_n <= 0:
        raise SystemExit("--worst-n must be > 0.")

    impact = load_or_rebuild_impact(args.impact, args.dataset, args.model)
    if impact.is_empty():
        raise SystemExit("Impact output is empty; cannot audit CT/T perspective consistency.")

    require_columns(
        impact,
        [
            "event_id",
            "side",
            "win_prob_before",
            "win_prob_after",
            "win_prob_delta",
            "kill_context_type",
            "bomb_planted_before",
        ],
    )

    consistency, missing_side_event_ids, null_event_id_rows = build_consistency(impact)
    unique_event_ids = impact.filter(pl.col("event_id").is_not_null()).select(pl.col("event_id").n_unique()).item()

    before_error_summary = metric_summary(consistency, "prob_sum_before_error")
    after_error_summary = metric_summary(consistency, "prob_sum_after_error")
    delta_sum_abs_summary = metric_summary(consistency, "delta_sum_abs")

    LOGGER.info("unique_event_ids=%s", unique_event_ids)
    LOGGER.info("event_ids_with_both_CT_T_rows=%s", consistency.height)
    LOGGER.info("missing_side_event_ids=%s", missing_side_event_ids)
    LOGGER.info("null_event_id_rows=%s", null_event_id_rows)
    log_metric_summary("prob_sum_before_error", before_error_summary)
    log_metric_summary("prob_sum_after_error", after_error_summary)
    log_metric_summary("delta_sum_abs", delta_sum_abs_summary)

    worst_after_error = consistency.sort(
        ["prob_sum_after_error", "delta_sum_abs", "event_id"],
        descending=[True, True, False],
    ).head(args.worst_n)
    worst_delta_sum = consistency.sort(
        ["delta_sum_abs", "prob_sum_after_error", "event_id"],
        descending=[True, True, False],
    ).head(args.worst_n)

    log_worst_events("worst_prob_sum_after_error", worst_after_error)
    log_worst_events("worst_delta_sum_abs", worst_delta_sum)

    distribution_summaries = {
        "by_kill_context_type": delta_summary(impact, "kill_context_type"),
        "by_side": delta_summary(impact, "side"),
        "by_bomb_planted_before": delta_summary(impact, "bomb_planted_before"),
        "normal_kill": delta_summary(impact.filter(pl.col("kill_context_type") == "normal_kill")),
        "world_death": delta_summary(impact.filter(pl.col("kill_context_type") == "world_death")),
    }
    LOGGER.info("delta_by_kill_context_type=%s", distribution_summaries["by_kill_context_type"])
    LOGGER.info("delta_by_side=%s", distribution_summaries["by_side"])
    LOGGER.info("delta_by_bomb_planted_before=%s", distribution_summaries["by_bomb_planted_before"])
    LOGGER.info("delta_normal_kill=%s", distribution_summaries["normal_kill"])
    LOGGER.info("delta_world_death=%s", distribution_summaries["world_death"])

    summary_payload = {
        "impact_path": args.impact,
        "impact_rows": impact.height,
        "unique_event_ids": unique_event_ids,
        "event_ids_with_both_CT_T_rows": consistency.height,
        "missing_side_event_ids": missing_side_event_ids,
        "null_event_id_rows": null_event_id_rows,
        "prob_sum_before_error": before_error_summary,
        "prob_sum_after_error": after_error_summary,
        "delta_sum_abs": delta_sum_abs_summary,
        "win_prob_delta_distributions": distribution_summaries,
    }
    worst_events_payload = {
        "worst_prob_sum_after_error": worst_after_error.to_dicts(),
        "worst_delta_sum_abs": worst_delta_sum.to_dicts(),
    }

    save_json(args.summary_out, summary_payload)
    save_json(args.worst_events_out, worst_events_payload)
    LOGGER.info("summary_out=%s", args.summary_out)
    LOGGER.info("worst_events_out=%s", args.worst_events_out)


if __name__ == "__main__":
    main()
