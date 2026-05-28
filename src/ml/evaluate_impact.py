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
from ml.format_impact import format_ml_impact_event
from ml.predict import DEFAULT_MODEL_PATH, load_round_win_model, predict_probabilities
from sectors.ml_impact import build_ml_impact_from_snapshots


REPO_ROOT = SRC_DIR.parent
DEFAULT_IMPACT_OUTPUT_PATH = REPO_ROOT / "data" / "ml" / "ml_event_impact.parquet"
DEFAULT_TOP_POSITIVE_OUTPUT_PATH = REPO_ROOT / "data" / "ml" / "top_positive_events.json"
DEFAULT_TOP_NEGATIVE_OUTPUT_PATH = REPO_ROOT / "data" / "ml" / "top_negative_events.json"
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate snapshot impact with the existing LightGBM round win model.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to the round snapshot parquet dataset (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained LightGBM model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--impact-out",
        type=Path,
        default=DEFAULT_IMPACT_OUTPUT_PATH,
        help=f"Where to save the full ML event impact parquet (default: {DEFAULT_IMPACT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--top-positive-out",
        type=Path,
        default=DEFAULT_TOP_POSITIVE_OUTPUT_PATH,
        help=f"Where to save the top positive event JSON (default: {DEFAULT_TOP_POSITIVE_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--top-negative-out",
        type=Path,
        default=DEFAULT_TOP_NEGATIVE_OUTPUT_PATH,
        help=f"Where to save the top negative event JSON (default: {DEFAULT_TOP_NEGATIVE_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many top positive and negative events to save (default: 10).",
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


def _rows_with_formatted_summary(frame: pl.DataFrame) -> list[dict[str, Any]]:
    rows = sanitize_for_json(frame.to_dicts())
    if not isinstance(rows, list):
        return []
    return [
        {
            **row,
            "formatted_summary": format_ml_impact_event(row),
        }
        for row in rows
        if isinstance(row, dict)
    ]


def save_json_rows(path: Path, frame: pl.DataFrame) -> list[dict[str, Any]]:
    rows = _rows_with_formatted_summary(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    return rows


def log_impact_sanity(impact: pl.DataFrame) -> None:
    LOGGER.info("impact_rows=%s", impact.height)
    if impact.is_empty():
        raise SystemExit("Impact is empty: no paired before/after snapshots produced win probability deltas.")

    stats = impact.select(
        [
            pl.col("win_prob_delta").min().alias("min_win_prob_delta"),
            pl.col("win_prob_delta").max().alias("max_win_prob_delta"),
            pl.col("win_prob_delta").mean().alias("mean_win_prob_delta"),
            pl.col("win_prob_before").null_count().alias("win_prob_before_null_count"),
            pl.col("win_prob_after").null_count().alias("win_prob_after_null_count"),
            pl.col("win_prob_delta").null_count().alias("win_prob_delta_null_count"),
        ]
    ).row(0, named=True)

    LOGGER.info("min_win_prob_delta=%s", stats["min_win_prob_delta"])
    LOGGER.info("max_win_prob_delta=%s", stats["max_win_prob_delta"])
    LOGGER.info("mean_win_prob_delta=%s", stats["mean_win_prob_delta"])
    LOGGER.info("win_prob_before_null_count=%s", stats["win_prob_before_null_count"])
    LOGGER.info("win_prob_after_null_count=%s", stats["win_prob_after_null_count"])

    delta_null_count = int(stats["win_prob_delta_null_count"])
    if delta_null_count:
        raise RuntimeError(f"win_prob_delta contains {delta_null_count} null value(s).")


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.top_n <= 0:
        raise SystemExit("--top-n must be > 0.")

    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    snapshots = pl.read_parquet(args.dataset)
    LOGGER.info("snapshots_loaded=%s path=%s", snapshots.height, args.dataset)
    if snapshots.is_empty():
        raise SystemExit(f"Dataset is empty: {args.dataset}")

    model = load_round_win_model(args.model)
    snapshots_with_probs = predict_probabilities(model, snapshots)
    impact_outputs = build_ml_impact_from_snapshots(snapshots_with_probs)

    impact = impact_outputs["ml_event_impact"]
    log_impact_sanity(impact)

    top_positive = impact.head(args.top_n)
    top_negative = impact.sort(
        ["win_prob_delta", "match_id", "round_num", "tick_after"],
        descending=[False, False, False, False],
    ).head(args.top_n)

    args.impact_out.parent.mkdir(parents=True, exist_ok=True)
    impact.write_parquet(args.impact_out)
    top_positive_rows = save_json_rows(args.top_positive_out, top_positive)
    top_negative_rows = save_json_rows(args.top_negative_out, top_negative)

    LOGGER.info("top_positive_deltas=%s", top_positive.select("win_prob_delta").to_series().to_list())
    LOGGER.info("top_negative_deltas=%s", top_negative.select("win_prob_delta").to_series().to_list())
    LOGGER.info(
        "top_positive_formatted_summaries=%s",
        [row["formatted_summary"] for row in top_positive_rows],
    )
    LOGGER.info(
        "top_negative_formatted_summaries=%s",
        [row["formatted_summary"] for row in top_negative_rows],
    )
    LOGGER.info("impact_out=%s", args.impact_out)
    LOGGER.info("top_positive_out=%s", args.top_positive_out)
    LOGGER.info("top_negative_out=%s", args.top_negative_out)


if __name__ == "__main__":
    main()
