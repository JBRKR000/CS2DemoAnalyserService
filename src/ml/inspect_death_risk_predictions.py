from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "ml" / "death_risk_timeseries_5s_predictions.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect exported death risk predictions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Prediction parquet path (default: {DEFAULT_INPUT_PATH})",
    )
    return parser.parse_args()


def _print_group_counts(frame: pl.DataFrame, column: str, title: str) -> None:
    print(title)
    if frame.is_empty() or column not in frame.columns:
        print("  none")
        return
    grouped = frame.group_by(column).agg(pl.len().alias("rows")).sort(column)
    for row in grouped.to_dicts():
        print(f"  {row[column]}: {row['rows']}")


def _print_rate_by_group(frame: pl.DataFrame, column: str) -> None:
    print(f"death rate per {column}")
    if frame.is_empty() or column not in frame.columns or "death_within_5s" not in frame.columns:
        print("  none")
        return
    grouped = (
        frame.group_by(column)
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("death_within_5s").cast(pl.Boolean).mean().alias("death_rate"),
            ]
        )
        .sort(column)
    )
    for row in grouped.to_dicts():
        print(f"  {row[column]}: rows={row['rows']} death_rate={row['death_rate']}")


def _print_probability_summary(frame: pl.DataFrame) -> None:
    if frame.is_empty() or "death_risk_5s" not in frame.columns:
        print("probability summary: missing")
        return
    summary = frame.select(
        [
            pl.len().alias("rows"),
            pl.col("death_risk_5s").min().alias("min"),
            pl.col("death_risk_5s").mean().alias("mean"),
            pl.col("death_risk_5s").max().alias("max"),
        ]
    ).to_dicts()[0]
    print(
        "probability summary: "
        f"rows={summary['rows']} min={summary['min']} mean={summary['mean']} max={summary['max']}"
    )


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.input.exists():
        raise SystemExit(f"Dataset not found: {args.input}")

    dataset = pl.read_parquet(args.input)
    print(f"total rows: {dataset.height}")
    _print_probability_summary(dataset)
    _print_group_counts(dataset, "risk_label", "rows per risk_label")
    _print_group_counts(dataset, "death_risk_bucket_global", "rows per risk_bucket")
    _print_rate_by_group(dataset, "risk_label")
    _print_rate_by_group(dataset, "death_risk_bucket_global")

    print("top 20 highest-risk rows")
    if dataset.is_empty():
        print("  empty dataset")
        return

    columns = [
        "match_id",
        "map_name",
        "round_num",
        "tick",
        "player_name",
        "side",
        "death_risk_5s",
        "death_within_5s",
        "weapon",
        "nearest_enemy_distance",
        "nearest_teammate_distance",
    ]
    preview = (
        dataset.select(columns)
        .sort("death_risk_5s", descending=True)
        .head(20)
    )
    print(preview)


if __name__ == "__main__":
    main()
