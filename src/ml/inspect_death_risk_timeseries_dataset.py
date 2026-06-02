from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "ml" / "death_risk_timeseries_5s.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect time-sampled death risk dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input parquet path (default: {DEFAULT_INPUT_PATH})",
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


def _print_bool_distribution(frame: pl.DataFrame, column: str) -> None:
    if frame.is_empty() or column not in frame.columns:
        print(f"  {column}: missing")
        return
    grouped = (
        frame.with_columns(pl.col(column).fill_null(False).cast(pl.Boolean).alias(column))
        .group_by(column)
        .agg(pl.len().alias("rows"))
        .sort(column)
    )
    values = {bool(row[column]): int(row["rows"]) for row in grouped.to_dicts()}
    print(f"  {column}: false={values.get(False, 0)} true={values.get(True, 0)}")


def _null_count(frame: pl.DataFrame, column: str) -> int | None:
    if column not in frame.columns:
        return None
    return int(frame.select(pl.col(column).is_null().sum()).item())


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Dataset not found: {args.input}")

    dataset = pl.read_parquet(args.input)
    print(f"total rows: {dataset.height}")
    _print_group_counts(dataset, "match_id", "rows per match")
    _print_group_counts(dataset, "map_name", "rows per map")
    _print_group_counts(dataset, "side", "rows per side")

    print("death_within_5s distribution")
    _print_bool_distribution(dataset, "death_within_5s")
    print("kill_within_5s distribution")
    _print_bool_distribution(dataset, "kill_within_5s")

    print("null counts")
    for column in dataset.columns:
        print(f"  {column}: {_null_count(dataset, column)}")

    print("first 10 rows preview")
    if dataset.is_empty():
        print("empty dataset")
    else:
        print(dataset.head(10))


if __name__ == "__main__":
    main()
