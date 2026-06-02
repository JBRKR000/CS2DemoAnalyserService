from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "ml" / "death_risk_timeseries_5s.parquet"

REQUIRED_COLUMNS = [
    "match_id",
    "map_name",
    "round_num",
    "tick",
    "steamid",
    "player_name",
    "side",
    "alive_team_at_snapshot",
    "alive_enemy_at_snapshot",
    "seconds_remaining_at_snapshot",
    "bomb_planted_at_snapshot",
    "player_alive",
    "sample_time_seconds",
    "build_version",
    "death_within_5s",
    "kill_within_5s",
]
REQUIRED_FEATURE_COLUMNS = [
    "match_id",
    "map_name",
    "round_num",
    "tick",
    "steamid",
    "side",
    "alive_team_at_snapshot",
    "alive_enemy_at_snapshot",
    "seconds_remaining_at_snapshot",
    "bomb_planted_at_snapshot",
    "player_alive",
    "sample_time_seconds",
    "build_version",
]
FORBIDDEN_COLUMNS = {
    "source_situation_type",
    "source_situation_id",
    "event_weapon",
    "ml_impact",
    "ml_impact_at_event",
    "action_value_class",
    "high_cost_death",
    "high_impact_kill",
    "low_cost_death",
    "low_impact_kill",
    "zero_damage_death",
    "was_untraded",
    "was_traded",
    "high_cost_death_within_5s",
    "high_impact_kill_within_5s",
    "opening_duel_within_5s",
    "opening_duel_won_within_5s",
    "event_tick",
    "snapshot_tick",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate time-sampled death risk dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Dataset parquet path (default: {DEFAULT_DATASET_PATH})",
    )
    return parser.parse_args()


def _null_count(frame: pl.DataFrame, column: str) -> int | None:
    if column not in frame.columns:
        return None
    return int(frame.select(pl.col(column).is_null().sum()).item())


def _bool_distribution(frame: pl.DataFrame, column: str) -> dict[bool, int]:
    if column not in frame.columns or frame.is_empty():
        return {False: 0, True: 0}
    grouped = (
        frame.with_columns(pl.col(column).fill_null(False).cast(pl.Boolean).alias(column))
        .group_by(column)
        .agg(pl.len().alias("rows"))
    )
    values = {bool(row[column]): int(row["rows"]) for row in grouped.to_dicts()}
    return {False: values.get(False, 0), True: values.get(True, 0)}


def _print_null_counts(dataset: pl.DataFrame, columns: list[str]) -> None:
    print("null counts for required features")
    for column in columns:
        count = _null_count(dataset, column)
        print(f"  {column}: {'missing' if count is None else count}")


def main() -> None:
    args = parse_args()
    failures: list[str] = []

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        print("VALIDATION FAILED")
        raise SystemExit(1)

    dataset = pl.read_parquet(args.dataset)
    print(f"dataset: {args.dataset}")
    print(f"total rows: {dataset.height}")
    if dataset.height <= 0:
        failures.append("Dataset has no rows.")

    missing = [column for column in REQUIRED_COLUMNS if column not in dataset.columns]
    print(f"required columns missing: {missing or 'none'}")
    if missing:
        failures.append(f"Missing required columns: {missing}")

    forbidden_present = sorted(FORBIDDEN_COLUMNS.intersection(dataset.columns))
    print(f"forbidden columns present: {forbidden_present or 'none'}")
    if forbidden_present:
        failures.append(f"Forbidden columns are present: {forbidden_present}")

    if "player_alive" in dataset.columns:
        not_alive = int(dataset.filter(~pl.col("player_alive").fill_null(False).cast(pl.Boolean)).height)
        print(f"player_alive false/null rows: {not_alive}")
        if not_alive:
            failures.append(f"Found {not_alive} rows where player_alive is not true.")

    duplicate_required = {"match_id", "round_num", "tick", "steamid"}
    if duplicate_required.issubset(dataset.columns):
        duplicates = (
            dataset.group_by(["match_id", "round_num", "tick", "steamid"])
            .agg(pl.len().alias("rows"))
            .filter(pl.col("rows") > 1)
        )
        print(f"duplicate match_id + round_num + tick + steamid rows: {duplicates.height}")
        if duplicates.height:
            failures.append(f"Found {duplicates.height} duplicate snapshot-player rows.")
    else:
        failures.append("Cannot check duplicates because key columns are missing.")

    for column in ["death_within_5s", "kill_within_5s"]:
        distribution = _bool_distribution(dataset, column)
        print(f"{column}: false={distribution[False]} true={distribution[True]}")
        if distribution[False] == 0 or distribution[True] == 0:
            failures.append(f"{column} must contain both classes.")

    _print_null_counts(dataset, REQUIRED_FEATURE_COLUMNS)
    for column in REQUIRED_FEATURE_COLUMNS:
        count = _null_count(dataset, column)
        if count is None:
            failures.append(f"Required feature column is missing: {column}")
        elif count > 0:
            failures.append(f"Required feature column has nulls: {column} ({count})")

    if failures:
        print("failures")
        for failure in failures:
            print(f"  - {failure}")
        print("VALIDATION FAILED")
        raise SystemExit(1)

    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
