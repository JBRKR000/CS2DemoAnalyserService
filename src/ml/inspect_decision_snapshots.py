from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "ml" / "decision_snapshots.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Decision Snapshot Dataset v1.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Input decision snapshot parquet.",
    )
    return parser.parse_args()


def _null_count(frame: pl.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return frame.height
    return int(frame.select(pl.col(column).is_null().sum()).item())


def _print_group_counts(frame: pl.DataFrame, column: str, title: str) -> None:
    print(title)
    if frame.is_empty() or column not in frame.columns:
        print("  none")
        return
    grouped = frame.group_by(column).agg(pl.len().alias("rows")).sort(column)
    for row in grouped.to_dicts():
        print(f"  {row[column]}: {row['rows']}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Dataset not found: {input_path}")

    dataset = pl.read_parquet(input_path)

    print(f"total rows: {dataset.height}")
    _print_group_counts(dataset, "match_id", "rows per match")
    _print_group_counts(dataset, "map_name", "rows per map")
    _print_group_counts(dataset, "source_situation_type", "rows per source_situation_type")
    _print_group_counts(dataset, "seconds_before_event", "rows per seconds_before_event")

    print("target distributions")
    for column in [
        "death_within_5s",
        "kill_within_5s",
        "opening_duel_within_5s",
        "opening_duel_won_within_5s",
        "high_cost_death_within_5s",
        "high_impact_kill_within_5s",
    ]:
        if column in dataset.columns:
            true_count = int(dataset.select(pl.col(column).cast(pl.Boolean).sum()).item())
            print(f"  {column}: {true_count}")
        else:
            print(f"  {column}: missing")

    _print_group_counts(dataset, "action_value_class", "action_value_class distribution")

    print("null counts")
    for column in [
        "alive_team_at_snapshot",
        "alive_enemy_at_snapshot",
        "seconds_remaining_at_snapshot",
        "bomb_planted_at_snapshot",
        "weapon",
        "event_weapon",
        "prior_round_phase",
        "ml_impact_at_event",
    ]:
        print(f"  {column}: {_null_count(dataset, column)}")

    print("first 10 rows preview")
    if dataset.is_empty():
        print("empty dataset")
    else:
        print(dataset.head(10))


if __name__ == "__main__":
    main()