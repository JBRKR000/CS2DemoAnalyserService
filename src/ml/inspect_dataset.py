from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import DEFAULT_OUTPUT_PATH
from features import MAX_PLAYERS_PER_SIDE, MAX_REASONABLE_ROUND_SECONDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the generated ML snapshot dataset.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to the parquet dataset (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="How many example rows to print from the start of the dataset.",
    )
    return parser.parse_args()


def format_schema(dataset: pl.DataFrame) -> str:
    return "\n".join(f"  - {name}: {dtype}" for name, dtype in dataset.schema.items())


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def print_frame(label: str, frame: pl.DataFrame) -> None:
    print(f"\n{label}")
    if frame.is_empty():
        print("  <empty>")
        return
    for row in frame.to_dicts():
        print(f"  {row}")


def inspect_dataset(dataset: pl.DataFrame, head: int) -> None:
    print_section("Overview")
    print(f"Rows: {dataset.height}")
    print(f"Columns: {dataset.width}")
    print(f"Estimated size: {dataset.estimated_size('mb'):.2f} MB")

    print_section("Schema")
    print(format_schema(dataset))

    print_section("Null Counts")
    print_frame("Per-column nulls", dataset.null_count())

    print_section("Example Rows")
    print_frame("Head", dataset.head(head))

    print_frame(
        "Rows per match",
        dataset.group_by("match_id").len().sort("len", descending=True),
    )
    print_frame(
        "Rows per map",
        dataset.group_by("map_name").len().sort("len", descending=True),
    )
    print_frame(
        "Rows per side / snapshot",
        dataset.group_by(["side", "snapshot_type"]).len().sort(["side", "snapshot_type"]),
    )
    print_frame(
        "Rows per round result",
        dataset.group_by("team_won_round").len().sort("team_won_round"),
    )

    numeric_columns = [
        "alive_team",
        "alive_enemy",
        "seconds_remaining",
        "round_num",
        "tick",
    ]
    print_frame(
        "Numeric summary",
        dataset.select(numeric_columns).describe(),
    )

    alive_anomalies = dataset.filter(
        (pl.col("alive_team") > MAX_PLAYERS_PER_SIDE)
        | (pl.col("alive_enemy") > MAX_PLAYERS_PER_SIDE)
        | (pl.col("alive_team") < 0)
        | (pl.col("alive_enemy") < 0)
    )
    time_anomalies = dataset.filter(
        (pl.col("seconds_remaining") > MAX_REASONABLE_ROUND_SECONDS)
        | (pl.col("seconds_remaining") < 0)
    )

    print_section("Anomaly Summary")
    print(f"alive_anomaly_count: {alive_anomalies.height}")
    print(f"time_anomaly_count: {time_anomalies.height}")
    print(f"max_alive_team: {dataset['alive_team'].max()}")
    print(f"max_alive_enemy: {dataset['alive_enemy'].max()}")
    print(f"max_seconds_remaining: {dataset['seconds_remaining'].max()}")
    if "is_time_anomaly" in dataset.columns:
        print(f"flagged_time_anomaly_count: {dataset.filter(pl.col('is_time_anomaly')).height}")

    print_section("Alive Count Anomalies")
    if alive_anomalies.is_empty():
        print("No alive count anomalies found.")
    else:
        print_frame(
            "Alive anomaly rows",
            alive_anomalies.select(
                [
                    "match_id",
                    "map_name",
                    "round_num",
                    "tick",
                    "snapshot_type",
                    "side",
                    "alive_team",
                    "alive_enemy",
                ]
            ).head(50),
        )

    print_section("Time Anomalies")
    if time_anomalies.is_empty():
        print("No time anomalies found.")
    else:
        print_frame(
            "Time anomaly rows",
            time_anomalies.select(
                [
                    "match_id",
                    "map_name",
                    "round_num",
                    "tick",
                    "seconds_remaining",
                ]
            ).head(50),
        )


def main() -> None:
    args = parse_args()
    dataset_path = args.path

    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    dataset = pl.read_parquet(dataset_path)
    if dataset.is_empty():
        print(f"Dataset exists but is empty: {dataset_path}")
        return

    print(f"Inspecting dataset: {dataset_path}")
    inspect_dataset(dataset, head=max(0, args.head))


if __name__ == "__main__":
    main()
