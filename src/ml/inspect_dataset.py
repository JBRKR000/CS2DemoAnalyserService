from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import DEFAULT_OUTPUT_PATH
from features import BOMB_TIMER_SECONDS, MAX_PLAYERS_PER_SIDE, MAX_REASONABLE_ROUND_SECONDS


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

    print_section("Event ID Checks")
    event_id_exists = "event_id" in dataset.columns
    print(f"event_id_exists: {event_id_exists}")
    if event_id_exists:
        event_id_counts = (
            dataset.group_by("event_id")
            .len()
            .rename({"len": "rows_per_event_id"})
            .sort(["rows_per_event_id", "event_id"], descending=[False, False])
        )
        distribution = (
            event_id_counts.group_by("rows_per_event_id")
            .agg(pl.len().alias("event_id_count"))
            .sort("rows_per_event_id")
        )
        print_frame(
            "event_id row count distribution",
            distribution,
        )
        anomalous_event_ids = event_id_counts.filter(pl.col("rows_per_event_id") != 4)
        print(f"event_id_anomaly_count: {anomalous_event_ids.height}")
        if anomalous_event_ids.is_empty():
            print("All event_ids have 4 rows.")
        else:
            print_frame(
                "Anomalous event_ids",
                anomalous_event_ids.head(100),
            )

    print_section("Kill Context Checks")
    context_columns = [
        "killer_steamid",
        "victim_steamid",
        "killer_name",
        "victim_name",
        "weapon",
        "killer_side",
        "victim_side",
        "kill_context_type",
    ]
    missing_context_columns = [column for column in context_columns if column not in dataset.columns]
    print(f"context_columns_present: {not missing_context_columns}")
    if missing_context_columns:
        print(f"missing_context_columns: {missing_context_columns}")
    else:
        print_frame(
            "Kill context null counts",
            dataset.select(
                [
                    pl.col("killer_steamid").is_null().sum().alias("killer_steamid_nulls"),
                    pl.col("victim_steamid").is_null().sum().alias("victim_steamid_nulls"),
                    pl.col("killer_name").is_null().sum().alias("killer_name_nulls"),
                    pl.col("victim_name").is_null().sum().alias("victim_name_nulls"),
                    pl.col("weapon").is_null().sum().alias("weapon_nulls"),
                ]
            ),
        )
        print_frame(
            "Sample kill context rows",
            dataset.select(
                [
                    "event_id",
                    "round_num",
                    "killer_name",
                    "victim_name",
                    "weapon",
                    "killer_side",
                    "victim_side",
                    "kill_context_type",
                ]
            ).unique().sort(["round_num", "event_id"]).head(10),
        )
        print_frame(
            "Rows per kill_context_type",
            dataset.group_by("kill_context_type").len().sort(["len", "kill_context_type"], descending=[True, False]),
        )
        non_normal_kill_context = (
            dataset.filter(pl.col("kill_context_type") != "normal_kill")
            .select(
                [
                    "event_id",
                    "round_num",
                    "killer_name",
                    "victim_name",
                    "weapon",
                    "killer_side",
                    "victim_side",
                    "kill_context_type",
                ]
            )
            .unique()
            .sort(["round_num", "event_id"])
        )
        print_frame(
            "Sample non-normal kill context rows",
            non_normal_kill_context.head(10),
        )
        kill_context_mismatch = dataset.filter(
            (pl.col("killer_side") == pl.col("victim_side"))
            & (pl.col("kill_context_type") == "normal_kill")
        )
        print(f"kill_context_type_mismatch_count: {kill_context_mismatch.height}")
        if not kill_context_mismatch.is_empty():
            print_frame(
                "Kill context type mismatches",
                kill_context_mismatch.select(
                    [
                        "event_id",
                        "round_num",
                        "killer_name",
                        "victim_name",
                        "weapon",
                        "killer_side",
                        "victim_side",
                        "kill_context_type",
                    ]
                ).unique().sort(["round_num", "event_id"]).head(10),
            )

    numeric_columns = [
        "alive_team",
        "alive_enemy",
        "seconds_remaining",
        "bomb_time_since_plant",
        "bomb_time_remaining",
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
    bomb_timer_anomalies = dataset.filter(
        (pl.col("bomb_time_since_plant") < 0)
        | (pl.col("bomb_time_remaining") < 0)
        | (pl.col("bomb_time_remaining") > BOMB_TIMER_SECONDS)
        | ((~pl.col("bomb_planted")) & (pl.col("bomb_time_since_plant") != 0.0))
        | ((~pl.col("bomb_planted")) & (pl.col("bomb_time_remaining") != 0.0))
    )

    print_section("Anomaly Summary")
    print(f"alive_anomaly_count: {alive_anomalies.height}")
    print(f"time_anomaly_count: {time_anomalies.height}")
    print(f"bomb_timer_anomaly_count: {bomb_timer_anomalies.height}")
    print(f"max_alive_team: {dataset['alive_team'].max()}")
    print(f"max_alive_enemy: {dataset['alive_enemy'].max()}")
    print(f"max_seconds_remaining: {dataset['seconds_remaining'].max()}")
    if "bomb_time_since_plant" in dataset.columns:
        print(f"max_bomb_time_since_plant: {dataset['bomb_time_since_plant'].max()}")
    if "bomb_time_remaining" in dataset.columns:
        print(f"max_bomb_time_remaining: {dataset['bomb_time_remaining'].max()}")
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

    print_section("Bomb Timer Anomalies")
    if bomb_timer_anomalies.is_empty():
        print("No bomb timer anomalies found.")
    else:
        print_frame(
            "Bomb timer anomaly rows",
            bomb_timer_anomalies.select(
                [
                    "match_id",
                    "map_name",
                    "round_num",
                    "event_id",
                    "tick",
                    "snapshot_type",
                    "side",
                    "bomb_planted",
                    "bomb_time_since_plant",
                    "bomb_time_remaining",
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
