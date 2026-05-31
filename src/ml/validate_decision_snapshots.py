from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "ml" / "decision_snapshots.parquet"

REQUIRED_COLUMNS = [
    "match_id",
    "map_name",
    "round_num",
    "steamid",
    "source_situation_id",
    "source_situation_type",
    "event_tick",
    "snapshot_tick",
    "seconds_before_event",
    "death_within_5s",
    "kill_within_5s",
    "high_cost_death_within_5s",
    "high_impact_kill_within_5s",
    "action_value_class",
]
OPTIONAL_TARGET_COLUMNS = [
    "opening_duel_within_5s",
    "opening_duel_won_within_5s",
]
LABEL_COLUMNS = {
    "death_within_5s",
    "kill_within_5s",
    "opening_duel_within_5s",
    "opening_duel_won_within_5s",
    "high_cost_death_within_5s",
    "high_impact_kill_within_5s",
    "action_value_class",
}
FORBIDDEN_FEATURE_COLUMNS = {
    "ml_impact_at_event",
    "high_cost_death_within_5s",
    "high_impact_kill_within_5s",
    "action_value_class",
}
FEATURE_COLUMNS = [
    "alive_team_at_snapshot",
    "alive_enemy_at_snapshot",
    "seconds_remaining_at_snapshot",
    "bomb_planted_at_snapshot",
    "player_side",
    "weapon",
    "weapon_fallback_to_event",
    "is_awp_event",
    "is_rifle_event",
    "is_pistol_event",
    "prior_round_phase",
    "was_opening_context",
    "map_name",
    "seconds_before_event",
]
ALLOWED_OFFSETS = {1, 3, 5}
EXPECTED_SOURCE_TYPES = {
    "death_situation",
    "kill_situation",
    "opening_duel_situation",
}
NULL_CHECK_COLUMNS = [
    "alive_team_at_snapshot",
    "alive_enemy_at_snapshot",
    "seconds_remaining_at_snapshot",
    "bomb_planted_at_snapshot",
    "weapon",
    "event_weapon",
    "prior_round_phase",
    "ml_impact_at_event",
]
TARGET_DISTRIBUTION_COLUMNS = [
    "death_within_5s",
    "kill_within_5s",
    "opening_duel_within_5s",
    "opening_duel_won_within_5s",
    "high_cost_death_within_5s",
    "high_impact_kill_within_5s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Decision Snapshot Dataset v1 before model training.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the decision snapshot parquet dataset (default: {DEFAULT_DATASET_PATH})",
    )
    return parser.parse_args()


def _bool_expr(column: str) -> pl.Expr:
    return pl.col(column).fill_null(False).cast(pl.Boolean)


def _count(frame: pl.DataFrame, expression: pl.Expr) -> int:
    if frame.is_empty():
        return 0
    return int(frame.filter(expression).height)


def _null_count(frame: pl.DataFrame, column: str) -> int | None:
    if column not in frame.columns:
        return None
    return int(frame.select(pl.col(column).is_null().sum()).item())


def _print_group_counts(frame: pl.DataFrame, column: str, title: str) -> None:
    print(title)
    if frame.is_empty() or column not in frame.columns:
        print("  missing")
        return
    grouped = frame.group_by(column).agg(pl.len().alias("rows")).sort(column)
    for row in grouped.to_dicts():
        print(f"  {row[column]}: {row['rows']}")


def _print_bool_distribution(frame: pl.DataFrame, column: str) -> None:
    if column not in frame.columns:
        print(f"  {column}: missing")
        return
    grouped = (
        frame.with_columns(_bool_expr(column).alias(column))
        .group_by(column)
        .agg(pl.len().alias("rows"))
        .sort(column)
    )
    values = {row[column]: row["rows"] for row in grouped.to_dicts()}
    print(f"  {column}: false={values.get(False, 0)} true={values.get(True, 0)}")


def _print_pair_counts(frame: pl.DataFrame, left: str, right: str) -> None:
    print(f"{left} + {right}")
    if left not in frame.columns or right not in frame.columns:
        print("  missing")
        return
    grouped = (
        frame.select([_bool_expr(left).alias(left), _bool_expr(right).alias(right)])
        .group_by([left, right])
        .agg(pl.len().alias("rows"))
        .sort([left, right])
    )
    for row in grouped.to_dicts():
        print(f"  {left}={row[left]} {right}={row[right]}: {row['rows']}")


def _print_duplicate_pairs(frame: pl.DataFrame, failures: list[str]) -> None:
    print("duplicate source_situation_id + seconds_before_event pairs")
    required = {"source_situation_id", "seconds_before_event"}
    if not required.issubset(frame.columns):
        print("  skipped: required columns missing")
        return

    duplicates = (
        frame.group_by(["source_situation_id", "seconds_before_event"])
        .agg(pl.len().alias("rows"))
        .filter(pl.col("rows") > 1)
        .sort(["source_situation_id", "seconds_before_event"])
    )
    if duplicates.is_empty():
        print("  none")
        return

    failures.append(
        f"Found {duplicates.height} duplicated source_situation_id + seconds_before_event pairs."
    )
    for row in duplicates.head(20).to_dicts():
        print(
            "  "
            f"source_situation_id={row['source_situation_id']} "
            f"seconds_before_event={row['seconds_before_event']} rows={row['rows']}"
        )
    if duplicates.height > 20:
        print(f"  ... {duplicates.height - 20} more")


def _validate_basic_structure(dataset_path: Path, dataset: pl.DataFrame, failures: list[str]) -> None:
    print("basic structure")
    print(f"  dataset: {dataset_path}")
    print(f"  total rows: {dataset.height}")
    if dataset.height <= 0:
        failures.append("Dataset has no rows.")

    missing = [column for column in REQUIRED_COLUMNS if column not in dataset.columns]
    if missing:
        failures.append(f"Missing required columns: {missing}")
        print(f"  missing required columns: {missing}")
    else:
        print("  required columns: ok")


def _validate_anti_leakage(dataset: pl.DataFrame, failures: list[str]) -> None:
    print("anti-leakage checks")
    if {"snapshot_tick", "event_tick"}.issubset(dataset.columns):
        leaked_rows = _count(dataset, pl.col("snapshot_tick") >= pl.col("event_tick"))
        print(f"  rows with snapshot_tick >= event_tick: {leaked_rows}")
        if leaked_rows:
            failures.append(f"Found {leaked_rows} rows where snapshot_tick >= event_tick.")
    else:
        print("  tick ordering: skipped because required columns are missing")

    forbidden_used = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(FEATURE_COLUMNS))
    label_features = sorted(LABEL_COLUMNS.intersection(FEATURE_COLUMNS))
    print(f"  feature columns defined: {len(FEATURE_COLUMNS)}")
    print(f"  forbidden feature columns used: {forbidden_used or 'none'}")
    print(f"  label columns used as features: {label_features or 'none'}")
    if forbidden_used:
        failures.append(f"Forbidden leakage columns are present in FEATURE_COLUMNS: {forbidden_used}")
    if label_features:
        failures.append(f"Label columns are present in FEATURE_COLUMNS: {label_features}")


def _validate_offsets(dataset: pl.DataFrame, failures: list[str]) -> None:
    print("offset checks")
    if "seconds_before_event" in dataset.columns:
        invalid_offsets = (
            dataset.filter(~pl.col("seconds_before_event").is_in(sorted(ALLOWED_OFFSETS)))
            .select("seconds_before_event")
            .unique()
            .sort("seconds_before_event")
            .to_series()
            .to_list()
        )
        print(f"  allowed offsets: {sorted(ALLOWED_OFFSETS)}")
        print(f"  invalid offsets: {invalid_offsets or 'none'}")
        if invalid_offsets:
            failures.append(f"Invalid seconds_before_event values: {invalid_offsets}")
    else:
        print("  skipped: seconds_before_event missing")
    _print_duplicate_pairs(dataset, failures)


def _validate_source_type_consistency(
    dataset: pl.DataFrame,
    failures: list[str],
    warnings: list[str],
) -> None:
    print("source type consistency")
    if "source_situation_type" not in dataset.columns:
        print("  skipped: source_situation_type missing")
        return

    required_for_consistency = [
        "death_within_5s",
        "kill_within_5s",
        "high_cost_death_within_5s",
        "high_impact_kill_within_5s",
        "action_value_class",
    ]
    missing_for_consistency = [
        column for column in required_for_consistency if column not in dataset.columns
    ]
    if missing_for_consistency:
        print(f"  skipped: required label columns missing: {missing_for_consistency}")
        return

    unexpected_types = (
        dataset.filter(~pl.col("source_situation_type").is_in(sorted(EXPECTED_SOURCE_TYPES)))
        .select("source_situation_type")
        .unique()
        .sort("source_situation_type")
        .to_series()
        .to_list()
    )
    print(f"  unexpected source_situation_type values: {unexpected_types or 'none'}")
    if unexpected_types:
        warnings.append(f"Unexpected source_situation_type values: {unexpected_types}")

    death_rows = dataset.filter(pl.col("source_situation_type") == "death_situation")
    if not death_rows.is_empty():
        death_not_true = _count(death_rows, ~_bool_expr("death_within_5s"))
        death_bad_action = _count(death_rows, ~pl.col("action_value_class").is_in(["bad", "neutral"]))
        death_high_impact_without_kill = _count(
            death_rows,
            _bool_expr("high_impact_kill_within_5s") & ~_bool_expr("kill_within_5s"),
        )
        print(f"  death_situation death_within_5s false: {death_not_true}")
        print(f"  death_situation action_value_class not bad/neutral: {death_bad_action}")
        print(f"  death_situation high-impact kill without kill target: {death_high_impact_without_kill}")
        if death_not_true:
            warnings.append(f"{death_not_true} death_situation rows have death_within_5s != true.")
        if death_bad_action:
            warnings.append(
                f"{death_bad_action} death_situation rows have action_value_class outside bad/neutral."
            )
        if death_high_impact_without_kill:
            warnings.append(
                f"{death_high_impact_without_kill} death_situation rows have high_impact_kill true without kill_within_5s."
            )

    kill_rows = dataset.filter(pl.col("source_situation_type") == "kill_situation")
    if not kill_rows.is_empty():
        kill_not_true = _count(kill_rows, ~_bool_expr("kill_within_5s"))
        kill_bad_action = _count(
            kill_rows,
            ~pl.col("action_value_class").is_in(["excellent", "good", "neutral"]),
        )
        kill_high_cost_without_death = _count(
            kill_rows,
            _bool_expr("high_cost_death_within_5s") & ~_bool_expr("death_within_5s"),
        )
        print(f"  kill_situation kill_within_5s false: {kill_not_true}")
        print(f"  kill_situation action_value_class not excellent/good/neutral: {kill_bad_action}")
        print(f"  kill_situation high-cost death without death target: {kill_high_cost_without_death}")
        if kill_not_true:
            warnings.append(f"{kill_not_true} kill_situation rows have kill_within_5s != true.")
        if kill_bad_action:
            warnings.append(
                f"{kill_bad_action} kill_situation rows have action_value_class outside excellent/good/neutral."
            )
        if kill_high_cost_without_death:
            warnings.append(
                f"{kill_high_cost_without_death} kill_situation rows have high_cost_death true without death_within_5s."
            )

    opening_rows = dataset.filter(pl.col("source_situation_type") == "opening_duel_situation")
    if not opening_rows.is_empty():
        missing_opening_columns = [
            column for column in OPTIONAL_TARGET_COLUMNS if column not in opening_rows.columns
        ]
        if missing_opening_columns:
            print(f"  opening_duel_situation checks skipped columns: {missing_opening_columns}")
            warnings.append(f"Opening duel target columns are missing: {missing_opening_columns}")
            return
        opening_not_true = _count(opening_rows, ~_bool_expr("opening_duel_within_5s"))
        opening_result_null = _count(opening_rows, pl.col("opening_duel_won_within_5s").is_null())
        opening_bad_action = _count(opening_rows, ~pl.col("action_value_class").is_in(["good", "bad"]))
        print(f"  opening_duel_situation opening_duel_within_5s false: {opening_not_true}")
        print(f"  opening_duel_situation opening_duel_won_within_5s null: {opening_result_null}")
        print(f"  opening_duel_situation action_value_class not good/bad: {opening_bad_action}")
        if opening_not_true:
            warnings.append(
                f"{opening_not_true} opening_duel_situation rows have opening_duel_within_5s != true."
            )
        if opening_result_null:
            warnings.append(
                f"{opening_result_null} opening_duel_situation rows have null opening_duel_won_within_5s."
            )
        if opening_bad_action:
            warnings.append(
                f"{opening_bad_action} opening_duel_situation rows have action_value_class outside good/bad."
            )

    missing_expected = [
        source_type
        for source_type in sorted(EXPECTED_SOURCE_TYPES)
        if dataset.filter(pl.col("source_situation_type") == source_type).is_empty()
    ]
    if missing_expected:
        warnings.append(f"Expected source types have no rows: {missing_expected}")


def _print_target_combinations(dataset: pl.DataFrame) -> None:
    print("target combination counts")
    _print_pair_counts(dataset, "death_within_5s", "kill_within_5s")
    _print_pair_counts(dataset, "death_within_5s", "high_cost_death_within_5s")
    _print_pair_counts(dataset, "kill_within_5s", "high_impact_kill_within_5s")
    _print_pair_counts(dataset, "opening_duel_within_5s", "opening_duel_won_within_5s")


def _print_null_checks(dataset: pl.DataFrame) -> None:
    print("null counts")
    for column in NULL_CHECK_COLUMNS:
        count = _null_count(dataset, column)
        value = "missing" if count is None else count
        print(f"  {column}: {value}")


def _print_distributions(dataset: pl.DataFrame) -> None:
    print("distribution checks")
    _print_group_counts(dataset, "match_id", "rows per match")
    _print_group_counts(dataset, "map_name", "rows per map")
    _print_group_counts(dataset, "source_situation_type", "rows per source_situation_type")
    _print_group_counts(dataset, "seconds_before_event", "rows per seconds_before_event")
    _print_group_counts(dataset, "action_value_class", "action_value_class distribution")
    print("target distributions")
    for column in TARGET_DISTRIBUTION_COLUMNS:
        _print_bool_distribution(dataset, column)


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset
    failures: list[str] = []
    warnings: list[str] = []

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        print("VALIDATION FAILED")
        raise SystemExit(1)

    dataset = pl.read_parquet(dataset_path)
    _validate_basic_structure(dataset_path, dataset, failures)
    _validate_anti_leakage(dataset, failures)
    _validate_offsets(dataset, failures)
    _validate_source_type_consistency(dataset, failures, warnings)
    _print_target_combinations(dataset)
    _print_null_checks(dataset)
    _print_distributions(dataset)

    if warnings:
        print("warnings")
        for warning in warnings:
            print(f"  - {warning}")

    if failures:
        print("failures")
        for failure in failures:
            print(f"  - {failure}")
        print("VALIDATION FAILED")
        raise SystemExit(1)

    if warnings:
        print("VALIDATION PASSED WITH WARNINGS")
        return

    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
