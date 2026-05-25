from __future__ import annotations

import polars as pl


TOP_EVENT_LIMIT = 10

ML_EVENT_IMPACT_SCHEMA: dict[str, pl.DataType] = {
    "match_id": pl.Utf8,
    "map_name": pl.Utf8,
    "round_num": pl.Int64,
    "side": pl.Utf8,
    "event_type": pl.Utf8,
    "event_id": pl.Utf8,
    "killer_steamid": pl.UInt64,
    "victim_steamid": pl.UInt64,
    "killer_name": pl.Utf8,
    "victim_name": pl.Utf8,
    "weapon": pl.Utf8,
    "killer_side": pl.Utf8,
    "victim_side": pl.Utf8,
    "kill_context_type": pl.Utf8,
    "tick_before": pl.Int64,
    "tick_after": pl.Int64,
    "event_tick": pl.Int64,
    "alive_team_before": pl.Int64,
    "alive_enemy_before": pl.Int64,
    "seconds_remaining_before": pl.Float64,
    "bomb_planted_before": pl.Boolean,
    "bomb_time_since_plant_before": pl.Float64,
    "bomb_time_remaining_before": pl.Float64,
    "opening_kill_for_team_before": pl.Boolean,
    "team_won_round": pl.Boolean,
    "win_prob_before": pl.Float64,
    "alive_team_after": pl.Int64,
    "alive_enemy_after": pl.Int64,
    "seconds_remaining_after": pl.Float64,
    "bomb_planted_after": pl.Boolean,
    "bomb_time_since_plant_after": pl.Float64,
    "bomb_time_remaining_after": pl.Float64,
    "opening_kill_for_team_after": pl.Boolean,
    "win_prob_after": pl.Float64,
    "win_prob_delta": pl.Float64,
    "alive_team_delta": pl.Int64,
    "alive_enemy_delta": pl.Int64,
    "alive_diff_before": pl.Int64,
    "alive_diff_after": pl.Int64,
}


def _empty_ml_event_impact() -> pl.DataFrame:
    return pl.DataFrame(schema=ML_EVENT_IMPACT_SCHEMA)


def _ensure_columns(dataset: pl.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Snapshots are missing required columns: {missing_columns}")


def _normalize_snapshots(snapshots: pl.DataFrame) -> pl.DataFrame:
    required_columns = [
        "match_id",
        "round_num",
        "tick",
        "snapshot_type",
        "side",
        "event_type",
        "alive_team",
        "alive_enemy",
        "seconds_remaining",
        "bomb_planted",
        "bomb_time_since_plant",
        "bomb_time_remaining",
        "opening_kill_for_team",
        "team_won_round",
        "win_probability",
    ]
    _ensure_columns(snapshots, required_columns)

    expressions: list[pl.Expr] = [
        pl.col("match_id").cast(pl.Utf8),
        pl.col("round_num").cast(pl.Int64),
        pl.col("tick").cast(pl.Int64),
        pl.col("snapshot_type").cast(pl.Utf8),
        pl.col("side").cast(pl.Utf8),
        pl.col("event_type").cast(pl.Utf8),
        pl.col("alive_team").cast(pl.Int64),
        pl.col("alive_enemy").cast(pl.Int64),
        pl.col("seconds_remaining").cast(pl.Float64),
        pl.col("bomb_planted").cast(pl.Boolean),
        pl.col("bomb_time_since_plant").cast(pl.Float64),
        pl.col("bomb_time_remaining").cast(pl.Float64),
        pl.col("opening_kill_for_team").cast(pl.Boolean),
        pl.col("team_won_round").cast(pl.Boolean),
        pl.col("win_probability").cast(pl.Float64),
    ]
    optional_context_columns: list[tuple[str, pl.DataType]] = [
        ("killer_steamid", pl.UInt64),
        ("victim_steamid", pl.UInt64),
        ("killer_name", pl.Utf8),
        ("victim_name", pl.Utf8),
        ("weapon", pl.Utf8),
        ("killer_side", pl.Utf8),
        ("victim_side", pl.Utf8),
        ("kill_context_type", pl.Utf8),
    ]
    for column_name, dtype in optional_context_columns:
        if column_name in snapshots.columns:
            expressions.append(pl.col(column_name).cast(dtype, strict=False))
        else:
            expressions.append(pl.lit(None, dtype=dtype).alias(column_name))
    if "event_id" in snapshots.columns:
        expressions.append(pl.col("event_id").cast(pl.Utf8))
    else:
        expressions.append(pl.lit(None, dtype=pl.Utf8).alias("event_id"))
    if "map_name" in snapshots.columns:
        expressions.append(pl.col("map_name").fill_null("unknown").cast(pl.Utf8))
    else:
        expressions.append(pl.lit("unknown", dtype=pl.Utf8).alias("map_name"))

    return snapshots.with_columns(expressions)


def _before_snapshot_columns(pairing_keys: list[str]) -> list[pl.Expr | str]:
    return [
        *pairing_keys,
        "map_name",
        "killer_steamid",
        "victim_steamid",
        "killer_name",
        "victim_name",
        "weapon",
        "killer_side",
        "victim_side",
        "kill_context_type",
        pl.col("tick").alias("tick_before"),
        pl.col("alive_team").alias("alive_team_before"),
        pl.col("alive_enemy").alias("alive_enemy_before"),
        pl.col("seconds_remaining").alias("seconds_remaining_before"),
        pl.col("bomb_planted").alias("bomb_planted_before"),
        pl.col("bomb_time_since_plant").alias("bomb_time_since_plant_before"),
        pl.col("bomb_time_remaining").alias("bomb_time_remaining_before"),
        pl.col("opening_kill_for_team").alias("opening_kill_for_team_before"),
        "team_won_round",
        pl.col("win_probability").alias("win_prob_before"),
    ]


def _after_snapshot_columns(pairing_keys: list[str]) -> list[pl.Expr | str]:
    return [
        *pairing_keys,
        "map_name",
        "killer_steamid",
        "victim_steamid",
        "killer_name",
        "victim_name",
        "weapon",
        "killer_side",
        "victim_side",
        "kill_context_type",
        pl.col("tick").alias("tick_after"),
        pl.col("alive_team").alias("alive_team_after"),
        pl.col("alive_enemy").alias("alive_enemy_after"),
        pl.col("seconds_remaining").alias("seconds_remaining_after"),
        pl.col("bomb_planted").alias("bomb_planted_after"),
        pl.col("bomb_time_since_plant").alias("bomb_time_since_plant_after"),
        pl.col("bomb_time_remaining").alias("bomb_time_remaining_after"),
        pl.col("opening_kill_for_team").alias("opening_kill_for_team_after"),
        pl.col("win_probability").alias("win_prob_after"),
    ]


def _finalize_impact(impact: pl.DataFrame) -> pl.DataFrame:
    return (
        impact.with_columns(
            [
                pl.coalesce([pl.col("map_name"), pl.col("map_name_after")]).alias("map_name"),
                pl.col("tick_after").alias("event_tick"),
                (pl.col("win_prob_after") - pl.col("win_prob_before")).alias("win_prob_delta"),
                (pl.col("alive_team_after") - pl.col("alive_team_before")).alias("alive_team_delta"),
                (pl.col("alive_enemy_after") - pl.col("alive_enemy_before")).alias("alive_enemy_delta"),
                (pl.col("alive_team_before") - pl.col("alive_enemy_before")).alias("alive_diff_before"),
                (pl.col("alive_team_after") - pl.col("alive_enemy_after")).alias("alive_diff_after"),
            ]
        )
        .select(
            [
                "match_id",
                "map_name",
                "round_num",
                "side",
                "event_type",
                "event_id",
                "killer_steamid",
                "victim_steamid",
                "killer_name",
                "victim_name",
                "weapon",
                "killer_side",
                "victim_side",
                "kill_context_type",
                "tick_before",
                "tick_after",
                "event_tick",
                "alive_team_before",
                "alive_enemy_before",
                "seconds_remaining_before",
                "bomb_planted_before",
                "bomb_time_since_plant_before",
                "bomb_time_remaining_before",
                "opening_kill_for_team_before",
                "team_won_round",
                "win_prob_before",
                "alive_team_after",
                "alive_enemy_after",
                "seconds_remaining_after",
                "bomb_planted_after",
                "bomb_time_since_plant_after",
                "bomb_time_remaining_after",
                "opening_kill_for_team_after",
                "win_prob_after",
                "win_prob_delta",
                "alive_team_delta",
                "alive_enemy_delta",
                "alive_diff_before",
                "alive_diff_after",
            ]
        )
    )


def _build_impact_with_event_id(normalized: pl.DataFrame) -> pl.DataFrame:
    pairing_keys = ["match_id", "round_num", "side", "event_type", "event_id"]

    before = (
        normalized
        .filter(pl.col("snapshot_type") == "before_kill")
        .select(_before_snapshot_columns(pairing_keys))
    )
    after = (
        normalized
        .filter(pl.col("snapshot_type") == "after_kill")
        .select(_after_snapshot_columns(pairing_keys))
    )

    return (
        before.join(
            after,
            on=pairing_keys,
            how="inner",
            suffix="_after",
        )
        .filter(pl.col("tick_after") >= pl.col("tick_before"))
        .pipe(_finalize_impact)
    )


def _build_impact_with_event_rank(normalized: pl.DataFrame) -> pl.DataFrame:
    pairing_keys = ["match_id", "round_num", "side", "event_type"]

    before = (
        normalized
        .filter(pl.col("snapshot_type") == "before_kill")
        .sort([*pairing_keys, "tick"])
        .with_columns(
            pl.int_range(pl.len()).over(pairing_keys).alias("event_rank")
        )
        .select([*_before_snapshot_columns(pairing_keys), "event_rank"])
    )
    after = (
        normalized
        .filter(pl.col("snapshot_type") == "after_kill")
        .sort([*pairing_keys, "tick"])
        .with_columns(
            pl.int_range(pl.len()).over(pairing_keys).alias("event_rank")
        )
        .select([*_after_snapshot_columns(pairing_keys), "event_rank"])
    )

    return (
        before.join(
            after,
            on=[*pairing_keys, "event_rank"],
            how="left",
            suffix="_after",
        )
        .drop_nulls(["tick_after", "win_prob_after"])
        .filter(pl.col("tick_after") > pl.col("tick_before"))
        .with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("event_id"),
            ]
        )
        .pipe(_finalize_impact)
    )


def build_ml_impact_from_snapshots(snapshots_with_probs: pl.DataFrame) -> dict[str, pl.DataFrame]:
    normalized = _normalize_snapshots(snapshots_with_probs)
    if normalized.is_empty():
        empty = _empty_ml_event_impact()
        return {
            "ml_event_impact": empty,
            "top_positive_events": empty,
            "top_negative_events": empty,
        }

    if "event_id" in snapshots_with_probs.columns:
        impact = _build_impact_with_event_id(normalized)
    else:
        impact = _build_impact_with_event_rank(normalized)

    if impact.is_empty():
        empty = _empty_ml_event_impact()
        return {
            "ml_event_impact": empty,
            "top_positive_events": empty,
            "top_negative_events": empty,
        }

    impact = impact.sort(
        ["win_prob_delta", "match_id", "round_num", "tick_after"],
        descending=[True, False, False, False],
    )

    top_positive_events = impact.head(TOP_EVENT_LIMIT)
    top_negative_events = impact.sort(
        ["win_prob_delta", "match_id", "round_num", "tick_after"],
        descending=[False, False, False, False],
    ).head(TOP_EVENT_LIMIT)

    return {
        "ml_event_impact": impact,
        "top_positive_events": top_positive_events,
        "top_negative_events": top_negative_events,
    }
