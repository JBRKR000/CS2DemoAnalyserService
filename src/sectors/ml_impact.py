from __future__ import annotations

import polars as pl


TOP_EVENT_LIMIT = 10

ML_EVENT_IMPACT_SCHEMA: dict[str, pl.DataType] = {
    "match_id": pl.Utf8,
    "map_name": pl.Utf8,
    "round_num": pl.Int64,
    "side": pl.Utf8,
    "event_type": pl.Utf8,
    "tick_before": pl.Int64,
    "tick_after": pl.Int64,
    "event_tick": pl.Int64,
    "win_prob_before": pl.Float64,
    "win_prob_after": pl.Float64,
    "win_prob_delta": pl.Float64,
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
        pl.col("win_probability").cast(pl.Float64),
    ]
    if "map_name" in snapshots.columns:
        expressions.append(pl.col("map_name").fill_null("unknown").cast(pl.Utf8))
    else:
        expressions.append(pl.lit("unknown", dtype=pl.Utf8).alias("map_name"))

    return snapshots.with_columns(expressions)


def build_ml_impact_from_snapshots(snapshots_with_probs: pl.DataFrame) -> dict[str, pl.DataFrame]:
    normalized = _normalize_snapshots(snapshots_with_probs)
    if normalized.is_empty():
        empty = _empty_ml_event_impact()
        return {
            "ml_event_impact": empty,
            "top_positive_events": empty,
            "top_negative_events": empty,
        }

    pairing_keys = ["match_id", "round_num", "side", "event_type"]

    before = (
        normalized
        .filter(pl.col("snapshot_type") == "before_kill")
        .sort([*pairing_keys, "tick"])
        .with_columns(
            pl.int_range(pl.len()).over(pairing_keys).alias("event_rank")
        )
        .select(
            [
                *pairing_keys,
                "event_rank",
                "map_name",
                pl.col("tick").alias("tick_before"),
                pl.col("win_probability").alias("win_prob_before"),
            ]
        )
    )
    after = (
        normalized
        .filter(pl.col("snapshot_type") == "after_kill")
        .sort([*pairing_keys, "tick"])
        .with_columns(
            pl.int_range(pl.len()).over(pairing_keys).alias("event_rank")
        )
        .select(
            [
                *pairing_keys,
                "event_rank",
                "map_name",
                pl.col("tick").alias("tick_after"),
                pl.col("win_probability").alias("win_prob_after"),
            ]
        )
    )

    if before.is_empty() or after.is_empty():
        empty = _empty_ml_event_impact()
        return {
            "ml_event_impact": empty,
            "top_positive_events": empty,
            "top_negative_events": empty,
        }

    impact = (
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
                pl.coalesce([pl.col("map_name"), pl.col("map_name_after")]).alias("map_name"),
                pl.col("tick_after").alias("event_tick"),
                (pl.col("win_prob_after") - pl.col("win_prob_before")).alias("win_prob_delta"),
            ]
        )
        .select(
            [
                "match_id",
                "map_name",
                "round_num",
                "side",
                "event_type",
                "tick_before",
                "tick_after",
                "event_tick",
                "win_prob_before",
                "win_prob_after",
                "win_prob_delta",
            ]
        )
        .sort(["win_prob_delta", "match_id", "round_num", "tick_after"], descending=[True, False, False, False])
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
