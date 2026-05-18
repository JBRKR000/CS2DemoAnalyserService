"""Round timeline and player impact statistics for CS2 demos using Polars."""

from __future__ import annotations

from typing import Any

import polars as pl

from coach_metrics import _damages_df, _has_cols, _kills_df, _safe_df


TRADE_WINDOW_TICKS: int = 640

TIMELINE_SCHEMA: dict[str, pl.DataType] = {
    "round_num": pl.Int64,
    "tick": pl.Int64,
    "event_type": pl.Utf8,
    "steamid": pl.UInt64,
    "name": pl.Utf8,
    "side": pl.Utf8,
    "target_steamid": pl.UInt64,
    "target_name": pl.Utf8,
    "target_side": pl.Utf8,
    "weapon": pl.Utf8,
    "is_headshot": pl.Boolean,
    "is_opening_kill": pl.Boolean,
    "is_opening_death": pl.Boolean,
    "is_traded_death": pl.Boolean,
    "is_trade_kill": pl.Boolean,
    "trade_delay_ticks": pl.Int64,
    "damage_before_death": pl.Float64,
    "round_phase": pl.Utf8,
}

PLAYER_IMPACT_SCHEMA: dict[str, pl.DataType] = {
    "steamid": pl.UInt64,
    "name": pl.Utf8,
    "side": pl.Utf8,
    "kills": pl.Int64,
    "deaths": pl.Int64,
    "opening_kills": pl.Int64,
    "opening_deaths": pl.Int64,
    "opening_duels": pl.Int64,
    "opening_duel_win_pct": pl.Float64,
    "traded_deaths": pl.Int64,
    "untraded_deaths": pl.Int64,
    "untraded_death_rate": pl.Float64,
    "trade_kills": pl.Int64,
    "avg_damage_before_death": pl.Float64,
    "deaths_with_0_damage": pl.Int64,
    "deaths_under_40_damage": pl.Int64,
    "early_deaths": pl.Int64,
    "mid_deaths": pl.Int64,
    "late_deaths": pl.Int64,
}

ROUND_IMPACT_SCHEMA: dict[str, pl.DataType] = {
    "round_num": pl.Int64,
    "side": pl.Utf8,
    "opening_kill_steamid": pl.UInt64,
    "opening_death_steamid": pl.UInt64,
    "opening_kill_side": pl.Utf8,
    "opening_death_side": pl.Utf8,
    "kills": pl.Int64,
    "deaths": pl.Int64,
    "traded_deaths": pl.Int64,
    "untraded_deaths": pl.Int64,
    "trade_kills": pl.Int64,
}


def _empty_timeline_events() -> pl.DataFrame:
    return pl.DataFrame(schema=TIMELINE_SCHEMA)


def _empty_player_impact_summary() -> pl.DataFrame:
    return pl.DataFrame(schema=PLAYER_IMPACT_SCHEMA)


def _empty_round_impact_summary() -> pl.DataFrame:
    return pl.DataFrame(schema=ROUND_IMPACT_SCHEMA)


def _safe_rounds_df(demo: Any) -> pl.DataFrame:
    rounds = _safe_df(getattr(demo, "rounds", None))
    if rounds.is_empty() or "round_num" not in rounds.columns:
        return pl.DataFrame()

    start_col = "freeze_end" if "freeze_end" in rounds.columns else ("start" if "start" in rounds.columns else None)
    end_col = "end" if "end" in rounds.columns else ("official_end" if "official_end" in rounds.columns else None)
    if start_col is None or end_col is None:
        return pl.DataFrame()

    return (
        rounds.select(
            [
                pl.col("round_num").cast(pl.Int64, strict=False),
                pl.col(start_col).cast(pl.Float64, strict=False).alias("round_start"),
                pl.col(end_col).cast(pl.Float64, strict=False).alias("round_end"),
            ]
        )
        .drop_nulls(["round_num", "round_start", "round_end"])
        .filter(pl.col("round_end") > pl.col("round_start"))
        .unique(subset=["round_num"], keep="first")
    )


def _prepare_kills(demo: Any) -> pl.DataFrame:
    kills = _kills_df(demo)
    required = [
        "round_num",
        "tick",
        "attacker_steamid",
        "attacker_name",
        "attacker_side",
        "victim_steamid",
        "victim_name",
        "victim_side",
        "weapon",
        "headshot",
    ]
    if kills.is_empty() or not _has_cols(kills, ["round_num", "tick", "attacker_steamid", "victim_steamid"]):
        return pl.DataFrame()

    for col, dtype in {
        "round_num": pl.Int64,
        "tick": pl.Int64,
        "attacker_steamid": pl.UInt64,
        "victim_steamid": pl.UInt64,
    }.items():
        if col in kills.columns:
            kills = kills.with_columns(pl.col(col).cast(dtype, strict=False))

    defaults: list[pl.Expr] = []
    if "attacker_name" not in kills.columns:
        defaults.append(pl.lit(None, dtype=pl.Utf8).alias("attacker_name"))
    if "attacker_side" not in kills.columns:
        defaults.append(pl.lit(None, dtype=pl.Utf8).alias("attacker_side"))
    if "victim_name" not in kills.columns:
        defaults.append(pl.lit(None, dtype=pl.Utf8).alias("victim_name"))
    if "victim_side" not in kills.columns:
        defaults.append(pl.lit(None, dtype=pl.Utf8).alias("victim_side"))
    if "weapon" not in kills.columns:
        defaults.append(pl.lit(None, dtype=pl.Utf8).alias("weapon"))
    if "headshot" not in kills.columns:
        defaults.append(pl.lit(False, dtype=pl.Boolean).alias("headshot"))
    if defaults:
        kills = kills.with_columns(defaults)

    return (
        kills.select(required)
        .drop_nulls(["round_num", "tick", "attacker_steamid", "victim_steamid"])
        .with_columns(
            [
                pl.col("attacker_side").cast(pl.Utf8).str.to_uppercase(),
                pl.col("victim_side").cast(pl.Utf8).str.to_uppercase(),
                pl.col("headshot").fill_null(False).cast(pl.Boolean),
            ]
        )
        .sort(["round_num", "tick"])
        .with_row_count("kill_id")
    )


def _opening_flags(kills: pl.DataFrame) -> pl.DataFrame:
    opening = (
        kills.group_by("round_num", maintain_order=True)
        .agg(pl.first("kill_id").alias("opening_kill_id"))
    )
    return kills.join(opening, on="round_num", how="left").with_columns(
        (pl.col("kill_id") == pl.col("opening_kill_id")).alias("is_opening")
    )


def _trade_resolution(kills: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    prior = kills.select(
        [
            "kill_id",
            "round_num",
            pl.col("tick").alias("death_tick"),
            pl.col("attacker_steamid").alias("original_killer_steamid"),
            pl.col("victim_steamid").alias("dead_player_steamid"),
            pl.col("victim_side").alias("dead_player_side"),
        ]
    )
    later = kills.select(
        [
            pl.col("kill_id").alias("trade_kill_id"),
            "round_num",
            pl.col("tick").alias("trade_tick"),
            pl.col("attacker_steamid").alias("trade_attacker_steamid"),
            pl.col("attacker_side").alias("trade_attacker_side"),
            pl.col("victim_steamid").alias("trade_victim_steamid"),
        ]
    )

    trade_candidates = (
        prior.join(later, on="round_num", how="inner")
        .filter(pl.col("trade_tick") > pl.col("death_tick"))
        .filter(pl.col("trade_tick") <= (pl.col("death_tick") + TRADE_WINDOW_TICKS))
        .filter(pl.col("trade_attacker_side") == pl.col("dead_player_side"))
        .filter(pl.col("trade_victim_steamid") == pl.col("original_killer_steamid"))
        .with_columns((pl.col("trade_tick") - pl.col("death_tick")).alias("trade_delay_ticks"))
        .sort(["kill_id", "trade_tick", "trade_kill_id"])
        .group_by("kill_id", maintain_order=True)
        .agg(
            [
                pl.first("trade_kill_id").alias("trade_kill_id"),
                pl.first("trade_tick").alias("trade_tick"),
                pl.first("trade_delay_ticks").alias("trade_delay_ticks"),
            ]
        )
    )

    death_flags = (
        prior.join(trade_candidates, on="kill_id", how="left")
        .select(
            [
                "kill_id",
                pl.col("trade_kill_id").is_not_null().alias("is_traded_death"),
                pl.col("trade_delay_ticks").cast(pl.Int64, strict=False).alias("death_trade_delay_ticks"),
                "trade_kill_id",
            ]
        )
    )

    kill_flags = (
        trade_candidates.select(
            [
                pl.col("trade_kill_id").alias("kill_id"),
                pl.lit(True).alias("is_trade_kill"),
                pl.col("trade_delay_ticks").cast(pl.Int64, strict=False).alias("kill_trade_delay_ticks"),
            ]
        )
        .group_by("kill_id")
        .agg(
            [
                pl.max("is_trade_kill").alias("is_trade_kill"),
                pl.min("kill_trade_delay_ticks").alias("kill_trade_delay_ticks"),
            ]
        )
    )

    return death_flags, kill_flags


def _damage_before_death(demo: Any, kills: pl.DataFrame) -> pl.DataFrame:
    damages = _damages_df(demo)
    if damages.is_empty() or not _has_cols(damages, ["round_num", "tick", "attacker_steamid", "damage"]):
        return kills.select(["kill_id"]).with_columns(pl.lit(0.0, dtype=pl.Float64).alias("damage_before_death"))

    dmg = (
        damages.select(
            [
                pl.col("round_num").cast(pl.Int64, strict=False),
                pl.col("tick").cast(pl.Int64, strict=False).alias("damage_tick"),
                pl.col("attacker_steamid").cast(pl.UInt64, strict=False).alias("steamid"),
                pl.col("damage").cast(pl.Float64, strict=False),
            ]
        )
        .drop_nulls(["round_num", "damage_tick", "steamid", "damage"])
    )

    deaths = (
        kills.select(
            [
                "kill_id",
                "round_num",
                pl.col("tick").alias("death_tick"),
                pl.col("victim_steamid").alias("steamid"),
            ]
        )
    )

    damage_sum = (
        deaths.join(dmg, on=["round_num", "steamid"], how="left")
        .filter(pl.col("damage_tick") < pl.col("death_tick"))
        .group_by("kill_id")
        .agg(pl.col("damage").sum().alias("damage_before_death"))
    )

    return (
        kills.select(["kill_id"])
        .join(damage_sum, on="kill_id", how="left")
        .with_columns(pl.col("damage_before_death").fill_null(0.0).cast(pl.Float64, strict=False))
    )


def _apply_round_phase(kills: pl.DataFrame, rounds: pl.DataFrame) -> pl.DataFrame:
    if rounds.is_empty():
        return kills.with_columns(pl.lit("unknown", dtype=pl.Utf8).alias("round_phase"))

    with_bounds = kills.join(rounds, on="round_num", how="left")
    return with_bounds.with_columns(
        [
            pl.when(
                pl.col("round_start").is_null()
                | pl.col("round_end").is_null()
                | (pl.col("round_end") <= pl.col("round_start"))
            )
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise((pl.col("tick") - pl.col("round_start")) / (pl.col("round_end") - pl.col("round_start")))
            .alias("round_progress")
        ]
    ).with_columns(
        pl.when(pl.col("round_progress").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("round_progress") <= 0.30)
        .then(pl.lit("early"))
        .when(pl.col("round_progress") < 0.70)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("late"))
        .alias("round_phase")
    )


def _build_timeline_events(kills: pl.DataFrame) -> pl.DataFrame:
    kill_rows = kills.select(
        [
            "round_num",
            "tick",
            pl.lit("kill").alias("event_type"),
            pl.col("attacker_steamid").alias("steamid"),
            pl.col("attacker_name").alias("name"),
            pl.col("attacker_side").alias("side"),
            pl.col("victim_steamid").alias("target_steamid"),
            pl.col("victim_name").alias("target_name"),
            pl.col("victim_side").alias("target_side"),
            "weapon",
            pl.col("headshot").alias("is_headshot"),
            pl.col("is_opening").alias("is_opening_kill"),
            pl.lit(False).alias("is_opening_death"),
            pl.lit(False).alias("is_traded_death"),
            pl.col("is_trade_kill"),
            pl.col("kill_trade_delay_ticks").alias("trade_delay_ticks"),
            pl.lit(None, dtype=pl.Float64).alias("damage_before_death"),
            "round_phase",
        ]
    )

    death_rows = kills.select(
        [
            "round_num",
            "tick",
            pl.lit("death").alias("event_type"),
            pl.col("victim_steamid").alias("steamid"),
            pl.col("victim_name").alias("name"),
            pl.col("victim_side").alias("side"),
            pl.col("attacker_steamid").alias("target_steamid"),
            pl.col("attacker_name").alias("target_name"),
            pl.col("attacker_side").alias("target_side"),
            "weapon",
            pl.col("headshot").alias("is_headshot"),
            pl.lit(False).alias("is_opening_kill"),
            pl.col("is_opening").alias("is_opening_death"),
            pl.col("is_traded_death"),
            pl.lit(False).alias("is_trade_kill"),
            pl.col("death_trade_delay_ticks").alias("trade_delay_ticks"),
            pl.col("damage_before_death"),
            "round_phase",
        ]
    )

    return pl.concat([kill_rows, death_rows], how="vertical_relaxed").select(list(TIMELINE_SCHEMA.keys()))


def _player_impact_summary(timeline: pl.DataFrame) -> pl.DataFrame:
    if timeline.is_empty():
        return _empty_player_impact_summary()

    summary = (
        timeline.group_by(["steamid", "name", "side"])
        .agg(
            [
                (pl.col("event_type") == "kill").sum().cast(pl.Int64).alias("kills"),
                (pl.col("event_type") == "death").sum().cast(pl.Int64).alias("deaths"),
                ((pl.col("event_type") == "kill") & pl.col("is_opening_kill")).sum().cast(pl.Int64).alias("opening_kills"),
                ((pl.col("event_type") == "death") & pl.col("is_opening_death")).sum().cast(pl.Int64).alias("opening_deaths"),
                ((pl.col("event_type") == "death") & pl.col("is_traded_death")).sum().cast(pl.Int64).alias("traded_deaths"),
                ((pl.col("event_type") == "death") & (~pl.col("is_traded_death"))).sum().cast(pl.Int64).alias("untraded_deaths"),
                ((pl.col("event_type") == "kill") & pl.col("is_trade_kill")).sum().cast(pl.Int64).alias("trade_kills"),
                pl.when(pl.col("event_type") == "death").then(pl.col("damage_before_death")).otherwise(None).mean().alias("avg_damage_before_death"),
                ((pl.col("event_type") == "death") & (pl.col("damage_before_death").fill_null(0.0) <= 0.0)).sum().cast(pl.Int64).alias("deaths_with_0_damage"),
                ((pl.col("event_type") == "death") & (pl.col("damage_before_death").fill_null(0.0) < 40.0)).sum().cast(pl.Int64).alias("deaths_under_40_damage"),
                ((pl.col("event_type") == "death") & (pl.col("round_phase") == "early")).sum().cast(pl.Int64).alias("early_deaths"),
                ((pl.col("event_type") == "death") & (pl.col("round_phase") == "mid")).sum().cast(pl.Int64).alias("mid_deaths"),
                ((pl.col("event_type") == "death") & (pl.col("round_phase") == "late")).sum().cast(pl.Int64).alias("late_deaths"),
            ]
        )
        .with_columns((pl.col("opening_kills") + pl.col("opening_deaths")).alias("opening_duels"))
        .with_columns(
            [
                pl.when(pl.col("opening_duels") > 0)
                .then((pl.col("opening_kills") / pl.col("opening_duels") * 100.0))
                .otherwise(0.0)
                .alias("opening_duel_win_pct"),
                pl.when(pl.col("deaths") > 0)
                .then((pl.col("untraded_deaths") / pl.col("deaths") * 100.0))
                .otherwise(0.0)
                .alias("untraded_death_rate"),
            ]
        )
    )

    all_rows = (
        summary.group_by(["steamid", "name"])
        .agg(
            [
                pl.lit("ALL").alias("side"),
                pl.col("kills").sum().alias("kills"),
                pl.col("deaths").sum().alias("deaths"),
                pl.col("opening_kills").sum().alias("opening_kills"),
                pl.col("opening_deaths").sum().alias("opening_deaths"),
                pl.col("traded_deaths").sum().alias("traded_deaths"),
                pl.col("untraded_deaths").sum().alias("untraded_deaths"),
                pl.col("trade_kills").sum().alias("trade_kills"),
                pl.col("deaths_with_0_damage").sum().alias("deaths_with_0_damage"),
                pl.col("deaths_under_40_damage").sum().alias("deaths_under_40_damage"),
                pl.col("early_deaths").sum().alias("early_deaths"),
                pl.col("mid_deaths").sum().alias("mid_deaths"),
                pl.col("late_deaths").sum().alias("late_deaths"),
                (pl.col("avg_damage_before_death") * pl.col("deaths")).sum().alias("damage_weighted_sum"),
            ]
        )
        .with_columns((pl.col("opening_kills") + pl.col("opening_deaths")).alias("opening_duels"))
        .with_columns(
            [
                pl.when(pl.col("opening_duels") > 0)
                .then((pl.col("opening_kills") / pl.col("opening_duels") * 100.0))
                .otherwise(0.0)
                .alias("opening_duel_win_pct"),
                pl.when(pl.col("deaths") > 0)
                .then((pl.col("untraded_deaths") / pl.col("deaths") * 100.0))
                .otherwise(0.0)
                .alias("untraded_death_rate"),
                pl.when(pl.col("deaths") > 0)
                .then(pl.col("damage_weighted_sum") / pl.col("deaths"))
                .otherwise(0.0)
                .alias("avg_damage_before_death"),
            ]
        )
        .drop("damage_weighted_sum")
    )

    ordered_cols = list(PLAYER_IMPACT_SCHEMA.keys())
    summary_aligned = summary.select(ordered_cols)
    all_rows_aligned = all_rows.select(ordered_cols)

    return pl.concat([summary_aligned, all_rows_aligned], how="vertical_relaxed").sort(["name", "side"])


def _round_impact_summary(kills: pl.DataFrame, timeline: pl.DataFrame) -> pl.DataFrame:
    if timeline.is_empty():
        return _empty_round_impact_summary()

    opening_per_round = (
        kills.filter(pl.col("is_opening"))
        .select(
            [
                "round_num",
                pl.col("attacker_steamid").alias("opening_kill_steamid"),
                pl.col("victim_steamid").alias("opening_death_steamid"),
                pl.col("attacker_side").alias("opening_kill_side"),
                pl.col("victim_side").alias("opening_death_side"),
            ]
        )
    )

    agg = (
        timeline.group_by(["round_num", "side"])
        .agg(
            [
                (pl.col("event_type") == "kill").sum().cast(pl.Int64).alias("kills"),
                (pl.col("event_type") == "death").sum().cast(pl.Int64).alias("deaths"),
                ((pl.col("event_type") == "death") & pl.col("is_traded_death")).sum().cast(pl.Int64).alias("traded_deaths"),
                ((pl.col("event_type") == "death") & (~pl.col("is_traded_death"))).sum().cast(pl.Int64).alias("untraded_deaths"),
                ((pl.col("event_type") == "kill") & pl.col("is_trade_kill")).sum().cast(pl.Int64).alias("trade_kills"),
            ]
        )
        .join(opening_per_round, on="round_num", how="left")
        .select(list(ROUND_IMPACT_SCHEMA.keys()))
        .sort(["round_num", "side"])
    )
    return agg


def build_round_timeline_stats(demo: Any) -> dict[str, pl.DataFrame]:
    """Build per-event timeline rows and derived impact summaries for a CS2 demo."""
    kills = _prepare_kills(demo)
    if kills.is_empty():
        return {
            "timeline_events": _empty_timeline_events(),
            "player_impact_summary": _empty_player_impact_summary(),
            "round_impact_summary": _empty_round_impact_summary(),
        }

    rounds = _safe_rounds_df(demo)
    kills = _opening_flags(kills)

    death_flags, kill_flags = _trade_resolution(kills)
    kills = kills.join(death_flags, on="kill_id", how="left").join(kill_flags, on="kill_id", how="left")
    kills = kills.with_columns(
        [
            pl.col("is_traded_death").fill_null(False),
            pl.col("is_trade_kill").fill_null(False),
            pl.col("death_trade_delay_ticks").cast(pl.Int64, strict=False),
            pl.col("kill_trade_delay_ticks").cast(pl.Int64, strict=False),
        ]
    )

    kills = kills.join(_damage_before_death(demo, kills), on="kill_id", how="left")
    kills = _apply_round_phase(kills, rounds)

    timeline = _build_timeline_events(kills)
    player_summary = _player_impact_summary(timeline)
    round_summary = _round_impact_summary(kills, timeline)

    return {
        "timeline_events": timeline,
        "player_impact_summary": player_summary,
        "round_impact_summary": round_summary,
    }
