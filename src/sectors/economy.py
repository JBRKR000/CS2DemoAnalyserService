"""Economy analysis for CS2 demos.

Computes per-round economy context and aggregated economy summary per player.
Uses awpy v2 + polars dataframes only.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from coach_metrics import _has_cols, _kills_df, _safe_df


_WEAPON_VALUES: dict[str, int] = {
    "glock": 200,
    "usp_silencer": 200,
    "usp": 200,
    "p2000": 200,
    "p250": 300,
    "five_seven": 500,
    "tec9": 500,
    "cz75a": 500,
    "deagle": 700,
    "revolver": 600,
    "nova": 1050,
    "xm1014": 2000,
    "mag7": 1300,
    "sawedoff": 1100,
    "mp9": 1250,
    "mac10": 1050,
    "mp7": 1500,
    "mp5sd": 1500,
    "ump45": 1200,
    "p90": 2350,
    "bizon": 1400,
    "famas": 2050,
    "galilar": 1800,
    "m4a1": 3100,
    "m4a1_silencer": 2900,
    "ak47": 2700,
    "aug": 3300,
    "sg556": 3000,
    "ssg08": 1700,
    "awp": 4750,
    "g3sg1": 5000,
    "scar20": 5000,
    "m249": 5200,
    "negev": 1700,
    "hegrenade": 300,
    "flashbang": 200,
    "smokegrenade": 300,
    "incgrenade": 600,
    "molotov": 400,
    "decoy": 50,
}


def _empty_economy_per_round() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "steamid": pl.UInt64,
            "name": pl.Utf8,
            "side": pl.Utf8,
            "round_num": pl.Int64,
            "equip_value": pl.Int64,
            "buy_type": pl.Utf8,
            "round_winner": pl.Boolean,
            "team_spent_avg": pl.Float64,
            "relative_spend": pl.Float64,
        }
    )


def _empty_economy_summary() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "steamid": pl.UInt64,
            "name": pl.Utf8,
            "full_buy_win_rate": pl.Float64,
            "force_win_rate": pl.Float64,
            "eco_kills": pl.Int64,
            "broken_economy_rounds": pl.Int64,
            "save_rounds": pl.Int64,
        }
    )


def _classify_buy_type_expr() -> pl.Expr:
    return (
        pl.when(pl.col("equip_value") <= 999)
        .then(pl.lit("eco"))
        .when(pl.col("equip_value") <= 2499)
        .then(pl.lit("semi_eco"))
        .when(pl.col("equip_value") <= 3999)
        .then(pl.lit("force"))
        .otherwise(pl.lit("full_buy"))
        .alias("buy_type")
    )


def _players_per_round_from_ticks(ticks: pl.DataFrame) -> pl.DataFrame:
    required = ["steamid", "name", "side", "round_num"]
    if not _has_cols(ticks, required):
        return pl.DataFrame(schema={"steamid": pl.UInt64, "name": pl.Utf8, "side": pl.Utf8, "round_num": pl.Int64})
    return ticks.select(required).drop_nulls(required).unique()


def _equip_from_ticks(ticks: pl.DataFrame) -> pl.DataFrame:
    if not _has_cols(ticks, ["steamid", "round_num", "current_equip_value"]):
        return pl.DataFrame(schema={"steamid": pl.UInt64, "round_num": pl.Int64, "equip_value": pl.Int64})

    # Preferred: snapshot right after freeze time ends (first non-freeze tick in round).
    if _has_cols(ticks, ["steamid", "round_num", "current_equip_value", "tick", "is_freeze_period"]):
        post_freeze = (
            ticks.select(["steamid", "round_num", "tick", "current_equip_value", "is_freeze_period"])
            .drop_nulls(["steamid", "round_num", "tick", "current_equip_value", "is_freeze_period"])
            .with_columns(pl.col("is_freeze_period").cast(pl.Boolean))
            .filter(~pl.col("is_freeze_period"))
            .sort(["round_num", "steamid", "tick"])
            .group_by(["steamid", "round_num"], maintain_order=True)
            .agg(pl.first("current_equip_value").cast(pl.Int64).alias("equip_value"))
        )
        if not post_freeze.is_empty():
            return post_freeze

    # Fallback: earliest tick snapshot in round (less leakage than max over whole round).
    if _has_cols(ticks, ["steamid", "round_num", "current_equip_value", "tick"]):
        return (
            ticks.select(["steamid", "round_num", "tick", "current_equip_value"])
            .drop_nulls(["steamid", "round_num", "tick", "current_equip_value"])
            .sort(["round_num", "steamid", "tick"])
            .group_by(["steamid", "round_num"], maintain_order=True)
            .agg(pl.first("current_equip_value").cast(pl.Int64).alias("equip_value"))
        )

    # Last-resort compatibility fallback for sparse tick schemas.
    return (
        ticks.select(["steamid", "round_num", "current_equip_value"])
        .drop_nulls(["steamid", "round_num", "current_equip_value"])
        .group_by(["steamid", "round_num"])
        .agg(pl.col("current_equip_value").max().cast(pl.Int64).alias("equip_value"))
    )


def _equip_from_events_fallback(demo: Any, players_round: pl.DataFrame) -> pl.DataFrame:
    # Approximation fallback:
    # if current_equip_value is unavailable, estimate round equipment from
    # weapons observed in kills and grenades (lower bound, event-based only).
    kills = _kills_df(demo)
    grenades = _safe_df(getattr(demo, "grenades", None))

    kill_values = pl.DataFrame(schema={"steamid": pl.UInt64, "round_num": pl.Int64, "weapon_value": pl.Int64})
    if _has_cols(kills, ["attacker_steamid", "round_num", "weapon"]):
        kv = (
            kills.select(
                [
                    pl.col("attacker_steamid").alias("steamid"),
                    "round_num",
                    pl.col("weapon").cast(pl.Utf8).str.to_lowercase().alias("weapon"),
                ]
            )
            .drop_nulls(["steamid", "round_num", "weapon"])
            .with_columns(
                pl.col("weapon")
                .replace(_WEAPON_VALUES, default=0)
                .cast(pl.Int64)
                .alias("weapon_value")
            )
            .group_by(["steamid", "round_num"])
            .agg(pl.col("weapon_value").max())
        )
        kill_values = kv

    grenade_values = pl.DataFrame(schema={"steamid": pl.UInt64, "round_num": pl.Int64, "grenade_value": pl.Int64})
    if _has_cols(grenades, ["thrower_steamid", "round_num", "grenade_type"]):
        gv = (
            grenades.select(
                [
                    pl.col("thrower_steamid").alias("steamid"),
                    "round_num",
                    pl.col("grenade_type").cast(pl.Utf8).str.to_lowercase().alias("grenade_type"),
                ]
            )
            .drop_nulls(["steamid", "round_num", "grenade_type"])
            .with_columns(
                pl.col("grenade_type")
                .replace(_WEAPON_VALUES, default=0)
                .cast(pl.Int64)
                .alias("grenade_value")
            )
            .group_by(["steamid", "round_num"])
            .agg(pl.col("grenade_value").sum())
        )
        grenade_values = gv

    estimated = players_round.select(["steamid", "round_num"]).unique()
    if not kill_values.is_empty():
        estimated = estimated.join(kill_values, on=["steamid", "round_num"], how="left")
    if not grenade_values.is_empty():
        estimated = estimated.join(grenade_values, on=["steamid", "round_num"], how="left")

    return estimated.with_columns(
        (
            pl.col("weapon_value").fill_null(0) + pl.col("grenade_value").fill_null(0)
        )
        .cast(pl.Int64)
        .alias("equip_value")
    ).select(["steamid", "round_num", "equip_value"])


def _round_winner_lookup(demo: Any) -> pl.DataFrame:
    rounds = _safe_df(getattr(demo, "rounds", None))
    if not _has_cols(rounds, ["round_num", "winner"]):
        return pl.DataFrame(schema={"round_num": pl.Int64, "winner_side": pl.Utf8})
    return (
        rounds.select(["round_num", "winner"])
        .drop_nulls(["round_num"])
        .with_columns(pl.col("winner").cast(pl.Utf8).str.to_lowercase().alias("winner_side"))
        .select(["round_num", "winner_side"])
    )


def build_economy_stats(demo: Any) -> dict[str, pl.DataFrame]:
    ticks = _safe_df(getattr(demo, "ticks", None))
    players_round = _players_per_round_from_ticks(ticks)
    if players_round.is_empty():
        return {
            "economy_per_round": _empty_economy_per_round(),
            "economy_summary": _empty_economy_summary(),
        }

    equip = _equip_from_ticks(ticks)
    if equip.is_empty():
        equip = _equip_from_events_fallback(demo, players_round)

    per_round = players_round.join(equip, on=["steamid", "round_num"], how="left").with_columns(
        pl.col("equip_value").fill_null(0).cast(pl.Int64)
    )

    per_round = per_round.with_columns(_classify_buy_type_expr())

    per_round = per_round.join(
        per_round.group_by(["round_num", "side"]).agg(pl.col("equip_value").mean().alias("team_spent_avg")),
        on=["round_num", "side"],
        how="left",
    )

    per_round = per_round.with_columns(
        pl.when(pl.col("team_spent_avg") > 0)
        .then(((pl.col("equip_value") - pl.col("team_spent_avg")) / pl.col("team_spent_avg")).round(4))
        .otherwise(0.0)
        .alias("relative_spend")
    )

    winners = _round_winner_lookup(demo)
    per_round = per_round.join(winners, on="round_num", how="left").with_columns(
        (pl.col("side").cast(pl.Utf8).str.to_lowercase() == pl.col("winner_side"))
        .fill_null(False)
        .alias("round_winner")
    )

    economy_per_round = per_round.select(
        [
            "steamid",
            "name",
            "side",
            "round_num",
            "equip_value",
            "buy_type",
            "round_winner",
            "team_spent_avg",
            "relative_spend",
        ]
    )

    kills = _kills_df(demo)
    eco_kills = pl.DataFrame(schema={"steamid": pl.UInt64, "eco_kills": pl.Int64})
    if _has_cols(kills, ["attacker_steamid", "round_num"]):
        eco_kills = (
            kills.select([pl.col("attacker_steamid").alias("steamid"), "round_num"])
            .drop_nulls(["steamid", "round_num"])
            .join(
                economy_per_round.filter(pl.col("buy_type") == "eco").select(["steamid", "round_num"]),
                on=["steamid", "round_num"],
                how="inner",
            )
            .group_by("steamid")
            .agg(pl.len().cast(pl.Int64).alias("eco_kills"))
        )

    full_buy = (
        economy_per_round.filter(pl.col("buy_type") == "full_buy")
        .group_by("steamid")
        .agg(
            [
                pl.len().alias("full_buy_total"),
                pl.col("round_winner").cast(pl.Int64).sum().alias("full_buy_wins"),
            ]
        )
    )
    force = (
        economy_per_round.filter(pl.col("buy_type") == "force")
        .group_by("steamid")
        .agg(
            [
                pl.len().alias("force_total"),
                pl.col("round_winner").cast(pl.Int64).sum().alias("force_wins"),
            ]
        )
    )
    broken = (
        economy_per_round.with_columns((pl.col("relative_spend").abs() >= 0.4).alias("broken"))
        .group_by("steamid")
        .agg(pl.col("broken").cast(pl.Int64).sum().cast(pl.Int64).alias("broken_economy_rounds"))
    )
    saves = (
        economy_per_round.with_columns((pl.col("equip_value") < 500).alias("save"))
        .group_by("steamid")
        .agg(pl.col("save").cast(pl.Int64).sum().cast(pl.Int64).alias("save_rounds"))
    )

    summary = economy_per_round.select(["steamid", "name"]).unique()
    for frame in (full_buy, force, eco_kills, broken, saves):
        if not frame.is_empty():
            summary = summary.join(frame, on="steamid", how="left")

    for column in (
        "full_buy_total",
        "full_buy_wins",
        "force_total",
        "force_wins",
        "eco_kills",
        "broken_economy_rounds",
        "save_rounds",
    ):
        if column not in summary.columns:
            summary = summary.with_columns(pl.lit(0).alias(column))

    summary = summary.with_columns(
        [
            pl.col("full_buy_total").fill_null(0),
            pl.col("full_buy_wins").fill_null(0),
            pl.col("force_total").fill_null(0),
            pl.col("force_wins").fill_null(0),
            pl.col("eco_kills").fill_null(0),
            pl.col("broken_economy_rounds").fill_null(0),
            pl.col("save_rounds").fill_null(0),
        ]
    ).with_columns(
        [
            pl.when(pl.col("full_buy_total") > 0)
            .then((pl.col("full_buy_wins") / pl.col("full_buy_total") * 100.0).round(2))
            .otherwise(0.0)
            .alias("full_buy_win_rate"),
            pl.when(pl.col("force_total") > 0)
            .then((pl.col("force_wins") / pl.col("force_total") * 100.0).round(2))
            .otherwise(0.0)
            .alias("force_win_rate"),
        ]
    )

    economy_summary = summary.select(
        [
            "steamid",
            "name",
            "full_buy_win_rate",
            "force_win_rate",
            "eco_kills",
            "broken_economy_rounds",
            "save_rounds",
        ]
    )

    return {
        "economy_per_round": economy_per_round,
        "economy_summary": economy_summary,
    }
