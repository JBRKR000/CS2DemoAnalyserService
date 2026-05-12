"""
coach_metrics.py — statystyki dla awpy v2

Kolumny awpy v2 (Demo.kills, Demo.damages, Demo.ticks, Demo.grenades):

kills:
    tick, round_num,
    attacker_steamid, attacker_name, attacker_side,
    victim_steamid,   victim_name,   victim_side,
    assister_steamid, assister_name, assister_side,
    weapon, headshot, flash_assist

damages:
    tick, round_num,
    attacker_steamid, attacker_name, attacker_side,
    victim_steamid,   victim_name,   victim_side,
    weapon, damage, hitgroup

grenades:
    tick, round_num,
    thrower_steamid, thrower, thrower_side (opcjonalnie),
    grenade_type, entity_id, X, Y, Z

ticks (domyslne):
    tick, steamid, name, side, team_name, round_num
    (+ player_props jesli podane przy parse())

rounds:
    round_num, start, freeze_end, end, official_end,
    winner, reason, bomb_plant, bomb_site
"""

import logging
from typing import Iterable

import polars as pl


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_cols(df: pl.DataFrame, cols: Iterable[str]) -> bool:
    return all(col in df.columns for col in cols)


def _safe_df(obj) -> pl.DataFrame:
    if isinstance(obj, pl.DataFrame):
        return obj
    return pl.DataFrame()


def _pick_first_col(columns: Iterable[str], *candidates: str) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Raw frame accessors
# ---------------------------------------------------------------------------

def _kills_df(demo) -> pl.DataFrame:
    """Returns demo.kills normalised to awpy-v2 column names."""
    df = _safe_df(getattr(demo, "kills", None))
    if df.is_empty():
        return pl.DataFrame(schema={
            "tick": pl.Int64,
            "round_num": pl.Int64,
            "attacker_steamid": pl.UInt64,
            "attacker_side": pl.Utf8,
            "victim_steamid": pl.UInt64,
            "victim_side": pl.Utf8,
            "assister_steamid": pl.UInt64,
            "assister_side": pl.Utf8,
            "weapon": pl.Utf8,
            "headshot": pl.Boolean,
            "flash_assist": pl.Boolean,
        })

    for col in ("attacker_steamid", "victim_steamid", "assister_steamid"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.UInt64, strict=False))

    if "headshot" not in df.columns:
        df = df.with_columns(pl.lit(False).alias("headshot"))
    if "flash_assist" not in df.columns:
        df = df.with_columns(pl.lit(False).alias("flash_assist"))

    return df


def _damages_df(demo) -> pl.DataFrame:
    """Returns demo.damages normalised to awpy-v2 column names.
    awpy v2 uses 'damage', NOT 'hp_damage'.
    """
    df = _safe_df(getattr(demo, "damages", None))
    if df.is_empty():
        return pl.DataFrame(schema={
            "tick": pl.Int64,
            "round_num": pl.Int64,
            "attacker_steamid": pl.UInt64,
            "attacker_side": pl.Utf8,
            "victim_steamid": pl.UInt64,
            "victim_side": pl.Utf8,
            "weapon": pl.Utf8,
            "damage": pl.Int64,
        })

    # awpy v2: 'damage'; accept legacy names too
    dmg_col = _pick_first_col(df.columns, "damage", "hp_damage", "hpDamage", "dmg_health")
    if dmg_col and dmg_col != "damage":
        df = df.rename({dmg_col: "damage"})
    elif dmg_col is None:
        df = df.with_columns(pl.lit(0).cast(pl.Int64).alias("damage"))

    for col in ("attacker_steamid", "victim_steamid"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.UInt64, strict=False))

    return df


# ---------------------------------------------------------------------------
# _build_players_base
# ---------------------------------------------------------------------------

def _build_players_base(demo) -> pl.DataFrame:
    ticks = _safe_df(getattr(demo, "ticks", None))

    required = ["round_num", "steamid", "name", "side"]
    if not _has_cols(ticks, required):
        logger.warning("ticks dataframe does not contain required columns: %s", required)
        return pl.DataFrame(schema={
            "steamid": pl.UInt64,
            "name": pl.Utf8,
            "start_side": pl.Utf8,
            "first_round": pl.Int64,
        })

    ticks_clean = (
        ticks
        .select(required)
        .drop_nulls(required)
        .unique()
    )

    players = (
        ticks_clean
        .sort(["steamid", "round_num"])
        .group_by("steamid")
        .agg([
            pl.first("name").alias("name"),
            pl.first("side").alias("start_side"),
            pl.first("round_num").cast(pl.Int64).alias("first_round"),
        ])
        .sort("name")
    )

    return players


# ---------------------------------------------------------------------------
# _player_round_presence
# ---------------------------------------------------------------------------

def _player_round_presence(demo) -> pl.DataFrame:
    ticks = _safe_df(getattr(demo, "ticks", None))
    required = ["steamid", "name", "round_num"]

    if not _has_cols(ticks, required):
        return pl.DataFrame(schema={
            "steamid": pl.UInt64,
            "name": pl.Utf8,
            "rounds_played": pl.UInt32,
        })

    return (
        ticks
        .select(required)
        .drop_nulls(required)
        .unique()
        .group_by(["steamid", "name"])
        .agg(pl.n_unique("round_num").alias("rounds_played"))
    )


# ---------------------------------------------------------------------------
# _kills_stats
# ---------------------------------------------------------------------------

def _kills_stats(demo) -> pl.DataFrame:
    """Computes base overall metrics: kills, deaths, assists, hs_kills."""
    kills = _kills_df(demo)
    empty = pl.DataFrame(schema={
        "steamid": pl.UInt64,
        "kills": pl.UInt32,
        "deaths": pl.UInt32,
        "assists": pl.UInt32,
        "hs_kills": pl.UInt32,
    })

    if kills.is_empty():
        return empty

    frames: list[pl.DataFrame] = []

    if "attacker_steamid" in kills.columns:
        frames.append(
            kills.filter(pl.col("attacker_steamid").is_not_null())
            .group_by("attacker_steamid")
            .agg(pl.len().alias("kills"))
            .rename({"attacker_steamid": "steamid"})
        )

    if "victim_steamid" in kills.columns:
        frames.append(
            kills.filter(pl.col("victim_steamid").is_not_null())
            .group_by("victim_steamid")
            .agg(pl.len().alias("deaths"))
            .rename({"victim_steamid": "steamid"})
        )

    if "assister_steamid" in kills.columns:
        frames.append(
            kills.filter(pl.col("assister_steamid").is_not_null())
            .group_by("assister_steamid")
            .agg(pl.len().alias("assists"))
            .rename({"assister_steamid": "steamid"})
        )

    if _has_cols(kills, ["attacker_steamid", "headshot"]):
        frames.append(
            kills.filter(pl.col("attacker_steamid").is_not_null() & pl.col("headshot"))
            .group_by("attacker_steamid")
            .agg(pl.len().alias("hs_kills"))
            .rename({"attacker_steamid": "steamid"})
        )

    if not frames:
        return empty

    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on="steamid", how="full", coalesce=True)

    numeric_cols = [c for c in result.columns if c != "steamid"]
    result = result.with_columns([pl.col(c).fill_null(0) for c in numeric_cols])

    for col in ["kills", "deaths", "assists", "hs_kills"]:
        if col not in result.columns:
            result = result.with_columns(pl.lit(0).cast(pl.UInt32).alias(col))

    return result


# ---------------------------------------------------------------------------
# _adr_stats
# ---------------------------------------------------------------------------

def _adr_stats(demo) -> pl.DataFrame:
    """
    Computes raw total damage per player.
    """
    damages = _damages_df(demo)
    empty = pl.DataFrame(schema={"steamid": pl.UInt64, "total_damage": pl.Int64})

    if damages.is_empty() or not _has_cols(damages, ["attacker_steamid", "damage"]):
        return empty

    # Exclude team damage if side info is available
    dmg_filtered = damages.filter(pl.col("attacker_steamid").is_not_null())
    if _has_cols(damages, ["attacker_side", "victim_side"]):
        dmg_filtered = dmg_filtered.filter(
            pl.col("attacker_side").is_not_null()
            & pl.col("victim_side").is_not_null()
            & (pl.col("attacker_side") != pl.col("victim_side"))
        )

    damage_per_player = (
        dmg_filtered
        .group_by("attacker_steamid")
        .agg(pl.col("damage").sum().cast(pl.Int64).alias("total_damage"))
        .rename({"attacker_steamid": "steamid"})
    )

    return damage_per_player


# ---------------------------------------------------------------------------
# _kast_stats
# ---------------------------------------------------------------------------

def _kast_stats(demo) -> pl.DataFrame:
    """
    KAST = % of rounds where the player had a Kill, Assist, Survived, or was Traded.
    """
    TRADE_WINDOW_TICKS = 640

    ticks = _safe_df(getattr(demo, "ticks", None))
    kills = _kills_df(demo)
    empty = pl.DataFrame(schema={"steamid": pl.UInt64, "kast_rounds": pl.UInt32})

    if not _has_cols(ticks, ["steamid", "round_num"]):
        return empty

    player_rounds = (
        ticks
        .select(["steamid", "round_num"])
        .drop_nulls()
        .unique()
    )

    kast_events: list[pl.DataFrame] = []

    # K
    if _has_cols(kills, ["attacker_steamid", "round_num"]):
        kast_events.append(
            kills
            .filter(pl.col("attacker_steamid").is_not_null())
            .select([pl.col("attacker_steamid").alias("steamid"), "round_num"])
            .unique()
            .with_columns(pl.lit(True).alias("had_kill"))
        )

    # A
    if _has_cols(kills, ["assister_steamid", "round_num"]):
        kast_events.append(
            kills
            .filter(pl.col("assister_steamid").is_not_null())
            .select([pl.col("assister_steamid").alias("steamid"), "round_num"])
            .unique()
            .with_columns(pl.lit(True).alias("had_assist"))
        )

    # T — was traded
    if _has_cols(kills, ["tick", "round_num", "attacker_steamid", "attacker_side",
                          "victim_steamid", "victim_side"]):
        k = kills.select([
            "tick", "round_num", "attacker_steamid", "attacker_side", "victim_steamid",
        ]).filter(
            pl.col("attacker_steamid").is_not_null() & pl.col("victim_steamid").is_not_null()
        )
        prior = k.rename({
            "tick": "prior_tick",
            "attacker_steamid": "prior_killer",
            "attacker_side": "prior_killer_side",
            "victim_steamid": "prior_victim",
        })
        traded = (
            k
            .join(prior, on="round_num", how="inner")
            .filter(pl.col("victim_steamid") == pl.col("prior_killer"))
            .filter(pl.col("prior_tick") < pl.col("tick"))
            .filter((pl.col("tick") - pl.col("prior_tick")) <= TRADE_WINDOW_TICKS)
            .select([pl.col("prior_victim").alias("steamid"), "round_num"])
            .unique()
            .with_columns(pl.lit(True).alias("was_traded"))
        )
        if not traded.is_empty():
            kast_events.append(traded)

    # Deaths by round
    deaths_by_round = pl.DataFrame(schema={"steamid": pl.UInt64, "round_num": pl.Int64})
    if _has_cols(kills, ["victim_steamid", "round_num"]):
        deaths_by_round = (
            kills
            .filter(pl.col("victim_steamid").is_not_null())
            .select([pl.col("victim_steamid").alias("steamid"), "round_num"])
            .unique()
        )

    # S — survived
    survived_rounds = (
        player_rounds
        .join(deaths_by_round, on=["steamid", "round_num"], how="anti")
        .with_columns(pl.lit(True).alias("survived"))
    )

    base = player_rounds.join(survived_rounds, on=["steamid", "round_num"], how="left")

    for event_df in kast_events:
        extra_cols = [c for c in event_df.columns if c not in ("steamid", "round_num")]
        base = base.join(event_df, on=["steamid", "round_num"], how="left")
        base = base.with_columns([pl.col(c).fill_null(False) for c in extra_cols])

    for col in ["survived", "had_kill", "had_assist", "was_traded"]:
        if col not in base.columns:
            base = base.with_columns(pl.lit(False).alias(col))
        else:
            base = base.with_columns(pl.col(col).fill_null(False))

    base = base.with_columns(
        (pl.col("had_kill") | pl.col("had_assist") | pl.col("survived") | pl.col("was_traded"))
        .alias("kast_round")
    )

    return (
        base
        .group_by("steamid")
        .agg(pl.col("kast_round").sum().alias("kast_rounds"))
        .select(["steamid", "kast_rounds"])
    )


# ---------------------------------------------------------------------------
# build_raw_overall_stats
# ---------------------------------------------------------------------------

def build_raw_overall_stats(demo, stats_tables: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    stats_tables = stats_tables or {}

    def _get(key, fallback):
        v = stats_tables.get(key)
        return fallback() if v is None else v

    players         = _get("players",        lambda: _build_players_base(demo))
    rounds_presence = _get("rounds_presence", lambda: _player_round_presence(demo))
    kills_stats     = _get("kills_stats",     lambda: _kills_stats(demo))
    adr_stats       = _get("adr_stats",       lambda: _adr_stats(demo))
    kast_stats      = _get("kast_stats",      lambda: _kast_stats(demo))
    # Filter ghost players out of kast_stats (players seen in ticks but not in players table)
    if not kast_stats.is_empty() and not players.is_empty():
        kast_stats = kast_stats.join(players.select("steamid"), on="steamid", how="inner")

    rp = rounds_presence
    if "name" in rp.columns:
        rp = rp.drop("name")

    result = (
        players
        .join(rp, on="steamid", how="left")
        .join(kills_stats, on="steamid", how="left")
        .join(adr_stats, on="steamid", how="left")
        .join(kast_stats, on="steamid", how="left")
    )

    numeric_defaults: dict[str, int | float] = {
        "rounds_played": 0,
        "kills": 0,
        "deaths": 0,
        "assists": 0,
        "hs_kills": 0,
        "total_damage": 0,
        "kast_rounds": 0,
    }

    for col, default in numeric_defaults.items():
        if col in result.columns:
            result = result.with_columns(pl.col(col).fill_null(default))

    select_order = [
        "steamid", "name", "start_side", "rounds_played",
        "kills", "deaths", "assists",
        "hs_kills",
        "total_damage",
        "kast_rounds",
    ]

    existing_cols = [c for c in select_order if c in result.columns]

    return result.select(existing_cols).sort(["kills", "hs_kills"], descending=[True, True])
