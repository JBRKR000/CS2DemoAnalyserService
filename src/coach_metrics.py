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


def _safe_len(obj) -> int:
    if obj is None:
        return 0
    if isinstance(obj, pl.DataFrame):
        return obj.height
    try:
        return len(obj)
    except TypeError:
        return 0


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
# Public table builders
# ---------------------------------------------------------------------------

def collect_stats_tables(demo) -> dict[str, pl.DataFrame]:
    return {
        "players": _build_players_base(demo),
        "rounds_presence": _player_round_presence(demo),
        "kills_stats": _kills_stats(demo),
        "adr_stats": _adr_stats(demo),
        "kast_stats": _kast_stats(demo),
        "flash_stats": _flash_stats(demo),
        "utility_deaths": _unused_utility_at_death(demo),
    }


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
    """
    Computes: kills, deaths, assists, hs_kills, flash_assists_kill,
              opening_kills, opening_deaths, trade_kills, traded_deaths.

    awpy v2 kill columns:
        - headshot      (bool)
        - flash_assist  (bool) — the assist was a flash assist
        - NO is_trade / player_traded_steamid natively.
          Trade kills are detected manually: a kill is a "trade" when the
          attacker's team-mate was killed by the victim within TRADE_WINDOW
          ticks before this kill.
    """
    TRADE_WINDOW_TICKS = 640  # ~5 s at 128 tick

    kills = _kills_df(demo)
    empty = pl.DataFrame(schema={
        "steamid": pl.UInt64,
        "kills": pl.UInt32,
        "deaths": pl.UInt32,
        "assists": pl.UInt32,
        "hs_kills": pl.UInt32,
        "flash_assists_kill": pl.UInt32,
        "opening_kills": pl.UInt32,
        "opening_deaths": pl.UInt32,
        "trade_kills": pl.UInt32,
        "traded_deaths": pl.UInt32,
    })

    if kills.is_empty():
        return empty

    frames: list[pl.DataFrame] = []

    # kills
    if "attacker_steamid" in kills.columns:
        frames.append(
            kills
            .filter(pl.col("attacker_steamid").is_not_null())
            .group_by("attacker_steamid")
            .agg(pl.len().alias("kills"))
            .rename({"attacker_steamid": "steamid"})
        )

    # deaths
    if "victim_steamid" in kills.columns:
        frames.append(
            kills
            .filter(pl.col("victim_steamid").is_not_null())
            .group_by("victim_steamid")
            .agg(pl.len().alias("deaths"))
            .rename({"victim_steamid": "steamid"})
        )

    # assists
    if "assister_steamid" in kills.columns:
        frames.append(
            kills
            .filter(pl.col("assister_steamid").is_not_null())
            .group_by("assister_steamid")
            .agg(pl.len().alias("assists"))
            .rename({"assister_steamid": "steamid"})
        )

    # headshot kills
    if _has_cols(kills, ["attacker_steamid", "headshot"]):
        frames.append(
            kills
            .filter(pl.col("attacker_steamid").is_not_null() & pl.col("headshot"))
            .group_by("attacker_steamid")
            .agg(pl.len().alias("hs_kills"))
            .rename({"attacker_steamid": "steamid"})
        )

    # flash assists (kills where assister got a flash assist credit)
    if _has_cols(kills, ["assister_steamid", "flash_assist"]):
        frames.append(
            kills
            .filter(pl.col("assister_steamid").is_not_null() & pl.col("flash_assist"))
            .group_by("assister_steamid")
            .agg(pl.len().alias("flash_assists_kill"))
            .rename({"assister_steamid": "steamid"})
        )

    # opening kills / opening deaths (first kill of each round)
    if _has_cols(kills, ["attacker_steamid", "victim_steamid", "tick", "round_num"]):
        first_kill_per_round = (
            kills
            .filter(
                pl.col("attacker_steamid").is_not_null()
                & pl.col("victim_steamid").is_not_null()
            )
            .sort(["round_num", "tick"])
            .group_by("round_num")
            .first()
        )
        frames.append(
            first_kill_per_round
            .group_by("attacker_steamid")
            .agg(pl.len().alias("opening_kills"))
            .rename({"attacker_steamid": "steamid"})
        )
        frames.append(
            first_kill_per_round
            .group_by("victim_steamid")
            .agg(pl.len().alias("opening_deaths"))
            .rename({"victim_steamid": "steamid"})
        )

    # trade kills & traded deaths
    if _has_cols(kills, ["tick", "round_num", "attacker_steamid", "attacker_side",
                          "victim_steamid", "victim_side"]):
        k = kills.select([
            "tick", "round_num",
            "attacker_steamid", "attacker_side",
            "victim_steamid", "victim_side",
        ]).filter(
            pl.col("attacker_steamid").is_not_null()
            & pl.col("victim_steamid").is_not_null()
        )

        prior = k.rename({
            "tick": "prior_tick",
            "attacker_steamid": "prior_killer",
            "attacker_side": "prior_killer_side",
            "victim_steamid": "prior_victim",
            "victim_side": "prior_victim_side",
        })

        crossed = (
            k
            .join(prior, on="round_num", how="inner")
            # current victim == prior killer (victim killed the prior killer)
            .filter(pl.col("victim_steamid") == pl.col("prior_killer"))
            # attacker is on same side as the player who was killed previously
            .filter(pl.col("attacker_side") == pl.col("prior_victim_side"))
            # prior kill was before this kill
            .filter(pl.col("prior_tick") < pl.col("tick"))
            # within trade window
            .filter((pl.col("tick") - pl.col("prior_tick")) <= TRADE_WINDOW_TICKS)
        )

        if not crossed.is_empty():
            frames.append(
                crossed.group_by("attacker_steamid")
                .agg(pl.len().alias("trade_kills"))
                .rename({"attacker_steamid": "steamid"})
            )
            frames.append(
                crossed.group_by("victim_steamid")
                .agg(pl.len().alias("traded_deaths"))
                .rename({"victim_steamid": "steamid"})
            )

    if not frames:
        return empty

    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on="steamid", how="full", coalesce=True)

    numeric_cols = [c for c in result.columns if c != "steamid"]
    result = result.with_columns([pl.col(c).fill_null(0) for c in numeric_cols])

    for col in ["kills", "deaths", "assists", "hs_kills", "flash_assists_kill",
                "opening_kills", "opening_deaths", "trade_kills", "traded_deaths"]:
        if col not in result.columns:
            result = result.with_columns(pl.lit(0).cast(pl.UInt32).alias(col))

    return result


# ---------------------------------------------------------------------------
# _adr_stats
# ---------------------------------------------------------------------------

def _adr_stats(demo) -> pl.DataFrame:
    """
    ADR = total_damage / rounds_played_by_that_player.

    awpy v2 damage column: 'damage' (not 'hp_damage').
    Team damage is excluded.
    """
    damages = _damages_df(demo)
    empty = pl.DataFrame(schema={"steamid": pl.UInt64, "total_damage": pl.Int64, "adr": pl.Float64})

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

    # Divide by each player's own rounds_played (not global round count)
    rounds_presence = _player_round_presence(demo)

    if rounds_presence.is_empty():
        fallback_rounds = max(_safe_len(getattr(demo, "rounds", None)), 1)
        return damage_per_player.with_columns(
            (pl.col("total_damage") / pl.lit(fallback_rounds)).round(2).alias("adr")
        )

    return (
        damage_per_player
        .join(rounds_presence.select(["steamid", "rounds_played"]), on="steamid", how="left")
        .with_columns(pl.col("rounds_played").fill_null(1))
        .with_columns(
            (pl.col("total_damage") / pl.col("rounds_played")).round(2).alias("adr")
        )
        .drop("rounds_played")
    )


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
    empty = pl.DataFrame(schema={"steamid": pl.UInt64, "kast_rounds": pl.UInt32, "kast": pl.Float64})

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

    rounds_per_player = player_rounds.group_by("steamid").agg(pl.len().alias("rounds_played"))

    return (
        base
        .group_by("steamid")
        .agg(pl.col("kast_round").sum().alias("kast_rounds"))
        .join(rounds_per_player, on="steamid", how="left")
        .with_columns(
            ((pl.col("kast_rounds") / pl.col("rounds_played")) * 100.0).round(2).alias("kast")
        )
        .select(["steamid", "kast_rounds", "kast"])
    )


# ---------------------------------------------------------------------------
# _flash_stats
# ---------------------------------------------------------------------------

def _flash_stats(demo) -> pl.DataFrame:
    """
    Flash statistics:
    - flash_assists: kills where flash_assist == True (from demo.kills)
    - team_flashes / self_flashes: from player_blind event if available
      (awpy v2 does NOT expose player_blind by default — zeros otherwise)
    """
    empty = pl.DataFrame(schema={
        "steamid": pl.UInt64,
        "flash_assists": pl.UInt32,
        "team_flashes": pl.UInt32,
        "self_flashes": pl.UInt32,
    })

    frames: list[pl.DataFrame] = []

    # Flash assists from kills.flash_assist
    kills = _kills_df(demo)
    if _has_cols(kills, ["assister_steamid", "flash_assist"]):
        fa = (
            kills
            .filter(pl.col("assister_steamid").is_not_null() & pl.col("flash_assist"))
            .group_by("assister_steamid")
            .agg(pl.len().alias("flash_assists"))
            .rename({"assister_steamid": "steamid"})
        )
        if not fa.is_empty():
            frames.append(fa)

    # team_flashes / self_flashes from player_blind event (if present)
    blind_df = pl.DataFrame()
    events = getattr(demo, "events", None)
    if isinstance(events, dict):
        for key in ("player_blind", "flashed"):
            if key in events:
                blind_df = _safe_df(events[key])
                if not blind_df.is_empty():
                    break

    if not blind_df.is_empty():
        attacker_col = _pick_first_col(blind_df.columns,
                                       "attacker_steamid", "flasher_steamid", "thrower_steamid")
        victim_col = _pick_first_col(blind_df.columns,
                                     "user_steamid", "victim_steamid", "blinded_steamid")
        att_side_col = _pick_first_col(blind_df.columns, "attacker_side", "thrower_side")
        vic_side_col = _pick_first_col(blind_df.columns, "user_side", "victim_side")

        if attacker_col and victim_col and att_side_col and vic_side_col:
            frames.append(
                blind_df
                .filter(
                    pl.col(attacker_col).is_not_null()
                    & pl.col(victim_col).is_not_null()
                    & (pl.col(att_side_col) == pl.col(vic_side_col))
                    & (pl.col(attacker_col) != pl.col(victim_col))
                )
                .group_by(attacker_col)
                .agg(pl.len().alias("team_flashes"))
                .rename({attacker_col: "steamid"})
            )
            frames.append(
                blind_df
                .filter(
                    pl.col(attacker_col).is_not_null()
                    & (pl.col(attacker_col) == pl.col(victim_col))
                )
                .group_by(attacker_col)
                .agg(pl.len().alias("self_flashes"))
                .rename({attacker_col: "steamid"})
            )

    if not frames:
        return empty

    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on="steamid", how="full", coalesce=True)

    for col in ["flash_assists", "team_flashes", "self_flashes"]:
        if col not in result.columns:
            result = result.with_columns(pl.lit(0).cast(pl.UInt32).alias(col))
        else:
            result = result.with_columns(pl.col(col).fill_null(0))

    return result.select(["steamid", "flash_assists", "team_flashes", "self_flashes"])


# ---------------------------------------------------------------------------
# _unused_utility_at_death
# ---------------------------------------------------------------------------

def _unused_utility_at_death(demo) -> pl.DataFrame:
    """
    Counts rounds where a player died while still holding utility.

    Requires 'inventory' in ticks, which is only present when parse() is
    called with player_props=['inventory']. Returns empty DataFrame otherwise.
    """
    ticks = _safe_df(getattr(demo, "ticks", None))
    kills = _kills_df(demo)

    empty = pl.DataFrame(schema={"steamid": pl.UInt64, "deaths_with_utility": pl.Int64})

    if "inventory" not in ticks.columns:
        logger.info(
            "No 'inventory' column in ticks. "
            "Re-parse with player_props=['inventory'] to enable deaths_with_utility."
        )
        return empty

    if not _has_cols(ticks, ["steamid", "round_num", "tick", "inventory"]):
        return empty

    if "victim_steamid" not in kills.columns:
        return empty

    UTILITY_TYPES = {"flashbang", "hegrenade", "smokegrenade", "molotov", "incgrenade", "decoy"}

    def _has_utility(inv) -> bool:
        if inv is None:
            return False
        try:
            return any(str(item).lower() in UTILITY_TYPES for item in inv)
        except Exception:
            return False

    deaths = (
        kills
        .filter(pl.col("victim_steamid").is_not_null())
        .select([
            pl.col("victim_steamid").alias("steamid"),
            "round_num",
            pl.col("tick").alias("death_tick"),
        ])
    )

    tick_snapshots = (
        ticks
        .select(["steamid", "round_num", "tick", "inventory"])
        .drop_nulls(["steamid", "round_num", "tick"])
    )

    joined = (
        deaths
        .join(tick_snapshots, on=["steamid", "round_num"], how="left")
        .filter(pl.col("tick") <= pl.col("death_tick"))
    )

    if joined.is_empty():
        return empty

    last_before_death = (
        joined
        .sort(["steamid", "round_num", "tick"])
        .group_by(["steamid", "round_num"])
        .agg(pl.last("inventory").alias("inventory_at_death"))
    )

    result_rows = [
        {"steamid": row["steamid"], "round_num": row["round_num"]}
        for row in last_before_death.to_dicts()
        if _has_utility(row.get("inventory_at_death"))
    ]

    if not result_rows:
        return empty

    return (
        pl.DataFrame(result_rows, schema={"steamid": pl.UInt64, "round_num": pl.Int64})
        .group_by("steamid")
        .agg(pl.len().cast(pl.Int64).alias("deaths_with_utility"))
    )


# ---------------------------------------------------------------------------
# build_coach_scoreboard
# ---------------------------------------------------------------------------

def build_coach_scoreboard(demo, stats_tables: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    stats_tables = stats_tables or {}

    def _get(key, fallback):
        v = stats_tables.get(key)
        return fallback() if v is None else v

    players         = _get("players",        lambda: _build_players_base(demo))
    rounds_presence = _get("rounds_presence", lambda: _player_round_presence(demo))
    kills_stats     = _get("kills_stats",     lambda: _kills_stats(demo))
    adr_stats       = _get("adr_stats",       lambda: _adr_stats(demo))
    kast_stats      = _get("kast_stats",      lambda: _kast_stats(demo))
    flash_stats     = _get("flash_stats",     lambda: _flash_stats(demo))
    utility_deaths  = _get("utility_deaths",  lambda: _unused_utility_at_death(demo))

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
        .join(flash_stats, on="steamid", how="left")
        .join(utility_deaths, on="steamid", how="left")
    )

    numeric_defaults: dict[str, int | float] = {
        "rounds_played": 0,
        "kills": 0,
        "deaths": 0,
        "assists": 0,
        "hs_kills": 0,
        "flash_assists_kill": 0,
        "opening_kills": 0,
        "opening_deaths": 0,
        "trade_kills": 0,
        "traded_deaths": 0,
        "total_damage": 0,
        "adr": 0.0,
        "kast_rounds": 0,
        "kast": 0.0,
        "flash_assists": 0,
        "team_flashes": 0,
        "self_flashes": 0,
        "deaths_with_utility": 0,
    }

    for col, default in numeric_defaults.items():
        if col in result.columns:
            result = result.with_columns(pl.col(col).fill_null(default))

    result = result.with_columns([
        pl.when(pl.col("kills") > 0)
        .then((pl.col("hs_kills") / pl.col("kills") * 100.0).round(2))
        .otherwise(0.0)
        .alias("hs_percent"),

        pl.when(pl.col("rounds_played") > 0)
        .then((pl.col("kills") / pl.col("rounds_played")).round(2))
        .otherwise(0.0)
        .alias("kpr"),

        pl.when(pl.col("rounds_played") > 0)
        .then((pl.col("deaths") / pl.col("rounds_played")).round(2))
        .otherwise(0.0)
        .alias("dpr"),

        pl.when((pl.col("opening_kills") + pl.col("opening_deaths")) > 0)
        .then(
            (pl.col("opening_kills") / (pl.col("opening_kills") + pl.col("opening_deaths")) * 100.0)
            .round(2)
        )
        .otherwise(0.0)
        .alias("opening_duel_win_pct"),
    ])

    select_order = [
        "steamid", "name", "start_side", "rounds_played",
        "kills", "deaths", "assists",
        "kpr", "dpr", "adr", "kast",
        "hs_kills", "hs_percent",
        "opening_kills", "opening_deaths", "opening_duel_win_pct",
        "trade_kills", "traded_deaths",
        "flash_assists", "flash_assists_kill",
        "team_flashes", "self_flashes",
        "deaths_with_utility",
    ]

    existing_cols = [c for c in select_order if c in result.columns]

    return result.select(existing_cols).sort(
        ["adr", "kast", "kills"],
        descending=[True, True, True],
    )