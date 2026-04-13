from pathlib import Path
import logging
import sys
from typing import Any, Iterable

import polars as pl

from Parser import load_cached_demo

logger = logging.getLogger(__name__)


def load_demo_for_analysis(
    cache_key_path: str = "./last_cache_key.txt",
    cache_dir: str = ".cache",
    verbose: bool = True,
):
    cache_key = Path(cache_key_path).read_text(encoding="utf-8").strip()
    demo = load_cached_demo(cache_key, cache_dir=cache_dir)

    header_info = getattr(demo, "header", {}) or {}
    map_name = header_info.get("map_name", None) if isinstance(header_info, dict) else None

    if verbose:
        logger.info(
            "Loaded from cache for analysis | cache_key=%s | type=%s | map=%s",
            cache_key,
            type(demo).__name__,
            map_name,
        )

    return demo


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


def _collect_stats_tables(demo) -> dict[str, pl.DataFrame]:
    return {
        "players": _build_players_base(demo),
        "rounds_presence": _player_round_presence(demo),
        "kills_stats": _kills_stats(demo),
        "adr_stats": _adr_stats(demo),
        "kast_stats": _kast_stats(demo),
        "flash_stats": _flash_stats(demo),
        "utility_deaths": _unused_utility_at_death(demo),
    }


def _build_players_base(demo) -> pl.DataFrame:
    ticks = _safe_df(getattr(demo, "ticks", None))

    required = ["round_num", "steamid", "name", "side"]
    if not _has_cols(ticks, required):
        logger.warning("ticks dataframe does not contain required columns: %s", required)
        return pl.DataFrame(
            schema={
                "steamid": pl.Int64,
                "name": pl.Utf8,
                "start_side": pl.Utf8,
                "first_round": pl.Int64,
            }
        )

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
            pl.first("round_num").alias("first_round"),
        ])
        .sort("name")
    )

    return players


def _player_round_presence(demo) -> pl.DataFrame:
    ticks = _safe_df(getattr(demo, "ticks", None))
    required = ["steamid", "name", "round_num"]

    if not _has_cols(ticks, required):
        return pl.DataFrame(
            schema={"steamid": pl.Int64, "name": pl.Utf8, "rounds_played": pl.Int64}
        )

    return (
        ticks
        .select(required)
        .drop_nulls(required)
        .unique()
        .group_by(["steamid", "name"])
        .agg(pl.n_unique("round_num").alias("rounds_played"))
    )


def _kills_stats(demo) -> pl.DataFrame:
    kills = _safe_df(getattr(demo, "kills", None))

    if kills.is_empty():
        return pl.DataFrame(
            schema={
                "steamid": pl.Int64,
                "kills": pl.Int64,
                "deaths": pl.Int64,
                "assists": pl.Int64,
                "hs_kills": pl.Int64,
                "opening_kills": pl.Int64,
                "opening_deaths": pl.Int64,
                "trade_kills": pl.Int64,
                "traded_deaths": pl.Int64,
            }
        )

    attacker_id_col = "attacker_steamid" if "attacker_steamid" in kills.columns else "attackerSteamID"
    victim_id_col = "victim_steamid" if "victim_steamid" in kills.columns else "victimSteamID"
    assister_id_col = "assister_steamid" if "assister_steamid" in kills.columns else "assisterSteamID"
    headshot_col = "is_headshot" if "is_headshot" in kills.columns else "isHeadshot"
    first_kill_col = "is_first_kill" if "is_first_kill" in kills.columns else "isFirstKill"
    trade_col = "is_trade" if "is_trade" in kills.columns else "isTrade"
    traded_player_col = "player_traded_steamid" if "player_traded_steamid" in kills.columns else "playerTradedSteamID"

    kill_frames = []

    if attacker_id_col in kills.columns:
        kill_frames.append(
            kills
            .filter(pl.col(attacker_id_col).is_not_null())
            .group_by(attacker_id_col)
            .agg(pl.len().alias("kills"))
            .rename({attacker_id_col: "steamid"})
        )

    if victim_id_col in kills.columns:
        kill_frames.append(
            kills
            .filter(pl.col(victim_id_col).is_not_null())
            .group_by(victim_id_col)
            .agg(pl.len().alias("deaths"))
            .rename({victim_id_col: "steamid"})
        )

    if assister_id_col in kills.columns:
        kill_frames.append(
            kills
            .filter(pl.col(assister_id_col).is_not_null())
            .group_by(assister_id_col)
            .agg(pl.len().alias("assists"))
            .rename({assister_id_col: "steamid"})
        )

    if attacker_id_col in kills.columns and headshot_col in kills.columns:
        kill_frames.append(
            kills
            .filter(pl.col(attacker_id_col).is_not_null() & (pl.col(headshot_col) == True))
            .group_by(attacker_id_col)
            .agg(pl.len().alias("hs_kills"))
            .rename({attacker_id_col: "steamid"})
        )

    if attacker_id_col in kills.columns and first_kill_col in kills.columns:
        kill_frames.append(
            kills
            .filter(pl.col(attacker_id_col).is_not_null() & (pl.col(first_kill_col) == True))
            .group_by(attacker_id_col)
            .agg(pl.len().alias("opening_kills"))
            .rename({attacker_id_col: "steamid"})
        )

    if victim_id_col in kills.columns and first_kill_col in kills.columns:
        kill_frames.append(
            kills
            .filter(pl.col(victim_id_col).is_not_null() & (pl.col(first_kill_col) == True))
            .group_by(victim_id_col)
            .agg(pl.len().alias("opening_deaths"))
            .rename({victim_id_col: "steamid"})
        )

    if attacker_id_col in kills.columns and trade_col in kills.columns:
        kill_frames.append(
            kills
            .filter(pl.col(attacker_id_col).is_not_null() & (pl.col(trade_col) == True))
            .group_by(attacker_id_col)
            .agg(pl.len().alias("trade_kills"))
            .rename({attacker_id_col: "steamid"})
        )

    if traded_player_col in kills.columns:
        kill_frames.append(
            kills
            .filter(pl.col(traded_player_col).is_not_null())
            .group_by(traded_player_col)
            .agg(pl.len().alias("traded_deaths"))
            .rename({traded_player_col: "steamid"})
        )

    if not kill_frames:
        return pl.DataFrame({"steamid": []})

    result = kill_frames[0]
    for frame in kill_frames[1:]:
        result = result.join(frame, on="steamid", how="full", coalesce=True)

    numeric_cols = [c for c in result.columns if c != "steamid"]
    result = result.with_columns([pl.col(c).fill_null(0) for c in numeric_cols])

    return result


def _adr_stats(demo) -> pl.DataFrame:
    damages = _safe_df(getattr(demo, "damages", None))

    if damages.is_empty():
        return pl.DataFrame(schema={"steamid": pl.Int64, "total_damage": pl.Int64, "adr": pl.Float64})

    attacker_id_col = "attacker_steamid" if "attacker_steamid" in damages.columns else "attackerSteamID"
    damage_col = "hp_damage" if "hp_damage" in damages.columns else "hpDamage"

    if not _has_cols(damages, [attacker_id_col, damage_col]):
        return pl.DataFrame(schema={"steamid": pl.Int64, "total_damage": pl.Int64, "adr": pl.Float64})

    rounds_played = _safe_len(getattr(demo, "rounds", None))

    if rounds_played <= 0:
        rounds_played = 1

    return (
        damages
        .filter(pl.col(attacker_id_col).is_not_null())
        .group_by(attacker_id_col)
        .agg(pl.col(damage_col).sum().alias("total_damage"))
        .rename({attacker_id_col: "steamid"})
        .with_columns(
            (pl.col("total_damage") / pl.lit(rounds_played)).round(2).alias("adr")
        )
    )


def _kast_stats(demo) -> pl.DataFrame:
    """
    KAST = round with at least one of:
    - Kill
    - Assist
    - Survived
    - Traded death
    """
    players = _build_players_base(demo)
    ticks = _safe_df(getattr(demo, "ticks", None))
    kills = _safe_df(getattr(demo, "kills", None))

    if players.is_empty():
        return pl.DataFrame(schema={"steamid": pl.Int64, "kast_rounds": pl.Int64, "kast": pl.Float64})

    if not _has_cols(ticks, ["steamid", "round_num"]):
        return pl.DataFrame(schema={"steamid": pl.Int64, "kast_rounds": pl.Int64, "kast": pl.Float64})

    player_rounds = (
        ticks
        .select(["steamid", "round_num"])
        .drop_nulls(["steamid", "round_num"])
        .unique()
    )

    attacker_id_col = "attacker_steamid" if "attacker_steamid" in kills.columns else "attackerSteamID"
    victim_id_col = "victim_steamid" if "victim_steamid" in kills.columns else "victimSteamID"
    assister_id_col = "assister_steamid" if "assister_steamid" in kills.columns else "assisterSteamID"
    traded_player_col = "player_traded_steamid" if "player_traded_steamid" in kills.columns else "playerTradedSteamID"
    round_col = "round_num"

    kast_events = []

    if _has_cols(kills, [attacker_id_col, round_col]):
        kast_events.append(
            kills
            .filter(pl.col(attacker_id_col).is_not_null())
            .select([
                pl.col(attacker_id_col).alias("steamid"),
                pl.col(round_col),
            ])
            .unique()
            .with_columns(pl.lit(True).alias("had_kill"))
        )

    if _has_cols(kills, [assister_id_col, round_col]):
        kast_events.append(
            kills
            .filter(pl.col(assister_id_col).is_not_null())
            .select([
                pl.col(assister_id_col).alias("steamid"),
                pl.col(round_col),
            ])
            .unique()
            .with_columns(pl.lit(True).alias("had_assist"))
        )

    if _has_cols(kills, [traded_player_col, round_col]):
        kast_events.append(
            kills
            .filter(pl.col(traded_player_col).is_not_null())
            .select([
                pl.col(traded_player_col).alias("steamid"),
                pl.col(round_col),
            ])
            .unique()
            .with_columns(pl.lit(True).alias("was_traded"))
        )

    deaths_by_round = pl.DataFrame(schema={"steamid": pl.Int64, "round_num": pl.Int64})
    if _has_cols(kills, [victim_id_col, round_col]):
        deaths_by_round = (
            kills
            .filter(pl.col(victim_id_col).is_not_null())
            .select([
                pl.col(victim_id_col).alias("steamid"),
                pl.col(round_col),
            ])
            .unique()
        )

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

    if "survived" not in base.columns:
        base = base.with_columns(pl.lit(False).alias("survived"))
    else:
        base = base.with_columns(pl.col("survived").fill_null(False))

    for col in ["had_kill", "had_assist", "was_traded"]:
        if col not in base.columns:
            base = base.with_columns(pl.lit(False).alias(col))

    base = base.with_columns(
        (
            pl.col("had_kill")
            | pl.col("had_assist")
            | pl.col("survived")
            | pl.col("was_traded")
        ).alias("kast_round")
    )

    rounds_per_player = (
        player_rounds
        .group_by("steamid")
        .agg(pl.len().alias("rounds_played"))
    )

    kast = (
        base
        .group_by("steamid")
        .agg(pl.col("kast_round").sum().alias("kast_rounds"))
        .join(rounds_per_player, on="steamid", how="left")
        .with_columns(
            ((pl.col("kast_rounds") / pl.col("rounds_played")) * 100.0).round(2).alias("kast")
        )
        .select(["steamid", "kast_rounds", "kast"])
    )

    return kast


def _flash_stats(demo) -> pl.DataFrame:
    damages = _safe_df(getattr(demo, "damages", None))

    if damages.is_empty():
        return pl.DataFrame(
            schema={
                "steamid": pl.Int64,
                "flash_assists": pl.Int64,
                "team_flashes": pl.Int64,
                "self_flashes": pl.Int64,
            }
        )

    attacker_id_col = "attacker_steamid" if "attacker_steamid" in damages.columns else "attackerSteamID"
    victim_id_col = "victim_steamid" if "victim_steamid" in damages.columns else "victimSteamID"
    attacker_side_col = "attacker_side" if "attacker_side" in damages.columns else "attackerSide"
    victim_side_col = "victim_side" if "victim_side" in damages.columns else "victimSide"
    weapon_col = "weapon" if "weapon" in damages.columns else "weapon"
    hp_damage_col = "hp_damage" if "hp_damage" in damages.columns else "hpDamage"
    blinded_col = "is_attacker_blinded" if "is_attacker_blinded" in damages.columns else None

    frames = []

    # Flash assists usually better from kills via flash thrower, but if you only have damages,
    # we keep this part empty unless such fields exist.
    flash_thrower_col = "flashThrowerSteamID" if "flashThrowerSteamID" in damages.columns else None
    if flash_thrower_col:
        frames.append(
            damages
            .filter(pl.col(flash_thrower_col).is_not_null())
            .group_by(flash_thrower_col)
            .agg(pl.len().alias("flash_assists"))
            .rename({flash_thrower_col: "steamid"})
        )

    # Team flashes / self flashes through flashbang damage-like events are parser-dependent.
    # This block is heuristic and may need adjustment to your exact schema.
    if _has_cols(damages, [attacker_id_col, victim_id_col, attacker_side_col, victim_side_col, weapon_col]):
        flash_rows = damages.filter(pl.col(weapon_col) == "Flashbang")

        frames.append(
            flash_rows
            .filter(
                pl.col(attacker_id_col).is_not_null()
                & pl.col(victim_id_col).is_not_null()
                & (pl.col(attacker_side_col) == pl.col(victim_side_col))
                & (pl.col(attacker_id_col) != pl.col(victim_id_col))
            )
            .group_by(attacker_id_col)
            .agg(pl.len().alias("team_flashes"))
            .rename({attacker_id_col: "steamid"})
        )

        frames.append(
            flash_rows
            .filter(
                pl.col(attacker_id_col).is_not_null()
                & pl.col(victim_id_col).is_not_null()
                & (pl.col(attacker_id_col) == pl.col(victim_id_col))
            )
            .group_by(attacker_id_col)
            .agg(pl.len().alias("self_flashes"))
            .rename({attacker_id_col: "steamid"})
        )

    if not frames:
        return pl.DataFrame({"steamid": []})

    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on="steamid", how="full", coalesce=True)

    numeric_cols = [c for c in result.columns if c != "steamid"]
    result = result.with_columns([pl.col(c).fill_null(0) for c in numeric_cols])

    return result


def _unused_utility_at_death(demo) -> pl.DataFrame:
    """
    Heurystyka:
    bierzemy ostatni tick gracza w rundzie, w której zginął,
    i sprawdzamy czy miał jeszcze utility.
    """
    ticks = _safe_df(getattr(demo, "ticks", None))
    kills = _safe_df(getattr(demo, "kills", None))

    required_tick_cols = ["steamid", "round_num", "tick"]
    victim_id_col = "victim_steamid" if "victim_steamid" in kills.columns else "victimSteamID"

    # Spróbujmy znaleźć kolumnę z liczbą utility
    utility_col = None
    for candidate in ["total_utility", "totalUtility"]:
        if candidate in ticks.columns:
            utility_col = candidate
            break

    if utility_col is None or not _has_cols(ticks, required_tick_cols) or victim_id_col not in kills.columns:
        return pl.DataFrame(schema={"steamid": pl.Int64, "deaths_with_utility": pl.Int64})

    deaths = (
        kills
        .filter(pl.col(victim_id_col).is_not_null())
        .select([
            pl.col(victim_id_col).alias("steamid"),
            pl.col("round_num"),
            pl.col("tick").alias("death_tick"),
        ])
    )

    tick_snapshots = (
        ticks
        .select(["steamid", "round_num", "tick", utility_col])
        .drop_nulls(["steamid", "round_num", "tick"])
    )

    # Ostatni tick <= death_tick
    joined = deaths.join(
        tick_snapshots,
        on=["steamid", "round_num"],
        how="left",
    ).filter(pl.col("tick") <= pl.col("death_tick"))

    if joined.is_empty():
        return pl.DataFrame(schema={"steamid": pl.Int64, "deaths_with_utility": pl.Int64})

    last_before_death = (
        joined
        .sort(["steamid", "round_num", "tick"])
        .group_by(["steamid", "round_num"])
        .agg(pl.last(utility_col).alias("utility_left"))
    )

    return (
        last_before_death
        .filter(pl.col("utility_left") > 0)
        .group_by("steamid")
        .agg(pl.len().alias("deaths_with_utility"))
    )


def build_coach_scoreboard(demo, stats_tables: dict[str, pl.DataFrame] | None = None) -> pl.DataFrame:
    stats_tables = stats_tables or {}

    players = stats_tables.get("players")
    rounds_presence = stats_tables.get("rounds_presence")
    kills_stats = stats_tables.get("kills_stats")
    adr_stats = stats_tables.get("adr_stats")
    kast_stats = stats_tables.get("kast_stats")
    flash_stats = stats_tables.get("flash_stats")
    utility_deaths = stats_tables.get("utility_deaths")

    if players is None:
        players = _build_players_base(demo)
    if rounds_presence is None:
        rounds_presence = _player_round_presence(demo)
    if kills_stats is None:
        kills_stats = _kills_stats(demo)
    if adr_stats is None:
        adr_stats = _adr_stats(demo)
    if kast_stats is None:
        kast_stats = _kast_stats(demo)
    if flash_stats is None:
        flash_stats = _flash_stats(demo)
    if utility_deaths is None:
        utility_deaths = _unused_utility_at_death(demo)

    result = (
        players
        .join(rounds_presence, on=["steamid", "name"], how="left")
        .join(kills_stats, on="steamid", how="left")
        .join(adr_stats, on="steamid", how="left")
        .join(kast_stats, on="steamid", how="left")
        .join(flash_stats, on="steamid", how="left")
        .join(utility_deaths, on="steamid", how="left")
    )

    numeric_defaults = {
        "rounds_played": 0,
        "kills": 0,
        "deaths": 0,
        "assists": 0,
        "hs_kills": 0,
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
            (
                pl.col("opening_kills")
                / (pl.col("opening_kills") + pl.col("opening_deaths"))
                * 100.0
            ).round(2)
        )
        .otherwise(0.0)
        .alias("opening_duel_win_pct"),
    ])

    select_order = [
        "steamid",
        "name",
        "start_side",
        "rounds_played",
        "kills",
        "deaths",
        "assists",
        "kpr",
        "dpr",
        "adr",
        "kast",
        "hs_kills",
        "hs_percent",
        "opening_kills",
        "opening_deaths",
        "opening_duel_win_pct",
        "trade_kills",
        "traded_deaths",
        "flash_assists",
        "team_flashes",
        "self_flashes",
        "deaths_with_utility",
    ]

    existing_cols = [c for c in select_order if c in result.columns]

    return result.select(existing_cols).sort(
        ["adr", "kast", "kills"],
        descending=[True, True, True],
    )


def _log_stats_tables(stats_tables: dict[str, pl.DataFrame]) -> None:
    for table_name, table_df in stats_tables.items():
        logger.info("%s:\n%s", table_name, table_df)


def _select_player(scoreboard: pl.DataFrame, player_selector: str | int | None = None) -> pl.DataFrame:
    if scoreboard.is_empty():
        return scoreboard

    players_index = scoreboard.select(["steamid", "name"]).with_row_index("idx")
    logger.info("Available players for detailed analysis:\n%s", players_index)

    selector: Any = player_selector

    if selector is None:
        if sys.stdin is not None and sys.stdin.isatty():
            try:
                user_value = input("Select player [idx/steamid/name], Enter=0: ").strip()
            except EOFError:
                user_value = ""
            selector = user_value or "0"
        else:
            selector = "0"

    selected = pl.DataFrame()

    if isinstance(selector, int):
        if 0 <= selector < scoreboard.height:
            selected = scoreboard.slice(selector, 1)
        else:
            selected = scoreboard.filter(pl.col("steamid") == selector)
    else:
        selector_text = str(selector).strip()

        if not selector_text:
            selected = scoreboard.slice(0, 1)
        elif selector_text.isdigit():
            numeric_value = int(selector_text)
            if 0 <= numeric_value < scoreboard.height:
                selected = scoreboard.slice(numeric_value, 1)
            if selected.is_empty():
                selected = scoreboard.filter(pl.col("steamid") == numeric_value)

        if selected.is_empty():
            selected = scoreboard.filter(
                pl.col("name").str.to_lowercase() == selector_text.lower()
            )

        if selected.is_empty():
            selected = scoreboard.filter(
                pl.col("name").str.to_lowercase().str.contains(selector_text.lower(), literal=True)
            )

    if selected.is_empty():
        logger.warning("Player selector '%s' not found. Fallback to first player.", selector)
        selected = scoreboard.slice(0, 1)

    return selected


def _log_selected_player_stats(selected_player: pl.DataFrame) -> dict[str, Any]:
    if selected_player.is_empty():
        return {}

    row = selected_player.to_dicts()[0]
    player_name = row.get("name", "Unknown")
    player_id = row.get("steamid", "Unknown")

    logger.info("Detailed stats for player: %s (%s)", player_name, player_id)
    for stat_name, stat_value in row.items():
        logger.info("%s=%s", stat_name, stat_value)

    return row


def analyse_demo(demo, player_selector: str | int | None = None):
    header_info = getattr(demo, "header", {}) or {}
    map_name = header_info.get("map_name", None) if isinstance(header_info, dict) else None
    rounds_played = _safe_len(getattr(demo, "rounds", None))

    stats_tables = _collect_stats_tables(demo)
    players = stats_tables["players"]
    team_a = players.filter(pl.col("start_side") == "ct").sort("name")
    team_b = players.filter(pl.col("start_side") == "t").sort("name")

    scoreboard = build_coach_scoreboard(demo, stats_tables=stats_tables)
    selected_player = _select_player(scoreboard, player_selector=player_selector)
    selected_player_stats = _log_selected_player_stats(selected_player)

    logger.info("Analyzing demo on map: %s", map_name or "Unknown")
    logger.info("Rounds played: %d", rounds_played)
    logger.info("Team A (CT start):\n%s", team_a)
    logger.info("Team B (T start):\n%s", team_b)
    _log_stats_tables(stats_tables)
    logger.info("Coach scoreboard:\n%s", scoreboard)

    return {
        "map_name": map_name,
        "rounds_played": rounds_played,
        "team_a": team_a,
        "team_b": team_b,
        "scoreboard": scoreboard,
        "selected_player": selected_player,
        "selected_player_stats": selected_player_stats,
    }


if __name__ == "__main__":
    analyse_demo(load_demo_for_analysis())
