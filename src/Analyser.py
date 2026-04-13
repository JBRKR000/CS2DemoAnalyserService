from pathlib import Path
import logging
import sys
from typing import Any

import polars as pl

from Parser import load_cached_demo
from coach_metrics import build_coach_scoreboard, collect_stats_tables


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


def _safe_len(obj) -> int:
    if obj is None:
        return 0
    if isinstance(obj, pl.DataFrame):
        return obj.height
    try:
        return len(obj)
    except TypeError:
        return 0


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

    stats_tables = collect_stats_tables(demo)
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
