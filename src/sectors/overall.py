import logging
import sys
from typing import Any

import polars as pl


logger = logging.getLogger(__name__)

def build_overall_table(raw_stats: pl.DataFrame) -> pl.DataFrame:
    if raw_stats.is_empty():
        return raw_stats

    result = raw_stats.with_columns([
        pl.col("rounds_played").fill_null(0),
        pl.col("kills").fill_null(0),
        pl.col("deaths").fill_null(0),
        pl.col("assists").fill_null(0),
        pl.col("hs_kills").fill_null(0),
        pl.col("total_damage").fill_null(0),
        pl.col("kast_rounds").fill_null(0),
    ])

    result = result.with_columns([
        pl.when(pl.col("rounds_played") > 0)
        .then((pl.col("total_damage") / pl.col("rounds_played")).round(2))
        .otherwise(0.0)
        .alias("adr"),
        pl.when(pl.col("rounds_played") > 0)
        .then(((pl.col("kast_rounds") / pl.col("rounds_played")) * 100.0).round(2))
        .otherwise(0.0)
        .alias("kast"),
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
    ])

    select_order = [
        "steamid", "name", "start_side", "rounds_played",
        "kills", "deaths", "assists",
        "kpr", "dpr", "adr", "kast",
        "hs_kills", "hs_percent",
    ]
    existing_cols = [c for c in select_order if c in result.columns]
    return result.select(existing_cols).sort(["adr", "kast", "kills"], descending=[True, True, True])


def select_player(scoreboard: pl.DataFrame, player_selector: str | int | None = None) -> pl.DataFrame:
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
            selected = scoreboard.filter(pl.col("name").str.to_lowercase() == selector_text.lower())

        if selected.is_empty():
            selected = scoreboard.filter(
                pl.col("name").str.to_lowercase().str.contains(selector_text.lower(), literal=True)
            )

    if selected.is_empty():
        logger.warning("Player selector '%s' not found. Fallback to first player.", selector)
        selected = scoreboard.slice(0, 1)

    return selected


def get_overall_player_stats(
    scoreboard: pl.DataFrame,
    player_selector: str | int | None = None,
    log_stats: bool = True,
) -> dict[str, Any]:
    selected_player = select_player(scoreboard, player_selector=player_selector)
    return selected_player_overall_stats(selected_player, log_stats=log_stats)


def selected_player_overall_stats(selected_player: pl.DataFrame, log_stats: bool = True) -> dict[str, Any]:
    if selected_player.is_empty():
        return {}

    row = selected_player.to_dicts()[0]
    if log_stats:
        player_name = row.get("name", "Unknown")
        player_id = row.get("steamid", "Unknown")
        logger.info("Overall stats for player: %s (%s)", player_name, player_id)
        for stat_name, stat_value in row.items():
            logger.info("%s=%s", stat_name, stat_value)

    return row
