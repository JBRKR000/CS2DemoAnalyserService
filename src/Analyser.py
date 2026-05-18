from pathlib import Path
import logging
from typing import Any

import polars as pl

from Parser import load_cached_demo
from benchmarks import (
    append_match_samples,
    evaluate_player,
    is_match_analyzed,
    load_analyzed_matches,
    load_benchmark_samples,
    make_contextual_match_samples,
    mark_match_analyzed,
)
from coach_metrics import _damages_df, _kills_df, _safe_df, build_raw_overall_stats
from sectors.clutch import build_clutch_stats
from sectors.economy import build_economy_stats
from sectors.feedback import generate_feedback
from sectors.overall import build_overall_table, select_player, selected_player_overall_stats


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


def _summary_row_for_player(summary_df: pl.DataFrame, steamid: int | None) -> dict[str, Any]:
    if summary_df.is_empty() or steamid is None or "steamid" not in summary_df.columns:
        return {}
    row = summary_df.filter(pl.col("steamid") == steamid)
    if row.is_empty():
        return {}
    return row.to_dicts()[0]


def _round_counts_by_side(ticks: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (player_round_side, rounds_side). player_round_side is empty when ticks lack required columns."""
    _empty_prs = pl.DataFrame(schema={"steamid": pl.UInt64, "name": pl.Utf8, "side": pl.Utf8, "round_num": pl.Int64})
    _empty_rs = pl.DataFrame(schema={"steamid": pl.UInt64, "name": pl.Utf8, "side": pl.Utf8, "rounds_played": pl.UInt32})
    if ticks.is_empty() or not all(col in ticks.columns for col in ["steamid", "name", "side", "round_num"]):
        return _empty_prs, _empty_rs
    player_round_side = (
        ticks.select(["steamid", "name", "side", "round_num"])
        .drop_nulls(["steamid", "side", "round_num"])
        .with_columns(pl.col("side").cast(pl.Utf8).str.to_uppercase().alias("side"))
        .filter(pl.col("side").is_in(["CT", "T"]))
        .unique()
    )
    rounds_side = (
        player_round_side.group_by(["steamid", "name", "side"])
        .agg(pl.len().alias("rounds_played"))
    )
    return player_round_side, rounds_side


def _kill_stats_by_side(
    kills: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return (kills_side, deaths_side, assists_side, hs_kills_side)."""
    kills_side = pl.DataFrame(schema={"steamid": pl.UInt64, "side": pl.Utf8, "kills": pl.Int64})
    deaths_side = pl.DataFrame(schema={"steamid": pl.UInt64, "side": pl.Utf8, "deaths": pl.Int64})
    assists_side = pl.DataFrame(schema={"steamid": pl.UInt64, "side": pl.Utf8, "assists": pl.Int64})
    hs_kills_side = pl.DataFrame(schema={"steamid": pl.UInt64, "side": pl.Utf8, "hs_kills": pl.Int64})
    if kills.is_empty():
        return kills_side, deaths_side, assists_side, hs_kills_side
    if all(c in kills.columns for c in ["attacker_steamid", "attacker_side"]):
        kills_side = (
            kills.select(
                [pl.col("attacker_steamid").alias("steamid"), pl.col("attacker_side").cast(pl.Utf8).str.to_uppercase().alias("side")]
            )
            .drop_nulls(["steamid", "side"])
            .filter(pl.col("side").is_in(["CT", "T"]))
            .group_by(["steamid", "side"])
            .agg(pl.len().alias("kills"))
        )
    if all(c in kills.columns for c in ["victim_steamid", "victim_side"]):
        deaths_side = (
            kills.select(
                [pl.col("victim_steamid").alias("steamid"), pl.col("victim_side").cast(pl.Utf8).str.to_uppercase().alias("side")]
            )
            .drop_nulls(["steamid", "side"])
            .filter(pl.col("side").is_in(["CT", "T"]))
            .group_by(["steamid", "side"])
            .agg(pl.len().alias("deaths"))
        )
    if all(c in kills.columns for c in ["assister_steamid", "assister_side"]):
        assists_side = (
            kills.select(
                [pl.col("assister_steamid").alias("steamid"), pl.col("assister_side").cast(pl.Utf8).str.to_uppercase().alias("side")]
            )
            .drop_nulls(["steamid", "side"])
            .filter(pl.col("side").is_in(["CT", "T"]))
            .group_by(["steamid", "side"])
            .agg(pl.len().alias("assists"))
        )
    if all(c in kills.columns for c in ["attacker_steamid", "attacker_side", "headshot"]):
        hs_kills_side = (
            kills.select(
                [pl.col("attacker_steamid").alias("steamid"), pl.col("attacker_side").cast(pl.Utf8).str.to_uppercase().alias("side"), "headshot"]
            )
            .drop_nulls(["steamid", "side"])
            .filter(pl.col("side").is_in(["CT", "T"]) & pl.col("headshot"))
            .group_by(["steamid", "side"])
            .agg(pl.len().alias("hs_kills"))
        )
    return kills_side, deaths_side, assists_side, hs_kills_side


def _opening_duel_stats_by_side(kills: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (opening_duels_side, opening_duels_won_side)."""
    opening_duels_side = pl.DataFrame(schema={"steamid": pl.UInt64, "side": pl.Utf8, "opening_duels": pl.Int64})
    opening_duels_won_side = pl.DataFrame(schema={"steamid": pl.UInt64, "side": pl.Utf8, "opening_duels_won": pl.Int64})
    if kills.is_empty() or not all(
        c in kills.columns for c in ["round_num", "tick", "attacker_steamid", "attacker_side", "victim_steamid", "victim_side"]
    ):
        return opening_duels_side, opening_duels_won_side
    opening_kills = (
        kills.select(
            ["round_num", "tick", pl.col("attacker_steamid").alias("steamid"), pl.col("attacker_side").cast(pl.Utf8).str.to_uppercase().alias("side")]
        )
        .drop_nulls(["round_num", "tick", "steamid", "side"])
        .sort(["round_num", "tick"])
        .group_by("round_num", maintain_order=True)
        .agg([pl.first("steamid").alias("steamid"), pl.first("side").alias("side")])
        .filter(pl.col("side").is_in(["CT", "T"]))
        .group_by(["steamid", "side"])
        .agg(pl.len().alias("opening_duels_won"))
    )
    opening_deaths = (
        kills.select(
            ["round_num", "tick", pl.col("victim_steamid").alias("steamid"), pl.col("victim_side").cast(pl.Utf8).str.to_uppercase().alias("side")]
        )
        .drop_nulls(["round_num", "tick", "steamid", "side"])
        .sort(["round_num", "tick"])
        .group_by("round_num", maintain_order=True)
        .agg([pl.first("steamid").alias("steamid"), pl.first("side").alias("side")])
        .filter(pl.col("side").is_in(["CT", "T"]))
        .group_by(["steamid", "side"])
        .agg(pl.len().alias("opening_duels_lost"))
    )
    opening_agg = (
        pl.concat(
            [
                opening_kills.with_columns(pl.lit(0).cast(pl.Int64).alias("opening_duels_lost")).select(
                    ["steamid", "side", "opening_duels_won", "opening_duels_lost"]
                ),
                opening_deaths.with_columns(pl.lit(0).cast(pl.Int64).alias("opening_duels_won")).select(
                    ["steamid", "side", "opening_duels_won", "opening_duels_lost"]
                ),
            ],
            how="vertical_relaxed",
        )
        .group_by(["steamid", "side"])
        .agg(
            [
                pl.col("opening_duels_won").sum().cast(pl.Int64).alias("opening_duels_won"),
                pl.col("opening_duels_lost").sum().cast(pl.Int64).alias("opening_duels_lost"),
            ]
        )
        .with_columns((pl.col("opening_duels_won") + pl.col("opening_duels_lost")).alias("opening_duels"))
    )
    return opening_agg.select(["steamid", "side", "opening_duels"]), opening_agg.select(["steamid", "side", "opening_duels_won"])


def _damage_stats_by_side(damages: pl.DataFrame) -> pl.DataFrame:
    """Return dmg_side with total enemy damage per player per side."""
    dmg_side = pl.DataFrame(schema={"steamid": pl.UInt64, "side": pl.Utf8, "total_damage": pl.Int64})
    if damages.is_empty() or not all(c in damages.columns for c in ["attacker_steamid", "attacker_side", "damage"]):
        return dmg_side
    d = damages.select(
        [
            pl.col("attacker_steamid").alias("steamid"),
            pl.col("attacker_side").cast(pl.Utf8).str.to_uppercase().alias("side"),
            "damage",
            pl.col("victim_side").cast(pl.Utf8).str.to_uppercase().alias("victim_side")
            if "victim_side" in damages.columns
            else pl.lit(None).alias("victim_side"),
        ]
    ).drop_nulls(["steamid", "side"])
    d = d.filter(pl.col("side").is_in(["CT", "T"]))
    if "victim_side" in d.columns:
        d = d.filter(
            pl.col("victim_side").is_null()
            | ~pl.col("victim_side").is_in(["CT", "T"])
            | (pl.col("side") != pl.col("victim_side"))
        )
    return d.group_by(["steamid", "side"]).agg(pl.col("damage").sum().cast(pl.Int64).alias("total_damage"))


def _economy_stats_by_side(demo: Any) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (econ_full, econ_force) — win rates per buy type, per player per side."""
    econ_full = pl.DataFrame(
        schema={"steamid": pl.UInt64, "side": pl.Utf8, "full_buy_rounds": pl.Int64, "full_buy_wins": pl.Int64, "full_buy_win_rate": pl.Float64}
    )
    econ_force = pl.DataFrame(
        schema={"steamid": pl.UInt64, "side": pl.Utf8, "force_rounds": pl.Int64, "force_wins": pl.Int64, "force_win_rate": pl.Float64}
    )
    economy_stats = build_economy_stats(demo)
    econ_per_round = economy_stats.get("economy_per_round", pl.DataFrame())
    if econ_per_round.is_empty() or not all(c in econ_per_round.columns for c in ["steamid", "side", "buy_type", "round_winner"]):
        return econ_full, econ_force
    econ_base = econ_per_round.with_columns(pl.col("side").cast(pl.Utf8).str.to_uppercase())
    econ_full = (
        econ_base.filter(pl.col("buy_type") == "full_buy")
        .group_by(["steamid", "side"])
        .agg(
            [
                pl.len().cast(pl.Int64).alias("full_buy_rounds"),
                pl.col("round_winner").cast(pl.Int64).sum().cast(pl.Int64).alias("full_buy_wins"),
                ((pl.col("round_winner").cast(pl.Int64).sum() / pl.len()) * 100.0).round(2).alias("full_buy_win_rate"),
            ]
        )
    )
    econ_force = (
        econ_base.filter(pl.col("buy_type") == "force")
        .group_by(["steamid", "side"])
        .agg(
            [
                pl.len().cast(pl.Int64).alias("force_rounds"),
                pl.col("round_winner").cast(pl.Int64).sum().cast(pl.Int64).alias("force_wins"),
                ((pl.col("round_winner").cast(pl.Int64).sum() / pl.len()) * 100.0).round(2).alias("force_win_rate"),
            ]
        )
    )
    return econ_full, econ_force


def _clutch_stats_by_side(demo: Any) -> pl.DataFrame:
    """Return clutch_side — clutch attempts, wins and win rate per player per side."""
    clutch_side = pl.DataFrame(
        schema={"steamid": pl.UInt64, "side": pl.Utf8, "clutch_attempts": pl.Int64, "clutches_won": pl.Int64, "clutch_win_rate": pl.Float64}
    )
    clutch_stats = build_clutch_stats(demo)
    clutch_rounds = clutch_stats.get("clutch_rounds", pl.DataFrame())
    if clutch_rounds.is_empty() or not all(c in clutch_rounds.columns for c in ["steamid", "side", "won"]):
        return clutch_side
    return (
        clutch_rounds.with_columns(pl.col("side").cast(pl.Utf8).str.to_uppercase())
        .group_by(["steamid", "side"])
        .agg(
            [
                pl.len().cast(pl.Int64).alias("clutch_attempts"),
                pl.col("won").cast(pl.Int64).sum().cast(pl.Int64).alias("clutches_won"),
                ((pl.col("won").cast(pl.Int64).sum() / pl.len()) * 100.0).round(2).alias("clutch_win_rate"),
            ]
        )
    )


def _kast_by_side(kills: pl.DataFrame, player_round_side: pl.DataFrame) -> pl.DataFrame:
    """Return kast_side — KAST% per player per side."""
    had_kill = pl.DataFrame(schema={"steamid": pl.UInt64, "side": pl.Utf8, "round_num": pl.Int64, "had_kill": pl.Boolean})
    had_assist = pl.DataFrame(schema={"steamid": pl.UInt64, "side": pl.Utf8, "round_num": pl.Int64, "had_assist": pl.Boolean})
    deaths = pl.DataFrame(schema={"steamid": pl.UInt64, "side": pl.Utf8, "round_num": pl.Int64})
    if not kills.is_empty():
        if all(c in kills.columns for c in ["attacker_steamid", "attacker_side", "round_num"]):
            had_kill = (
                kills.select(
                    [pl.col("attacker_steamid").alias("steamid"), pl.col("attacker_side").cast(pl.Utf8).str.to_uppercase().alias("side"), "round_num"]
                )
                .drop_nulls(["steamid", "side", "round_num"])
                .filter(pl.col("side").is_in(["CT", "T"]))
                .unique()
                .with_columns(pl.lit(True).alias("had_kill"))
            )
        if all(c in kills.columns for c in ["assister_steamid", "assister_side", "round_num"]):
            had_assist = (
                kills.select(
                    [pl.col("assister_steamid").alias("steamid"), pl.col("assister_side").cast(pl.Utf8).str.to_uppercase().alias("side"), "round_num"]
                )
                .drop_nulls(["steamid", "side", "round_num"])
                .filter(pl.col("side").is_in(["CT", "T"]))
                .unique()
                .with_columns(pl.lit(True).alias("had_assist"))
            )
        if all(c in kills.columns for c in ["victim_steamid", "victim_side", "round_num"]):
            deaths = (
                kills.select(
                    [pl.col("victim_steamid").alias("steamid"), pl.col("victim_side").cast(pl.Utf8).str.to_uppercase().alias("side"), "round_num"]
                )
                .drop_nulls(["steamid", "side", "round_num"])
                .filter(pl.col("side").is_in(["CT", "T"]))
                .unique()
            )
    prs = player_round_side.select(["steamid", "side", "round_num"]).unique()
    kast_base = prs
    for frame in (had_kill, had_assist):
        if not frame.is_empty():
            kast_base = kast_base.join(frame, on=["steamid", "side", "round_num"], how="left")
    kast_base = kast_base.join(
        deaths.with_columns(pl.lit(True).alias("died")) if not deaths.is_empty() else deaths,
        on=["steamid", "side", "round_num"],
        how="left",
    )
    kast_base = kast_base.with_columns(
        [
            pl.col("had_kill").fill_null(False),
            pl.col("had_assist").fill_null(False),
            pl.col("died").fill_null(False),
        ]
    ).with_columns((~pl.col("died")).alias("survived"))
    kast_base = kast_base.with_columns(
        (pl.col("had_kill") | pl.col("had_assist") | pl.col("survived")).alias("kast_round")
    )
    return (
        kast_base.group_by(["steamid", "side"])
        .agg((pl.col("kast_round").cast(pl.Int64).sum() / pl.len() * 100.0).round(2).alias("kast"))
    )


def _build_benchmark_player_rows(
    demo: Any,
) -> list[dict[str, Any]]:
    ticks = _safe_df(getattr(demo, "ticks", None))
    kills = _kills_df(demo)
    damages = _damages_df(demo)

    player_round_side, rounds_side = _round_counts_by_side(ticks)
    if player_round_side.is_empty():
        return []

    kills_side, deaths_side, assists_side, hs_kills_side = _kill_stats_by_side(kills)
    opening_duels_side, opening_duels_won_side = _opening_duel_stats_by_side(kills)
    dmg_side = _damage_stats_by_side(damages)
    econ_full, econ_force = _economy_stats_by_side(demo)
    clutch_side = _clutch_stats_by_side(demo)

    base = rounds_side
    for frame in (kills_side, deaths_side, assists_side, hs_kills_side, opening_duels_side, opening_duels_won_side, dmg_side, econ_full, econ_force, clutch_side):
        if not frame.is_empty():
            base = base.join(frame, on=["steamid", "side"], how="left")

    base = base.with_columns(
        [
            pl.col("kills").fill_null(0),
            pl.col("deaths").fill_null(0),
            pl.col("assists").fill_null(0),
            pl.col("hs_kills").fill_null(0),
            pl.col("opening_duels").fill_null(0),
            pl.col("opening_duels_won").fill_null(0),
            pl.col("total_damage").fill_null(0),
            pl.col("full_buy_rounds").fill_null(0),
            pl.col("full_buy_wins").fill_null(0),
            pl.col("force_rounds").fill_null(0),
            pl.col("force_wins").fill_null(0),
            pl.col("clutch_attempts").fill_null(0),
            pl.col("clutches_won").fill_null(0),
        ]
    ).with_columns(
        [
            pl.when(pl.col("rounds_played") > 0)
            .then((pl.col("total_damage") / pl.col("rounds_played")).round(2))
            .otherwise(0.0)
            .alias("adr"),
            pl.when(pl.col("rounds_played") > 0)
            .then((pl.col("kills") / pl.col("rounds_played")).round(2))
            .otherwise(0.0)
            .alias("kpr"),
            pl.when(pl.col("kills") > 0)
            .then((pl.col("hs_kills") / pl.col("kills") * 100.0).round(2))
            .otherwise(0.0)
            .alias("hs_percent"),
            pl.when(pl.col("opening_duels") > 0)
            .then((pl.col("opening_duels_won") / pl.col("opening_duels") * 100.0).round(2))
            .otherwise(0.0)
            .alias("opening_duel_win_pct"),
        ]
    )

    kast_side = _kast_by_side(kills, player_round_side)
    base = base.join(kast_side, on=["steamid", "side"], how="left").with_columns(pl.col("kast").fill_null(0.0))

    cols = [
        "steamid", "name", "side", "rounds_played", "kills", "deaths", "assists", "hs_kills",
        "adr", "kast", "hs_percent", "kpr", "opening_duels", "opening_duels_won",
        "opening_duel_win_pct", "full_buy_rounds", "full_buy_wins", "full_buy_win_rate",
        "force_rounds", "force_wins", "force_win_rate", "clutch_attempts", "clutches_won",
        "clutch_win_rate",
    ]
    return (
        base.select([c for c in cols if c in base.columns])
        .rename({"hs_kills": "headshot_kills"} if "hs_kills" in base.columns else {})
        .to_dicts()
    )



def analyse_demo(demo, player_selector: str | int | None = None, match_id: str | None = None):
    header_info = getattr(demo, "header", {}) or {}
    map_name = header_info.get("map_name", None) if isinstance(header_info, dict) else None
    rounds_played = _safe_len(getattr(demo, "rounds", None))

    raw_overall = build_raw_overall_stats(demo)
    overall = build_overall_table(raw_overall)
    selected_player = select_player(overall, player_selector=player_selector)
    selected_player_stats = selected_player_overall_stats(selected_player, log_stats=True)
    selected_steamid = selected_player_stats.get("steamid")

    economy_stats = build_economy_stats(demo)
    clutch_stats = build_clutch_stats(demo)

    economy_summary_row = _summary_row_for_player(
        economy_stats.get("economy_summary", pl.DataFrame()),
        selected_steamid,
    )
    clutch_summary_row = _summary_row_for_player(
        clutch_stats.get("clutch_summary", pl.DataFrame()),
        selected_steamid,
    )

    side_rows = _build_benchmark_player_rows(demo)
    ct_rows = [row for row in side_rows if str(row.get("side", "")).upper() == "CT"]
    t_rows = [row for row in side_rows if str(row.get("side", "")).upper() == "T"]
    sample_match_id = match_id or "temporary_unpersisted_match"
    match_samples = make_contextual_match_samples(
        match_id=sample_match_id,
        map_name=map_name,
        round_count=rounds_played if rounds_played > 0 else None,
        ct_player_stats=ct_rows,
        t_player_stats=t_rows,
    )
    historical_samples = load_benchmark_samples()
    evaluation_samples = [
        sample
        for sample in historical_samples
        if match_id is None or str(sample.get("match_id")) != str(match_id)
    ]
    benchmark_pool_source = "historical"
    analyzed_registry = load_analyzed_matches()
    match_already_analyzed = bool(match_id) and is_match_analyzed(str(match_id), analyzed_registry)

    def _selected_match_sample(side: str) -> dict[str, Any]:
        side_upper = side.upper()
        return next(
            (
                sample
                for sample in match_samples
                if str(sample.get("steamid")) == str(selected_steamid)
                and str(sample.get("side", "ALL")).upper() == side_upper
            ),
            {},
        )

    def _selected_side_row(side: str) -> dict[str, Any]:
        return next(
            (
                row
                for row in side_rows
                if str(row.get("steamid")) == str(selected_steamid)
                and str(row.get("side", "")).upper() == side
            ),
            {},
        )

    def _evaluate_for_side(side: str) -> dict[str, Any]:
        side_upper = side.upper()
        sample = _selected_match_sample(side_upper)
        metrics = sample.get("metrics") if isinstance(sample.get("metrics"), dict) else {}
        counts = sample.get("counts") if isinstance(sample.get("counts"), dict) else {}
        rounds = sample.get("rounds_played")

        if not metrics and side_upper != "ALL":
            row = _selected_side_row(side_upper)
            metrics = {
                "adr": row.get("adr"),
                "kast": row.get("kast"),
                "hs_percent": row.get("hs_percent"),
                "kpr": row.get("kpr"),
                "opening_duel_win_pct": row.get("opening_duel_win_pct"),
                "full_buy_win_rate": row.get("full_buy_win_rate"),
                "force_win_rate": row.get("force_win_rate"),
                "clutch_win_rate": row.get("clutch_win_rate"),
            }
            counts = {
                "kills": row.get("kills"),
                "opening_duels": row.get("opening_duels"),
                "full_buy_rounds": row.get("full_buy_rounds"),
                "force_rounds": row.get("force_rounds"),
                "clutch_attempts": row.get("clutch_attempts"),
            }
            rounds = row.get("rounds_played")

        return evaluate_player(
            metrics,
            evaluation_samples,
            map_name=map_name,
            side=side_upper if side_upper in {"CT", "T", "ALL"} else "ALL",
            player_counts=counts,
            rounds_played=rounds,
        )

    benchmark_evaluations_all = _evaluate_for_side("ALL")
    benchmark_evaluations_ct = _evaluate_for_side("CT")
    benchmark_evaluations_t = _evaluate_for_side("T")
    benchmark_evaluations = benchmark_evaluations_all
    samples_appended = False
    all_samples = historical_samples
    if match_id and not match_already_analyzed:
        all_samples = append_match_samples(match_samples)
        samples_appended = True
        mark_match_analyzed(
            str(match_id),
            {
                "map_name": map_name,
                "round_count": rounds_played,
                "samples_written": len(match_samples),
            },
        )

    feedback = generate_feedback(
        {
            "overall_stats": selected_player_stats,
            "economy_summary": economy_summary_row,
            "clutch_summary": clutch_summary_row,
            "benchmark_evaluations": benchmark_evaluations,
        }
    )

    logger.info("Analyzing demo on map: %s", map_name or "Unknown")
    logger.info("Rounds played: %d", rounds_played)

    return {
        "map_name": map_name,
        "rounds_played": rounds_played,
        "overall": overall,
        "selected_player": selected_player,
        "selected_player_stats": selected_player_stats,
        "economy_stats": economy_stats,
        "clutch_stats": clutch_stats,
        "economy_summary_selected": economy_summary_row,
        "clutch_summary_selected": clutch_summary_row,
        "benchmark_evaluations": benchmark_evaluations,
        "benchmark_evaluations_all": benchmark_evaluations_all,
        "benchmark_evaluations_ct": benchmark_evaluations_ct,
        "benchmark_evaluations_t": benchmark_evaluations_t,
        "benchmark_pool_source": benchmark_pool_source,
        "benchmark_pool_size_before_append": len(historical_samples),
        "benchmark_pool_size_after_append": len(all_samples),
        "benchmark_samples_appended": samples_appended,
        "benchmark_match_already_analyzed": match_already_analyzed,
        "side_breakdown": {
            "CT": benchmark_evaluations_ct,
            "T": benchmark_evaluations_t,
        },
        "feedback": feedback,
    }


if __name__ == "__main__":
    analyse_demo(load_demo_for_analysis())
