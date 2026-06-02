from __future__ import annotations

import argparse
import logging
import math
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Parser import load_cached_demo
from coach_metrics import _damages_df, _kills_df, _safe_df
from ml.dataset import DEFAULT_CACHE_DIR, DEFAULT_CACHE_KEY_PATH, discover_cache_keys
from ml.features import (
    _bomb_plant_ticks,
    _build_round_time_bounds,
    _clamp_alive_count,
    _demo_tickrate,
    _normalize_side,
    _round_rosters,
    _seconds_remaining_for_snapshot,
)


LOGGER = logging.getLogger(__name__)
REPO_ROOT = SRC_DIR.parent
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "ml" / "death_risk_timeseries_5s.parquet"
BUILD_VERSION = "death_risk_timeseries_v1"

REQUIRED_COLUMNS = [
    "match_id",
    "map_name",
    "round_num",
    "tick",
    "steamid",
    "player_name",
    "side",
    "alive_team_at_snapshot",
    "alive_enemy_at_snapshot",
    "seconds_remaining_at_snapshot",
    "bomb_planted_at_snapshot",
    "player_alive",
    "sample_time_seconds",
    "build_version",
    "death_within_5s",
    "kill_within_5s",
]
OPTIONAL_COLUMNS = [
    "player_hp",
    "weapon",
    "has_armor",
    "has_helmet",
    "money",
    "equipment_value",
    "nearest_teammate_distance",
    "nearest_enemy_distance",
    "prior_round_phase",
    "damage_dealt_next_5s",
    "damage_taken_next_5s",
]
OUTPUT_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build time-sampled death risk dataset with one row per alive player per sampled tick.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory to scan (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--sample-every-seconds",
        type=float,
        default=1.0,
        help="Sample spacing in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--horizon-seconds",
        type=float,
        default=5.0,
        help="Future label horizon in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--limit-matches",
        type=int,
        default=None,
        help="Optional cap on cached matches to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue after match errors (default: true).",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _seconds_to_ticks(seconds: float, tickrate: float) -> int:
    return max(1, int(round(float(seconds) * float(tickrate))))


def _distance(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    lx = _safe_float(left.get("X"))
    ly = _safe_float(left.get("Y"))
    rx = _safe_float(right.get("X"))
    ry = _safe_float(right.get("Y"))
    if lx is None or ly is None or rx is None or ry is None:
        return None

    lz = _safe_float(left.get("Z"))
    rz = _safe_float(right.get("Z"))
    dz = 0.0 if lz is None or rz is None else lz - rz
    return float(math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2 + dz**2))


def _nearest_distance(player: dict[str, Any], others: list[dict[str, Any]]) -> float | None:
    distances = [
        distance
        for other in others
        if str(other.get("steamid")) != str(player.get("steamid"))
        for distance in [_distance(player, other)]
        if distance is not None
    ]
    return min(distances) if distances else None


def _round_phase(
    *,
    tick: int,
    round_start: int,
    plant_tick: int | None,
    tickrate: float,
) -> str:
    if plant_tick is not None and plant_tick <= tick:
        return "post_plant"
    elapsed = max(0.0, float(tick - round_start) / tickrate)
    if elapsed < 20.0:
        return "early"
    if elapsed < 60.0:
        return "mid"
    return "late"


def _tick_select_exprs(ticks: pl.DataFrame) -> list[pl.Expr]:
    optional_defaults: dict[str, pl.Expr] = {
        "health": pl.lit(None, dtype=pl.Int64).alias("health"),
        "active_weapon": pl.lit(None, dtype=pl.Utf8).alias("active_weapon"),
        "armor": pl.lit(None, dtype=pl.Int64).alias("armor"),
        "has_helmet": pl.lit(None, dtype=pl.Boolean).alias("has_helmet"),
        "money": pl.lit(None, dtype=pl.Int64).alias("money"),
        "current_equip_value": pl.lit(None, dtype=pl.Int64).alias("current_equip_value"),
        "X": pl.lit(None, dtype=pl.Float64).alias("X"),
        "Y": pl.lit(None, dtype=pl.Float64).alias("Y"),
        "Z": pl.lit(None, dtype=pl.Float64).alias("Z"),
    }
    expressions: list[pl.Expr] = [
        pl.col("round_num").cast(pl.Int64, strict=False),
        pl.col("tick").cast(pl.Int64, strict=False),
        pl.col("steamid").cast(pl.UInt64, strict=False),
        pl.col("name").cast(pl.Utf8, strict=False).alias("player_name")
        if "name" in ticks.columns
        else pl.lit(None, dtype=pl.Utf8).alias("player_name"),
        pl.col("side").cast(pl.Utf8, strict=False),
    ]
    for column, default_expr in optional_defaults.items():
        if column in ticks.columns:
            dtype = default_expr.meta.output_name()
            if column in {"X", "Y", "Z"}:
                expressions.append(pl.col(column).cast(pl.Float64, strict=False))
            elif column in {"active_weapon"}:
                expressions.append(pl.col(column).cast(pl.Utf8, strict=False))
            elif column == "has_helmet":
                expressions.append(pl.col(column).cast(pl.Boolean, strict=False))
            else:
                expressions.append(pl.col(column).cast(pl.Int64, strict=False))
        else:
            expressions.append(default_expr)
    return expressions


def _prepare_ticks(ticks: pl.DataFrame) -> pl.DataFrame:
    required = {"round_num", "tick", "steamid", "side"}
    if ticks.is_empty() or not required.issubset(ticks.columns):
        return pl.DataFrame()
    return (
        ticks.select(_tick_select_exprs(ticks))
        .drop_nulls(["round_num", "tick", "steamid", "side"])
        .with_columns(
            [
                pl.col("side").map_elements(_normalize_side, return_dtype=pl.Utf8),
                pl.col("steamid").cast(pl.Utf8),
            ]
        )
        .drop_nulls(["side"])
        .sort(["round_num", "tick", "steamid"])
    )


def _normalize_kill_events(kills: pl.DataFrame) -> dict[int, list[dict[str, Any]]]:
    required = {"round_num", "tick", "victim_steamid"}
    if kills.is_empty() or not required.issubset(kills.columns):
        return {}

    expressions: list[pl.Expr] = [
        pl.col("round_num").cast(pl.Int64, strict=False),
        pl.col("tick").cast(pl.Int64, strict=False),
        pl.col("victim_steamid").cast(pl.UInt64, strict=False),
        pl.col("victim_side").cast(pl.Utf8, strict=False)
        if "victim_side" in kills.columns
        else pl.lit(None, dtype=pl.Utf8).alias("victim_side"),
        pl.col("attacker_steamid").cast(pl.UInt64, strict=False)
        if "attacker_steamid" in kills.columns
        else pl.lit(None, dtype=pl.UInt64).alias("attacker_steamid"),
        pl.col("attacker_side").cast(pl.Utf8, strict=False)
        if "attacker_side" in kills.columns
        else pl.lit(None, dtype=pl.Utf8).alias("attacker_side"),
    ]
    rows = (
        kills.select(expressions)
        .drop_nulls(["round_num", "tick", "victim_steamid"])
        .sort(["round_num", "tick"])
        .to_dicts()
    )

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        attacker_side = _normalize_side(row.get("attacker_side"))
        victim_side = _normalize_side(row.get("victim_side"))
        grouped[int(row["round_num"])].append(
            {
                "tick": int(row["tick"]),
                "victim_steamid": str(int(row["victim_steamid"])),
                "victim_side": victim_side,
                "attacker_steamid": str(int(row["attacker_steamid"]))
                if row.get("attacker_steamid") is not None
                else None,
                "attacker_side": attacker_side,
                "is_enemy_kill": attacker_side is not None
                and victim_side is not None
                and attacker_side != victim_side,
            }
        )
    return dict(grouped)


def _normalize_damage_events(damages: pl.DataFrame) -> dict[int, list[dict[str, Any]]]:
    required = {"round_num", "tick", "attacker_steamid", "victim_steamid", "damage"}
    if damages.is_empty() or not required.issubset(damages.columns):
        return {}

    expressions: list[pl.Expr] = [
        pl.col("round_num").cast(pl.Int64, strict=False),
        pl.col("tick").cast(pl.Int64, strict=False),
        pl.col("attacker_steamid").cast(pl.UInt64, strict=False),
        pl.col("victim_steamid").cast(pl.UInt64, strict=False),
        pl.col("damage").cast(pl.Float64, strict=False),
    ]
    rows = (
        damages.select(expressions)
        .drop_nulls(["round_num", "tick", "attacker_steamid", "victim_steamid", "damage"])
        .sort(["round_num", "tick"])
        .to_dicts()
    )

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["round_num"])].append(
            {
                "tick": int(row["tick"]),
                "attacker_steamid": str(int(row["attacker_steamid"])),
                "victim_steamid": str(int(row["victim_steamid"])),
                "damage": float(row["damage"] or 0.0),
            }
        )
    return dict(grouped)


def _event_ticks(events: list[dict[str, Any]]) -> list[int]:
    return [int(event["tick"]) for event in events]


def _events_in_horizon(
    events: list[dict[str, Any]],
    event_ticks: list[int],
    start_tick: int,
    end_tick: int,
) -> list[dict[str, Any]]:
    start_index = bisect_right(event_ticks, start_tick)
    end_index = bisect_right(event_ticks, end_tick)
    return events[start_index:end_index]


def _alive_after_prior_kills(
    roster: set[int],
    kill_events: list[dict[str, Any]],
    tick: int,
) -> set[str]:
    alive = {str(steamid) for steamid in roster}
    for event in kill_events:
        if int(event["tick"]) >= tick:
            break
        alive.discard(str(event.get("victim_steamid")))
    return alive


def _sample_ticks_for_round(
    ticks: pl.DataFrame,
    *,
    round_num: int,
    round_start: int,
    round_end: int,
    step_ticks: int,
) -> list[int]:
    round_ticks = (
        ticks.filter(
            (pl.col("round_num") == round_num)
            & (pl.col("tick") >= round_start)
            & (pl.col("tick") < round_end)
        )
        .select("tick")
        .unique()
        .sort("tick")
        .to_series()
        .to_list()
    )
    if not round_ticks:
        return []

    samples: list[int] = []
    next_target = round_start
    index = bisect_left(round_ticks, next_target)
    while index < len(round_ticks):
        tick = int(round_ticks[index])
        samples.append(tick)
        next_target = tick + step_ticks
        index = bisect_left(round_ticks, next_target, lo=index + 1)
    return samples


def _alive_counts_for_player(
    *,
    side: str,
    alive_ct: set[str],
    alive_t: set[str],
) -> tuple[int, int]:
    if side == "CT":
        return _clamp_alive_count(len(alive_ct)), _clamp_alive_count(len(alive_t))
    return _clamp_alive_count(len(alive_t)), _clamp_alive_count(len(alive_ct))


def _row_for_player(
    *,
    match_id: str,
    map_name: str | None,
    round_num: int,
    tick: int,
    round_start: int,
    round_end: int,
    tickrate: float,
    plant_tick: int | None,
    player: dict[str, Any],
    tick_players: list[dict[str, Any]],
    alive_ct: set[str],
    alive_t: set[str],
    kill_events: list[dict[str, Any]],
    kill_ticks: list[int],
    damage_events: list[dict[str, Any]],
    damage_ticks: list[int],
    horizon_ticks: int,
) -> dict[str, Any] | None:
    steamid = _safe_str(player.get("steamid"))
    side = _normalize_side(player.get("side"))
    if steamid is None or side is None:
        return None

    alive_side = alive_ct if side == "CT" else alive_t
    if steamid not in alive_side:
        return None

    horizon_end = min(round_end, tick + horizon_ticks)
    future_kills = _events_in_horizon(kill_events, kill_ticks, tick, horizon_end)
    future_damages = _events_in_horizon(damage_events, damage_ticks, tick, horizon_end)

    death_within_5s = any(event.get("victim_steamid") == steamid for event in future_kills)
    kill_within_5s = any(
        event.get("attacker_steamid") == steamid and event.get("is_enemy_kill") is True
        for event in future_kills
    )
    damage_dealt = sum(
        float(event.get("damage") or 0.0)
        for event in future_damages
        if event.get("attacker_steamid") == steamid
    )
    damage_taken = sum(
        float(event.get("damage") or 0.0)
        for event in future_damages
        if event.get("victim_steamid") == steamid
    )

    alive_team, alive_enemy = _alive_counts_for_player(
        side=side,
        alive_ct=alive_ct,
        alive_t=alive_t,
    )
    seconds_remaining, _ = _seconds_remaining_for_snapshot(
        snapshot_tick=tick,
        round_num=round_num,
        tickrate=tickrate,
        round_time_bounds={round_num: {"start": round_start, "end": round_end}},
    )
    bomb_planted = plant_tick is not None and plant_tick <= tick
    teammates = [row for row in tick_players if _normalize_side(row.get("side")) == side]
    enemies = [row for row in tick_players if _normalize_side(row.get("side")) != side]
    armor = _safe_int(player.get("armor"))
    health = _safe_int(player.get("health"))

    return {
        "match_id": match_id,
        "map_name": map_name,
        "round_num": round_num,
        "tick": tick,
        "steamid": steamid,
        "player_name": _safe_str(player.get("player_name")),
        "side": side,
        "alive_team_at_snapshot": alive_team,
        "alive_enemy_at_snapshot": alive_enemy,
        "seconds_remaining_at_snapshot": seconds_remaining,
        "bomb_planted_at_snapshot": bomb_planted,
        "player_alive": True,
        "sample_time_seconds": float(tick - round_start) / tickrate,
        "build_version": BUILD_VERSION,
        "player_hp": health,
        "weapon": _safe_str(player.get("active_weapon")),
        "has_armor": None if armor is None else armor > 0,
        "has_helmet": player.get("has_helmet"),
        "money": _safe_int(player.get("money")),
        "equipment_value": _safe_int(player.get("current_equip_value")),
        "nearest_teammate_distance": _nearest_distance(player, teammates),
        "nearest_enemy_distance": _nearest_distance(player, enemies),
        "prior_round_phase": _round_phase(
            tick=tick,
            round_start=round_start,
            plant_tick=plant_tick,
            tickrate=tickrate,
        ),
        "death_within_5s": death_within_5s,
        "kill_within_5s": kill_within_5s,
        "damage_dealt_next_5s": damage_dealt,
        "damage_taken_next_5s": damage_taken,
    }


def build_match_rows(cache_key: str, *, cache_dir: Path, sample_every_seconds: float, horizon_seconds: float) -> list[dict[str, Any]]:
    demo = load_cached_demo(cache_key, cache_dir=str(cache_dir))
    kills = _kills_df(demo)
    damages = _damages_df(demo)
    ticks = _prepare_ticks(_safe_df(getattr(demo, "ticks", None)))
    rounds = _safe_df(getattr(demo, "rounds", None))
    bomb = _safe_df(getattr(demo, "bomb", None))
    header = getattr(demo, "header", None)
    map_name = header.get("map_name") if isinstance(header, dict) else None
    tickrate = _demo_tickrate(demo)
    step_ticks = _seconds_to_ticks(sample_every_seconds, tickrate)
    horizon_ticks = _seconds_to_ticks(horizon_seconds, tickrate)

    if ticks.is_empty():
        LOGGER.warning("Skipping match with no usable ticks | match_id=%s", cache_key)
        return []

    round_time_bounds = _build_round_time_bounds(rounds)
    rosters = _round_rosters(ticks, kills, round_time_bounds=round_time_bounds, match_id=cache_key)
    plant_ticks = _bomb_plant_ticks(rounds, bomb)
    kill_events_by_round = _normalize_kill_events(kills)
    damage_events_by_round = _normalize_damage_events(damages)
    output_rows: list[dict[str, Any]] = []

    for round_num, bounds in sorted(round_time_bounds.items()):
        round_start = bounds.get("start")
        round_end = bounds.get("end")
        if round_start is None or round_end is None or round_end <= round_start:
            continue

        roster = rosters.get(round_num, {"CT": set(), "T": set()})
        if not roster.get("CT") or not roster.get("T"):
            continue

        sample_ticks = _sample_ticks_for_round(
            ticks,
            round_num=round_num,
            round_start=round_start,
            round_end=round_end,
            step_ticks=step_ticks,
        )
        if not sample_ticks:
            continue

        sampled_ticks = set(sample_ticks)
        sampled_rows_by_tick: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in (
            ticks.filter((pl.col("round_num") == round_num) & pl.col("tick").is_in(sample_ticks))
            .sort(["tick", "steamid"])
            .to_dicts()
        ):
            tick = int(row["tick"])
            if tick in sampled_ticks:
                sampled_rows_by_tick[tick].append(row)

        kill_events = kill_events_by_round.get(round_num, [])
        kill_ticks = _event_ticks(kill_events)
        damage_events = damage_events_by_round.get(round_num, [])
        damage_ticks = _event_ticks(damage_events)
        plant_tick = plant_ticks.get(round_num)

        for tick in sample_ticks:
            alive_ct = _alive_after_prior_kills(roster.get("CT", set()), kill_events, tick)
            alive_t = _alive_after_prior_kills(roster.get("T", set()), kill_events, tick)
            tick_players = sampled_rows_by_tick.get(tick, [])
            for player in tick_players:
                row = _row_for_player(
                    match_id=cache_key,
                    map_name=_safe_str(map_name),
                    round_num=round_num,
                    tick=tick,
                    round_start=round_start,
                    round_end=round_end,
                    tickrate=tickrate,
                    plant_tick=plant_tick,
                    player=player,
                    tick_players=tick_players,
                    alive_ct=alive_ct,
                    alive_t=alive_t,
                    kill_events=kill_events,
                    kill_ticks=kill_ticks,
                    damage_events=damage_events,
                    damage_ticks=damage_ticks,
                    horizon_ticks=horizon_ticks,
                )
                if row is not None:
                    output_rows.append(row)

    return output_rows


def _empty_dataset() -> pl.DataFrame:
    schema = {
        "match_id": pl.Utf8,
        "map_name": pl.Utf8,
        "round_num": pl.Int64,
        "tick": pl.Int64,
        "steamid": pl.Utf8,
        "player_name": pl.Utf8,
        "side": pl.Utf8,
        "alive_team_at_snapshot": pl.Int64,
        "alive_enemy_at_snapshot": pl.Int64,
        "seconds_remaining_at_snapshot": pl.Float64,
        "bomb_planted_at_snapshot": pl.Boolean,
        "player_alive": pl.Boolean,
        "sample_time_seconds": pl.Float64,
        "build_version": pl.Utf8,
        "player_hp": pl.Int64,
        "weapon": pl.Utf8,
        "has_armor": pl.Boolean,
        "has_helmet": pl.Boolean,
        "money": pl.Int64,
        "equipment_value": pl.Int64,
        "nearest_teammate_distance": pl.Float64,
        "nearest_enemy_distance": pl.Float64,
        "prior_round_phase": pl.Utf8,
        "death_within_5s": pl.Boolean,
        "kill_within_5s": pl.Boolean,
        "damage_dealt_next_5s": pl.Float64,
        "damage_taken_next_5s": pl.Float64,
    }
    return pl.DataFrame(schema=schema).select(OUTPUT_COLUMNS)


def _dataset_from_rows(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return _empty_dataset()
    return (
        pl.from_dicts(rows, infer_schema_length=None)
        .with_columns(
            [
                pl.col("match_id").cast(pl.Utf8),
                pl.col("map_name").cast(pl.Utf8),
                pl.col("round_num").cast(pl.Int64),
                pl.col("tick").cast(pl.Int64),
                pl.col("steamid").cast(pl.Utf8),
                pl.col("player_name").cast(pl.Utf8),
                pl.col("side").cast(pl.Utf8),
                pl.col("alive_team_at_snapshot").cast(pl.Int64),
                pl.col("alive_enemy_at_snapshot").cast(pl.Int64),
                pl.col("seconds_remaining_at_snapshot").cast(pl.Float64),
                pl.col("bomb_planted_at_snapshot").cast(pl.Boolean),
                pl.col("player_alive").cast(pl.Boolean),
                pl.col("sample_time_seconds").cast(pl.Float64),
                pl.col("build_version").cast(pl.Utf8),
                pl.col("player_hp").cast(pl.Int64, strict=False),
                pl.col("weapon").cast(pl.Utf8),
                pl.col("has_armor").cast(pl.Boolean, strict=False),
                pl.col("has_helmet").cast(pl.Boolean, strict=False),
                pl.col("money").cast(pl.Int64, strict=False),
                pl.col("equipment_value").cast(pl.Int64, strict=False),
                pl.col("nearest_teammate_distance").cast(pl.Float64, strict=False),
                pl.col("nearest_enemy_distance").cast(pl.Float64, strict=False),
                pl.col("prior_round_phase").cast(pl.Utf8),
                pl.col("death_within_5s").cast(pl.Boolean),
                pl.col("kill_within_5s").cast(pl.Boolean),
                pl.col("damage_dealt_next_5s").cast(pl.Float64),
                pl.col("damage_taken_next_5s").cast(pl.Float64),
            ]
        )
        .select(OUTPUT_COLUMNS)
        .sort(["match_id", "round_num", "tick", "steamid"])
    )


def _null_count(frame: pl.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return frame.height
    return int(frame.select(pl.col(column).is_null().sum()).item())


def _log_summary(
    dataset: pl.DataFrame,
    *,
    cached_demos_found: int,
    matches_processed: int,
    rows_per_match: Counter[str],
    errors: list[dict[str, str]],
    output_path: Path,
) -> None:
    LOGGER.info("cached demos found=%d", cached_demos_found)
    LOGGER.info("matches processed=%d", matches_processed)
    LOGGER.info("rows generated=%d", dataset.height)
    for match_id, count in sorted(rows_per_match.items()):
        LOGGER.info("rows per match | match_id=%s rows=%d", match_id, count)

    if not dataset.is_empty() and "map_name" in dataset.columns:
        for row in dataset.group_by("map_name").agg(pl.len().alias("rows")).sort("map_name").to_dicts():
            LOGGER.info("rows per map | map_name=%s rows=%d", row["map_name"], row["rows"])

    for column in ["death_within_5s", "kill_within_5s"]:
        if column in dataset.columns and not dataset.is_empty():
            positives = int(dataset.select(pl.col(column).cast(pl.Boolean).sum()).item())
            rate = positives / dataset.height if dataset.height else 0.0
            LOGGER.info("positive %s count=%d rate=%.6f", column, positives, rate)

    for column in [
        "alive_team_at_snapshot",
        "alive_enemy_at_snapshot",
        "seconds_remaining_at_snapshot",
        "bomb_planted_at_snapshot",
        "player_hp",
        "weapon",
        "has_armor",
        "equipment_value",
        "nearest_teammate_distance",
        "nearest_enemy_distance",
        "prior_round_phase",
    ]:
        LOGGER.info("null_count_%s=%d", column, _null_count(dataset, column))

    LOGGER.info("errors count=%d", len(errors))
    for error in errors[:20]:
        LOGGER.info("error match_id=%s message=%s", error.get("match_id"), error.get("error"))
    if len(errors) > 20:
        LOGGER.info("errors truncated=%d", len(errors) - 20)
    LOGGER.info("output path=%s", output_path)


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.sample_every_seconds <= 0:
        raise SystemExit("--sample-every-seconds must be > 0.")
    if args.horizon_seconds <= 0:
        raise SystemExit("--horizon-seconds must be > 0.")
    if args.limit_matches is not None and args.limit_matches < 0:
        raise SystemExit("--limit-matches must be >= 0.")
    if args.output.exists() and not args.force:
        raise SystemExit(f"Output already exists: {args.output}. Use --force to overwrite.")

    cache_keys_all = [
        cache_key
        for cache_key in discover_cache_keys(
            cache_dir=args.cache_dir,
            cache_key_path=DEFAULT_CACHE_KEY_PATH,
        )
        if (args.cache_dir / f"{cache_key}.pkl").exists()
    ]
    cache_keys = (
        cache_keys_all[: max(args.limit_matches, 0)]
        if args.limit_matches is not None
        else cache_keys_all
    )

    rows: list[dict[str, Any]] = []
    rows_per_match: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    matches_processed = 0
    LOGGER.info("cached demos found=%d", len(cache_keys_all))

    for cache_key in cache_keys:
        try:
            match_rows = build_match_rows(
                cache_key,
                cache_dir=args.cache_dir,
                sample_every_seconds=args.sample_every_seconds,
                horizon_seconds=args.horizon_seconds,
            )
            matches_processed += 1
            rows.extend(match_rows)
            rows_per_match[cache_key] = len(match_rows)
            LOGGER.info("processed match_id=%s rows=%d", cache_key, len(match_rows))
        except Exception as exc:
            LOGGER.exception("Failed processing match_id=%s", cache_key)
            errors.append({"match_id": cache_key, "error": str(exc)})
            if not args.continue_on_error:
                raise

    dataset = _dataset_from_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(args.output)
    _log_summary(
        dataset,
        cached_demos_found=len(cache_keys_all),
        matches_processed=matches_processed,
        rows_per_match=rows_per_match,
        errors=errors,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
