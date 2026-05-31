from __future__ import annotations

import argparse
import logging
import math
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Parser import load_cached_demo
from coach_metrics import _kills_df, _safe_df
from ml.dataset import DEFAULT_CACHE_DIR
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
DEFAULT_SITUATIONS_PATH = REPO_ROOT / "data" / "situations" / "all_situations.parquet"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "ml" / "decision_snapshots.parquet"
BUILD_VERSION = "decision_snapshots_v1"
CREATED_FROM = "all_situations"
SOURCE_SITUATION_TYPES = {
    "death_situation",
    "kill_situation",
    "opening_duel_situation",
}
SNAPSHOT_OFFSETS_SECONDS = (5, 3, 1)
RIFLE_WEAPONS = {
    "ak-47",
    "aug",
    "famas",
    "galil ar",
    "m4a1",
    "m4a1-s",
    "sg 553",
}
PISTOL_WEAPONS = {
    "cz75-auto",
    "deagle",
    "dual berettas",
    "five-seven",
    "glock-18",
    "p2000",
    "p250",
    "r8 revolver",
    "tec-9",
    "usp-s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Decision Snapshot Dataset v1 from normalized situations.",
    )
    parser.add_argument(
        "--situations",
        default=str(DEFAULT_SITUATIONS_PATH),
        help="Input normalized situations parquet.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output parquet path for decision snapshots.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional cap on source situation rows processed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


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


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return bool(value)


def _safe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_weapon_name(value: Any) -> str | None:
    text = _safe_str(value)
    return text.lower() if text is not None else None


def _event_key(row: dict[str, Any], situation_type: str | None = None) -> tuple[str | None, str | None, int | None, int | None, str]:
    return (
        _safe_str(row.get("match_id")),
        _safe_str(row.get("steamid")),
        _safe_int(row.get("round_num")),
        _safe_int(row.get("tick")),
        situation_type or str(row.get("situation_type") or ""),
    )


def _interval_has_tick(ticks: list[int], start_exclusive: int, end_inclusive: int) -> bool:
    if not ticks or end_inclusive <= start_exclusive:
        return False
    start_idx = bisect_right(ticks, start_exclusive)
    return start_idx < len(ticks) and ticks[start_idx] <= end_inclusive


def _round_start_tick(round_time_bounds: dict[int, dict[str, int]], round_num: int) -> int | None:
    return round_time_bounds.get(round_num, {}).get("start")


def _seconds_to_ticks(seconds: int, tickrate: float) -> int:
    return max(1, int(round(float(seconds) * float(tickrate))))


def _snapshot_candidates(event_tick: int, round_start: int | None, tickrate: float) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for seconds_before_event in SNAPSHOT_OFFSETS_SECONDS:
        snapshot_tick = event_tick - _seconds_to_ticks(seconds_before_event, tickrate)
        if snapshot_tick >= event_tick:
            continue
        if round_start is not None and snapshot_tick < round_start:
            continue
        if snapshot_tick < 0:
            continue
        candidates.append((seconds_before_event, snapshot_tick))
    return candidates


def _is_awp_weapon(event_weapon: str | None) -> bool:
    return event_weapon == "awp"


def _is_rifle_weapon(event_weapon: str | None) -> bool:
    return event_weapon in RIFLE_WEAPONS


def _is_pistol_weapon(event_weapon: str | None) -> bool:
    return event_weapon in PISTOL_WEAPONS


def _load_situations(path: Path, limit_rows: int | None) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Situations parquet not found: {path}")

    dataset = pl.read_parquet(path)
    filtered = (
        dataset
        .filter(pl.col("situation_type").is_in(sorted(SOURCE_SITUATION_TYPES)))
        .filter(pl.col("tick").is_not_null())
        .sort(["match_id", "source_match_index", "source_player_index", "round_num", "tick"])
    )
    if limit_rows is not None:
        filtered = filtered.head(max(limit_rows, 0))
    return _dedupe_source_situations(filtered)


def _source_dedupe_key(row: dict[str, Any]) -> tuple[str, str | tuple[str, ...]]:
    situation_id = _safe_str(row.get("situation_id"))
    if situation_id is not None:
        return ("situation_id", situation_id)
    return (
        "composite",
        (
            _safe_str(row.get("match_id")) or "",
            _safe_str(row.get("steamid")) or "",
            str(_safe_int(row.get("round_num")) or ""),
            str(_safe_int(row.get("tick")) or ""),
            _safe_str(row.get("situation_type")) or "",
            _safe_str(row.get("opponent_steamid")) or "",
            _safe_str(row.get("weapon")) or "",
        ),
    )


def _source_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _safe_str(row.get("match_id")) or "",
        _safe_int(row.get("source_match_index")) or 0,
        _safe_int(row.get("source_player_index")) or 0,
        _safe_int(row.get("round_num")) or 0,
        _safe_int(row.get("tick")) if row.get("tick") is not None else -1,
        _safe_str(row.get("situation_type")) or "",
        _safe_str(row.get("steamid")) or "",
        _safe_str(row.get("situation_id")) or "",
    )


def _dedupe_source_situations(situations: pl.DataFrame) -> pl.DataFrame:
    if situations.is_empty():
        return situations

    rows = sorted(situations.to_dicts(), key=_source_sort_key)
    key_counts = Counter(_source_dedupe_key(row) for row in rows)
    duplicate_key_count = sum(1 for count in key_counts.values() if count > 1)
    if duplicate_key_count == 0:
        return situations

    deduped_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str | tuple[str, ...]]] = set()
    for row in rows:
        key = _source_dedupe_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_rows.append(row)

    LOGGER.info("source_situations_before_dedupe=%d", len(rows))
    LOGGER.info("source_duplicate_situation_id_count=%d", duplicate_key_count)
    LOGGER.info("source_situations_after_dedupe=%d", len(deduped_rows))
    return pl.from_dicts(deduped_rows, schema=situations.schema)


def _build_match_situation_index(match_rows: list[dict[str, Any]]) -> dict[str, Any]:
    event_lookup: dict[tuple[str | None, str | None, int | None, int | None, str], dict[str, Any]] = {}
    player_round_ticks: dict[tuple[str | None, int | None], dict[str, list[int]]] = defaultdict(
        lambda: {"kill_situation": [], "death_situation": [], "opening_duel_situation": []}
    )

    for row in match_rows:
        situation_type = str(row.get("situation_type") or "")
        key = _event_key(row)
        event_lookup[key] = row
        player_round_key = (_safe_str(row.get("steamid")), _safe_int(row.get("round_num")))
        tick = _safe_int(row.get("tick"))
        if tick is not None and situation_type in player_round_ticks[player_round_key]:
            player_round_ticks[player_round_key][situation_type].append(tick)

    for row_sets in player_round_ticks.values():
        for ticks in row_sets.values():
            ticks.sort()

    return {
        "event_lookup": event_lookup,
        "player_round_ticks": player_round_ticks,
    }


def _normalize_kill_rows(kills: pl.DataFrame) -> dict[int, list[dict[str, Any]]]:
    required = [
        "round_num",
        "tick",
        "victim_steamid",
        "victim_side",
    ]
    if kills.is_empty() or not all(column in kills.columns for column in required):
        return {}

    select_exprs: list[pl.Expr | str] = [
        pl.col("round_num").cast(pl.Int64, strict=False),
        pl.col("tick").cast(pl.Int64, strict=False),
        pl.col("victim_steamid").cast(pl.UInt64, strict=False),
        pl.col("victim_side").cast(pl.Utf8, strict=False),
    ]
    if "attacker_steamid" in kills.columns:
        select_exprs.append(pl.col("attacker_steamid").cast(pl.UInt64, strict=False))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.UInt64).alias("attacker_steamid"))
    if "attacker_side" in kills.columns:
        select_exprs.append(pl.col("attacker_side").cast(pl.Utf8, strict=False))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.Utf8).alias("attacker_side"))

    rows = (
        kills.select(select_exprs)
        .drop_nulls(["round_num", "tick", "victim_steamid", "victim_side"])
        .sort(["round_num", "tick"])
        .to_dicts()
    )

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        round_num = int(row["round_num"])
        grouped[round_num].append(
            {
                "tick": int(row["tick"]),
                "victim_steamid": int(row["victim_steamid"]),
                "victim_side": _normalize_side(row.get("victim_side")),
                "attacker_steamid": _safe_int(row.get("attacker_steamid")),
                "attacker_side": _normalize_side(row.get("attacker_side")),
            }
        )
    return dict(grouped)


def _build_player_tick_index(ticks: pl.DataFrame) -> dict[tuple[int, str], dict[str, list[Any]]]:
    if ticks.is_empty() or not all(column in ticks.columns for column in ["round_num", "steamid", "tick"]):
        return {}

    select_exprs: list[pl.Expr | str] = [
        pl.col("round_num").cast(pl.Int64, strict=False),
        pl.col("steamid").cast(pl.UInt64, strict=False),
        pl.col("tick").cast(pl.Int64, strict=False),
    ]
    if "active_weapon" in ticks.columns:
        select_exprs.append(pl.col("active_weapon").cast(pl.Utf8, strict=False))
    else:
        select_exprs.append(pl.lit(None, dtype=pl.Utf8).alias("active_weapon"))

    rows = (
        ticks.select(select_exprs)
        .drop_nulls(["round_num", "steamid", "tick"])
        .sort(["round_num", "steamid", "tick"])
        .to_dicts()
    )

    index: dict[tuple[int, str], dict[str, list[Any]]] = {}
    for row in rows:
        key = (int(row["round_num"]), str(int(row["steamid"])))
        bucket = index.setdefault(key, {"ticks": [], "weapons": []})
        bucket["ticks"].append(int(row["tick"]))
        bucket["weapons"].append(_safe_str(row.get("active_weapon")))
    return index


def _lookup_weapon_at_snapshot(
    player_tick_index: dict[tuple[int, str], dict[str, list[Any]]],
    round_num: int,
    steamid: str,
    snapshot_tick: int,
) -> str | None:
    bucket = player_tick_index.get((round_num, steamid))
    if not bucket:
        return None
    tick_values = bucket["ticks"]
    index = bisect_right(tick_values, snapshot_tick) - 1
    if index < 0:
        return None
    return _safe_str(bucket["weapons"][index])


def _alive_counts_at_snapshot(
    rosters: dict[int, dict[str, set[int]]],
    kill_rows_by_round: dict[int, list[dict[str, Any]]],
    state_cache: dict[tuple[int, int], tuple[int, int]],
    round_num: int,
    snapshot_tick: int,
) -> tuple[int | None, int | None]:
    key = (round_num, snapshot_tick)
    if key not in state_cache:
        roster = rosters.get(round_num)
        if roster is None:
            state_cache[key] = (None, None)
        else:
            alive_ct = set(roster.get("CT", set()))
            alive_t = set(roster.get("T", set()))
            for kill in kill_rows_by_round.get(round_num, []):
                kill_tick = int(kill["tick"])
                if kill_tick >= snapshot_tick:
                    break
                victim_side = kill.get("victim_side")
                victim_steamid = kill.get("victim_steamid")
                if victim_side == "CT" and victim_steamid is not None:
                    alive_ct.discard(int(victim_steamid))
                elif victim_side == "T" and victim_steamid is not None:
                    alive_t.discard(int(victim_steamid))
            state_cache[key] = (_clamp_alive_count(len(alive_ct)), _clamp_alive_count(len(alive_t)))
    return state_cache[key]


def _build_match_context(cache_key: str) -> dict[str, Any]:
    demo = load_cached_demo(cache_key, cache_dir=str(DEFAULT_CACHE_DIR))
    kills = _kills_df(demo)
    ticks = _safe_df(getattr(demo, "ticks", None))
    rounds = _safe_df(getattr(demo, "rounds", None))
    bomb = _safe_df(getattr(demo, "bomb", None))
    header = getattr(demo, "header", None)
    map_name = header.get("map_name") if isinstance(header, dict) else None

    round_time_bounds = _build_round_time_bounds(rounds)
    rosters = _round_rosters(ticks, kills, round_time_bounds=round_time_bounds, match_id=cache_key)
    return {
        "map_name": _safe_str(map_name),
        "tickrate": _demo_tickrate(demo),
        "round_time_bounds": round_time_bounds,
        "plant_ticks": _bomb_plant_ticks(rounds, bomb),
        "rosters": rosters,
        "kill_rows_by_round": _normalize_kill_rows(kills),
        "player_tick_index": _build_player_tick_index(ticks),
        "state_cache": {},
    }


def _exact_related_row(match_index: dict[str, Any], row: dict[str, Any], situation_type: str) -> dict[str, Any] | None:
    return match_index["event_lookup"].get(_event_key(row, situation_type=situation_type))


def _label_block(
    row: dict[str, Any],
    match_index: dict[str, Any],
    snapshot_tick: int,
    tickrate: float,
) -> dict[str, Any]:
    steamid = _safe_str(row.get("steamid"))
    round_num = _safe_int(row.get("round_num"))
    situation_type = str(row.get("situation_type") or "")
    event_tick = _safe_int(row.get("tick")) or 0
    five_second_horizon = snapshot_tick + _seconds_to_ticks(5, tickrate)
    player_round_key = (steamid, round_num)
    row_sets = match_index["player_round_ticks"].get(player_round_key, {})
    matched_opening = _exact_related_row(match_index, row, "opening_duel_situation")
    matched_kill = _exact_related_row(match_index, row, "kill_situation")
    matched_death = _exact_related_row(match_index, row, "death_situation")

    opening_duel_within_5s = matched_opening is not None or situation_type == "opening_duel_situation"
    opening_duel_won_within_5s = str((matched_opening or row).get("result") or "") == "won" if opening_duel_within_5s else False

    labels = {
        "death_within_5s": False,
        "kill_within_5s": False,
        "opening_duel_within_5s": opening_duel_within_5s,
        "opening_duel_won_within_5s": opening_duel_won_within_5s,
        "high_cost_death_within_5s": False,
        "high_impact_kill_within_5s": False,
        "ml_impact_at_event": _safe_float((matched_kill or matched_death or matched_opening or row).get("ml_impact")),
        "action_value_class": "neutral",
    }

    if situation_type == "death_situation":
        labels["death_within_5s"] = True
        labels["kill_within_5s"] = _interval_has_tick(
            row_sets.get("kill_situation", []),
            snapshot_tick,
            event_tick,
        )
        labels["high_cost_death_within_5s"] = _safe_bool(row.get("high_cost_death"))
        labels["action_value_class"] = (
            "bad"
            if _safe_bool(row.get("high_cost_death"))
            or _safe_bool(row.get("zero_damage_death"))
            or _safe_bool(row.get("was_untraded"))
            else "neutral"
        )
        return labels

    if situation_type == "kill_situation":
        labels["kill_within_5s"] = True
        labels["death_within_5s"] = _interval_has_tick(
            row_sets.get("death_situation", []),
            snapshot_tick,
            five_second_horizon,
        )
        labels["high_impact_kill_within_5s"] = _safe_bool(row.get("high_impact_kill"))
        if _safe_bool(row.get("high_impact_kill")):
            labels["action_value_class"] = "excellent"
        elif labels["ml_impact_at_event"] is not None and labels["ml_impact_at_event"] > 0.05:
            labels["action_value_class"] = "good"
        else:
            labels["action_value_class"] = "neutral"
        return labels

    if situation_type == "opening_duel_situation":
        result = str(row.get("result") or "")
        labels["kill_within_5s"] = result == "won"
        labels["death_within_5s"] = result == "lost"
        labels["high_impact_kill_within_5s"] = _safe_bool((matched_kill or {}).get("high_impact_kill"))
        labels["high_cost_death_within_5s"] = _safe_bool((matched_death or {}).get("high_cost_death"))
        labels["action_value_class"] = "good" if result == "won" else "bad"
        return labels

    return labels


def _feature_block(
    row: dict[str, Any],
    match_context: dict[str, Any],
    snapshot_tick: int,
) -> dict[str, Any]:
    round_num = _safe_int(row.get("round_num")) or 0
    steamid = _safe_str(row.get("steamid")) or ""
    side = _normalize_side(row.get("side")) or _safe_str(row.get("side"))
    alive_ct, alive_t = _alive_counts_at_snapshot(
        match_context["rosters"],
        match_context["kill_rows_by_round"],
        match_context["state_cache"],
        round_num,
        snapshot_tick,
    )
    if side == "CT":
        alive_team = alive_ct
        alive_enemy = alive_t
    elif side == "T":
        alive_team = alive_t
        alive_enemy = alive_ct
    else:
        alive_team = None
        alive_enemy = None

    round_bounds = match_context["round_time_bounds"].get(round_num, {})
    if "end" in round_bounds:
        seconds_remaining, _ = _seconds_remaining_for_snapshot(
            snapshot_tick=snapshot_tick,
            round_num=round_num,
            tickrate=match_context["tickrate"],
            round_time_bounds=match_context["round_time_bounds"],
        )
    else:
        seconds_remaining = None

    plant_tick = match_context["plant_ticks"].get(round_num)
    bomb_planted = plant_tick is not None and plant_tick <= snapshot_tick

    tick_weapon = _lookup_weapon_at_snapshot(
        match_context["player_tick_index"],
        round_num=round_num,
        steamid=steamid,
        snapshot_tick=snapshot_tick,
    )
    event_weapon = _safe_str(row.get("weapon"))
    weapon_fallback_to_event = tick_weapon is None and event_weapon is not None
    weapon = tick_weapon if tick_weapon is not None else event_weapon
    normalized_event_weapon = _normalize_weapon_name(event_weapon)
    related_opening = row.get("situation_type") == "opening_duel_situation"
    was_opening_context = related_opening or _safe_bool(row.get("is_opening_kill"))

    return {
        "alive_team_at_snapshot": alive_team,
        "alive_enemy_at_snapshot": alive_enemy,
        "seconds_remaining_at_snapshot": seconds_remaining,
        "bomb_planted_at_snapshot": bomb_planted,
        "player_side": side,
        "weapon": weapon,
        "weapon_fallback_to_event": weapon_fallback_to_event,
        "event_weapon": event_weapon,
        "is_awp_event": _is_awp_weapon(normalized_event_weapon),
        "is_rifle_event": _is_rifle_weapon(normalized_event_weapon),
        "is_pistol_event": _is_pistol_weapon(normalized_event_weapon),
        "prior_round_phase": _safe_str(row.get("death_timing")),
        "was_opening_context": was_opening_context,
    }


def _snapshot_row(
    row: dict[str, Any],
    match_context: dict[str, Any],
    match_index: dict[str, Any],
    *,
    snapshot_tick: int,
    seconds_before_event: int,
) -> dict[str, Any]:
    base = {
        "match_id": _safe_str(row.get("match_id")),
        "map_name": _safe_str(row.get("map_name")) or match_context["map_name"],
        "round_num": _safe_int(row.get("round_num")),
        "steamid": _safe_str(row.get("steamid")),
        "player_name": _safe_str(row.get("player_name")),
        "side": _safe_str(row.get("side")),
        "source_situation_id": _safe_str(row.get("situation_id")),
        "source_situation_type": _safe_str(row.get("situation_type")),
        "event_tick": _safe_int(row.get("tick")),
        "snapshot_tick": snapshot_tick,
        "seconds_before_event": seconds_before_event,
        "opponent_steamid": _safe_str(row.get("opponent_steamid")),
        "opponent_name": _safe_str(row.get("opponent_name")),
        "opponent_side": _safe_str(row.get("opponent_side")),
        "build_version": BUILD_VERSION,
        "created_from": CREATED_FROM,
        "source_match_index": _safe_int(row.get("source_match_index")),
        "source_player_index": _safe_int(row.get("source_player_index")),
    }
    features = _feature_block(row, match_context, snapshot_tick)
    labels = _label_block(row, match_index, snapshot_tick, match_context["tickrate"])
    return {**base, **features, **labels}


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("match_id") or ""),
            _safe_int(row.get("round_num")) or 0,
            _safe_int(row.get("snapshot_tick")) or -1,
            _safe_int(row.get("seconds_before_event")) or 0,
            str(row.get("source_situation_type") or ""),
            str(row.get("steamid") or ""),
        ),
    )


def build_decision_snapshots(situations: pl.DataFrame) -> list[dict[str, Any]]:
    rows = situations.to_dicts()
    rows_by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        match_id = _safe_str(row.get("match_id"))
        if match_id is None:
            continue
        rows_by_match[match_id].append(row)

    output_rows: list[dict[str, Any]] = []
    for match_id, match_rows in rows_by_match.items():
        LOGGER.info("Building decision snapshots for match_id=%s source_rows=%d", match_id, len(match_rows))
        match_context = _build_match_context(match_id)
        match_index = _build_match_situation_index(match_rows)

        for row in match_rows:
            round_num = _safe_int(row.get("round_num"))
            event_tick = _safe_int(row.get("tick"))
            if round_num is None or event_tick is None:
                continue
            round_start = _round_start_tick(match_context["round_time_bounds"], round_num)
            for seconds_before_event, snapshot_tick in _snapshot_candidates(
                event_tick=event_tick,
                round_start=round_start,
                tickrate=match_context["tickrate"],
            ):
                output_rows.append(
                    _snapshot_row(
                        row,
                        match_context,
                        match_index,
                        snapshot_tick=snapshot_tick,
                        seconds_before_event=seconds_before_event,
                    )
                )

    return _sort_rows(output_rows)


def _null_count(frame: pl.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return frame.height
    return int(frame.select(pl.col(column).is_null().sum()).item())


def _log_summary(dataset: pl.DataFrame, output_path: Path) -> None:
    LOGGER.info("snapshots_created=%d", dataset.height)

    for row in dataset.group_by("source_situation_type").agg(pl.len().alias("rows")).sort("source_situation_type").to_dicts():
        LOGGER.info("source_situation_type=%s rows=%d", row["source_situation_type"], row["rows"])

    for row in dataset.group_by("seconds_before_event").agg(pl.len().alias("rows")).sort("seconds_before_event").to_dicts():
        LOGGER.info("seconds_before_event=%s rows=%d", row["seconds_before_event"], row["rows"])

    for column in [
        "death_within_5s",
        "kill_within_5s",
        "high_cost_death_within_5s",
        "high_impact_kill_within_5s",
    ]:
        true_count = int(dataset.select(pl.col(column).cast(pl.Boolean).sum()).item()) if column in dataset.columns else 0
        LOGGER.info("target_%s_true=%d", column, true_count)

    if "action_value_class" in dataset.columns:
        for row in dataset.group_by("action_value_class").agg(pl.len().alias("rows")).sort("action_value_class").to_dicts():
            LOGGER.info("action_value_class=%s rows=%d", row["action_value_class"], row["rows"])

    for column in [
        "alive_team_at_snapshot",
        "alive_enemy_at_snapshot",
        "seconds_remaining_at_snapshot",
        "bomb_planted_at_snapshot",
        "weapon",
        "prior_round_phase",
    ]:
        LOGGER.info("null_count_%s=%d", column, _null_count(dataset, column))

    LOGGER.info("output path=%s", output_path)


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.limit_rows is not None and args.limit_rows < 0:
        raise SystemExit("--limit-rows must be >= 0.")

    situations_path = Path(args.situations)
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        raise SystemExit(f"Output already exists: {output_path}. Use --force to overwrite.")

    situations = _load_situations(situations_path, args.limit_rows)
    LOGGER.info("situations_loaded=%d", situations.height)

    rows = build_decision_snapshots(situations)
    dataset = pl.from_dicts(rows) if rows else pl.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_parquet(output_path)

    _log_summary(dataset, output_path)


if __name__ == "__main__":
    main()
