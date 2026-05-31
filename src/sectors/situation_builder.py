from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _to_float(v: Any, default: float | None = None) -> float | None:
    try:
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int | None = None) -> int | None:
    try:
        return default if v is None else int(v)
    except (TypeError, ValueError):
        return default


def _safe_bool(v: Any) -> bool:
    return bool(v) if not isinstance(v, bool) else v


def _safe_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _make_situation_id(
    match_id: str | None,
    steamid: str | None,
    round_num: int,
    situation_type: str,
    tick: int | None,
) -> str:
    mid = (str(match_id) if match_id else "noid")[:16]
    sid = str(steamid) if steamid is not None else "0"
    t = str(tick) if tick is not None else "x"
    return f"{mid}_{sid}_r{round_num}_{situation_type}_{t}"


# ---------------------------------------------------------------------------
# Extraction from report_data
# ---------------------------------------------------------------------------

def _extract_selected_info(
    report_data: dict[str, Any],
) -> tuple[int | None, str | None, str | None]:
    """Returns (raw_steamid_for_filter, str_steamid, player_name)."""
    selected_player = report_data.get("selected_player")
    if selected_player is not None:
        try:
            if hasattr(selected_player, "height") and selected_player.height > 0:
                row = selected_player.row(0, named=True)
                raw_sid = row.get("steamid")
                name = row.get("name")
                if raw_sid is not None:
                    player_name = str(name).strip() if name is not None else None
                    return int(raw_sid), str(raw_sid), player_name
        except Exception:
            pass

    stats = _safe_dict(report_data.get("selected_player_stats"))
    raw_sid = stats.get("steamid")
    name = stats.get("name")
    if raw_sid is not None:
        player_name = str(name).strip() if name else None
        return int(raw_sid), str(raw_sid), player_name

    return None, None, None


def _get_timeline_rows(report_data: dict[str, Any], raw_steamid: int | None) -> list[dict[str, Any]]:
    if raw_steamid is None:
        return []
    round_timeline = _safe_dict(report_data.get("round_timeline"))
    timeline_df = round_timeline.get("timeline_events")
    if not isinstance(timeline_df, pl.DataFrame) or timeline_df.is_empty():
        return []
    try:
        filtered = timeline_df.filter(pl.col("steamid") == raw_steamid)
        return filtered.to_dicts()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# ML impact lookups
# ---------------------------------------------------------------------------

_TICK_TOLERANCE = 128  # ticks; event_tick is snapshot-after so may lag the kill tick by up to ~64 ticks


def _index_ml_events_by_round(events: Any) -> dict[int, list[dict[str, Any]]]:
    idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not isinstance(events, list):
        return dict(idx)
    for evt in events:
        if not isinstance(evt, dict):
            continue
        rn = _safe_int(evt.get("round_num"))
        if rn is not None:
            idx[rn].append(evt)
    return dict(idx)


def _build_ml_lookups(player_ml_impact: dict[str, Any]) -> dict[str, Any]:
    ml = _safe_dict(player_ml_impact)
    all_kills = ml.get("kill_events") or []
    best_kills = ml.get("best_kills") or []
    worst_deaths = ml.get("worst_deaths") or []

    return {
        # Event-level lookup: all kill events, best kill events, and worst death events
        "kill_events_by_round": _index_ml_events_by_round(all_kills),
        "best_kills_by_round": _index_ml_events_by_round(best_kills),
        "death_events_by_round": _index_ml_events_by_round(worst_deaths),
    }


def _match_ml_event(
    candidates: list[dict[str, Any]],
    *,
    killer_steamid: str | None,
    victim_steamid: str | None,
    weapon: str | None,
    tick: int | None,
) -> tuple[float | None, str | None]:
    """
    Match a single kill/death event against ML candidates in the same round.

    Returns (win_prob_delta, ambiguity_flag).
    ambiguity_flag is 'ml_ambiguous_match' when no reliable single match can be identified.
    """
    matched: list[dict[str, Any]] = []
    for evt in candidates:
        ks = _safe_str(evt.get("killer_steamid"))
        vs = _safe_str(evt.get("victim_steamid"))
        wp = _safe_str(evt.get("weapon"))
        if killer_steamid is not None and ks != killer_steamid:
            continue
        if victim_steamid is not None and vs != victim_steamid:
            continue
        # weapon filter only when both sides have a value
        if weapon is not None and wp is not None and weapon.lower() != wp.lower():
            continue
        matched.append(evt)

    if not matched:
        return None, None
    if len(matched) == 1:
        return _to_float(matched[0].get("win_prob_delta")), None

    # Disambiguate by tick proximity when a timeline tick is available
    if tick is not None:
        scored: list[tuple[int, dict[str, Any]]] = []
        for evt in matched:
            et = _safe_int(evt.get("event_tick")) or _safe_int(evt.get("tick_after"))
            if et is not None:
                diff = abs(et - tick)
                scored.append((diff, evt))
        if scored:
            scored.sort(key=lambda x: x[0])
            min_diff = scored[0][0]
            if min_diff <= _TICK_TOLERANCE:
                tied = [e for d, e in scored if d == min_diff]
                if len(tied) == 1:
                    return _to_float(tied[0].get("win_prob_delta")), None
                deltas = {_to_float(e.get("win_prob_delta")) for e in tied}
                if len(deltas) == 1:
                    return next(iter(deltas)), None
                return None, "ml_ambiguous_match"

    # No tick or no candidate within tolerance — require all remaining to agree
    deltas = {_to_float(e.get("win_prob_delta")) for e in matched}
    if len(deltas) == 1:
        return next(iter(deltas)), None
    return None, "ml_ambiguous_match"


# ---------------------------------------------------------------------------
# VOD priority lookup
# ---------------------------------------------------------------------------

def _build_vod_lookup(vod_review_priority: Any) -> dict[int, tuple[int, dict[str, Any]]]:
    result: dict[int, tuple[int, dict[str, Any]]] = {}
    if not isinstance(vod_review_priority, list):
        return result
    for rank, entry in enumerate(vod_review_priority, start=1):
        if not isinstance(entry, dict):
            continue
        rn = _safe_int(entry.get("round_num"))
        if rn is not None:
            result[rn] = (rank, entry)
    return result


# ---------------------------------------------------------------------------
# Situation builders
# ---------------------------------------------------------------------------

def _build_death_situation(
    row: dict[str, Any],
    steamid: str,
    player_name: str | None,
    match_id: str | None,
    map_name: str | None,
    ml: dict[str, Any],
    vod_lookup: dict[int, tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    round_num = _safe_int(row.get("round_num")) or 0
    tick = _safe_int(row.get("tick"))
    damage = _to_float(row.get("damage_before_death"))
    opponent_steamid = _safe_str(row.get("target_steamid"))
    weapon = _safe_str(row.get("weapon"))

    candidates = ml["death_events_by_round"].get(round_num, [])
    ml_impact, ambiguity_flag = _match_ml_event(
        candidates,
        killer_steamid=opponent_steamid,
        victim_steamid=steamid,
        weapon=weapon,
        tick=tick,
    )

    high_cost_death = ml_impact is not None and ml_impact <= -0.20
    low_cost_death = ml_impact is not None and -0.03 <= ml_impact <= 0.0

    source_flags: list[str] = ["timeline"]
    if ml_impact is not None:
        source_flags.append("ml_enriched")
    if ambiguity_flag:
        source_flags.append(ambiguity_flag)
    if high_cost_death:
        source_flags.append("high_cost_death")
    if low_cost_death:
        source_flags.append("low_cost_death")
    if round_num in vod_lookup:
        source_flags.append("vod_priority")

    vod_rank: int | None = vod_lookup[round_num][0] if round_num in vod_lookup else None

    return {
        "situation_id": _make_situation_id(match_id, steamid, round_num, "death_situation", tick),
        "match_id": match_id,
        "map_name": map_name,
        "round_num": round_num,
        "tick": tick,
        "steamid": steamid,
        "player_name": player_name,
        "side": _safe_str(row.get("side")),
        "situation_type": "death_situation",
        "opponent_steamid": _safe_str(row.get("target_steamid")),
        "opponent_name": _safe_str(row.get("target_name")),
        "opponent_side": _safe_str(row.get("target_side")),
        "weapon": _safe_str(row.get("weapon")),
        "damage_before_death": damage,
        "zero_damage_death": damage is not None and damage <= 0.0,
        "low_damage_death": damage is not None and 0.0 < damage < 40.0,
        "was_traded": _safe_bool(row.get("is_traded_death")),
        "was_untraded": not _safe_bool(row.get("is_traded_death")),
        "death_timing": _safe_str(row.get("round_phase")),
        "trade_delay_ticks": _safe_int(row.get("trade_delay_ticks")),
        "ml_impact": ml_impact,
        "high_cost_death": high_cost_death,
        "low_cost_death": low_cost_death,
        "vod_priority_rank": vod_rank,
        "source_flags": source_flags,
    }


def _build_kill_situation(
    row: dict[str, Any],
    steamid: str,
    player_name: str | None,
    match_id: str | None,
    map_name: str | None,
    ml: dict[str, Any],
    vod_lookup: dict[int, tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    round_num = _safe_int(row.get("round_num")) or 0
    tick = _safe_int(row.get("tick"))
    opponent_steamid = _safe_str(row.get("target_steamid"))
    weapon = _safe_str(row.get("weapon"))

    candidates = ml["kill_events_by_round"].get(round_num, [])
    ml_impact, ambiguity_flag = _match_ml_event(
        candidates,
        killer_steamid=steamid,
        victim_steamid=opponent_steamid,
        weapon=weapon,
        tick=tick,
    )

    # Check if this exact event matches a best_kill entry (reliable identity match)
    best_kill_candidates = ml["best_kills_by_round"].get(round_num, [])
    best_kill_match, _ = _match_ml_event(
        best_kill_candidates,
        killer_steamid=steamid,
        victim_steamid=opponent_steamid,
        weapon=weapon,
        tick=tick,
    )

    high_impact_kill = (ml_impact is not None and ml_impact >= 0.20) or (best_kill_match is not None)
    low_impact_kill = ml_impact is not None and 0.0 <= ml_impact <= 0.03

    source_flags: list[str] = ["timeline"]
    if ml_impact is not None:
        source_flags.append("ml_enriched")
    if ambiguity_flag:
        source_flags.append(ambiguity_flag)
    if high_impact_kill:
        source_flags.append("high_impact_kill")
    if low_impact_kill:
        source_flags.append("low_impact_kill")
    if round_num in vod_lookup:
        source_flags.append("vod_priority")

    return {
        "situation_id": _make_situation_id(match_id, steamid, round_num, "kill_situation", tick),
        "match_id": match_id,
        "map_name": map_name,
        "round_num": round_num,
        "tick": tick,
        "steamid": steamid,
        "player_name": player_name,
        "side": _safe_str(row.get("side")),
        "situation_type": "kill_situation",
        "opponent_steamid": _safe_str(row.get("target_steamid")),
        "opponent_name": _safe_str(row.get("target_name")),
        "opponent_side": _safe_str(row.get("target_side")),
        "weapon": _safe_str(row.get("weapon")),
        "is_headshot": _safe_bool(row.get("is_headshot")),
        "is_opening_kill": _safe_bool(row.get("is_opening_kill")),
        "is_trade_kill": _safe_bool(row.get("is_trade_kill")),
        "trade_delay_ticks": _safe_int(row.get("trade_delay_ticks")),
        "ml_impact": ml_impact,
        "high_impact_kill": high_impact_kill,
        "low_impact_kill": low_impact_kill,
        "source_flags": source_flags,
    }


def _build_opening_duel_situation(
    row: dict[str, Any],
    steamid: str,
    player_name: str | None,
    match_id: str | None,
    map_name: str | None,
    ml: dict[str, Any],
) -> dict[str, Any]:
    round_num = _safe_int(row.get("round_num")) or 0
    tick = _safe_int(row.get("tick"))
    event_type = str(row.get("event_type") or "").lower()
    result = "won" if event_type == "kill" else "lost"
    opponent_steamid = _safe_str(row.get("target_steamid"))
    weapon = _safe_str(row.get("weapon"))

    if result == "won":
        candidates = ml["kill_events_by_round"].get(round_num, [])
        ml_impact, ambiguity_flag = _match_ml_event(
            candidates,
            killer_steamid=steamid,
            victim_steamid=opponent_steamid,
            weapon=weapon,
            tick=tick,
        )
    else:
        candidates = ml["death_events_by_round"].get(round_num, [])
        ml_impact, ambiguity_flag = _match_ml_event(
            candidates,
            killer_steamid=opponent_steamid,
            victim_steamid=steamid,
            weapon=weapon,
            tick=tick,
        )

    source_flags: list[str] = ["timeline"]
    if ml_impact is not None:
        source_flags.append("ml_enriched")
    if ambiguity_flag:
        source_flags.append(ambiguity_flag)

    return {
        "situation_id": _make_situation_id(match_id, steamid, round_num, "opening_duel_situation", tick),
        "match_id": match_id,
        "map_name": map_name,
        "round_num": round_num,
        "tick": tick,
        "steamid": steamid,
        "player_name": player_name,
        "side": _safe_str(row.get("side")),
        "situation_type": "opening_duel_situation",
        "result": result,
        "opponent_name": _safe_str(row.get("target_name")),
        "weapon": _safe_str(row.get("weapon")),
        "ml_impact": ml_impact,
        "source_flags": source_flags,
    }


def _build_vod_review_situations(
    vod_review_priority: list[dict[str, Any]],
    steamid: str,
    player_name: str | None,
    match_id: str | None,
    map_name: str | None,
) -> list[dict[str, Any]]:
    situations: list[dict[str, Any]] = []
    if not isinstance(vod_review_priority, list):
        return situations
    for rank, entry in enumerate(vod_review_priority, start=1):
        if not isinstance(entry, dict):
            continue
        round_num = _safe_int(entry.get("round_num")) or 0
        ml_val = entry.get("ml_impact")
        ml_impact = _to_float(ml_val) if ml_val is not None else None
        reasons_raw = entry.get("reasons")
        reasons = (
            [str(r).strip() for r in reasons_raw if str(r).strip()]
            if isinstance(reasons_raw, list)
            else []
        )
        situations.append({
            "situation_id": _make_situation_id(match_id, steamid, round_num, "vod_review_situation", rank),
            "match_id": match_id,
            "map_name": map_name,
            "round_num": round_num,
            "side": _safe_str(entry.get("side")),
            "steamid": steamid,
            "player_name": player_name,
            "situation_type": "vod_review_situation",
            "priority": _safe_str(entry.get("priority")),
            "review_type": _safe_str(entry.get("review_type")),
            "reasons": reasons,
            "summary": _safe_str(entry.get("summary")),
            "ml_impact": ml_impact,
            "source_flags": ["vod_priority"],
        })
    return situations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_player_situations(report_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build normalized situation dicts for the selected player from analysis output.

    Input: report_data dict from Analyser.analyse_demo (with match_id added by caller,
    and optionally vod_review_priority / decision_simulation populated by build_match_report).
    """
    safe_data = _safe_dict(report_data)

    raw_steamid, str_steamid, player_name = _extract_selected_info(safe_data)
    match_id = _safe_str(safe_data.get("match_id"))
    map_name = _safe_str(safe_data.get("map_name"))

    ml = _build_ml_lookups(_safe_dict(safe_data.get("player_ml_impact")))
    vod_priority = safe_data.get("vod_review_priority") or []
    vod_lookup = _build_vod_lookup(vod_priority)

    timeline_rows = _get_timeline_rows(safe_data, raw_steamid)

    situations: list[dict[str, Any]] = []
    seen_opening_rounds: set[int] = set()

    for row in timeline_rows:
        event_type = str(row.get("event_type") or "").lower()
        round_num = _safe_int(row.get("round_num")) or 0

        if event_type == "death":
            situations.append(
                _build_death_situation(row, str_steamid or "", player_name, match_id, map_name, ml, vod_lookup)
            )
        elif event_type == "kill":
            situations.append(
                _build_kill_situation(row, str_steamid or "", player_name, match_id, map_name, ml, vod_lookup)
            )

        is_opening = _safe_bool(row.get("is_opening_kill")) or _safe_bool(row.get("is_opening_death"))
        if is_opening and round_num > 0 and round_num not in seen_opening_rounds:
            seen_opening_rounds.add(round_num)
            situations.append(
                _build_opening_duel_situation(row, str_steamid or "", player_name, match_id, map_name, ml)
            )

    situations.extend(
        _build_vod_review_situations(vod_priority, str_steamid or "", player_name, match_id, map_name)
    )

    return situations


def situations_to_frame(situations: list[dict[str, Any]]) -> pl.DataFrame:
    if not situations:
        return pl.DataFrame()
    # Collect all keys across all situation types for a consistent schema
    all_keys: list[str] = []
    seen: set[str] = set()
    for sit in situations:
        for k in sit:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    normalized = [{k: sit.get(k) for k in all_keys} for sit in situations]
    try:
        return pl.from_dicts(normalized)
    except Exception:
        return pl.DataFrame()


def save_situations_json(situations: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(situations, fh, ensure_ascii=False, indent=2, default=str)


def save_situations_parquet(situations: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = situations_to_frame(situations)
    df.write_parquet(str(path))
