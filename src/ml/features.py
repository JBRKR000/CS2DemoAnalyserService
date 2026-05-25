from __future__ import annotations

import logging
from typing import Any

import polars as pl

from coach_metrics import _kills_df, _safe_df


LOGGER = logging.getLogger(__name__)
MAX_PLAYERS_PER_SIDE = 5
MAX_REASONABLE_ROUND_SECONDS = 120.0
BOMB_TIMER_SECONDS = 40.0

SNAPSHOT_COLUMNS = [
    "match_id",
    "map_name",
    "round_num",
    "event_id",
    "tick",
    "snapshot_type",
    "side",
    "event_type",
    "killer_steamid",
    "victim_steamid",
    "killer_name",
    "victim_name",
    "weapon",
    "killer_side",
    "victim_side",
    "kill_context_type",
    "alive_team",
    "alive_enemy",
    "seconds_remaining",
    "bomb_time_since_plant",
    "bomb_time_remaining",
    "is_time_anomaly",
    "bomb_planted",
    "opening_kill_for_team",
    "team_won_round",
]

SNAPSHOT_SCHEMA: dict[str, pl.DataType] = {
    "match_id": pl.Utf8,
    "map_name": pl.Utf8,
    "round_num": pl.Int64,
    "event_id": pl.Utf8,
    "tick": pl.Int64,
    "snapshot_type": pl.Utf8,
    "side": pl.Utf8,
    "event_type": pl.Utf8,
    "killer_steamid": pl.UInt64,
    "victim_steamid": pl.UInt64,
    "killer_name": pl.Utf8,
    "victim_name": pl.Utf8,
    "weapon": pl.Utf8,
    "killer_side": pl.Utf8,
    "victim_side": pl.Utf8,
    "kill_context_type": pl.Utf8,
    "alive_team": pl.Int64,
    "alive_enemy": pl.Int64,
    "seconds_remaining": pl.Float64,
    "bomb_time_since_plant": pl.Float64,
    "bomb_time_remaining": pl.Float64,
    "is_time_anomaly": pl.Boolean,
    "bomb_planted": pl.Boolean,
    "opening_kill_for_team": pl.Boolean,
    "team_won_round": pl.Boolean,
}


def empty_snapshot_dataset() -> pl.DataFrame:
    return pl.DataFrame(schema=SNAPSHOT_SCHEMA)


def _normalize_side(value: Any) -> str | None:
    if value is None:
        return None
    side = str(value).strip().upper()
    if side in {"CT", "COUNTERTERRORIST", "COUNTER-TERRORIST"}:
        return "CT"
    if side in {"T", "TERRORIST"}:
        return "T"
    return None


def _valid_steamid(value: Any) -> bool:
    try:
        steamid = int(value)
    except (TypeError, ValueError):
        return False
    return steamid > 0


def _clamp_alive_count(value: int) -> int:
    return min(max(int(value), 0), MAX_PLAYERS_PER_SIDE)


def _clamp_seconds_remaining(value: float) -> float:
    return min(max(float(value), 0.0), MAX_REASONABLE_ROUND_SECONDS)


def _build_round_time_bounds(rounds: pl.DataFrame) -> dict[int, dict[str, int]]:
    if rounds.is_empty() or "round_num" not in rounds.columns:
        return {}

    bounds: dict[int, dict[str, int]] = {}
    columns = set(rounds.columns)
    for row in rounds.to_dicts():
        if row.get("round_num") is None:
            continue

        round_num = int(row["round_num"])
        start_tick = None
        if "freeze_end" in columns and row.get("freeze_end") is not None:
            start_tick = int(row["freeze_end"])
        elif "start" in columns and row.get("start") is not None:
            start_tick = int(row["start"])

        end_candidates: list[int] = []
        if "end" in columns and row.get("end") is not None:
            end_candidates.append(int(row["end"]))
        if "official_end" in columns and row.get("official_end") is not None:
            end_candidates.append(int(row["official_end"]))

        end_tick = max(end_candidates) if end_candidates else None
        if start_tick is None and end_tick is None:
            continue

        bounds[round_num] = {}
        if start_tick is not None:
            bounds[round_num]["start"] = start_tick
        if end_tick is not None:
            bounds[round_num]["end"] = end_tick

    return bounds


def _round_rosters(
    ticks: pl.DataFrame,
    kills: pl.DataFrame,
    round_time_bounds: dict[int, dict[str, int]],
    match_id: str,
) -> dict[int, dict[str, set[int]]]:
    rosters: dict[int, dict[str, set[int]]] = {}

    if not ticks.is_empty() and all(col in ticks.columns for col in ["round_num", "steamid", "side"]):
        tick_rows = []
        for row in ticks.select(["round_num", "steamid", "side", "tick"]).drop_nulls(["round_num", "steamid", "side", "tick"]).to_dicts():
            round_num = int(row["round_num"])
            side = _normalize_side(row.get("side"))
            if side is None or not _valid_steamid(row.get("steamid")):
                continue

            tick = int(row["tick"])
            round_bounds = round_time_bounds.get(round_num, {})
            round_start = round_bounds.get("start")
            round_end = round_bounds.get("end")
            if round_start is not None and tick < round_start:
                continue
            if round_end is not None and tick >= round_end:
                continue

            tick_rows.append(
                {
                    "round_num": round_num,
                    "side": side,
                    "steamid": int(row["steamid"]),
                    "tick": tick,
                }
            )

        if tick_rows:
            roster_frame = (
                pl.DataFrame(tick_rows)
                .group_by(["round_num", "side", "steamid"])
                .agg(
                    [
                        pl.len().alias("tick_rows"),
                        pl.min("tick").alias("first_tick"),
                    ]
                )
                .sort(["round_num", "side", "tick_rows", "first_tick"], descending=[False, False, True, False])
            )

            for row in roster_frame.to_dicts():
                round_num = int(row["round_num"])
                side = str(row["side"])
                rosters.setdefault(round_num, {"CT": set(), "T": set()})
                if len(rosters[round_num][side]) < MAX_PLAYERS_PER_SIDE:
                    rosters[round_num][side].add(int(row["steamid"]))

            overflow = (
                roster_frame.group_by(["round_num", "side"])
                .agg(pl.len().alias("candidate_players"))
                .filter(pl.col("candidate_players") > MAX_PLAYERS_PER_SIDE)
            )
            for row in overflow.to_dicts():
                LOGGER.warning(
                    "Roster candidates exceed %d; truncating to top %d by tick frequency | match_id=%s round_num=%s side=%s candidates=%s",
                    MAX_PLAYERS_PER_SIDE,
                    MAX_PLAYERS_PER_SIDE,
                    match_id,
                    int(row["round_num"]),
                    str(row["side"]),
                    int(row["candidate_players"]),
                )

    if not kills.is_empty():
        kill_cols = [
            "round_num",
            "tick",
            "attacker_steamid",
            "attacker_side",
            "victim_steamid",
            "victim_side",
        ]
        if all(col in kills.columns for col in kill_cols):
            for row in kills.select(kill_cols).drop_nulls(["round_num", "tick"]).to_dicts():
                round_num = int(row["round_num"])
                rosters.setdefault(round_num, {"CT": set(), "T": set()})
                tick = int(row["tick"])
                round_bounds = round_time_bounds.get(round_num, {})
                round_start = round_bounds.get("start")
                round_end = round_bounds.get("end")
                if round_start is not None and tick < round_start:
                    continue
                if round_end is not None and tick >= round_end:
                    continue

                attacker_side = _normalize_side(row.get("attacker_side"))
                attacker = row.get("attacker_steamid")
                if attacker_side is not None and _valid_steamid(attacker) and len(rosters[round_num][attacker_side]) < MAX_PLAYERS_PER_SIDE:
                    rosters[round_num][attacker_side].add(int(attacker))

                victim_side = _normalize_side(row.get("victim_side"))
                victim = row.get("victim_steamid")
                if victim_side is not None and _valid_steamid(victim) and len(rosters[round_num][victim_side]) < MAX_PLAYERS_PER_SIDE:
                    rosters[round_num][victim_side].add(int(victim))

    return rosters


def _round_winners(rounds: pl.DataFrame) -> dict[int, str]:
    if rounds.is_empty() or not all(col in rounds.columns for col in ["round_num", "winner"]):
        return {}

    out: dict[int, str] = {}
    for row in rounds.select(["round_num", "winner"]).drop_nulls(["round_num"]).to_dicts():
        winner = _normalize_side(row.get("winner"))
        if winner is not None:
            out[int(row["round_num"])] = winner
    return out


def _bomb_plant_ticks(rounds: pl.DataFrame, bomb: pl.DataFrame) -> dict[int, int]:
    plant_ticks: dict[int, int] = {}

    if not rounds.is_empty() and all(col in rounds.columns for col in ["round_num", "bomb_plant"]):
        for row in rounds.select(["round_num", "bomb_plant"]).drop_nulls(["round_num", "bomb_plant"]).to_dicts():
            plant_ticks[int(row["round_num"])] = int(row["bomb_plant"])

    if not bomb.is_empty() and all(col in bomb.columns for col in ["round_num", "tick", "event"]):
        planted = (
            bomb.select(["round_num", "tick", "event"])
            .drop_nulls(["round_num", "tick", "event"])
            .with_columns(pl.col("event").cast(pl.Utf8).str.to_lowercase().alias("event"))
            .filter(pl.col("event") == "plant")
            .sort(["round_num", "tick"])
            .group_by("round_num", maintain_order=True)
            .agg(pl.first("tick").alias("plant_tick"))
        )
        for row in planted.to_dicts():
            plant_ticks.setdefault(int(row["round_num"]), int(row["plant_tick"]))

    return plant_ticks


def _demo_tickrate(demo: Any) -> float:
    tickrate = getattr(demo, "tickrate", None)
    try:
        value = float(tickrate)
    except (TypeError, ValueError):
        value = 128.0
    return value if value > 0 else 128.0


def _seconds_remaining_for_snapshot(
    snapshot_tick: int,
    round_num: int,
    tickrate: float,
    round_time_bounds: dict[int, dict[str, int]],
) -> tuple[float, bool]:
    bounds = round_time_bounds.get(round_num, {})
    round_start = bounds.get("start")
    round_end = bounds.get("end")
    if round_end is None:
        return 0.0, False

    effective_tick = snapshot_tick
    if round_start is not None:
        effective_tick = max(snapshot_tick, round_start)

    raw_seconds_remaining = max(0, round_end - effective_tick) / tickrate
    is_time_anomaly = raw_seconds_remaining > MAX_REASONABLE_ROUND_SECONDS or raw_seconds_remaining < 0
    return _clamp_seconds_remaining(raw_seconds_remaining), is_time_anomaly


def _bomb_timer_context(
    snapshot_tick: int,
    plant_tick: int | None,
    tickrate: float,
) -> tuple[float, float]:
    if plant_tick is None or snapshot_tick < plant_tick:
        return 0.0, 0.0

    bomb_time_since_plant = max(0.0, float(snapshot_tick - plant_tick) / tickrate)
    bomb_time_remaining = max(0.0, BOMB_TIMER_SECONDS - bomb_time_since_plant)
    return bomb_time_since_plant, bomb_time_remaining


def _kill_context_select_columns(kills: pl.DataFrame) -> list[pl.Expr | str]:
    columns: list[pl.Expr | str] = [
        "round_num",
        "tick",
        "attacker_side",
        "victim_side",
        "victim_steamid",
    ]
    if "attacker_steamid" in kills.columns:
        columns.append("attacker_steamid")
    else:
        columns.append(pl.lit(None, dtype=pl.UInt64).alias("attacker_steamid"))
    if "attacker_name" in kills.columns:
        columns.append("attacker_name")
    else:
        columns.append(pl.lit(None, dtype=pl.Utf8).alias("attacker_name"))
    if "victim_name" in kills.columns:
        columns.append("victim_name")
    else:
        columns.append(pl.lit(None, dtype=pl.Utf8).alias("victim_name"))
    if "weapon" in kills.columns:
        columns.append("weapon")
    else:
        columns.append(pl.lit(None, dtype=pl.Utf8).alias("weapon"))
    return columns


def _classify_kill_context_type(
    killer_steamid: int | None,
    victim_steamid: int | None,
    weapon: str | None,
    killer_side: str | None,
    victim_side: str | None,
) -> str:
    normalized_weapon = (weapon or "").strip().lower()
    if normalized_weapon == "world":
        return "world_death"
    if killer_steamid is not None and victim_steamid is not None and killer_steamid == victim_steamid:
        return "self_kill"
    if killer_side is not None and victim_side is not None and killer_side == victim_side:
        return "teamkill"
    return "normal_kill"


def build_round_snapshot_rows(demo: Any, match_id: str) -> pl.DataFrame:
    kills = _kills_df(demo)
    rounds = _safe_df(getattr(demo, "rounds", None))
    ticks = _safe_df(getattr(demo, "ticks", None))
    bomb = _safe_df(getattr(demo, "bomb", None))

    required_kill_cols = ["round_num", "tick", "attacker_side", "victim_side"]
    if kills.is_empty() or not all(col in kills.columns for col in required_kill_cols):
        return empty_snapshot_dataset()

    header_info = getattr(demo, "header", {}) or {}
    map_name = ""
    if isinstance(header_info, dict):
        map_name = str(header_info.get("map_name") or "")

    round_time_bounds = _build_round_time_bounds(rounds)
    rosters = _round_rosters(ticks, kills, round_time_bounds=round_time_bounds, match_id=match_id)
    winners = _round_winners(rounds)
    plant_ticks = _bomb_plant_ticks(rounds, bomb)
    tickrate = _demo_tickrate(demo)

    kill_rows = (
        kills.select(_kill_context_select_columns(kills))
        .drop_nulls(["round_num", "tick", "attacker_side", "victim_side", "victim_steamid"])
        .with_columns(
            [
                pl.col("attacker_side").map_elements(_normalize_side, return_dtype=pl.Utf8),
                pl.col("victim_side").map_elements(_normalize_side, return_dtype=pl.Utf8),
                pl.col("attacker_name").cast(pl.Utf8, strict=False),
                pl.col("victim_name").cast(pl.Utf8, strict=False),
                pl.col("weapon").cast(pl.Utf8, strict=False),
            ]
        )
        .drop_nulls(["attacker_side", "victim_side"])
        .sort(["round_num", "tick"])
        .to_dicts()
    )

    opening_side_by_round: dict[int, str] = {}
    for row in kill_rows:
        opening_side_by_round.setdefault(int(row["round_num"]), str(row["attacker_side"]))

    rows: list[dict[str, Any]] = []
    alive_state: dict[int, dict[str, set[int]]] = {}
    kill_index_by_round: dict[int, int] = {}

    for kill in kill_rows:
        round_num = int(kill["round_num"])
        kill_tick = int(kill["tick"])
        attacker_side = str(kill["attacker_side"])
        victim_side = str(kill["victim_side"])
        attacker_steamid = kill.get("attacker_steamid")
        victim_steamid = int(kill["victim_steamid"])
        attacker_name = kill.get("attacker_name")
        victim_name = kill.get("victim_name")
        weapon = kill.get("weapon")
        normalized_attacker_steamid = int(attacker_steamid) if attacker_steamid is not None else None
        kill_context_type = _classify_kill_context_type(
            killer_steamid=normalized_attacker_steamid,
            victim_steamid=victim_steamid,
            weapon=str(weapon) if weapon is not None else None,
            killer_side=attacker_side,
            victim_side=victim_side,
        )
        round_bounds = round_time_bounds.get(round_num, {})
        round_start = round_bounds.get("start")
        round_end = round_bounds.get("end")

        if round_start is not None and kill_tick < round_start:
            LOGGER.warning(
                "Skipping kill before round live start | match_id=%s round_num=%s tick=%s round_start=%s",
                match_id,
                round_num,
                kill_tick,
                round_start,
            )
            continue
        if round_end is not None and kill_tick >= round_end:
            LOGGER.warning(
                "Skipping kill at/after round end | match_id=%s round_num=%s tick=%s round_end=%s",
                match_id,
                round_num,
                kill_tick,
                round_end,
            )
            continue

        if round_num not in alive_state:
            roster = rosters.get(round_num, {"CT": set(), "T": set()})
            alive_state[round_num] = {
                "CT": set(roster.get("CT", set())),
                "T": set(roster.get("T", set())),
            }

        before_ct = len(alive_state[round_num]["CT"])
        before_t = len(alive_state[round_num]["T"])
        after_ct_players = set(alive_state[round_num]["CT"])
        after_t_players = set(alive_state[round_num]["T"])

        if victim_side == "CT":
            after_ct_players.discard(victim_steamid)
        elif victim_side == "T":
            after_t_players.discard(victim_steamid)

        after_ct = len(after_ct_players)
        after_t = len(after_t_players)

        kill_index = kill_index_by_round.get(round_num, 0) + 1
        kill_index_by_round[round_num] = kill_index
        event_id = f"{match_id}:{round_num}:{kill_index}"

        opening_side = opening_side_by_round.get(round_num)
        plant_tick = plant_ticks.get(round_num)
        team_winner = winners.get(round_num)

        snapshots = [
            ("before_kill", max(0, kill_tick - 1), before_ct, before_t),
            ("after_kill", kill_tick, after_ct, after_t),
        ]

        for snapshot_type, snapshot_tick, alive_ct, alive_t in snapshots:
            bomb_planted = plant_tick is not None and plant_tick <= snapshot_tick
            seconds_remaining, is_time_anomaly = _seconds_remaining_for_snapshot(
                snapshot_tick=snapshot_tick,
                round_num=round_num,
                tickrate=tickrate,
                round_time_bounds=round_time_bounds,
            )
            bomb_time_since_plant, bomb_time_remaining = _bomb_timer_context(
                snapshot_tick=snapshot_tick,
                plant_tick=plant_tick,
                tickrate=tickrate,
            )

            for side in ("CT", "T"):
                raw_alive_team = alive_ct if side == "CT" else alive_t
                raw_alive_enemy = alive_t if side == "CT" else alive_ct
                alive_team = _clamp_alive_count(raw_alive_team)
                alive_enemy = _clamp_alive_count(raw_alive_enemy)
                if alive_team != raw_alive_team or alive_enemy != raw_alive_enemy:
                    LOGGER.warning(
                        "Clamped alive counts | match_id=%s round_num=%s tick=%s snapshot_type=%s side=%s raw_alive_team=%s raw_alive_enemy=%s",
                        match_id,
                        round_num,
                        snapshot_tick,
                        snapshot_type,
                        side,
                        raw_alive_team,
                        raw_alive_enemy,
                    )
                rows.append(
                    {
                        "match_id": match_id,
                        "map_name": map_name,
                        "round_num": round_num,
                        "event_id": event_id,
                        "tick": snapshot_tick,
                        "snapshot_type": snapshot_type,
                        "side": side,
                        "event_type": "kill",
                        "killer_steamid": normalized_attacker_steamid,
                        "victim_steamid": victim_steamid,
                        "killer_name": str(attacker_name) if attacker_name is not None else None,
                        "victim_name": str(victim_name) if victim_name is not None else None,
                        "weapon": str(weapon) if weapon is not None else None,
                        "killer_side": attacker_side,
                        "victim_side": victim_side,
                        "kill_context_type": kill_context_type,
                        "alive_team": alive_team,
                        "alive_enemy": alive_enemy,
                        "seconds_remaining": seconds_remaining,
                        "bomb_time_since_plant": bomb_time_since_plant,
                        "bomb_time_remaining": bomb_time_remaining,
                        "is_time_anomaly": is_time_anomaly,
                        "bomb_planted": bomb_planted,
                        "opening_kill_for_team": opening_side == side,
                        "team_won_round": team_winner == side,
                    }
                )

        alive_state[round_num]["CT"] = after_ct_players
        alive_state[round_num]["T"] = after_t_players

    if not rows:
        return empty_snapshot_dataset()

    return (
        pl.DataFrame(rows, schema=SNAPSHOT_SCHEMA, orient="row")
        .select(SNAPSHOT_COLUMNS)
        .sort(["match_id", "round_num", "tick", "snapshot_type", "side"])
    )
