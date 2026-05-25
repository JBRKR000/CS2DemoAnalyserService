from __future__ import annotations

from typing import Any


def _safe_name(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text or text.lower() == "unknown":
        return fallback
    return text


def format_probability(prob: float) -> str:
    return f"{float(prob) * 100.0:.1f}%"


def format_delta(delta: float) -> str:
    return f"{float(delta) * 100.0:+.1f} pp"


def format_alive_state(alive_team: int, alive_enemy: int) -> str:
    return f"{int(alive_team)}v{int(alive_enemy)}"


def format_kill_description(event: dict[str, Any]) -> str:
    kill_context_type = str(event.get("kill_context_type") or "normal_kill").strip().lower()
    killer_name = _safe_name(event.get("killer_name"), "Unknown player")
    victim_name = _safe_name(event.get("victim_name"), "Unknown player")
    weapon = _safe_name(event.get("weapon"), "unknown weapon")

    if kill_context_type == "world_death":
        return f"{victim_name} died to world damage"
    if kill_context_type == "self_kill":
        return f"{victim_name} killed themselves"
    if kill_context_type == "teamkill":
        return f"{killer_name} teamkilled {victim_name} with {weapon}"
    return f"{killer_name} killed {victim_name} with {weapon}"


def format_ml_impact_event(event: dict[str, Any]) -> str:
    round_num = int(event.get("round_num") or 0)
    side = _safe_name(event.get("side"), "unknown")
    description = format_kill_description(event)
    alive_before = format_alive_state(
        event.get("alive_team_before") or 0,
        event.get("alive_enemy_before") or 0,
    )
    alive_after = format_alive_state(
        event.get("alive_team_after") or 0,
        event.get("alive_enemy_after") or 0,
    )
    prob_before = format_probability(event.get("win_prob_before") or 0.0)
    prob_after = format_probability(event.get("win_prob_after") or 0.0)
    delta = format_delta(event.get("win_prob_delta") or 0.0)
    return (
        f"Round {round_num} | {side} | {description} | "
        f"{alive_before} -> {alive_after} | {prob_before} -> {prob_after} ({delta})"
    )
