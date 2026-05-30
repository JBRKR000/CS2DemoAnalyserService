from __future__ import annotations

from typing import Any

import polars as pl


NORMAL_KILL_CONTEXT = "normal_kill"
EXCLUDED_CONTEXTS = ["teamkill", "world_death"]
LOW_IMPACT_THRESHOLD = 0.01


def _steamid_text(steamid: int | str) -> str:
    return str(steamid).strip()


def _ensure_columns(frame: pl.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"ML event impact is missing required columns: {missing_columns}")


def _filtered_by_steamid(frame: pl.DataFrame, column: str, selected_steamid: str) -> pl.DataFrame:
    return frame.filter(pl.col(column).cast(pl.Utf8) == selected_steamid)


def _events_to_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return frame.to_dicts()


def _sum_or_zero(frame: pl.DataFrame, column: str) -> float:
    if frame.is_empty():
        return 0.0
    value = frame.select(pl.col(column).sum()).item()
    return float(value or 0.0)


def _mean_or_none(frame: pl.DataFrame, column: str) -> float | None:
    if frame.is_empty():
        return None
    value = frame.select(pl.col(column).mean()).item()
    return None if value is None else float(value)


def _context_counts(frame: pl.DataFrame, selected_steamid: str) -> dict[str, int]:
    player_rows = frame.filter(
        (pl.col("killer_steamid").cast(pl.Utf8) == selected_steamid)
        | (pl.col("victim_steamid").cast(pl.Utf8) == selected_steamid)
    )
    counts = (
        player_rows.filter(pl.col("kill_context_type").is_in(EXCLUDED_CONTEXTS))
        .group_by("kill_context_type")
        .agg(pl.col("event_id").n_unique().alias("events"))
        .to_dicts()
    )
    by_context = {context: 0 for context in EXCLUDED_CONTEXTS}
    by_context.update({str(row["kill_context_type"]): int(row["events"]) for row in counts})
    return by_context


def build_player_ml_impact_summary(
    ml_event_impact: pl.DataFrame,
    selected_steamid: int | str,
    top_n: int = 5,
) -> dict[str, Any]:
    if top_n <= 0:
        raise ValueError("top_n must be > 0.")

    _ensure_columns(
        ml_event_impact,
        [
            "event_id",
            "kill_context_type",
            "killer_steamid",
            "victim_steamid",
            "killer_side",
            "victim_side",
            "side",
            "win_prob_delta",
        ],
    )

    selected = _steamid_text(selected_steamid)
    normal_impact = ml_event_impact.filter(pl.col("kill_context_type") == NORMAL_KILL_CONTEXT)

    positive_actions = _filtered_by_steamid(normal_impact, "killer_steamid", selected).filter(
        pl.col("side") == pl.col("killer_side")
    )
    negative_actions = _filtered_by_steamid(normal_impact, "victim_steamid", selected).filter(
        pl.col("side") == pl.col("victim_side")
    )

    total_kill_impact = _sum_or_zero(positive_actions, "win_prob_delta")
    total_death_impact = _sum_or_zero(negative_actions, "win_prob_delta")

    best_kills = positive_actions.sort(
        ["win_prob_delta", "match_id", "round_num", "tick_after"],
        descending=[True, False, False, False],
    ).head(top_n)
    low_impact_kills = positive_actions.filter(pl.col("win_prob_delta") <= LOW_IMPACT_THRESHOLD).sort(
        ["win_prob_delta", "match_id", "round_num", "tick_after"],
        descending=[False, False, False, False],
    ).head(top_n)
    worst_deaths = negative_actions.sort(
        ["win_prob_delta", "match_id", "round_num", "tick_after"],
        descending=[False, False, False, False],
    ).head(top_n)

    return {
        "selected_steamid": selected,
        "kill_count": positive_actions.height,
        "death_count": negative_actions.height,
        "total_kill_impact": total_kill_impact,
        "avg_kill_impact": _mean_or_none(positive_actions, "win_prob_delta"),
        "total_death_impact": total_death_impact,
        "avg_death_impact": _mean_or_none(negative_actions, "win_prob_delta"),
        "net_ml_impact": total_kill_impact + total_death_impact,
        "kill_events": _events_to_rows(positive_actions),
        "best_kills": _events_to_rows(best_kills),
        "low_impact_kills": _events_to_rows(low_impact_kills),
        "worst_deaths": _events_to_rows(worst_deaths),
        "excluded_context_counts": _context_counts(ml_event_impact, selected),
    }


def _format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:+.1f} pp"


def _safe_text(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _format_event(row: dict[str, Any]) -> str:
    round_num = int(row.get("round_num") or 0)
    side = _safe_text(row.get("side"))
    killer = _safe_text(row.get("killer_name"), "Unknown killer")
    victim = _safe_text(row.get("victim_name"), "Unknown victim")
    weapon = _safe_text(row.get("weapon"), "unknown weapon")
    delta = _format_delta(row.get("win_prob_delta"))
    return f"Round {round_num} | {side} | {killer} killed {victim} with {weapon} | {delta}"


def _format_event_list(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [f"{title}:"]
    if not rows:
        lines.append("- none")
        return lines
    lines.extend(f"- {_format_event(row)}" for row in rows)
    return lines


def format_player_ml_impact_summary(summary: dict[str, Any]) -> str:
    excluded_counts = summary.get("excluded_context_counts") or {}
    lines = [
        f"Player ML impact summary for {summary.get('selected_steamid')}",
        f"Kills: {summary.get('kill_count', 0)} | Deaths: {summary.get('death_count', 0)}",
        (
            "Kill impact: "
            f"total {_format_delta(summary.get('total_kill_impact'))}, "
            f"avg {_format_delta(summary.get('avg_kill_impact'))}"
        ),
        (
            "Death impact: "
            f"total {_format_delta(summary.get('total_death_impact'))}, "
            f"avg {_format_delta(summary.get('avg_death_impact'))}"
        ),
        f"Net ML impact: {_format_delta(summary.get('net_ml_impact'))}",
        (
            "Excluded contexts: "
            f"teamkill={int(excluded_counts.get('teamkill', 0))}, "
            f"world_death={int(excluded_counts.get('world_death', 0))}"
        ),
        "",
        *_format_event_list("Best kills", summary.get("best_kills") or []),
        "",
        *_format_event_list("Low-impact kills", summary.get("low_impact_kills") or []),
        "",
        *_format_event_list("Worst deaths", summary.get("worst_deaths") or []),
    ]
    return "\n".join(lines)
