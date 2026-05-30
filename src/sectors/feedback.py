"""Actionable coaching feedback based on local percentile benchmarks."""

from __future__ import annotations

from typing import Any

import polars as pl


_METRIC_CATEGORY = {
    "adr": "impact",
    "hs_percent": "aim",
    "kast": "survivability",
    "kpr": "impact",
    "opening_duel_win_pct": "game_sense",
    "full_buy_win_rate": "economy",
    "force_win_rate": "economy",
    "clutch_win_rate": "clutch",
}

MAX_TIP_EXAMPLES = 3
_ROUND_PHASE_PRIORITY = {"late": 0, "mid": 1, "early": 2}
_HS_MEANINGFUL_WEAPONS = {
    "ak47",
    "aug",
    "bizon",
    "cz75a",
    "deagle",
    "elite",
    "famas",
    "fiveseven",
    "galilar",
    "glock",
    "hkp2000",
    "m249",
    "m4a1",
    "m4a1_silencer",
    "mac10",
    "mp5sd",
    "mp7",
    "mp9",
    "negev",
    "p250",
    "p90",
    "revolver",
    "sg556",
    "tec9",
    "ump45",
    "usp_silencer",
}
_HS_AWP_FALLBACK_WEAPONS = {"awp"}


def _tip(
    category: str,
    severity: str,
    title: str,
    message: str,
    metric: str,
    value: float,
    percentile: float | None,
    context: str,
    benchmark: float | None = None,
) -> dict[str, Any]:
    return {
        "category": category,
        "severity": severity,
        "title": title,
        "message": message,
        "metric": metric,
        "value": round(value, 2),
        "benchmark": round(benchmark, 2) if benchmark is not None else None,
        "percentile": round(percentile, 2) if percentile is not None else None,
        "context": context,
    }


def _metric_title(metric: str, severity: str) -> str:
    names = {
        "adr": "Round Damage Impact",
        "hs_percent": "Headshot Consistency",
        "kast": "Round Survival/Trade Value",
        "kpr": "Kill Conversion Pace",
        "opening_duel_win_pct": "Opening Duel Efficiency",
        "full_buy_win_rate": "Full-Buy Conversion",
        "force_win_rate": "Force-Buy Conversion",
        "clutch_win_rate": "Clutch Conversion",
    }
    base = names.get(metric, metric)
    if severity == "critical":
        return f"{base} Is Well Below Benchmark"
    if severity == "warning":
        return f"{base} Needs Improvement"
    if severity == "good":
        return f"{base} Is A Strength"
    return base


def _message(metric: str, value: float, percentile: float, context: str, severity: str) -> str:
    prefix = f"{metric} {value:.2f} is around the {percentile:.1f} percentile in your {context} pool."
    if metric == "adr":
        if severity in {"critical", "warning"}:
            return f"{prefix} Focus on better spacing for trade damage and pre-aim common off-angles before committing to swings."
        return f"{prefix} Keep taking first-contact fights only with teammate support so this impact converts into round wins."
    if metric == "hs_percent":
        if severity in {"critical", "warning"}:
            return f"{prefix} Your crosshair likely drops between fights; rehearse head-height clears and delay crouch-sprays until after first bullet contact."
        return f"{prefix} Keep prioritizing controlled bursts at mid range to preserve this headshot discipline."
    if metric == "kast":
        if severity in {"critical", "warning"}:
            return f"{prefix} You are missing too many rounds without impact; stay closer to trade range and avoid isolated dry re-peeks."
        return f"{prefix} Your round-to-round presence is strong; keep syncing timing with teammate contact."
    if metric == "kpr":
        if severity in {"critical", "warning"}:
            return f"{prefix} Look for earlier map-control fights with utility support instead of late isolated duels."
        return f"{prefix} Good frag pace; keep balancing aggression with safe repositioning after first contact."
    if metric == "full_buy_win_rate":
        if severity in {"critical", "warning"}:
            return f"{prefix} Tighten full-buy protocols and preserve late-round utility before final executes."
        return f"{prefix} Strong gun-round conversion; protect it by avoiding unnecessary re-force chains."
    if metric == "force_win_rate":
        if severity in {"critical", "warning"}:
            return f"{prefix} Use compact, utility-layered force plans instead of spread defaults."
        return f"{prefix} Your force rounds are a weapon; focus on weapon recovery after wins."
    if metric == "clutch_win_rate":
        if severity in {"critical", "warning"}:
            return f"{prefix} In 1vX, isolate one duel at a time and reposition after each frag to deny quick refrags."
        return f"{prefix} Clutch decision-making is strong; keep abusing timing pressure and off-angle repositioning."
    return f"{prefix} Keep refining this area with focused VOD review."


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_frame(value: Any) -> pl.DataFrame:
    return value if isinstance(value, pl.DataFrame) else pl.DataFrame()


def _safe_rows(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _safe_text(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _safe_round_num(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return bool(value)


def _normalize_side(value: Any) -> str:
    side = str(value or "").strip().upper()
    return side if side else "-"


def _format_damage(value: Any) -> str:
    damage = _to_float(value, 0.0)
    if float(damage).is_integer():
        return str(int(damage))
    return f"{damage:.1f}"


def _format_pp(value: Any) -> str | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value) * 100.0:+.1f} pp"
    return None


def _death_phase_priority(value: Any) -> int:
    return _ROUND_PHASE_PRIORITY.get(str(value or "").strip().lower(), len(_ROUND_PHASE_PRIORITY))


def _ml_deaths_by_round(player_ml_impact: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(player_ml_impact, dict):
        return {}

    rows = _safe_rows(player_ml_impact.get("worst_deaths"))
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        round_num = _safe_round_num(row.get("round_num"))
        if round_num > 0 and round_num not in result:
            result[round_num] = row
    return result


def _death_events(stats: dict[str, Any]) -> list[dict[str, Any]]:
    timeline_events = _safe_frame(stats.get("selected_player_timeline_events"))
    if timeline_events.is_empty() or "event_type" not in timeline_events.columns:
        return []
    death_rows = timeline_events.filter(pl.col("event_type") == "death")
    if death_rows.is_empty():
        return []
    sort_columns = [column for column in ["round_num", "tick"] if column in death_rows.columns]
    if sort_columns:
        death_rows = death_rows.sort(sort_columns)
    return death_rows.to_dicts()


def _kill_events(stats: dict[str, Any]) -> list[dict[str, Any]]:
    timeline_events = _safe_frame(stats.get("selected_player_timeline_events"))
    if timeline_events.is_empty() or "event_type" not in timeline_events.columns:
        return []
    kill_rows = timeline_events.filter(pl.col("event_type") == "kill")
    if kill_rows.is_empty():
        return []
    sort_columns = [column for column in ["round_num", "tick"] if column in kill_rows.columns]
    if sort_columns:
        kill_rows = kill_rows.sort(sort_columns)
    return kill_rows.to_dicts()


def _failed_clutch_rounds(stats: dict[str, Any]) -> list[dict[str, Any]]:
    clutch_rounds = _safe_frame(stats.get("selected_player_clutch_rounds"))
    if clutch_rounds.is_empty() or "won" not in clutch_rounds.columns:
        return []
    failed_rows = clutch_rounds.filter(~pl.col("won").fill_null(False))
    if failed_rows.is_empty():
        return []
    if "round_num" in failed_rows.columns:
        failed_rows = failed_rows.sort("round_num")
    return failed_rows.to_dicts()


def _build_untraded_examples(stats: dict[str, Any]) -> list[str]:
    ml_deaths = _ml_deaths_by_round(stats.get("player_ml_impact"))
    candidates = [row for row in _death_events(stats) if not _safe_bool(row.get("is_traded_death"))]
    if not candidates:
        return []

    if ml_deaths:
        candidates.sort(
            key=lambda row: (
                0 if _safe_round_num(row.get("round_num")) in ml_deaths else 1,
                _to_float(ml_deaths.get(_safe_round_num(row.get("round_num")), {}).get("win_prob_delta"), 0.0),
                _death_phase_priority(row.get("round_phase")),
                _safe_round_num(row.get("round_num")),
            )
        )
    else:
        candidates.sort(
            key=lambda row: (
                _death_phase_priority(row.get("round_phase")),
                _safe_round_num(row.get("round_num")),
            )
        )

    examples: list[str] = []
    for row in candidates[:MAX_TIP_EXAMPLES]:
        round_num = _safe_round_num(row.get("round_num"))
        ml_row = ml_deaths.get(round_num, {})
        parts = [
            f"Round {round_num}",
            _normalize_side(row.get("side")),
            f"died to {_safe_text(row.get('target_name'), 'Unknown killer')} with {_safe_text(row.get('weapon'), 'unknown weapon')}",
        ]
        phase = str(row.get("round_phase") or "").strip().lower()
        if phase in {"early", "mid", "late"}:
            parts.append(phase)
        ml_impact = _format_pp(ml_row.get("win_prob_delta"))
        if ml_impact is not None:
            parts.append(ml_impact)
        examples.append(" | ".join(parts))
    return examples


def _build_low_impact_examples(stats: dict[str, Any], max_damage: float) -> list[str]:
    ml_deaths = _ml_deaths_by_round(stats.get("player_ml_impact"))
    candidates = [
        row
        for row in _death_events(stats)
        if _to_float(row.get("damage_before_death"), 0.0) <= max_damage
    ]
    if not candidates:
        return []

    candidates.sort(
        key=lambda row: (
            0 if _safe_round_num(row.get("round_num")) in ml_deaths else 1,
            _to_float(ml_deaths.get(_safe_round_num(row.get("round_num")), {}).get("win_prob_delta"), 0.0),
            _to_float(row.get("damage_before_death"), 0.0),
            _safe_round_num(row.get("round_num")),
        )
    )

    examples: list[str] = []
    for row in candidates[:MAX_TIP_EXAMPLES]:
        round_num = _safe_round_num(row.get("round_num"))
        ml_row = ml_deaths.get(round_num, {})
        parts = [
            f"Round {round_num}",
            _normalize_side(row.get("side")),
            f"{_format_damage(row.get('damage_before_death'))} dmg before death",
            f"died to {_safe_text(row.get('target_name'), 'Unknown killer')} with {_safe_text(row.get('weapon'), 'unknown weapon')}",
        ]
        ml_impact = _format_pp(ml_row.get("win_prob_delta"))
        if ml_impact is not None:
            parts.append(ml_impact)
        examples.append(" | ".join(parts))
    return examples


def _build_clutch_examples(stats: dict[str, Any]) -> list[str]:
    ml_deaths = _ml_deaths_by_round(stats.get("player_ml_impact"))
    deaths_by_round = {_safe_round_num(row.get("round_num")): row for row in _death_events(stats)}
    examples: list[str] = []

    for clutch_row in _failed_clutch_rounds(stats)[:MAX_TIP_EXAMPLES]:
        round_num = _safe_round_num(clutch_row.get("round_num"))
        death_row = deaths_by_round.get(round_num, {})
        ml_row = ml_deaths.get(round_num, {})
        parts = [
            f"Round {round_num}",
            _normalize_side(clutch_row.get("side")),
            _safe_text(clutch_row.get("clutch_type"), "1vX"),
            "lost",
        ]
        if death_row:
            parts.append(
                f"died to {_safe_text(death_row.get('target_name'), 'Unknown killer')} with {_safe_text(death_row.get('weapon'), 'unknown weapon')}"
            )
        ml_impact = _format_pp(ml_row.get("win_prob_delta"))
        if ml_impact is not None:
            parts.append(ml_impact)
        examples.append(" | ".join(parts))
    return examples


def _normalize_weapon(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_rifle_weapon(weapon: str) -> bool:
    return weapon in {
        "ak47",
        "aug",
        "famas",
        "galilar",
        "m4a1",
        "m4a1_silencer",
        "sg556",
    }


def _ml_kills_by_key(player_ml_impact: Any) -> dict[tuple[int, str, str, str], dict[str, Any]]:
    if not isinstance(player_ml_impact, dict):
        return {}

    rows = _safe_rows(player_ml_impact.get("kill_events"))
    if not rows:
        rows = _safe_rows(player_ml_impact.get("best_kills"))

    result: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            _safe_round_num(row.get("round_num")),
            _normalize_side(row.get("side")),
            _safe_text(row.get("victim_name"), "Unknown victim"),
            _normalize_weapon(row.get("weapon")),
        )
        existing = result.get(key)
        if existing is None or _to_float(row.get("win_prob_delta"), 0.0) > _to_float(existing.get("win_prob_delta"), 0.0):
            result[key] = row
    return result


def _build_hs_examples(stats: dict[str, Any]) -> list[str]:
    ml_kills = _ml_kills_by_key(stats.get("player_ml_impact"))
    non_hs_kills = [row for row in _kill_events(stats) if not _safe_bool(row.get("is_headshot"))]
    if not non_hs_kills:
        return []

    primary_candidates = [
        row for row in non_hs_kills if _normalize_weapon(row.get("weapon")) in _HS_MEANINGFUL_WEAPONS
    ]
    fallback_candidates = [
        row for row in non_hs_kills if _normalize_weapon(row.get("weapon")) in _HS_AWP_FALLBACK_WEAPONS
    ]
    candidates = primary_candidates if primary_candidates else fallback_candidates
    if not candidates:
        return []

    candidates.sort(
        key=lambda row: (
            0
            if (
                _safe_round_num(row.get("round_num")),
                _normalize_side(row.get("side")),
                _safe_text(row.get("target_name"), "Unknown victim"),
                _normalize_weapon(row.get("weapon")),
            )
            in ml_kills
            else 1,
            -_to_float(
                ml_kills.get(
                    (
                        _safe_round_num(row.get("round_num")),
                        _normalize_side(row.get("side")),
                        _safe_text(row.get("target_name"), "Unknown victim"),
                        _normalize_weapon(row.get("weapon")),
                    ),
                    {},
                ).get("win_prob_delta"),
                0.0,
            ),
            0 if _is_rifle_weapon(_normalize_weapon(row.get("weapon"))) else 1,
            _safe_round_num(row.get("round_num")),
        )
    )

    examples: list[str] = []
    for row in candidates[:MAX_TIP_EXAMPLES]:
        key = (
            _safe_round_num(row.get("round_num")),
            _normalize_side(row.get("side")),
            _safe_text(row.get("target_name"), "Unknown victim"),
            _normalize_weapon(row.get("weapon")),
        )
        ml_row = ml_kills.get(key, {})
        parts = [
            f"Round {_safe_round_num(row.get('round_num'))}",
            _normalize_side(row.get("side")),
            f"killed {_safe_text(row.get('target_name'), 'Unknown victim')} with {_safe_text(row.get('weapon'), 'unknown weapon')}",
            "non-HS",
        ]
        ml_impact = _format_pp(ml_row.get("win_prob_delta"))
        if ml_impact is not None:
            parts.append(ml_impact)
        examples.append(" | ".join(parts))
    return examples


def _build_positive_examples(metric: str, stats: dict[str, Any]) -> list[str]:
    player_ml_impact = stats.get("player_ml_impact")
    if not isinstance(player_ml_impact, dict):
        return []

    candidates = _safe_rows(player_ml_impact.get("best_kills"))
    if metric == "opening_duel_win_pct":
        opening_candidates = [row for row in candidates if _safe_bool(row.get("is_opening"))]
        if opening_candidates:
            candidates = opening_candidates

    examples: list[str] = []
    for row in candidates[:MAX_TIP_EXAMPLES]:
        parts = [
            f"Round {_safe_round_num(row.get('round_num'))}",
            _normalize_side(row.get("side")),
            f"killed {_safe_text(row.get('victim_name'), 'Unknown victim')} with {_safe_text(row.get('weapon'), 'unknown weapon')}",
        ]
        ml_impact = _format_pp(row.get("win_prob_delta"))
        if ml_impact is not None:
            parts.append(ml_impact)
        examples.append(" | ".join(parts))
    return examples


def build_tip_evidence(tip: dict[str, Any], stats: dict[str, Any]) -> list[str]:
    metric = str(tip.get("metric") or "")
    severity = str(tip.get("severity") or "")

    if metric == "untraded_death_rate":
        return _build_untraded_examples(stats)
    if metric == "hs_percent" and severity in {"critical", "warning"}:
        return _build_hs_examples(stats)
    if metric == "deaths_with_0_damage":
        return _build_low_impact_examples(stats, max_damage=0.0)
    if metric == "deaths_under_40_damage":
        return _build_low_impact_examples(stats, max_damage=39.999)
    if metric == "clutch_win_rate" and severity in {"critical", "warning"}:
        return _build_clutch_examples(stats)
    if severity == "good" and metric in {"adr", "kpr", "opening_duel_win_pct"}:
        return _build_positive_examples(metric, stats)
    return []


def _attach_tip_evidence(tips: list[dict[str, Any]], stats: dict[str, Any]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for tip in tips:
        examples = build_tip_evidence(tip, stats)
        if examples:
            enriched.append({**tip, "evidence": examples[:MAX_TIP_EXAMPLES]})
        else:
            enriched.append(tip)
    return enriched


def _impact_tips(stats: dict[str, Any]) -> list[dict[str, Any]]:
    selected_impact = stats.get("selected_player_impact")
    if not isinstance(selected_impact, dict) or not selected_impact:
        return []
    impact_row = selected_impact

    deaths = int(_to_float(impact_row.get("deaths"), 0.0))
    opening_kills = int(_to_float(impact_row.get("opening_kills"), 0.0))
    untraded_deaths = int(_to_float(impact_row.get("untraded_deaths"), 0.0))
    opening_duels = int(_to_float(impact_row.get("opening_duels"), 0.0))
    opening_duel_win_pct = _to_float(impact_row.get("opening_duel_win_pct"), 0.0)
    untraded_death_rate = _to_float(impact_row.get("untraded_death_rate"), 0.0)
    trade_kills = int(_to_float(impact_row.get("trade_kills"), 0.0))
    deaths_with_0_damage = int(_to_float(impact_row.get("deaths_with_0_damage"), 0.0))
    deaths_under_40_damage = int(_to_float(impact_row.get("deaths_under_40_damage"), 0.0))
    early_deaths = int(_to_float(impact_row.get("early_deaths"), 0.0))

    tips: list[dict[str, Any]] = []

    if deaths >= 5 and untraded_death_rate >= 60.0:
        severity = "critical" if untraded_death_rate >= 75.0 else "warning"
        tips.append(
            _tip(
                category="positioning",
                severity=severity,
                title="Too many untraded deaths",
                message=(
                    f"{untraded_deaths}/{deaths} deaths were untraded ({untraded_death_rate:.1f}%). "
                    "You often die outside trade range or take isolated duels. Play tighter off teammates and avoid solo repeeks without support."
                ),
                metric="untraded_death_rate",
                value=untraded_death_rate,
                percentile=None,
                benchmark=60.0,
                context="timeline",
            )
        )

    if deaths_with_0_damage >= 3:
        zero_impact_message = (
            f"{deaths_with_0_damage} deaths came with zero damage dealt before dying. "
            "Take first contact with utility or teammate timing so you can create impact before going down."
        )
        if deaths_under_40_damage >= 4:
            zero_impact_message = (
                f"{deaths_with_0_damage} deaths came with zero damage, and "
                f"{deaths_under_40_damage} deaths were under 40 damage before dying. "
                "Take first contact with utility or teammate timing so you can create impact before going down."
            )
        tips.append(
            _tip(
                category="impact",
                severity="critical",
                title="Too many zero-impact deaths",
                message=zero_impact_message,
                metric="deaths_with_0_damage",
                value=float(deaths_with_0_damage),
                percentile=None,
                benchmark=3.0,
                context="timeline",
            )
        )
    elif deaths_under_40_damage >= 4:
        tips.append(
            _tip(
                category="impact",
                severity="warning",
                title="Too many low-impact deaths",
                message=(
                    f"{deaths_under_40_damage} deaths had under 40 damage before death. "
                    "You frequently die before creating useful impact; prioritize safer first bullets and disengage paths."
                ),
                metric="deaths_under_40_damage",
                value=float(deaths_under_40_damage),
                percentile=None,
                benchmark=4.0,
                context="timeline",
            )
        )

    if early_deaths >= 4:
        tips.append(
            _tip(
                category="game_sense",
                severity="warning",
                title="Too many early deaths",
                message=(
                    f"{early_deaths} early deaths are putting your team into early round disadvantage. "
                    "Slow down early map fights and wait for utility/timing support."
                ),
                metric="early_deaths",
                value=float(early_deaths),
                percentile=None,
                benchmark=4.0,
                context="timeline",
            )
        )

    if opening_duels >= 4 and opening_duel_win_pct < 40.0:
        severity = "critical" if opening_duel_win_pct < 30.0 else "warning"
        tips.append(
            _tip(
                category="entry",
                severity=severity,
                title="Poor opening duel conversion",
                message=(
                    f"Opening duels won: {opening_kills}/{opening_duels} ({opening_duel_win_pct:.1f}%). "
                    "First-contact conversion is below target; use flash/swing timing and take fewer raw openers."
                ),
                metric="opening_duel_win_pct",
                value=opening_duel_win_pct,
                percentile=None,
                benchmark=40.0,
                context="timeline",
            )
        )

    if trade_kills >= 4 and untraded_death_rate < 75.0:
        tips.append(
            _tip(
                category="teamplay",
                severity="good",
                title="Strong trade conversion",
                message=f"You converted {trade_kills} trade kills. Keep this spacing discipline to stabilize difficult rounds.",
                metric="trade_kills",
                value=float(trade_kills),
                percentile=None,
                benchmark=4.0,
                context="timeline",
            )
        )
    elif trade_kills >= 4 and untraded_death_rate >= 75.0:
        tips.append(
            _tip(
                category="teamplay",
                severity="warning",
                title="Trade conversion is situational",
                message=(
                    "You convert trades well when near teammates, but your own deaths are often isolated."
                ),
                metric="trade_kills",
                value=float(trade_kills),
                percentile=None,
                benchmark=4.0,
                context="timeline",
            )
        )

    return tips


def generate_feedback(stats: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark_evals = stats.get("benchmark_evaluations") or {}
    if not isinstance(benchmark_evals, dict):
        benchmark_evals = {}

    critical_warning: list[dict[str, Any]] = []
    positive: list[dict[str, Any]] = []

    for metric, evaluation in benchmark_evals.items():
        if not isinstance(evaluation, dict):
            continue
        # Feedback must be driven only by benchmark evaluation outcomes,
        # never by raw metric values from the match payload.
        rating = str(evaluation.get("rating", "unknown"))
        if rating == "unknown":
            continue
        if evaluation.get("reason") is not None:
            continue

        percentile = evaluation.get("percentile")
        value = evaluation.get("value")
        context = str(evaluation.get("context", "global"))
        if percentile is None or value is None:
            continue

        category = _METRIC_CATEGORY.get(metric, "game_sense")
        if rating == "critical":
            severity = "critical"
        elif rating == "warning":
            severity = "warning"
        elif rating in {"good", "excellent"}:
            severity = "good"
        else:
            continue

        tip = _tip(
            category=category,
            severity=severity,
            title=_metric_title(metric, severity),
            message=_message(metric, float(value), float(percentile), context, severity),
            metric=metric,
            value=float(value),
            percentile=float(percentile),
            context=context,
        )

        if severity in {"critical", "warning"}:
            critical_warning.append(tip)
        else:
            positive.append(tip)

    critical_warning = sorted(
        critical_warning,
        key=lambda t: (0 if t["severity"] == "critical" else 1, float(t.get("percentile", 100.0))),
    )
    positive = sorted(positive, key=lambda t: float(t.get("percentile", 0.0)), reverse=True)

    impact = _impact_tips(stats)
    impact_negative = [t for t in impact if t.get("severity") in {"critical", "warning"}]
    impact_positive = [t for t in impact if t.get("severity") == "good"]

    combined_negative = critical_warning + impact_negative
    combined_negative = sorted(
        combined_negative,
        key=lambda t: (0 if t["severity"] == "critical" else 1),
    )

    # Avoid noisy duplicates for the same metric.
    seen_metrics: set[str] = set()
    deduped_negative: list[dict[str, Any]] = []
    for tip in combined_negative:
        metric = str(tip.get("metric", ""))
        if metric in seen_metrics:
            continue
        seen_metrics.add(metric)
        deduped_negative.append(tip)

    positive_all = positive + impact_positive
    deduped_positive: list[dict[str, Any]] = []
    for tip in positive_all:
        metric = str(tip.get("metric", ""))
        if metric in seen_metrics:
            continue
        deduped_positive.append(tip)
        seen_metrics.add(metric)

    max_tips = 6
    result = deduped_negative[:max_tips]
    remaining = max_tips - len(result)
    if remaining > 0 and deduped_positive:
        # Keep positives rare and focused.
        result.extend(deduped_positive[:1])
    return _attach_tip_evidence(result, stats)
