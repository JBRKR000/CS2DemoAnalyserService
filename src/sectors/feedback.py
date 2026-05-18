"""Actionable coaching feedback based on local percentile benchmarks."""

from __future__ import annotations

from typing import Any


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
        tips.append(
            _tip(
                category="impact",
                severity="critical",
                title="Too many zero-impact deaths",
                message=(
                    f"{deaths_with_0_damage} deaths came with zero damage dealt before dying. "
                    "Take first contact with utility or teammate timing so you can create impact before going down."
                ),
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

    if trade_kills >= 4:
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

    return tips


def generate_feedback(stats: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark_evals = stats.get("benchmark_evaluations") or {}
    if not isinstance(benchmark_evals, dict):
        return []

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
    return result
