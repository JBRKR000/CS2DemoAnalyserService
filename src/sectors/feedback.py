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
) -> dict[str, Any]:
    return {
        "category": category,
        "severity": severity,
        "title": title,
        "message": message,
        "metric": metric,
        "value": round(value, 2),
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

    # Keep output focused: all negatives first, then at most one rare positive highlight.
    if positive:
        return critical_warning + positive[:1]
    return critical_warning
