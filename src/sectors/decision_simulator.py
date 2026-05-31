from __future__ import annotations

from typing import Any

_STRONGLY_NEGATIVE_THRESHOLD = -0.20  # -20 pp in raw win_prob_delta units


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    to_dicts = getattr(value, "to_dicts", None)
    if callable(to_dicts):
        rows = to_dicts()
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _safe_round_num(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _index_events_by_round(events: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    index: dict[int, list[dict[str, Any]]] = {}
    for evt in events:
        rn = _safe_round_num(evt.get("round_num"))
        if rn > 0:
            index.setdefault(rn, []).append(evt)
    return index


def _round_death_ml_impact(worst_deaths_by_round: dict[int, list[dict[str, Any]]], round_num: int) -> float | None:
    events = worst_deaths_by_round.get(round_num, [])
    if not events:
        return None
    worst = min(events, key=lambda e: _to_float(e.get("win_prob_delta"), 0.0))
    val = _to_float(worst.get("win_prob_delta"), 0.0)
    return val if val < 0.0 else None


def _is_late_round_death(round_num: int, timeline_by_round: dict[int, list[dict[str, Any]]]) -> bool:
    return any(
        str(e.get("event_type", "")).lower() == "death"
        and str(e.get("round_phase", "")).lower() == "late"
        for e in timeline_by_round.get(round_num, [])
    )


def _build_candidate_set(
    zero_damage: bool,
    untraded: bool,
    late_round: bool,
    strongly_negative: bool,
    allow_wide_peek: bool,
) -> set[str]:
    candidates: set[str] = set()
    if zero_damage:
        candidates.update({"fall_back", "wait_for_trade", "reposition"})
    if untraded:
        candidates.update({"wait_for_trade", "hold_angle", "fall_back"})
    if late_round:
        candidates.update({"play_time", "fall_back"})
    if strongly_negative:
        candidates.update({"fall_back", "wait_for_trade", "play_time"})
    if allow_wide_peek:
        candidates.add("wide_peek")
    return candidates


def _score_alternative(
    label: str,
    zero_damage: bool,
    untraded: bool,
    late_round: bool,
    strongly_negative: bool,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    is_high_risk = zero_damage or untraded or strongly_negative

    if label == "fall_back":
        if zero_damage:
            score += 0.3
        if strongly_negative:
            score += 0.2
        if zero_damage and strongly_negative:
            reasons.append("safer after zero-damage high-cost death")
        elif zero_damage:
            reasons.append("better option when current death created no damage")
        elif strongly_negative:
            reasons.append("avoids repeating a high-cost ML death")
        else:
            reasons.append("reduces isolation risk")
        if "reduces isolation risk" not in reasons:
            reasons.append("reduces isolation risk")

    elif label == "wait_for_trade":
        if untraded:
            score += 0.3
            reasons.append("keeps trade possibility alive")
        if strongly_negative:
            score += 0.15
            reasons.append("avoids repeating a high-cost ML death")
        if not reasons:
            reasons.append("reduces isolation risk")

    elif label == "play_time":
        if late_round:
            score += 0.2
        if strongly_negative:
            score += 0.1
        if late_round or strongly_negative:
            reasons.append("avoids unnecessary late-round risk" if strongly_negative else "safer late-round choice")
        else:
            reasons.append("reduces unnecessary risk")

    elif label == "hold_angle":
        score += 0.15
        reasons.append("reduces isolation risk")

    elif label == "reposition":
        score += 0.1
        reasons.append("better option when current death created no damage")

    elif label == "wide_peek":
        if is_high_risk:
            score -= 0.2
            reasons.append("high risk in this situation")
        else:
            score += 0.1
            reasons.append("viable when position pressure is needed")

    return round(score, 2), reasons


def _score_actual(
    zero_damage: bool,
    untraded: bool,
    ml_death_impact: float | None,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    if zero_damage:
        score -= 0.4
        reasons.append("died without dealing any damage")
    if untraded:
        score -= 0.3
        reasons.append("death went untraded")
    if isinstance(ml_death_impact, (int, float)) and ml_death_impact < 0.0:
        score -= abs(ml_death_impact)
        reasons.append(f"ML impact: {ml_death_impact * 100:.1f} pp")
    return round(score, 2), reasons


def simulate_decisions(
    report_data: dict[str, Any],
    vod_review_priority: list[dict[str, Any]],
    selected_player_impact: dict[str, Any] | None = None,
    player_ml_impact: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    safe_report = _safe_dict(report_data)

    ml = _safe_dict(player_ml_impact) if isinstance(player_ml_impact, dict) else _safe_dict(safe_report.get("ml_impact"))
    timeline_events = _safe_rows(safe_report.get("selected_player_timeline_events"))

    timeline_by_round = _index_events_by_round(timeline_events)
    worst_deaths_by_round = _index_events_by_round(_safe_rows(ml.get("worst_deaths")))
    low_impact_kills_by_round = _index_events_by_round(_safe_rows(ml.get("low_impact_kills")))
    best_kills_by_round = _index_events_by_round(_safe_rows(ml.get("best_kills")))

    results: list[dict[str, Any]] = []

    for entry in vod_review_priority[:5]:
        if len(results) >= 3:
            break

        round_num = _safe_round_num(entry.get("round_num"))
        if round_num <= 0:
            continue

        reasons_raw = entry.get("reasons")
        reasons = reasons_raw if isinstance(reasons_raw, list) else []
        review_type = str(entry.get("review_type") or "").strip().lower()
        entry_ml_impact = entry.get("ml_impact")

        zero_damage = "zero-damage death" in reasons
        untraded = "untraded death" in reasons
        late_round = _is_late_round_death(round_num, timeline_by_round)

        ml_death_impact = _round_death_ml_impact(worst_deaths_by_round, round_num)
        if ml_death_impact is None and isinstance(entry_ml_impact, (int, float)) and entry_ml_impact < 0.0:
            ml_death_impact = entry_ml_impact

        strongly_negative = isinstance(ml_death_impact, (int, float)) and ml_death_impact <= _STRONGLY_NEGATIVE_THRESHOLD

        has_low_impact_kill = round_num in low_impact_kills_by_round
        has_positive_kill = any(
            _to_float(e.get("win_prob_delta"), 0.0) > 0.0
            for e in best_kills_by_round.get(round_num, [])
        )
        is_high_risk = zero_damage or untraded or strongly_negative
        allow_wide_peek = (
            not is_high_risk
            and (has_low_impact_kill or review_type == "strength" or has_positive_kill)
        )

        candidate_set = _build_candidate_set(zero_damage, untraded, late_round, strongly_negative, allow_wide_peek)
        if not candidate_set:
            continue

        actual_score, actual_reasons = _score_actual(zero_damage, untraded, ml_death_impact)
        actual_decision: dict[str, Any] = {
            "label": "actual",
            "score": actual_score,
            "reasons": actual_reasons,
        }
        if isinstance(ml_death_impact, (int, float)):
            actual_decision["ml_impact"] = round(ml_death_impact, 4)

        alternatives: list[dict[str, Any]] = []
        for cand_label in sorted(candidate_set):
            cand_score, cand_reasons = _score_alternative(
                cand_label, zero_damage, untraded, late_round, strongly_negative
            )
            alternatives.append({
                "label": cand_label,
                "score": cand_score,
                "reasons": cand_reasons,
            })

        alternatives.sort(key=lambda x: x["score"], reverse=True)
        alternatives = alternatives[:3]

        results.append({
            "round_num": round_num,
            "side": str(entry.get("side") or "-").strip(),
            "review_type": review_type,
            "original_summary": str(entry.get("summary") or "").strip(),
            "actual_decision": actual_decision,
            "alternatives": alternatives,
        })

    return results
