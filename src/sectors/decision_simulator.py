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


def _apply_death_risk_adjustments(
    score: float,
    label: str,
    death_risk_label: str | None,
    death_risk_context: dict[str, Any] | None,
) -> float:
    """Apply ML death-risk-based boosts/penalties to an alternative score.

    Rules applied (external):
    - death_risk_label boosts: critical/high/medium -> specific boosts
    - death_risk_context.nearest_teammate_distance > 500: wait_for_trade -0.10, fall_back +0.10
    - death_risk_context.player_hp <= 30: fall_back +0.20, wide_peek -0.20, wait_for_trade +0.10
    """
    drl = (str(death_risk_label).strip().lower() if death_risk_label else "")
    ctx = death_risk_context if isinstance(death_risk_context, dict) else {}

    # label-based boosts
    if drl == "critical":
        if label == "fall_back":
            score += 0.35
        elif label == "wait_for_trade":
            score += 0.30
        elif label == "hold_angle":
            score += 0.10
    elif drl == "high":
        if label == "fall_back":
            score += 0.25
        elif label == "wait_for_trade":
            score += 0.20
        elif label == "hold_angle":
            score += 0.10
    elif drl == "medium":
        if label == "fall_back":
            score += 0.10
        elif label == "wait_for_trade":
            score += 0.10

    # context-based adjustments
    try:
        nearest_teammate = float(ctx.get("nearest_teammate_distance") or 0)
    except (TypeError, ValueError):
        nearest_teammate = 0.0
    try:
        player_hp = float(ctx.get("player_hp") or 9999)
    except (TypeError, ValueError):
        player_hp = 9999.0

    if nearest_teammate > 500:
        if label == "wait_for_trade":
            score -= 0.10
        elif label == "fall_back":
            score += 0.10

    if player_hp <= 30:
        if label == "fall_back":
            score += 0.20
        elif label == "wide_peek":
            score -= 0.20
        elif label == "wait_for_trade":
            score += 0.10

    return round(score, 2)


def build_death_risk_explanation(death_risk_context: dict[str, Any] | None, death_risk_label: str | None) -> str:
    """Build a short explanation for why death risk is elevated.

    Returns a comma-separated lowercase string of reasons, or a default sentence.
    """
    ctx = death_risk_context if isinstance(death_risk_context, dict) else {}
    reasons: list[str] = []
    try:
        player_hp = float(ctx.get("player_hp") if ctx.get("player_hp") is not None else 9999)
    except (TypeError, ValueError):
        player_hp = 9999.0
    try:
        nearest_enemy = float(ctx.get("nearest_enemy_distance") or 999999)
    except (TypeError, ValueError):
        nearest_enemy = 999999.0
    try:
        nearest_teammate = float(ctx.get("nearest_teammate_distance") or 999999)
    except (TypeError, ValueError):
        nearest_teammate = 999999.0

    if player_hp <= 30:
        reasons.append("low HP")
    if nearest_enemy <= 700:
        reasons.append("enemy close")
    if nearest_teammate > 500:
        reasons.append("teammate far from trade")
    if nearest_teammate <= 250:
        reasons.append("teammate nearby but duel still high-risk")
    if str(death_risk_label).strip().lower() == "critical":
        reasons.append("model marked this as top-tier risk")

    if not reasons:
        return "model detected elevated danger from round state"
    return ", ".join(reasons)


def _safe_name(row: Any, *keys: str, default: str = "Unknown") -> str:
    if not isinstance(row, dict):
        return default
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default


def _build_death_summary_from_row(row: dict[str, Any]) -> str:
    killer = _safe_name(row, "target_name", "killer_name", default="Unknown killer")
    weapon = _safe_name(row, "weapon", default="unknown weapon")
    try:
        damage = float(row.get("damage_before_death") or 0.0)
    except (TypeError, ValueError):
        damage = 0.0
    if damage <= 0.0:
        damage_text = "after creating no damage"
    elif damage < 40.0:
        damage_text = f"after only {_safe_round_num(damage)} damage"
    else:
        damage_text = f"after {_safe_round_num(damage)} damage"
    return f"died to {killer} with {weapon} {damage_text}."


def build_decision_actual_text(candidate: dict[str, Any]) -> str:
    """Build a succinct 'Actual' text for Decision Simulation using negative/death context.

    Priority:
    - death summary from `death_row` if present
    - any summary containing 'died to'
    - fallback: 'costly death/risk event in a mixed round'
    - otherwise return the original `summary`
    """
    if not isinstance(candidate, dict):
        return ""

    # prefer explicit death row
    death_row = candidate.get("death_row") if isinstance(candidate.get("death_row"), dict) else None
    if isinstance(death_row, dict):
        return _build_death_summary_from_row(death_row)

    # prefer any summary that contains 'died to'
    summary = str(candidate.get("summary") or "").strip()
    if "died to" in summary.lower():
        return summary

    # fallback to explicit risk wording when negative signals exist
    negative_ml = candidate.get("negative_ml")
    death_risk = candidate.get("death_risk_5s")
    zero_damage_flag = False
    dr_ctx = candidate.get("death_risk_context") if isinstance(candidate.get("death_risk_context"), dict) else {}
    try:
        dmg = float(dr_ctx.get("player_hp") or 9999)
    except (TypeError, ValueError):
        dmg = 9999.0
    # zero-damage death is not directly stored here; we conservatively check death_row damage if present earlier
    if isinstance(negative_ml, (int, float)) and negative_ml < 0.0:
        return "costly death/risk event in a mixed round"
    if death_risk is not None:
        return "costly death/risk event in a mixed round"

    # final fallback: return original summary
    return summary


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

        # death risk info from the VOD priority entry (may be missing)
        death_risk_label = entry.get("death_risk_label")
        death_risk_context = entry.get("death_risk_context")
        death_risk_prob = entry.get("death_risk_5s")

        alternatives: list[dict[str, Any]] = []
        for cand_label in sorted(candidate_set):
            cand_score, cand_reasons = _score_alternative(
                cand_label, zero_damage, untraded, late_round, strongly_negative
            )
            # apply ML death-risk based adjustments
            adj_score = _apply_death_risk_adjustments(cand_score, cand_label, death_risk_label, death_risk_context)
            alternatives.append({
                "label": cand_label,
                "score": adj_score,
                "reasons": cand_reasons,
            })

        alternatives.sort(key=lambda x: x["score"], reverse=True)
        alternatives = alternatives[:3]

        # Append risk info to the original summary so report 'Actual' line includes it.
        orig_summary = str(entry.get("summary") or "").strip()
        if death_risk_label and isinstance(death_risk_prob, (int, float)):
            try:
                prob_text = f"{float(death_risk_prob) * 100.0:.1f}%"
            except Exception:
                prob_text = "-"
            explanation = build_death_risk_explanation(death_risk_context, death_risk_label)
            # format: either '...; risk before death: X, Y (explanation)' or '...<period> Risk before...'
            risk_fragment = f"risk before death: {str(death_risk_label).strip()}, {prob_text}"
            if explanation:
                # attach explanation in parentheses
                risk_fragment = f"{risk_fragment} ({explanation})"
            if orig_summary:
                if orig_summary.endswith('.'):
                    orig_summary = f"{orig_summary} {risk_fragment[0].upper() + risk_fragment[1:]}"
                else:
                    orig_summary = f"{orig_summary}; {risk_fragment}"
            else:
                # start summary with capitalized fragment
                orig_summary = f"{risk_fragment[0].upper() + risk_fragment[1:]}"

        # Detect negative signals (death/ML/death-risk) and decide Actual text.
        try:
            has_negative_signal = (
                zero_damage or
                untraded or
                (isinstance(ml_death_impact, (int, float)) and ml_death_impact < 0.0) or
                (death_risk_label is not None and str(death_risk_label).strip() != "") or
                (entry.get("negative_ml") is not None)
            )
        except Exception:
            has_negative_signal = False

        # detect explicit negative reasons in the entry.reasons list
        try:
            neg_reasons = any(
                isinstance(r, str) and (
                    "untraded death" in r.lower()
                    or "death risk" in r.lower()
                    or ("ml impact" in r.lower() and r.strip().startswith("-"))
                )
                for r in (reasons or [])
            )
        except Exception:
            neg_reasons = False

        actual_summary = orig_summary

        # If this is a mixed review with negative signals, prefer the death/risk-focused text
        if review_type == "mixed" and has_negative_signal:
            alt = build_decision_actual_text(entry)
            if alt:
                actual_summary = alt

        # Final rendering adjustment: if the final `actual_summary` text starts with
        # a positive 'killed' but the computed actual score is negative, replace
        # the visible text with a neutral fallback while preserving any appended
        # 'risk before death' fragment. This prevents misleading positive kill
        # phrasing when the simulation judges the event as costly.
        try:
            starts_killed = isinstance(actual_summary, str) and actual_summary.strip().lower().startswith("killed ")
        except Exception:
            starts_killed = False

        if actual_score < 0 and starts_killed:
            # Build risk fragment (if available) using same formatting as above.
            risk_tail = ""
            if death_risk_label and isinstance(death_risk_prob, (int, float)):
                try:
                    prob_text = f"{float(death_risk_prob) * 100.0:.1f}%"
                except Exception:
                    prob_text = "-"
                explanation = build_death_risk_explanation(death_risk_context, death_risk_label)
                risk_fragment = f"risk before death: {str(death_risk_label).strip()}, {prob_text}"
                if explanation:
                    risk_fragment = f"{risk_fragment} ({explanation})"
                # Capitalize fragment head when attaching as a sentence.
                risk_tail = risk_fragment[0].upper() + risk_fragment[1:]

            actual_summary = "costly death/risk event in a mixed round"
            if risk_tail:
                actual_summary = f"{actual_summary}. {risk_tail}"

        results.append({
            "round_num": round_num,
            "side": str(entry.get("side") or "-").strip(),
            "review_type": review_type,
            "original_summary": orig_summary,
            "actual_summary": actual_summary,
            "actual_decision": actual_decision,
            "alternatives": alternatives,
        })

    return results
