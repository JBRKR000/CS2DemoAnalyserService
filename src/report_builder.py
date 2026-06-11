from __future__ import annotations

from datetime import datetime, timezone
import logging
import re
from typing import Any

from sectors.death_risk import find_player_pre_event_death_risk, load_death_risk_predictions
from sectors.decision_simulator import simulate_decisions, build_death_risk_explanation


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _format_number(value: Any, digits: int = 2) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return "-"


def _format_pct(value: Any, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}%"
    return "-"


def _format_impact_score(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):+.{digits}f} impact score"
    return "-"


def _format_pp_per_event(value: Any, digits: int = 2) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value) * 100.0:+.{digits}f} pp per event"
    return "-"


def _format_rounded_int(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(int(round(float(value))))
    return "-"


def _is_numeric_like_weapon(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, (int, float)):
        return True

    text = str(value).strip()
    if not text:
        return False
    if text.isdigit():
        return True

    stripped = text.replace("_", "").replace("-", "").replace(" ", "")
    return bool(stripped) and stripped.isdigit()


def _weapon_from_summary(summary: Any) -> str | None:
    text = str(summary or "").strip()
    if not text:
        return None

    match = re.search(r"\bwith\s+([A-Za-z0-9][A-Za-z0-9_\-]*)\b", text, re.IGNORECASE)
    if not match:
        return None

    weapon = match.group(1).strip()
    if weapon and not _is_numeric_like_weapon(weapon):
        return weapon
    return None


def format_risk_weapon(risk_context: dict[str, Any], candidate: dict[str, Any] | None = None) -> str:
    context = risk_context if isinstance(risk_context, dict) else {}
    candidate_data = candidate if isinstance(candidate, dict) else {}

    weapon_name = str(context.get("weapon_name") or "").strip()
    if weapon_name and not _is_numeric_like_weapon(weapon_name):
        return weapon_name

    weapon_value = context.get("weapon")
    if isinstance(weapon_value, str):
        weapon_text = weapon_value.strip()
        if weapon_text and not _is_numeric_like_weapon(weapon_text):
            return weapon_text
    elif isinstance(weapon_value, (int, float)):
        weapon_id = int(round(float(weapon_value)))
    else:
        weapon_id = None

    summary_weapon = _weapon_from_summary(candidate_data.get("summary"))
    if summary_weapon:
        return summary_weapon

    for field_name in ("death_risk_event_weapon", "event_weapon", "death_weapon", "weapon"):
        field_value = candidate_data.get(field_name)
        if isinstance(field_value, str):
            field_text = field_value.strip()
            if field_text and not _is_numeric_like_weapon(field_text):
                return field_text
        elif isinstance(field_value, (int, float)):
            weapon_id = int(round(float(field_value)))

    if weapon_value is not None and _is_numeric_like_weapon(weapon_value):
        return f"unknown_weapon_{int(round(float(weapon_value)))}"

    if weapon_id is not None:
        return f"unknown_weapon_{weapon_id}"

    return "unknown"


if __debug__:
    _risk_weapon_test_context = {"weapon": "5767522"}
    _risk_weapon_test_candidate = {"summary": "died to phzy with m4a1 in a high-cost ML swing."}
    assert format_risk_weapon(_risk_weapon_test_context, _risk_weapon_test_candidate) == "m4a1"


def _format_risk_before_death_line(item: dict[str, Any]) -> str | None:
    label = _safe_text(item.get("death_risk_label"), "")
    if not label or label == "-":
        return None

    probability = item.get("death_risk_5s")
    bucket = _safe_text(item.get("death_risk_bucket"), "-")
    context = item.get("death_risk_context")
    context_dict = context if isinstance(context, dict) else {}
    nearest_enemy = _format_rounded_int(context_dict.get("nearest_enemy_distance"))
    nearest_teammate = _format_rounded_int(context_dict.get("nearest_teammate_distance"))
    player_hp = _format_rounded_int(context_dict.get("player_hp"))
    weapon = format_risk_weapon(context_dict, item)
    probability_text = f"{float(probability) * 100.0:.1f}%" if isinstance(probability, (int, float)) else "-"

    return (
        f"{label}, {probability_text}, {bucket}, "
        f"enemy {nearest_enemy}u, teammate {nearest_teammate}u, hp {player_hp}, weapon {weapon}"
    )


def _format_event_pp(value: Any, digits: int = 2) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value) * 100.0:+.{digits}f} pp"
    return "-"


def _safe_text(value: Any, fallback: str = "-") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def _benchmark_status(benchmarks: dict[str, Any]) -> str:
    eval_dicts: list[dict[str, Any]] = []
    for side_key in ("all", "ct", "t"):
        side_evals = benchmarks.get(side_key)
        if isinstance(side_evals, dict):
            for evaluation in side_evals.values():
                if isinstance(evaluation, dict):
                    eval_dicts.append(evaluation)

    if not eval_dicts:
        return "unknown"

    if any(str(item.get("rating", "unknown")) != "unknown" for item in eval_dicts):
        return "available"

    reasons = {str(item.get("reason", "")) for item in eval_dicts}
    if "insufficient_population" in reasons:
        return "insufficient_population"
    if "not_enough_player_samples" in reasons:
        return "not_enough_player_samples"

    return "unknown"


def _benchmark_meta_from_evals(
    evaluations: dict[str, Any],
    map_name: Any,
    side: str = "ALL",
) -> dict[str, Any]:
    metric_entries: list[dict[str, Any]] = [
        item
        for item in evaluations.values()
        if isinstance(item, dict) and isinstance(item.get("metric"), str)
    ]

    metric_details: list[dict[str, Any]] = []
    available_contexts: set[str] = set()
    for entry in metric_entries:
        metric = entry.get("metric")
        context = entry.get("context")
        sample_size = entry.get("sample_size")
        percentile = entry.get("percentile")
        rating = entry.get("rating") if isinstance(entry.get("rating"), str) else "unknown"
        reason = entry.get("reason") if isinstance(entry.get("reason"), str) else None

        metric_details.append(
            {
                "metric": metric,
                "context": context,
                "sample_size": int(sample_size) if isinstance(sample_size, (int, float)) else None,
                "percentile": float(percentile) if isinstance(percentile, (int, float)) else None,
                "rating": rating,
                "reason": reason,
            }
        )

        if rating != "unknown" and reason is None and isinstance(context, str) and context:
            available_contexts.add(context)

    selected_context = None
    if len(available_contexts) == 1:
        selected_context = next(iter(available_contexts))
    elif len(available_contexts) > 1:
        selected_context = "mixed"

    return {
        "selected_context": selected_context,
        "map_name": map_name,
        "side": side,
        "evaluated_metrics_count": len(metric_details),
        "metric_details": metric_details,
    }


def _format_excluded_contexts(excluded_context_counts: Any) -> str:
    if not isinstance(excluded_context_counts, dict):
        return "-"

    teamkills = int(excluded_context_counts.get("teamkill") or 0)
    world_deaths = int(excluded_context_counts.get("world_death") or 0)
    other_contexts = [
        f"{context}={int(count or 0)}"
        for context, count in sorted(excluded_context_counts.items())
        if context not in {"teamkill", "world_death"}
    ]
    contexts = [f"teamkill={teamkills}", f"world_death={world_deaths}", *other_contexts]
    return ", ".join(contexts)


def _format_ml_event(row: dict[str, Any]) -> str:
    round_num = _format_number(row.get("round_num"), digits=0)
    side = _safe_text(row.get("side"))
    killer = _safe_text(row.get("killer_name"), "Unknown killer")
    victim = _safe_text(row.get("victim_name"), "Unknown victim")
    weapon = _safe_text(row.get("weapon"), "unknown weapon")
    impact = _format_event_pp(row.get("win_prob_delta"))
    return f"Round {round_num} | {side} | {killer} killed {victim} with {weapon} | {impact}"


def _append_ml_event_list(lines: list[str], title: str, rows: Any) -> None:
    lines.append(f"{title}:")
    if not isinstance(rows, list) or not rows:
        lines.append("- none")
        return
    for row in rows[:5]:
        if isinstance(row, dict):
            lines.append(f"- {_format_ml_event(row)}")


def format_tip_evidence(tip: dict[str, Any]) -> list[str]:
    evidence_raw = tip.get("evidence")
    if not isinstance(evidence_raw, list):
        return []

    evidence = [str(item).strip() for item in evidence_raw if str(item).strip()]
    if not evidence:
        return []

    return ["Examples:", *(f"- {item}" for item in evidence[:3])]


_ROUND_REFERENCE_RE = re.compile(r"\bRound\s+(\d+)\b", re.IGNORECASE)
_REVIEW_TYPE_ORDER = {"mistake": 0, "mixed": 1, "strength": 2}


def _safe_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]

    to_dicts = getattr(value, "to_dicts", None)
    if callable(to_dicts):
        rows = to_dicts()
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]

    return []


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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_side(value: Any) -> str:
    side = str(value or "").strip().upper()
    return side if side else "-"


def _parse_round_reference(text: Any) -> int:
    match = _ROUND_REFERENCE_RE.search(str(text or ""))
    if not match:
        return 0
    return _safe_round_num(match.group(1))


def _parse_tip_evidence_details(text: Any) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    round_num = _parse_round_reference(raw_text)
    if round_num <= 0:
        return {}

    parts = [part.strip() for part in raw_text.split("|")]
    side = ""
    damage_context = ""
    death_summary = ""

    if len(parts) >= 2:
        possible_side = _normalize_side(parts[1])
        if possible_side in {"T", "CT"}:
            side = possible_side

    for part in parts[2:]:
        lower_part = part.lower()
        if not damage_context and "dmg before death" in lower_part:
            damage_context = part
            continue
        if not death_summary and "died to " in lower_part:
            death_summary = part

    return {
        "round_num": round_num,
        "evidence_side": side,
        "evidence_damage_context": damage_context,
        "evidence_summary": death_summary,
    }


def _tip_reason(metric: str, severity: str) -> tuple[str, str, int] | None:
    if metric == "deaths_with_0_damage":
        return ("zero-damage death", "negative", 6)
    if metric == "deaths_under_40_damage":
        return ("low-damage death under 40", "negative", 4)
    if metric == "untraded_death_rate":
        return ("untraded death", "negative", 4)
    if metric == "clutch_win_rate" and severity in {"critical", "warning"}:
        return ("failed clutch", "negative", 3)
    if metric == "opening_duel_win_pct" and severity == "good":
        return ("strong opening duel", "positive", 3)
    if metric in {"adr", "kpr"} and severity == "good":
        return ("high-impact kill", "positive", 2)
    return None


def _ml_reason_weight(value: float, polarity: str) -> int:
    magnitude = abs(value) * 100.0
    if magnitude >= 30.0:
        return 4
    if magnitude >= 20.0:
        return 3
    if magnitude >= 10.0:
        return 2
    if polarity == "positive" and magnitude >= 5.0:
        return 1
    return 0


def _death_risk_reason(risk_label: Any, risk_bucket: Any) -> tuple[float, str | None, int]:
    label = str(risk_label or "").strip().lower()
    bucket = str(risk_bucket or "").strip().lower()

    boost = 0.0
    reason_label: str | None = None
    display_weight = 0

    if label == "critical":
        boost += 3.0
        reason_label = "critical death risk before death"
        display_weight = 4
    elif label == "high":
        boost += 2.0
        reason_label = "high death risk before death"
        display_weight = 3
    elif label == "medium":
        boost += 1.0
        reason_label = "medium death risk before death"
        display_weight = 2

    if bucket == "top_1_percent":
        boost += 2.0
    elif bucket == "top_5_percent":
        boost += 1.5
    elif bucket == "top_10_percent":
        boost += 1.0
    elif bucket == "top_20_percent":
        boost += 0.5

    return min(boost, 4.0), reason_label, display_weight


def _new_review_round(round_num: int) -> dict[str, Any]:
    return {
        "round_num": round_num,
        "side": "-",
        "reason_map": {},
        "evidence_side": "",
        "evidence_summary": "",
        "evidence_damage_context": "",
        "death_row": None,
        "opening_kill_row": None,
        "kill_rows": [],
        "negative_ml": None,
        "negative_ml_row": None,
        "positive_ml": None,
        "positive_ml_row": None,
    }


def _ensure_review_round(candidates: dict[int, dict[str, Any]], round_num: int) -> dict[str, Any]:
    entry = candidates.get(round_num)
    if entry is None:
        entry = _new_review_round(round_num)
        candidates[round_num] = entry
    return entry


def _update_round_side(entry: dict[str, Any], side: Any) -> None:
    normalized = _normalize_side(side)
    if entry.get("side") in {None, "-", ""} and normalized != "-":
        entry["side"] = normalized


def _update_evidence_details(entry: dict[str, Any], details: dict[str, Any]) -> None:
    evidence_side = _normalize_side(details.get("evidence_side"))
    if evidence_side in {"T", "CT"}:
        existing_side = _normalize_side(entry.get("evidence_side"))
        if existing_side not in {"T", "CT"}:
            entry["evidence_side"] = evidence_side
        _update_round_side(entry, evidence_side)

    evidence_summary = str(details.get("evidence_summary") or "").strip()
    if evidence_summary and not str(entry.get("evidence_summary") or "").strip():
        entry["evidence_summary"] = evidence_summary

    damage_context = str(details.get("evidence_damage_context") or "").strip()
    if damage_context and not str(entry.get("evidence_damage_context") or "").strip():
        entry["evidence_damage_context"] = damage_context


def _add_round_reason(entry: dict[str, Any], label: str, polarity: str, weight: int) -> None:
    if weight <= 0 or not label:
        return

    reason_map = entry["reason_map"]
    existing = reason_map.get(label)
    if existing is None or weight > int(existing.get("weight") or 0):
        reason_map[label] = {
            "label": label,
            "polarity": polarity,
            "weight": weight,
        }


def _death_actor_name(row: dict[str, Any]) -> str:
    return _safe_text(row.get("target_name") or row.get("killer_name"), "Unknown killer")


def _kill_actor_name(row: dict[str, Any]) -> str:
    return _safe_text(row.get("target_name") or row.get("victim_name"), "Unknown victim")


def _round_tick(row: dict[str, Any]) -> int:
    return _safe_round_num(row.get("tick") or row.get("tick_after") or row.get("tick_before"))


def _select_primary_ml_impact(item: dict[str, Any]) -> float | None:
    negative_ml = item.get("negative_ml")
    positive_ml = item.get("positive_ml")
    if isinstance(negative_ml, (int, float)) and isinstance(positive_ml, (int, float)):
        return float(negative_ml) if abs(float(negative_ml)) >= abs(float(positive_ml)) else float(positive_ml)
    if isinstance(negative_ml, (int, float)):
        return float(negative_ml)
    if isinstance(positive_ml, (int, float)):
        return float(positive_ml)
    return None


def _build_death_summary(row: dict[str, Any]) -> str:
    killer = _death_actor_name(row)
    weapon = _safe_text(row.get("weapon"), "unknown weapon")
    damage = _to_float(row.get("damage_before_death"), 0.0)
    if damage <= 0.0:
        damage_text = "after creating no damage"
    elif damage < 40.0:
        damage_text = f"after only {_format_number(damage, digits=0)} damage"
    else:
        damage_text = f"after {_format_number(damage, digits=0)} damage"
    return f"died to {killer} with {weapon} {damage_text}."


def _build_kill_summary(row: dict[str, Any], kill_count: int) -> str:
    victim = _kill_actor_name(row)
    weapon = _safe_text(row.get("weapon"), "unknown weapon")
    if _safe_bool(row.get("is_opening") or row.get("is_opening_kill")):
        suffix = " in a strong opening duel."
    elif kill_count >= 2:
        suffix = f" as part of a {kill_count}-kill round."
    else:
        suffix = " in a round-swinging duel."
    return f"killed {victim} with {weapon}{suffix}"


def _build_mixed_summary(item: dict[str, Any]) -> str:
    opening_row = item.get("opening_kill_row")
    death_row = item.get("death_row")
    if isinstance(opening_row, dict) and isinstance(death_row, dict):
        victim = _kill_actor_name(opening_row)
        weapon = _safe_text(opening_row.get("weapon"), "unknown weapon")
        if not _safe_bool(death_row.get("is_traded_death")):
            return f"won the opener on {victim} with {weapon} but later died untraded."
        return f"won the opener on {victim} with {weapon} but still lost impact later in the round."

    positive_ml_row = item.get("positive_ml_row")
    if isinstance(positive_ml_row, dict):
        return _build_kill_summary(positive_ml_row, len(item.get("kill_rows") or []))

    if isinstance(death_row, dict):
        return _build_death_summary(death_row)

    return "mixed round with both useful impact and a review-worthy mistake."


def _build_evidence_summary(item: dict[str, Any]) -> str | None:
    damage_context = str(item.get("evidence_damage_context") or "").strip()
    evidence_summary = str(item.get("evidence_summary") or "").strip()
    parts = [part for part in [damage_context, evidence_summary] if part]
    if not parts:
        return None

    summary = "; ".join(parts).rstrip(".;")
    return f"{summary}."


def _build_review_summary(item: dict[str, Any], review_type: str) -> str:
    death_row = item.get("death_row")
    negative_ml_row = item.get("negative_ml_row")
    positive_ml_row = item.get("positive_ml_row")
    kill_rows = item.get("kill_rows") or []

    if review_type == "mistake":
        if isinstance(death_row, dict):
            return _build_death_summary(death_row)
        if isinstance(negative_ml_row, dict):
            row = negative_ml_row
            killer = _death_actor_name(row)
            weapon = _safe_text(row.get("weapon"), "unknown weapon")
            return f"died to {killer} with {weapon} in a high-cost ML swing."
        evidence_summary = _build_evidence_summary(item)
        if evidence_summary is not None:
            return evidence_summary
        return "mistake-heavy round with multiple low-value outcomes."

    if review_type == "strength":
        if isinstance(positive_ml_row, dict):
            return _build_kill_summary(positive_ml_row, len(kill_rows))
        if kill_rows:
            return _build_kill_summary(kill_rows[0], len(kill_rows))
        return "strength round with clear positive impact worth repeating."

    return _build_mixed_summary(item)


def build_vod_review_priority(report_data: dict[str, Any], top_n: int = 5) -> list[dict[str, Any]]:
    if top_n <= 0:
        return []

    safe_report = _safe_dict(report_data)
    match = _safe_dict(safe_report.get("match"))
    player = _safe_dict(safe_report.get("player"))
    feedback = [tip for tip in safe_report.get("feedback", []) if isinstance(tip, dict)]
    timeline_events = _safe_rows(safe_report.get("selected_player_timeline_events"))
    ml_impact = _safe_dict(safe_report.get("ml_impact"))
    match_id = _safe_text(match.get("match_id"), "")
    player_steamid = _safe_text(player.get("steamid"), "")

    if not feedback and not timeline_events and not ml_impact:
        return []

    death_risk_predictions = load_death_risk_predictions()

    candidates: dict[int, dict[str, Any]] = {}

    for tip in feedback:
        metric = str(tip.get("metric") or "")
        severity = str(tip.get("severity") or "")
        reason = _tip_reason(metric, severity)
        evidence = tip.get("evidence")
        if reason is None or not isinstance(evidence, list):
            continue

        label, polarity, weight = reason
        for item in evidence:
            evidence_details = _parse_tip_evidence_details(item)
            round_num = _safe_round_num(evidence_details.get("round_num"))
            if round_num <= 0:
                continue
            entry = _ensure_review_round(candidates, round_num)
            _update_evidence_details(entry, evidence_details)
            _add_round_reason(entry, label, polarity, weight)

    for row in timeline_events:
        round_num = _safe_round_num(row.get("round_num"))
        if round_num <= 0:
            continue

        entry = _ensure_review_round(candidates, round_num)
        _update_round_side(entry, row.get("side"))
        event_type = str(row.get("event_type") or "").strip().lower()

        if event_type == "death":
            current_death = entry.get("death_row")
            if not isinstance(current_death, dict) or _round_tick(row) < _round_tick(current_death):
                entry["death_row"] = row

            damage_before_death = _to_float(row.get("damage_before_death"), 0.0)
            if damage_before_death <= 0.0:
                _add_round_reason(entry, "zero-damage death", "negative", 6)
            elif damage_before_death < 40.0:
                _add_round_reason(entry, "low-damage death under 40", "negative", 4)

            if not _safe_bool(row.get("is_traded_death")):
                _add_round_reason(entry, "untraded death", "negative", 4)

            if _safe_bool(row.get("is_opening_death")) or str(row.get("round_phase") or "").strip().lower() == "early":
                _add_round_reason(entry, "early death", "negative", 2)

        if event_type == "kill":
            kill_rows = entry["kill_rows"]
            kill_rows.append(row)
            if _safe_bool(row.get("is_opening_kill")) and not isinstance(entry.get("opening_kill_row"), dict):
                entry["opening_kill_row"] = row
                _add_round_reason(entry, "strong opening duel", "positive", 3)

    for row in _safe_rows(ml_impact.get("worst_deaths")):
        round_num = _safe_round_num(row.get("round_num"))
        if round_num <= 0:
            continue

        impact_value = _to_float(row.get("win_prob_delta"), 0.0)
        if impact_value >= 0.0:
            continue

        entry = _ensure_review_round(candidates, round_num)
        _update_round_side(entry, row.get("side"))
        current_negative = entry.get("negative_ml")
        if not isinstance(current_negative, (int, float)) or impact_value < float(current_negative):
            entry["negative_ml"] = impact_value
            entry["negative_ml_row"] = row

        ml_weight = _ml_reason_weight(impact_value, "negative")
        _add_round_reason(entry, f"{_format_event_pp(impact_value)} ML impact", "negative", ml_weight)

    for row in _safe_rows(ml_impact.get("best_kills")):
        round_num = _safe_round_num(row.get("round_num"))
        if round_num <= 0:
            continue

        impact_value = _to_float(row.get("win_prob_delta"), 0.0)
        if impact_value <= 0.0:
            continue

        entry = _ensure_review_round(candidates, round_num)
        _update_round_side(entry, row.get("side"))
        current_positive = entry.get("positive_ml")
        if not isinstance(current_positive, (int, float)) or impact_value > float(current_positive):
            entry["positive_ml"] = impact_value
            entry["positive_ml_row"] = row

        ml_weight = _ml_reason_weight(impact_value, "positive")
        _add_round_reason(entry, f"{_format_event_pp(impact_value)} ML impact", "positive", ml_weight)
        if _safe_bool(row.get("is_opening")):
            _add_round_reason(entry, "strong opening duel", "positive", 3)

    if death_risk_predictions is not None and match_id and player_steamid:
        for entry in candidates.values():
            round_num = _safe_round_num(entry.get("round_num"))
            if round_num <= 0:
                continue

            event_tick = 0
            for candidate_row_key in ("death_row", "negative_ml_row", "opening_kill_row"):
                candidate_row = entry.get(candidate_row_key)
                if isinstance(candidate_row, dict):
                    event_tick = _round_tick(candidate_row)
                    if event_tick > 0:
                        break

            risk_snapshot = find_player_pre_event_death_risk(
                death_risk_predictions,
                match_id=match_id,
                steamid=player_steamid,
                round_num=round_num,
                event_tick=event_tick if event_tick > 0 else None,
            )
            if risk_snapshot is None:
                continue

            entry["death_risk_5s"] = risk_snapshot.get("max_death_risk_5s")
            entry["death_risk_label"] = risk_snapshot.get("max_risk_label")
            entry["death_risk_bucket"] = risk_snapshot.get("max_risk_bucket")
            entry["death_risk_snapshot_tick"] = risk_snapshot.get("risk_snapshot_tick")
            entry["death_risk_context"] = {
                "nearest_enemy_distance": risk_snapshot.get("nearest_enemy_distance_at_max_risk"),
                "nearest_teammate_distance": risk_snapshot.get("nearest_teammate_distance_at_max_risk"),
                "player_hp": risk_snapshot.get("player_hp_at_max_risk"),
                "weapon": risk_snapshot.get("weapon_at_max_risk"),
                "weapon_name": risk_snapshot.get("weapon_name"),
            }

            event_weapon = None
            for candidate_row_key in ("death_row", "negative_ml_row", "opening_kill_row"):
                candidate_row = entry.get(candidate_row_key)
                if isinstance(candidate_row, dict):
                    candidate_weapon = candidate_row.get("weapon")
                    if isinstance(candidate_weapon, str) and candidate_weapon.strip():
                        event_weapon = candidate_weapon.strip()
                        break
                    if isinstance(candidate_weapon, (int, float)):
                        event_weapon = str(int(round(float(candidate_weapon))))
                        break

            if event_weapon is not None:
                entry["death_risk_event_weapon"] = event_weapon

            death_risk_boost, death_risk_reason, death_risk_reason_weight = _death_risk_reason(
                risk_snapshot.get("max_risk_label"),
                risk_snapshot.get("max_risk_bucket"),
            )
            if death_risk_boost > 0.0:
                current_boost = _to_float(entry.get("death_risk_boost"), 0.0)
                entry["death_risk_boost"] = max(current_boost, death_risk_boost)

            if death_risk_reason:
                _add_round_reason(entry, death_risk_reason, "negative", death_risk_reason_weight)
                reason_map = entry.get("reason_map")
                if isinstance(reason_map, dict):
                    risk_reason_entry = reason_map.get(death_risk_reason)
                    if isinstance(risk_reason_entry, dict):
                        risk_reason_entry["source"] = "death_risk"

    prioritized_rounds: list[dict[str, Any]] = []
    for entry in candidates.values():
        kill_rows = entry.get("kill_rows") or []
        if len(kill_rows) >= 2:
            if len(kill_rows) >= 3 or _to_float(entry.get("positive_ml"), 0.0) >= 0.15:
                _add_round_reason(entry, "high-impact multi-kill round", "positive", 3)
            else:
                _add_round_reason(entry, "high-impact multi-kill round", "positive", 2)

        reasons = [reason for reason in entry.get("reason_map", {}).values() if isinstance(reason, dict)]
        death_risk_boost = _to_float(entry.get("death_risk_boost"), 0.0)
        if not reasons and death_risk_boost <= 0.0:
            continue

        negative_score = sum(
            int(reason.get("weight") or 0)
            for reason in reasons
            if str(reason.get("polarity") or "") == "negative" and str(reason.get("source") or "") != "death_risk"
        )
        positive_score = sum(
            int(reason.get("weight") or 0)
            for reason in reasons
            if str(reason.get("polarity") or "") == "positive"
        )
        combined_negative_score = negative_score + death_risk_boost
        if combined_negative_score <= 0 and positive_score <= 0:
            continue

        if combined_negative_score > 0 and positive_score > 0:
            review_type = "mixed"
        elif combined_negative_score > 0:
            review_type = "mistake"
        else:
            review_type = "strength"

        total_score = combined_negative_score + positive_score
        if total_score >= 9 or combined_negative_score >= 8 or positive_score >= 8:
            priority = "high"
        elif total_score >= 5 or combined_negative_score >= 4 or positive_score >= 4:
            priority = "medium"
        else:
            priority = "low"

        # Enforce minimum priority based on death risk ML signals.
        # Do not lower an existing priority — only raise it when ML indicates elevated risk.
        try:
            dr_label = str(entry.get("death_risk_label") or "").strip().lower()
            dr_bucket = str(entry.get("death_risk_bucket") or "").strip().lower()
        except Exception:
            dr_label = ""
            dr_bucket = ""

        min_priority = None
        if dr_label == "critical" or dr_bucket == "top_1_percent":
            min_priority = "high"
        elif dr_label == "high" or dr_bucket in {"top_5_percent", "top_10_percent"}:
            min_priority = "medium"

        if min_priority is not None:
            _order = {"low": 0, "medium": 1, "high": 2}
            current_rank = _order.get(str(priority) if priority is not None else "low", 0)
            min_rank = _order.get(min_priority, 0)
            if current_rank < min_rank:
                priority = min_priority

        if review_type == "strength":
            reasons.sort(
                key=lambda reason: (
                    0 if str(reason.get("polarity") or "") == "positive" else 1,
                    -int(reason.get("weight") or 0),
                    str(reason.get("label") or ""),
                )
            )
        else:
            reasons.sort(
                key=lambda reason: (
                    0 if str(reason.get("polarity") or "") == "negative" else 1,
                    -int(reason.get("weight") or 0),
                    str(reason.get("label") or ""),
                )
            )

        prioritized_rounds.append(
            {
                "round_num": int(entry["round_num"]),
                "side": _normalize_side(entry.get("side") or entry.get("evidence_side")),
                "priority": priority,
                "review_type": review_type,
                "reasons": [str(reason.get("label") or "") for reason in reasons[:3]],
                "ml_impact": _select_primary_ml_impact(entry),
                "summary": _build_review_summary(entry, review_type),
                "score": total_score,
                "negative_score": combined_negative_score,
                "positive_score": positive_score,
                "death_risk_5s": entry.get("death_risk_5s"),
                "death_risk_label": entry.get("death_risk_label"),
                "death_risk_bucket": entry.get("death_risk_bucket"),
                "death_risk_snapshot_tick": entry.get("death_risk_snapshot_tick"),
                "death_risk_context": entry.get("death_risk_context"),
            }
        )

    prioritized_rounds.sort(
        key=lambda item: (
            _REVIEW_TYPE_ORDER.get(str(item.get("review_type") or ""), len(_REVIEW_TYPE_ORDER)),
            -_to_float(item.get("negative_score"), 0.0),
            -_to_float(item.get("positive_score"), 0.0),
            -_to_float(item.get("score"), 0.0),
            int(item.get("round_num") or 0),
        )
    )

    return [
        {
            "round_num": item["round_num"],
            "side": item["side"],
            "priority": item["priority"],
            "review_type": item["review_type"],
            "reasons": item["reasons"],
            "ml_impact": item["ml_impact"],
            "summary": item["summary"],
            "death_risk_5s": item.get("death_risk_5s"),
            "death_risk_label": item.get("death_risk_label"),
            "death_risk_bucket": item.get("death_risk_bucket"),
            "death_risk_snapshot_tick": item.get("death_risk_snapshot_tick"),
            "death_risk_context": item.get("death_risk_context"),
                "death_risk_event_weapon": item.get("death_risk_event_weapon"),
        }
        for item in prioritized_rounds[: min(top_n, 5)]
    ]


# --- Coach Summary v1 -------------------------------------------------------
# A short, product-facing layer that distils the existing report into five
# coach-style bullets. It only reads data already produced by other sections
# (feedback tips, benchmark evaluations, impact stats, VOD review priority,
# decision simulation, ML impact, death risk) and never recomputes them.

_COACH_WEAKNESS_SENTENCES = {
    "untraded_death_rate": "Too many costly deaths happen outside trade range.",
    "deaths_with_0_damage": "Too many deaths happen before you create any impact.",
    "deaths_under_40_damage": "Too many deaths happen before you create useful damage.",
    "early_deaths": "Too many early deaths put your team behind.",
    "opening_duel_win_pct": "Opening duel efficiency needs improvement.",
    "kast": "KAST is below the benchmark pool.",
    "adr": "Round damage impact is below the benchmark pool.",
    "hs_percent": "Headshot consistency is below the benchmark pool.",
    "kpr": "Kill pace is below the benchmark pool.",
    "full_buy_win_rate": "Full-buy conversion is below the benchmark pool.",
    "force_win_rate": "Force-buy conversion is below the benchmark pool.",
    "clutch_win_rate": "Clutch conversion is below the benchmark pool.",
}

_COACH_STRENGTH_SENTENCES = {
    "adr": "Round damage impact is a strength.",
    "kast": "Round survival/trade value is strong compared with the benchmark pool.",
    "kpr": "Kill pace is strong compared with the benchmark pool.",
    "hs_percent": "Headshot consistency is strong.",
    "opening_duel_win_pct": "Opening duel efficiency is strong.",
    "full_buy_win_rate": "Full-buy conversion is strong.",
    "force_win_rate": "Force-buy conversion is strong.",
    "clutch_win_rate": "Clutch conversion is strong.",
    "trade_kills": "Trade conversion is a strength.",
}

_COACH_PRACTICE_SENTENCES = {
    "untraded_death_rate": "Play 10 review rounds focusing only on teammate distance before contact.",
    "deaths_with_0_damage": "Before every first contact, require utility, trade timing, or off-angle advantage.",
    "deaths_under_40_damage": "Before every first contact, require utility, trade timing, or off-angle advantage.",
    "hs_percent": "Run 10 minutes of head-height pathing and first-bullet discipline.",
    "clutch_win_rate": "Review 1vX rounds and mark every missed reposition timing.",
    "full_buy_win_rate": "Review full-buy losses and check utility usage before final contact.",
    "force_win_rate": "Review full-buy losses and check utility usage before final contact.",
    "opening_duel_win_pct": "Review opening deaths and tag whether contact had utility or teammate timing.",
}

_COACH_PRACTICE_FALLBACK = (
    "Review the top 5 VOD rounds and mark teammate distance before every first contact."
)

_COACH_DECISION_PATTERN_SENTENCES = {
    "fall_back": "You often take fights where backing off would preserve round equity.",
    "wait_for_trade": "Your costly deaths often happen before teammate trade timing is ready.",
    "hold_angle": "You may be over-peeking instead of holding safer contact.",
    "play_time": "Late-round deaths suggest you should play time more often.",
}

_COACH_DECISION_PATTERN_PRIORITY = ("fall_back", "wait_for_trade", "hold_angle", "play_time")


def _coach_benchmark_evals(benchmarks: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten benchmark evaluations into a list of rated metric entries.

    Only entries with a real rating (not 'unknown' and without a blocking
    reason) and a numeric percentile are returned, so callers can pick the
    lowest/highest percentile without re-validating each entry.
    """
    evals: list[dict[str, Any]] = []
    for side_key in ("all", "ct", "t"):
        side_evals = benchmarks.get(side_key)
        if not isinstance(side_evals, dict):
            continue
        for metric_key, evaluation in side_evals.items():
            if not isinstance(evaluation, dict):
                continue
            rating = str(evaluation.get("rating") or "unknown")
            if rating == "unknown" or evaluation.get("reason") is not None:
                continue
            percentile = evaluation.get("percentile")
            if not isinstance(percentile, (int, float)) or isinstance(percentile, bool):
                continue
            metric_name = (
                evaluation.get("metric")
                if isinstance(evaluation.get("metric"), str)
                else str(metric_key)
            )
            evals.append({"metric": metric_name, "rating": rating, "percentile": float(percentile)})
    return evals


def _coach_main_weakness(
    feedback: list[dict[str, Any]],
    benchmarks: dict[str, Any],
    vod: list[dict[str, Any]],
    ml_impact: dict[str, Any],
) -> tuple[str, str]:
    """Return (sentence, metric) for the dominant weakness.

    Priority: critical tip -> warning tip -> lowest benchmark percentile ->
    largest negative VOD/ML issue.
    """
    for severity in ("critical", "warning"):
        for tip in feedback:
            if str(tip.get("severity")) == severity:
                metric = str(tip.get("metric") or "")
                sentence = _COACH_WEAKNESS_SENTENCES.get(metric) or _safe_text(tip.get("title"))
                return sentence, metric

    negative_evals = [e for e in _coach_benchmark_evals(benchmarks) if e["rating"] in {"critical", "warning"}]
    if negative_evals:
        worst = min(negative_evals, key=lambda e: (e["percentile"], e["metric"]))
        metric = worst["metric"]
        sentence = _COACH_WEAKNESS_SENTENCES.get(metric) or f"{metric} is below the benchmark pool."
        return sentence, metric

    for item in vod:
        if str(item.get("review_type")) in {"mistake", "mixed"}:
            reasons_raw = item.get("reasons")
            reasons = [str(r).strip() for r in reasons_raw if str(r).strip()] if isinstance(reasons_raw, list) else []
            reason = reasons[0] if reasons else ""
            round_num = _format_number(item.get("round_num"), digits=0)
            if reason:
                return f"Round {round_num} keeps surfacing a recurring issue: {reason}.", ""
            break

    net_ml = ml_impact.get("net_ml_impact")
    if isinstance(net_ml, (int, float)) and not isinstance(net_ml, bool) and net_ml < 0.0:
        return "Net ML impact across the match is negative.", ""

    return "No dominant weakness detected in this match.", ""


def _coach_best_strength(
    feedback: list[dict[str, Any]],
    benchmarks: dict[str, Any],
    ml_impact: dict[str, Any],
    overall: dict[str, Any],
) -> str:
    """Return a one-line best-strength sentence.

    Priority: good tip -> highest benchmark percentile -> positive ML impact ->
    strong raw KAST.
    """
    for tip in feedback:
        if str(tip.get("severity")) == "good":
            metric = str(tip.get("metric") or "")
            return _COACH_STRENGTH_SENTENCES.get(metric) or _safe_text(tip.get("title"))

    positive_evals = [e for e in _coach_benchmark_evals(benchmarks) if e["rating"] in {"good", "excellent"}]
    if positive_evals:
        best = max(positive_evals, key=lambda e: (e["percentile"], e["metric"]))
        metric = best["metric"]
        return _COACH_STRENGTH_SENTENCES.get(metric) or f"{metric} is strong compared with the benchmark pool."

    net_ml = ml_impact.get("net_ml_impact")
    if isinstance(net_ml, (int, float)) and not isinstance(net_ml, bool) and net_ml > 0.0:
        return "Net positive kill impact across the match."

    kast = overall.get("kast")
    if isinstance(kast, (int, float)) and not isinstance(kast, bool) and kast >= 70.0:
        return "Round survival/trade value (KAST) held up well this match."

    return "No standout strength detected in this match."


def _coach_top_vod_focus(vod: list[dict[str, Any]]) -> str | None:
    """Return the first VOD priority round as a single focus line, or None."""
    for item in vod:
        round_num = _format_number(item.get("round_num"), digits=0)
        reasons_raw = item.get("reasons")
        reasons = [str(r).strip() for r in reasons_raw if str(r).strip()] if isinstance(reasons_raw, list) else []
        parts = reasons[:2]
        label = _safe_text(item.get("death_risk_label"), "")
        if label and label != "-" and not any("death risk" in part.lower() for part in parts):
            parts.append(f"{label.lower()} death risk")
        detail = ", ".join(parts) if parts else _safe_text(item.get("summary"))
        return f"Round {round_num}: {detail}"
    return None


def _coach_decision_pattern(decision_simulation: Any, vod: list[dict[str, Any]]) -> str | None:
    """Infer the most repeated decision pattern from the simulation.

    Falls back to VOD death reasons when no simulation is available.
    """
    counts: dict[str, int] = {}
    if isinstance(decision_simulation, list):
        for sim in decision_simulation:
            if not isinstance(sim, dict):
                continue
            alternatives = sim.get("alternatives")
            if not isinstance(alternatives, list) or not alternatives:
                continue
            top = alternatives[0]
            if isinstance(top, dict):
                label = str(top.get("label") or "")
                if label in _COACH_DECISION_PATTERN_SENTENCES:
                    counts[label] = counts.get(label, 0) + 1

    if counts:
        best = max(_COACH_DECISION_PATTERN_PRIORITY, key=lambda label: counts.get(label, 0))
        if counts.get(best, 0) > 0:
            return _COACH_DECISION_PATTERN_SENTENCES[best]

    for item in vod:
        reasons_raw = item.get("reasons")
        reasons = [str(r).lower() for r in reasons_raw] if isinstance(reasons_raw, list) else []
        if any("untraded death" in reason for reason in reasons):
            return "Costly deaths often happen outside trade range."
        if any("zero-damage death" in reason for reason in reasons):
            return "Several deaths happen before creating impact."

    return None


def _coach_practice_focus(weakness_metric: str, feedback: list[dict[str, Any]]) -> str:
    """Pick one concrete practice habit, keyed off the main weakness metric."""
    if weakness_metric in _COACH_PRACTICE_SENTENCES:
        return _COACH_PRACTICE_SENTENCES[weakness_metric]

    for tip in feedback:
        if str(tip.get("severity")) in {"critical", "warning"}:
            metric = str(tip.get("metric") or "")
            if metric in _COACH_PRACTICE_SENTENCES:
                return _COACH_PRACTICE_SENTENCES[metric]

    return _COACH_PRACTICE_FALLBACK


def build_coach_summary(report_data: dict[str, Any]) -> dict[str, Any]:
    """Build the COACH SUMMARY v1 structured layer from existing report data.

    Returns a dict with main_weakness, best_strength, top_vod_focus,
    decision_pattern and practice_focus. Every field renders even when the
    underlying section is empty (top_vod_focus/decision_pattern may be None).
    """
    safe_report = _safe_dict(report_data)
    feedback = [tip for tip in safe_report.get("feedback", []) if isinstance(tip, dict)]
    benchmarks = _safe_dict(safe_report.get("benchmarks"))
    overall = _safe_dict(safe_report.get("overall"))
    ml_impact = _safe_dict(safe_report.get("ml_impact"))
    vod_raw = safe_report.get("vod_review_priority")
    vod = [item for item in vod_raw if isinstance(item, dict)] if isinstance(vod_raw, list) else []
    decision_simulation = safe_report.get("decision_simulation")

    main_weakness, weakness_metric = _coach_main_weakness(feedback, benchmarks, vod, ml_impact)

    return {
        "main_weakness": main_weakness,
        "best_strength": _coach_best_strength(feedback, benchmarks, ml_impact, overall),
        "top_vod_focus": _coach_top_vod_focus(vod),
        "decision_pattern": _coach_decision_pattern(decision_simulation, vod),
        "practice_focus": _coach_practice_focus(weakness_metric, feedback),
    }


# --- Structured report (API / frontend) ------------------------------------
# A stable, JSON-serialisable view of the report. It only re-shapes data that
# the existing sections already produced (no scoring, no recomputation) and
# converts numpy/polars scalars into native Python types so the result can be
# returned by an API or rendered by a frontend.

_STRUCTURED_SCHEMA_VERSION = "1.0"
_STRUCTURED_REPORT_TYPE = "cs2_coach_report"
_STRUCTURED_TOP_LEVEL_KEYS = (
    "meta",
    "player",
    "overall",
    "impact",
    "side_breakdown",
    "economy",
    "clutch",
    "benchmarks",
    "ml_impact",
    "tips",
    "vod_review_priority",
    "decision_simulation",
    "coach_summary",
)


def _json_safe(value: Any) -> Any:
    """Recursively convert a value into JSON-safe native Python types.

    Handles numpy/polars scalars (anything exposing ``.item()``) and nested
    dicts/lists. Falls back to ``str`` for anything otherwise unserialisable.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    item_getter = getattr(value, "item", None)
    if callable(item_getter):
        try:
            return _json_safe(item_getter())
        except Exception:
            return str(value)
    return str(value)


def _structured_pp(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value) * 100.0, 2)
    return None


def _structured_risk_before_death(item: dict[str, Any]) -> dict[str, Any] | None:
    """Build the risk_before_death object for a VOD/decision entry, or None."""
    label = _safe_text(item.get("death_risk_label"), "")
    if not label or label == "-":
        return None

    context = item.get("death_risk_context")
    context_dict = context if isinstance(context, dict) else {}
    probability = item.get("death_risk_5s")
    probability_native = (
        float(probability)
        if isinstance(probability, (int, float)) and not isinstance(probability, bool)
        else None
    )
    probability_percent = round(probability_native * 100.0, 1) if probability_native is not None else None
    explanation = build_death_risk_explanation(context_dict, item.get("death_risk_label"))

    return {
        "label": label,
        "probability": probability_native,
        "probability_percent": probability_percent,
        "bucket": _json_safe(item.get("death_risk_bucket")),
        "nearest_enemy_distance": _json_safe(context_dict.get("nearest_enemy_distance")),
        "nearest_teammate_distance": _json_safe(context_dict.get("nearest_teammate_distance")),
        "player_hp": _json_safe(context_dict.get("player_hp")),
        "weapon": format_risk_weapon(context_dict, item),
        "explanation": explanation or None,
    }


def _structured_ml_event(row: dict[str, Any]) -> dict[str, Any]:
    killer = _safe_text(row.get("killer_name"), "")
    victim = _safe_text(row.get("victim_name"), "")
    weapon = _safe_text(row.get("weapon"), "")
    return {
        "round_num": _json_safe(row.get("round_num")),
        "side": _json_safe(row.get("side")),
        "summary": _format_ml_event(row),
        "impact_pp": _structured_pp(row.get("win_prob_delta")),
        "killer": killer or None,
        "victim": victim or None,
        "weapon": weapon or None,
    }


def _structured_ml_event_list(rows: Any) -> list[dict[str, Any]]:
    return [_structured_ml_event(row) for row in _safe_rows(rows)][:5]


def _structured_side(impact: dict[str, Any]) -> dict[str, Any] | None:
    if not impact:
        return None
    return {
        "untraded_deaths": _json_safe(impact.get("untraded_deaths")),
        "total_deaths": _json_safe(impact.get("deaths")),
        "untraded_death_rate": _json_safe(impact.get("untraded_death_rate")),
        "opening_duel_win_pct": _json_safe(impact.get("opening_duel_win_pct")),
        "trade_kills": _json_safe(impact.get("trade_kills")),
        "early_deaths": _json_safe(impact.get("early_deaths")),
    }


def _structured_meta(match: dict[str, Any], report_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": _STRUCTURED_SCHEMA_VERSION,
        "report_type": _STRUCTURED_REPORT_TYPE,
        "map_name": _json_safe(match.get("map_name")),
        "match_id": _json_safe(match.get("match_id")),
        "generated_at": _json_safe(report_meta.get("generated_at")),
    }


def _structured_benchmarks(benchmarks: dict[str, Any]) -> dict[str, Any]:
    benchmark_all = _safe_dict(benchmarks.get("all"))
    metrics: list[dict[str, Any]] = []
    for metric_key, evaluation in benchmark_all.items():
        if not isinstance(evaluation, dict):
            continue
        metrics.append(
            {
                "metric": _json_safe(evaluation.get("metric") or metric_key),
                "value": _json_safe(evaluation.get("value")),
                "percentile": _json_safe(evaluation.get("percentile")),
                "rating": _json_safe(evaluation.get("rating")),
                "context": _json_safe(evaluation.get("context")),
                "reason": _json_safe(evaluation.get("reason")),
            }
        )
    metrics.sort(key=lambda entry: str(entry.get("metric") or ""))

    pools: list[dict[str, Any]] = []
    metric_details = benchmarks.get("metric_details")
    if isinstance(metric_details, list):
        for detail in metric_details:
            if not isinstance(detail, dict):
                continue
            pools.append(
                {
                    "metric": _json_safe(detail.get("metric")),
                    "context": _json_safe(detail.get("context")),
                    "sample_size": _json_safe(detail.get("sample_size")),
                    "percentile": _json_safe(detail.get("percentile")),
                    "rating": _json_safe(detail.get("rating")),
                    "reason": _json_safe(detail.get("reason")),
                }
            )

    return {
        "status": _json_safe(benchmarks.get("status")),
        "context": _json_safe(benchmarks.get("selected_context")),
        "metrics_evaluated": _json_safe(benchmarks.get("evaluated_metrics_count")),
        "metrics": metrics,
        "pools": pools,
    }


def _structured_ml_impact(ml_impact: Any) -> dict[str, Any]:
    if not isinstance(ml_impact, dict):
        return {"status": "missing"}

    kill_count = int(_to_float(ml_impact.get("kill_count"), 0.0))
    death_count = int(_to_float(ml_impact.get("death_count"), 0.0))
    if kill_count <= 0 and death_count <= 0:
        return {"status": "missing"}

    return {
        "status": "available",
        "net_impact_score": _json_safe(ml_impact.get("net_ml_impact")),
        "kill_impact_score": _json_safe(ml_impact.get("total_kill_impact")),
        "death_impact_score": _json_safe(ml_impact.get("total_death_impact")),
        "average_kill_impact_pp": _structured_pp(ml_impact.get("avg_kill_impact")),
        "average_death_impact_pp": _structured_pp(ml_impact.get("avg_death_impact")),
        "best_kills": _structured_ml_event_list(ml_impact.get("best_kills")),
        "worst_deaths": _structured_ml_event_list(ml_impact.get("worst_deaths")),
        "low_impact_kills": _structured_ml_event_list(ml_impact.get("low_impact_kills")),
    }


def build_structured_report(report_data: dict[str, Any]) -> dict[str, Any]:
    """Build a stable, JSON-serialisable structured report from report data.

    Re-shapes the already-computed sections into a fixed top-level schema. All
    values are converted to native Python types so the result is safe to return
    through an API or serialise to JSON.
    """
    safe_report = _safe_dict(report_data)
    match = _safe_dict(safe_report.get("match"))
    player = _safe_dict(safe_report.get("player"))
    overall = _safe_dict(safe_report.get("overall"))
    impact = _safe_dict(safe_report.get("impact"))
    economy = _safe_dict(safe_report.get("economy"))
    clutch = _safe_dict(safe_report.get("clutch"))
    benchmarks = _safe_dict(safe_report.get("benchmarks"))
    report_meta = _safe_dict(safe_report.get("meta"))
    side_breakdown = _safe_dict(safe_report.get("side_breakdown"))
    ct_impact = _safe_dict(_safe_dict(side_breakdown.get("CT")).get("impact"))
    t_impact = _safe_dict(_safe_dict(side_breakdown.get("T")).get("impact"))

    feedback = [tip for tip in safe_report.get("feedback", []) if isinstance(tip, dict)]
    vod_raw = safe_report.get("vod_review_priority")
    vod = [item for item in vod_raw if isinstance(item, dict)] if isinstance(vod_raw, list) else []
    decision_raw = safe_report.get("decision_simulation")
    decision_simulation = (
        [sim for sim in decision_raw if isinstance(sim, dict)] if isinstance(decision_raw, list) else []
    )
    coach_summary_raw = safe_report.get("coach_summary")
    coach_summary = (
        coach_summary_raw if isinstance(coach_summary_raw, dict) else build_coach_summary(safe_report)
    )

    structured_overall = {
        "kills": _json_safe(overall.get("kills")),
        "deaths": _json_safe(overall.get("deaths")),
        "assists": _json_safe(overall.get("assists")),
        "kpr": _json_safe(overall.get("kpr")),
        "dpr": _json_safe(overall.get("dpr")),
        "adr": _json_safe(overall.get("adr")),
        "kast": _json_safe(overall.get("kast")),
        "hs_kills": _json_safe(overall.get("hs_kills")),
        "hs_percent": _json_safe(overall.get("hs_percent")),
    }

    structured_impact = {
        "untraded_deaths": _json_safe(impact.get("untraded_deaths")),
        "total_deaths": _json_safe(impact.get("deaths")),
        "untraded_death_rate": _json_safe(impact.get("untraded_death_rate")),
        "opening_duels": _json_safe(impact.get("opening_duels")),
        "opening_duel_win_pct": _json_safe(impact.get("opening_duel_win_pct")),
        "trade_kills": _json_safe(impact.get("trade_kills")),
        "avg_damage_before_death": _json_safe(impact.get("avg_damage_before_death")),
        "zero_damage_deaths": _json_safe(impact.get("deaths_with_0_damage")),
        "deaths_under_40_damage": _json_safe(impact.get("deaths_under_40_damage")),
        "death_timing": {
            "early": _json_safe(impact.get("early_deaths")),
            "mid": _json_safe(impact.get("mid_deaths")),
            "late": _json_safe(impact.get("late_deaths")),
        },
    }

    structured_economy = {
        "full_buy_win_rate": _json_safe(economy.get("full_buy_win_rate")),
        "force_win_rate": _json_safe(economy.get("force_win_rate")),
        "eco_kills": _json_safe(economy.get("eco_kills")),
        "broken_economy_rounds": _json_safe(economy.get("broken_economy_rounds")),
        "save_rounds": _json_safe(economy.get("save_rounds")),
    }

    structured_clutch = {
        "total_clutches": _json_safe(clutch.get("total_clutches")),
        "clutches_won": _json_safe(clutch.get("clutches_won")),
        "win_rate": _json_safe(clutch.get("win_rate")),
        "v1": {"won": _json_safe(clutch.get("v1_won")), "total": _json_safe(clutch.get("v1_total"))},
        "v2": {"won": _json_safe(clutch.get("v2_won")), "total": _json_safe(clutch.get("v2_total"))},
        "v3plus": {
            "won": _json_safe(clutch.get("v3plus_won")),
            "total": _json_safe(clutch.get("v3plus_total")),
        },
    }

    structured_tips = [
        {
            "severity": _json_safe(tip.get("severity")),
            "category": _json_safe(tip.get("category")),
            "title": _json_safe(tip.get("title")),
            "message": _json_safe(tip.get("message")),
            "metric": _json_safe(tip.get("metric")),
            "value": _json_safe(tip.get("value")),
            "examples": [_json_safe(example) for example in tip.get("evidence")]
            if isinstance(tip.get("evidence"), list)
            else [],
        }
        for tip in feedback
    ]

    _vod_type_map = {"strength": "positive"}
    structured_vod: list[dict[str, Any]] = []
    risk_by_round: dict[int, dict[str, Any]] = {}
    for rank, item in enumerate(vod, start=1):
        review_type = str(item.get("review_type") or "")
        reasons_raw = item.get("reasons")
        reasons = (
            [_json_safe(reason) for reason in reasons_raw] if isinstance(reasons_raw, list) else []
        )
        risk = _structured_risk_before_death(item)
        round_key = _safe_round_num(item.get("round_num"))
        if risk is not None and round_key > 0 and round_key not in risk_by_round:
            risk_by_round[round_key] = risk
        structured_vod.append(
            {
                "rank": rank,
                "round_num": _json_safe(item.get("round_num")),
                "side": _json_safe(item.get("side")),
                "severity": _json_safe(item.get("priority")),
                "type": _vod_type_map.get(review_type, review_type),
                "reasons": reasons,
                "summary": _json_safe(item.get("summary")),
                "risk_before_death": risk,
            }
        )

    structured_decisions: list[dict[str, Any]] = []
    for rank, sim in enumerate(decision_simulation, start=1):
        actual_decision = _safe_dict(sim.get("actual_decision"))
        round_key = _safe_round_num(sim.get("round_num"))
        alternatives_raw = sim.get("alternatives")
        alternatives: list[dict[str, Any]] = []
        if isinstance(alternatives_raw, list):
            for alt in alternatives_raw:
                if not isinstance(alt, dict):
                    continue
                alt_reasons = alt.get("reasons") if isinstance(alt.get("reasons"), list) else []
                alternatives.append(
                    {
                        "name": _json_safe(alt.get("label")),
                        "score": _json_safe(alt.get("score")),
                        "reason": _safe_text(alt_reasons[0], "") or None if alt_reasons else None,
                    }
                )
        structured_decisions.append(
            {
                "rank": rank,
                "round_num": _json_safe(sim.get("round_num")),
                "side": _json_safe(sim.get("side")),
                "actual": {
                    "summary": _json_safe(sim.get("actual_summary") or sim.get("original_summary")),
                    "score": _json_safe(actual_decision.get("score")),
                    "risk_before_death": risk_by_round.get(round_key),
                },
                "alternatives": alternatives,
            }
        )

    structured_coach_summary = {
        "main_weakness": _json_safe(coach_summary.get("main_weakness")),
        "best_strength": _json_safe(coach_summary.get("best_strength")),
        "top_vod_focus": _json_safe(coach_summary.get("top_vod_focus")),
        "decision_pattern": _json_safe(coach_summary.get("decision_pattern")),
        "practice_focus": _json_safe(coach_summary.get("practice_focus")),
    }

    return {
        "meta": _structured_meta(match, report_meta),
        "player": {
            "steamid": _json_safe(player.get("steamid")),
            "name": _json_safe(player.get("name")),
            "start_side": _json_safe(player.get("start_side")),
            "rounds_played": _json_safe(match.get("rounds_played")),
        },
        "overall": structured_overall,
        "impact": structured_impact,
        "side_breakdown": {
            "ct": _structured_side(ct_impact),
            "t": _structured_side(t_impact),
        },
        "economy": structured_economy,
        "clutch": structured_clutch,
        "benchmarks": _structured_benchmarks(benchmarks),
        "ml_impact": _structured_ml_impact(safe_report.get("ml_impact")),
        "tips": structured_tips,
        "vod_review_priority": structured_vod,
        "decision_simulation": structured_decisions,
        "coach_summary": structured_coach_summary,
    }


def validate_structured_report(report: dict[str, Any]) -> list[str]:
    """Validate the structured report shape. Returns a list of problems.

    An empty list means the report passed all checks.
    """
    problems: list[str] = []
    if not isinstance(report, dict):
        return ["structured_report is not a dict"]

    for key in _STRUCTURED_TOP_LEVEL_KEYS:
        if key not in report:
            problems.append(f"missing top-level key: {key}")

    player = report.get("player")
    if not isinstance(player, dict):
        problems.append("player is not a dict")
    else:
        if player.get("steamid") in (None, ""):
            problems.append("player.steamid is missing")
        if player.get("name") in (None, ""):
            problems.append("player.name is missing")

    if not isinstance(report.get("overall"), dict):
        problems.append("overall is not a dict")
    if not isinstance(report.get("tips"), list):
        problems.append("tips is not a list")
    if not isinstance(report.get("vod_review_priority"), list):
        problems.append("vod_review_priority is not a list")
    if not isinstance(report.get("decision_simulation"), list):
        problems.append("decision_simulation is not a list")
    if not isinstance(report.get("coach_summary"), dict):
        problems.append("coach_summary is not a dict")

    return problems


def _sanitize_filename_part(value: Any, fallback: str) -> str:
    """Reduce a value to a safe filename fragment.

    Keeps letters/digits/underscore/hyphen and replaces everything else with
    an underscore. Returns ``fallback`` when nothing usable remains.
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return fallback
    cleaned = re.sub(r"[^0-9A-Za-z_-]", "_", text)
    cleaned = cleaned.strip("_")
    return cleaned if cleaned else fallback


def write_structured_report_json(
    report: dict[str, Any],
    output_dir: "str | Path" = "data/reports",
) -> "Path":
    """Write the structured report to a pretty JSON file and return its Path.

    Builds the structured report from ``report`` if it is not already present.
    The filename is ``<match_id>_<steamid>_structured_report.json`` with unsafe
    characters sanitised and ``unknown_match`` / ``unknown_player`` fallbacks.
    After writing, the file is re-read and checked for the core top-level keys.
    """
    import json
    from pathlib import Path

    safe_report = _safe_dict(report)
    structured = safe_report.get("structured_report")
    if not isinstance(structured, dict):
        structured = build_structured_report(safe_report)

    meta = _safe_dict(structured.get("meta"))
    player = _safe_dict(structured.get("player"))
    match_part = _sanitize_filename_part(meta.get("match_id"), "unknown_match")
    steam_part = _sanitize_filename_part(player.get("steamid"), "unknown_player")
    filename = f"{match_part}_{steam_part}_structured_report.json"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename
    file_path.write_text(json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        missing = [key for key in ("meta", "player", "overall", "coach_summary") if key not in loaded]
        if missing:
            logging.getLogger(__name__).warning(
                "Structured report file missing keys after write | path=%s | missing=%s",
                file_path,
                ", ".join(missing),
            )
    except (OSError, ValueError) as error:
        logging.getLogger(__name__).warning(
            "Structured report re-read validation failed | path=%s | error=%s", file_path, error
        )

    return file_path


def build_match_report(analysis: dict[str, Any]) -> dict[str, Any]:
    safe_analysis = _safe_dict(analysis)

    selected_player_stats = _safe_dict(safe_analysis.get("selected_player_stats"))
    selected_impact_all = _safe_dict(safe_analysis.get("selected_player_impact"))
    if not selected_impact_all:
        selected_impact_all = _safe_dict(
            _safe_dict(safe_analysis.get("selected_player_impact_by_side")).get("ALL")
        )
    if not selected_impact_all:
        selected_impact_all = _safe_dict(
            _safe_dict(safe_analysis.get("round_timeline")).get("selected_player_impact")
        )
    economy_summary = _safe_dict(safe_analysis.get("economy_summary_selected"))
    clutch_summary = _safe_dict(safe_analysis.get("clutch_summary_selected"))

    benchmark_all = _safe_dict(safe_analysis.get("benchmark_evaluations_all"))
    benchmark_ct = _safe_dict(safe_analysis.get("benchmark_evaluations_ct"))
    benchmark_t = _safe_dict(safe_analysis.get("benchmark_evaluations_t"))
    if not benchmark_all:
        benchmark_all = _safe_dict(safe_analysis.get("benchmark_evaluations"))
    benchmark_meta = _safe_dict(safe_analysis.get("benchmark_evaluation_meta"))
    if not benchmark_meta:
        benchmark_meta = _safe_dict(safe_analysis.get("benchmark_evaluation_meta_all"))
    if not benchmark_meta or not isinstance(benchmark_meta.get("metric_details"), list):
        benchmark_meta = _benchmark_meta_from_evals(
            benchmark_all,
            map_name=safe_analysis.get("map_name"),
            side="ALL",
        )

    feedback_raw = safe_analysis.get("feedback")
    feedback = feedback_raw if isinstance(feedback_raw, list) else []
    player_ml_impact = safe_analysis.get("player_ml_impact")
    selected_timeline_events = _safe_rows(safe_analysis.get("selected_player_timeline_events"))
    selected_impact_by_side = _safe_dict(safe_analysis.get("selected_player_impact_by_side"))
    selected_impact_ct = _safe_dict(selected_impact_by_side.get("CT"))
    selected_impact_t = _safe_dict(selected_impact_by_side.get("T"))

    report = {
        "match": {
            "match_id": safe_analysis.get("match_id"),
            "map_name": safe_analysis.get("map_name"),
            "rounds_played": safe_analysis.get("rounds_played"),
            "benchmark_source": safe_analysis.get("benchmark_pool_source"),
            "benchmark_samples_before": safe_analysis.get("benchmark_pool_size_before_append"),
            "benchmark_samples_after": safe_analysis.get("benchmark_pool_size_after_append"),
        },
        "player": {
            "steamid": selected_player_stats.get("steamid"),
            "name": selected_player_stats.get("name"),
            "start_side": selected_player_stats.get("start_side"),
        },
        "overall": {
            "kills": selected_player_stats.get("kills"),
            "deaths": selected_player_stats.get("deaths"),
            "assists": selected_player_stats.get("assists"),
            "kpr": selected_player_stats.get("kpr"),
            "dpr": selected_player_stats.get("dpr"),
            "adr": selected_player_stats.get("adr"),
            "kast": selected_player_stats.get("kast"),
            "hs_kills": selected_player_stats.get("hs_kills"),
            "hs_percent": selected_player_stats.get("hs_percent"),
        },
        "impact": {
            "deaths": selected_impact_all.get("deaths"),
            "opening_kills": selected_impact_all.get("opening_kills"),
            "opening_deaths": selected_impact_all.get("opening_deaths"),
            "opening_duels": selected_impact_all.get("opening_duels"),
            "opening_duel_win_pct": selected_impact_all.get("opening_duel_win_pct"),
            "traded_deaths": selected_impact_all.get("traded_deaths"),
            "untraded_deaths": selected_impact_all.get("untraded_deaths"),
            "untraded_death_rate": selected_impact_all.get("untraded_death_rate"),
            "trade_kills": selected_impact_all.get("trade_kills"),
            "avg_damage_before_death": selected_impact_all.get("avg_damage_before_death"),
            "deaths_with_0_damage": selected_impact_all.get("deaths_with_0_damage"),
            "deaths_under_40_damage": selected_impact_all.get("deaths_under_40_damage"),
            "early_deaths": selected_impact_all.get("early_deaths"),
            "mid_deaths": selected_impact_all.get("mid_deaths"),
            "late_deaths": selected_impact_all.get("late_deaths"),
        },
        "side_breakdown": {
            "CT": {
                "impact": selected_impact_ct,
                "benchmarks": benchmark_ct,
            },
            "T": {
                "impact": selected_impact_t,
                "benchmarks": benchmark_t,
            },
        },
        "economy": {
            "full_buy_win_rate": economy_summary.get("full_buy_win_rate"),
            "force_win_rate": economy_summary.get("force_win_rate"),
            "eco_kills": economy_summary.get("eco_kills"),
            "broken_economy_rounds": economy_summary.get("broken_economy_rounds"),
            "save_rounds": economy_summary.get("save_rounds"),
        },
        "clutch": {
            "total_clutches": clutch_summary.get("total_clutches"),
            "clutches_won": clutch_summary.get("clutches_won"),
            "win_rate": clutch_summary.get("win_rate"),
            "v1_won": clutch_summary.get("v1_won"),
            "v1_total": clutch_summary.get("v1_total"),
            "v2_won": clutch_summary.get("v2_won"),
            "v2_total": clutch_summary.get("v2_total"),
            "v3plus_won": clutch_summary.get("v3plus_won"),
            "v3plus_total": clutch_summary.get("v3plus_total"),
        },
        "benchmarks": {
            "status": "unknown",
            "all": benchmark_all,
            "ct": benchmark_ct,
            "t": benchmark_t,
            "selected_context": benchmark_meta.get("selected_context"),
            "map_name": benchmark_meta.get("map_name"),
            "side": benchmark_meta.get("side"),
            "evaluated_metrics_count": benchmark_meta.get("evaluated_metrics_count"),
            "metric_details": benchmark_meta.get("metric_details"),
        },
        "feedback": feedback,
        "selected_player_timeline_events": selected_timeline_events,
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_version": 1,
        },
    }

    if isinstance(player_ml_impact, dict):
        report["ml_impact"] = player_ml_impact

    report["benchmarks"]["status"] = _benchmark_status(_safe_dict(report.get("benchmarks")))
    vod_review_priority = build_vod_review_priority(report)
    if vod_review_priority:
        report["vod_review_priority"] = vod_review_priority
        analysis["vod_review_priority"] = vod_review_priority
        decision_simulation = simulate_decisions(
            report_data=report,
            vod_review_priority=vod_review_priority,
            player_ml_impact=player_ml_impact if isinstance(player_ml_impact, dict) else None,
        )
        if decision_simulation:
            report["decision_simulation"] = decision_simulation
            analysis["decision_simulation"] = decision_simulation

    coach_summary = build_coach_summary(report)
    report["coach_summary"] = coach_summary
    analysis["coach_summary"] = coach_summary

    structured_report = build_structured_report(report)
    report["structured_report"] = structured_report
    analysis["structured_report"] = structured_report
    return report


def format_report_text(report: dict[str, Any]) -> str:
    safe_report = _safe_dict(report)
    match = _safe_dict(safe_report.get("match"))
    player = _safe_dict(safe_report.get("player"))
    overall = _safe_dict(safe_report.get("overall"))
    impact = _safe_dict(safe_report.get("impact"))
    economy = _safe_dict(safe_report.get("economy"))
    clutch = _safe_dict(safe_report.get("clutch"))
    side_breakdown = _safe_dict(safe_report.get("side_breakdown"))
    ct_side = _safe_dict(side_breakdown.get("CT"))
    t_side = _safe_dict(side_breakdown.get("T"))
    ct_impact = _safe_dict(ct_side.get("impact"))
    t_impact = _safe_dict(t_side.get("impact"))
    benchmarks = _safe_dict(safe_report.get("benchmarks"))
    feedback_raw = safe_report.get("feedback")
    feedback = feedback_raw if isinstance(feedback_raw, list) else []
    ml_impact_raw = safe_report.get("ml_impact")
    ml_impact = ml_impact_raw if isinstance(ml_impact_raw, dict) else None
    vod_review_priority_raw = safe_report.get("vod_review_priority")
    vod_review_priority = (
        [item for item in vod_review_priority_raw if isinstance(item, dict)]
        if isinstance(vod_review_priority_raw, list)
        else []
    )

    def _format_death_risk_line(item: dict[str, Any]) -> str | None:
        label = _safe_text(item.get("death_risk_label"), "")
        if label == "-":
            return None

        probability = item.get("death_risk_5s")
        bucket = _safe_text(item.get("death_risk_bucket"), "-")
        context = item.get("death_risk_context")
        context_dict = context if isinstance(context, dict) else {}
        nearest_enemy = _format_rounded_int(context_dict.get("nearest_enemy_distance"))
        nearest_teammate = _format_rounded_int(context_dict.get("nearest_teammate_distance"))
        player_hp = _format_rounded_int(context_dict.get("player_hp"))
        weapon = format_risk_weapon(context_dict, item)
        probability_text = f"{float(probability) * 100.0:.1f}%" if isinstance(probability, (int, float)) else "-"

        return (
            f"{label}, {probability_text}, {bucket}, "
            f"enemy {nearest_enemy}u, teammate {nearest_teammate}u, hp {player_hp}, weapon {weapon}"
        )

    lines: list[str] = []
    lines.append("CS2 COACH REPORT")
    lines.append(f"Map: {match.get('map_name') if match.get('map_name') is not None else '-'}")
    lines.append(f"Player: {player.get('name') if player.get('name') is not None else '-'}")
    lines.append(f"SteamID: {player.get('steamid') if player.get('steamid') is not None else '-'}")
    lines.append(f"Start side: {player.get('start_side') if player.get('start_side') is not None else '-'}")
    lines.append(f"Rounds: {_format_number(match.get('rounds_played'), digits=0)}")
    lines.append("")

    lines.append("OVERALL")
    lines.append(
        "K/D/A: "
        f"{_format_number(overall.get('kills'), digits=0)} / "
        f"{_format_number(overall.get('deaths'), digits=0)} / "
        f"{_format_number(overall.get('assists'), digits=0)}"
    )
    lines.append(f"KPR: {_format_number(overall.get('kpr'))}")
    lines.append(f"DPR: {_format_number(overall.get('dpr'))}")
    lines.append(f"ADR: {_format_number(overall.get('adr'))}")
    lines.append(f"KAST: {_format_pct(overall.get('kast'))}")
    lines.append(f"HS kills: {_format_number(overall.get('hs_kills'), digits=0)}")
    lines.append(f"HS: {_format_pct(overall.get('hs_percent'))}")
    lines.append("")

    lines.append("IMPACT")
    lines.append(
        "Untraded deaths: "
        f"{_format_number(impact.get('untraded_deaths'), digits=0)}/"
        f"{_format_number(impact.get('deaths'), digits=0)} "
        f"({_format_pct(impact.get('untraded_death_rate'))})"
    )
    lines.append(f"Opening duels: {_format_number(impact.get('opening_duels'), digits=0)}")
    lines.append(f"Opening duel win: {_format_pct(impact.get('opening_duel_win_pct'))}")
    lines.append(f"Trade kills: {_format_number(impact.get('trade_kills'), digits=0)}")
    lines.append(f"Avg dmg before death: {_format_number(impact.get('avg_damage_before_death'))}")
    lines.append(f"Zero-damage deaths: {_format_number(impact.get('deaths_with_0_damage'), digits=0)}")
    lines.append(f"Deaths under 40 dmg: {_format_number(impact.get('deaths_under_40_damage'), digits=0)}")
    lines.append(
        "Death timing (early/mid/late): "
        f"{_format_number(impact.get('early_deaths'), digits=0)} / "
        f"{_format_number(impact.get('mid_deaths'), digits=0)} / "
        f"{_format_number(impact.get('late_deaths'), digits=0)}"
    )
    lines.append("")

    lines.append("SIDE BREAKDOWN")
    lines.append("CT:")
    lines.append(
        "- Untraded deaths: "
        f"{_format_number(ct_impact.get('untraded_deaths'), digits=0)}/"
        f"{_format_number(ct_impact.get('deaths'), digits=0)} "
        f"({_format_pct(ct_impact.get('untraded_death_rate'))})"
    )
    lines.append(f"- Opening duel win: {_format_pct(ct_impact.get('opening_duel_win_pct'))}")
    lines.append(f"- Trade kills: {_format_number(ct_impact.get('trade_kills'), digits=0)}")
    lines.append(f"- Early deaths: {_format_number(ct_impact.get('early_deaths'), digits=0)}")
    lines.append("")
    lines.append("T:")
    lines.append(
        "- Untraded deaths: "
        f"{_format_number(t_impact.get('untraded_deaths'), digits=0)}/"
        f"{_format_number(t_impact.get('deaths'), digits=0)} "
        f"({_format_pct(t_impact.get('untraded_death_rate'))})"
    )
    lines.append(f"- Opening duel win: {_format_pct(t_impact.get('opening_duel_win_pct'))}")
    lines.append(f"- Trade kills: {_format_number(t_impact.get('trade_kills'), digits=0)}")
    lines.append(f"- Early deaths: {_format_number(t_impact.get('early_deaths'), digits=0)}")
    lines.append("")

    lines.append("ECONOMY")
    lines.append(f"Full-buy win rate: {_format_pct(economy.get('full_buy_win_rate'))}")
    lines.append(f"Force win rate: {_format_pct(economy.get('force_win_rate'))}")
    lines.append(f"Eco kills: {_format_number(economy.get('eco_kills'), digits=0)}")
    lines.append(f"Broken economy rounds: {_format_number(economy.get('broken_economy_rounds'), digits=0)}")
    lines.append(f"Save rounds: {_format_number(economy.get('save_rounds'), digits=0)}")
    lines.append("")

    lines.append("CLUTCH")
    lines.append(
        "Total/won: "
        f"{_format_number(clutch.get('total_clutches'), digits=0)} / "
        f"{_format_number(clutch.get('clutches_won'), digits=0)}"
    )
    lines.append(f"Win rate: {_format_pct(clutch.get('win_rate'))}")
    lines.append(
        "1v1 won/total: "
        f"{_format_number(clutch.get('v1_won'), digits=0)} / "
        f"{_format_number(clutch.get('v1_total'), digits=0)}"
    )
    lines.append(
        "1v2 won/total: "
        f"{_format_number(clutch.get('v2_won'), digits=0)} / "
        f"{_format_number(clutch.get('v2_total'), digits=0)}"
    )
    lines.append(
        "1v3+ won/total: "
        f"{_format_number(clutch.get('v3plus_won'), digits=0)} / "
        f"{_format_number(clutch.get('v3plus_total'), digits=0)}"
    )
    lines.append("")

    lines.append("BENCHMARKS")
    lines.append(f"Status: {benchmarks.get('status') if benchmarks.get('status') is not None else '-'}")
    if str(benchmarks.get("status")) == "available":
        context_used = benchmarks.get("selected_context")
        evaluated_metrics_count = benchmarks.get("evaluated_metrics_count")
        metric_details_raw = benchmarks.get("metric_details")
        metric_details = metric_details_raw if isinstance(metric_details_raw, list) else []

        lines.append(f"Context: {_safe_text(context_used)}")
        if evaluated_metrics_count is not None:
            lines.append(f"Metrics evaluated: {_format_number(evaluated_metrics_count, digits=0)}")

        if metric_details:
            lines.append("")
            lines.append("Benchmark pools:")
            for detail in metric_details:
                if not isinstance(detail, dict):
                    continue
                metric = _safe_text(detail.get("metric"))
                context = _safe_text(detail.get("context"))
                sample_size = _format_number(detail.get("sample_size"), digits=0)
                rating = str(detail.get("rating") or "unknown")
                reason = detail.get("reason")
                if rating != "unknown" and reason is None:
                    lines.append(f"- {metric}: {context}, samples={sample_size}")
                elif reason:
                    lines.append(f"- {metric}: unavailable, reason={reason}")
    lines.append("")

    if ml_impact is not None:
        kill_count = int(ml_impact.get("kill_count") or 0)
        death_count = int(ml_impact.get("death_count") or 0)
        has_events = kill_count > 0 or death_count > 0

        lines.append("ML IMPACT (EXPERIMENTAL)")
        if not has_events:
            lines.append("No normal-kill ML impact data found for this player.")
            lines.append(
                "Excluded contexts: "
                f"{_format_excluded_contexts(ml_impact.get('excluded_context_counts'))}"
            )
            lines.append("")
        else:
            lines.append(f"Kills: {_format_number(kill_count, digits=0)}")
            lines.append(f"Deaths: {_format_number(death_count, digits=0)}")
            lines.append(f"Net impact score: {_format_impact_score(ml_impact.get('net_ml_impact'))}")
            lines.append(f"Kill impact score: {_format_impact_score(ml_impact.get('total_kill_impact'))}")
            lines.append(f"Death impact score: {_format_impact_score(ml_impact.get('total_death_impact'))}")
            lines.append(f"Average kill impact: {_format_pp_per_event(ml_impact.get('avg_kill_impact'))}")
            lines.append(f"Average death impact: {_format_pp_per_event(ml_impact.get('avg_death_impact'))}")
            lines.append(
                "Excluded contexts: "
                f"{_format_excluded_contexts(ml_impact.get('excluded_context_counts'))}"
            )
            lines.append("")
            _append_ml_event_list(lines, "Best kills", ml_impact.get("best_kills"))
            lines.append("")
            _append_ml_event_list(lines, "Worst deaths", ml_impact.get("worst_deaths"))
            lines.append("")
            _append_ml_event_list(lines, "Low-impact kills", ml_impact.get("low_impact_kills"))
            lines.append("")

    lines.append("TOP TIPS")
    if feedback:
        for tip in feedback[:3]:
            if not isinstance(tip, dict):
                continue
            severity = str(tip.get("severity", "-")).upper()
            title = str(tip.get("title", "-"))
            message = str(tip.get("message", "-"))
            lines.append(f"[{severity}] {title}")
            lines.append(message)
            lines.extend(format_tip_evidence(tip))
            lines.append("")
        if lines and lines[-1] == "":
            lines.pop()
    else:
        lines.append("No feedback tips generated yet.")

    if vod_review_priority:
        lines.append("")
        lines.append("VOD REVIEW PRIORITY")
        for index, item in enumerate(vod_review_priority[:5], start=1):
            round_num = _format_number(item.get("round_num"), digits=0)
            side = _safe_text(item.get("side"))
            priority = _safe_text(item.get("priority")).lower()
            review_type = _safe_text(item.get("review_type")).lower()
            reasons_raw = item.get("reasons")
            reasons = [str(reason).strip() for reason in reasons_raw if str(reason).strip()] if isinstance(reasons_raw, list) else []
            summary = _safe_text(item.get("summary"))

            lines.append(f"{index}. Round {round_num} | {side} | {priority} | {review_type}")
            if reasons:
                lines.append(f"   Reasons: {', '.join(reasons[:3])}")
            risk_line = _format_death_risk_line(item)
            if risk_line:
                lines.append(f"   Risk before death: {risk_line}")
                # add a short explanation why model marked this as risky
                context = item.get("death_risk_context") if isinstance(item.get("death_risk_context"), dict) else {}
                dr_label = item.get("death_risk_label")
                explanation = build_death_risk_explanation(context, dr_label)
                if explanation:
                    lines.append(f"   Risk explanation: {explanation}")
            lines.append(f"   Summary: {summary}")

    decision_simulation = safe_report.get("decision_simulation")
    if isinstance(decision_simulation, list) and decision_simulation:
        lines.append("")
        lines.append("DECISION SIMULATION (MVP)")
        for idx, sim in enumerate(decision_simulation[:3], start=1):
            sim_round = _format_number(sim.get("round_num"), digits=0)
            sim_side = _safe_text(sim.get("side"))
            actual = sim.get("actual_decision") if isinstance(sim.get("actual_decision"), dict) else {}
            # prefer the simulation's rendered actual summary (fallback to original)
            summary = _safe_text(sim.get("actual_summary") or sim.get("original_summary"))
            actual_score = actual.get("score", 0)
            actual_score_str = f"{actual_score:+.2f}" if isinstance(actual_score, (int, float)) else "-"
            alternatives = sim.get("alternatives") if isinstance(sim.get("alternatives"), list) else []

            lines.append(f"{idx}. Round {sim_round} | {sim_side}")
            lines.append(f"   Actual: {summary} | score {actual_score_str}")
            if alternatives:
                lines.append("   Better alternatives:")
                for alt in alternatives[:3]:
                    alt_label = _safe_text(alt.get("label"))
                    alt_score = alt.get("score", 0)
                    alt_score_str = f"{alt_score:+.2f}" if isinstance(alt_score, (int, float)) else "-"
                    alt_reasons = alt.get("reasons") if isinstance(alt.get("reasons"), list) else []
                    alt_reason = str(alt_reasons[0]).strip() if alt_reasons else ""
                    lines.append(f"   - {alt_label} | score {alt_score_str} | {alt_reason}")

    coach_summary_raw = safe_report.get("coach_summary")
    coach_summary = (
        coach_summary_raw if isinstance(coach_summary_raw, dict) else build_coach_summary(safe_report)
    )
    top_vod_focus = coach_summary.get("top_vod_focus")
    decision_pattern = coach_summary.get("decision_pattern")

    lines.append("")
    lines.append("COACH SUMMARY")
    lines.append("Main weakness:")
    lines.append(f"- {_safe_text(coach_summary.get('main_weakness'))}")
    lines.append("")
    lines.append("Best strength:")
    lines.append(f"- {_safe_text(coach_summary.get('best_strength'))}")
    lines.append("")
    lines.append("Top VOD focus:")
    lines.append(f"- {_safe_text(top_vod_focus) if top_vod_focus else 'No VOD priority rounds available.'}")
    lines.append("")
    lines.append("Decision pattern:")
    lines.append(
        f"- {_safe_text(decision_pattern) if decision_pattern else 'No repeated decision pattern detected.'}"
    )
    lines.append("")
    lines.append("Practice focus:")
    lines.append(f"- {_safe_text(coach_summary.get('practice_focus'))}")

    return "\n".join(lines)
