from __future__ import annotations

from datetime import datetime, timezone
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

    return "\n".join(lines)
