from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


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
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_version": 1,
        },
    }

    if isinstance(player_ml_impact, dict):
        report["ml_impact"] = player_ml_impact

    report["benchmarks"]["status"] = _benchmark_status(_safe_dict(report.get("benchmarks")))
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

    return "\n".join(lines)
