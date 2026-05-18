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

    feedback_raw = safe_analysis.get("feedback")
    feedback = feedback_raw if isinstance(feedback_raw, list) else []

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
        },
        "feedback": feedback,
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_version": 1,
        },
    }

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
    benchmarks = _safe_dict(safe_report.get("benchmarks"))
    feedback_raw = safe_report.get("feedback")
    feedback = feedback_raw if isinstance(feedback_raw, list) else []

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
            lines.append("")
        if lines and lines[-1] == "":
            lines.pop()
    else:
        lines.append("No feedback tips generated yet.")

    return "\n".join(lines)
