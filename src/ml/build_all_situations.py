from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Analyser import _build_benchmark_player_rows, _selected_impact_rows, _selected_rows, _summary_row_for_player
from Parser import load_cached_demo
from benchmarks import evaluate_player, load_benchmark_samples, make_contextual_match_samples
from coach_metrics import build_raw_overall_stats
from report_builder import build_match_report
from sectors.clutch import build_clutch_stats
from sectors.economy import build_economy_stats
from sectors.feedback import generate_feedback
from sectors.overall import build_overall_table
from sectors.player_ml_impact import build_player_ml_impact_summary
from sectors.round_timeline import build_round_timeline_stats
from sectors.situation_builder import build_player_situations, save_situations_json, save_situations_parquet
from ml.dataset import DEFAULT_CACHE_DIR, DEFAULT_CACHE_KEY_PATH, discover_cache_keys


LOGGER = logging.getLogger(__name__)
REPO_ROOT = SRC_DIR.parent
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "situations" / "all_situations.parquet"
DEFAULT_ML_EVENT_IMPACT_PATH = REPO_ROOT / "data" / "ml" / "ml_event_impact.parquet"
BUILD_VERSION = "situations_v1"
CREATED_FROM = "batch_cached_demos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one normalized situations dataset across all cached demos and players.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Cache directory to scan (default: repo .cache directory).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Parquet output path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write a JSON copy next to the parquet output.",
    )
    parser.add_argument(
        "--limit-matches",
        type=int,
        default=None,
        help="Optional cap on cached matches to process.",
    )
    parser.add_argument(
        "--limit-players",
        type=int,
        default=None,
        help="Optional cap on players processed per match.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue after match/player errors (default: true).",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def _safe_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, pl.DataFrame):
        return value.height
    try:
        return len(value)
    except TypeError:
        return 0


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_match_sort(value: Any) -> tuple[int, str]:
    if value is None:
        return (1, "")
    return (0, str(value))


def _json_output_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".json") if parquet_path.suffix else parquet_path.parent / f"{parquet_path.name}.json"


def _load_ml_event_impact() -> pl.DataFrame | None:
    if not DEFAULT_ML_EVENT_IMPACT_PATH.exists():
        LOGGER.warning("ML event impact parquet not found: %s", DEFAULT_ML_EVENT_IMPACT_PATH)
        return None
    try:
        return pl.read_parquet(DEFAULT_ML_EVENT_IMPACT_PATH)
    except Exception:
        LOGGER.exception("Failed to load ML event impact parquet from %s", DEFAULT_ML_EVENT_IMPACT_PATH)
        return None


def _match_ml_event_impact(ml_event_impact: pl.DataFrame | None, match_id: str) -> pl.DataFrame | None:
    if ml_event_impact is None or ml_event_impact.is_empty():
        return None
    if "match_id" not in ml_event_impact.columns:
        return ml_event_impact
    try:
        return ml_event_impact.filter(pl.col("match_id").cast(pl.Utf8) == str(match_id))
    except Exception:
        LOGGER.exception("Failed to filter ML event impact for match %s", match_id)
        return None


def _player_ml_impact_summary(match_ml_impact: pl.DataFrame | None, steamid: Any) -> dict[str, Any] | None:
    if match_ml_impact is None or match_ml_impact.is_empty() or steamid is None:
        return None
    try:
        return build_player_ml_impact_summary(match_ml_impact, selected_steamid=steamid, top_n=5)
    except Exception:
        LOGGER.exception("Failed to build ML impact summary for player %s", steamid)
        return None


def _select_player_rows(overall: pl.DataFrame, limit_players: int | None) -> list[dict[str, Any]]:
    if overall.is_empty():
        return []
    player_rows = overall.to_dicts()
    if limit_players is not None:
        return player_rows[: max(limit_players, 0)]
    return player_rows


def _build_match_context(
    cache_key: str,
    cache_dir: str,
    historical_samples: list[dict[str, Any]],
    ml_event_impact: pl.DataFrame | None,
) -> dict[str, Any]:
    demo = load_cached_demo(cache_key, cache_dir=cache_dir)
    header = getattr(demo, "header", None)
    map_name = header.get("map_name") if isinstance(header, dict) else None
    rounds_played = _safe_len(getattr(demo, "rounds", None))

    overall = build_overall_table(build_raw_overall_stats(demo))
    economy_stats = build_economy_stats(demo)
    clutch_stats = build_clutch_stats(demo)
    round_timeline = build_round_timeline_stats(demo)

    side_rows = _build_benchmark_player_rows(demo)
    ct_rows = [row for row in side_rows if str(row.get("side", "")).upper() == "CT"]
    t_rows = [row for row in side_rows if str(row.get("side", "")).upper() == "T"]
    match_samples = make_contextual_match_samples(
        match_id=cache_key,
        map_name=map_name,
        round_count=rounds_played if rounds_played > 0 else None,
        ct_player_stats=ct_rows,
        t_player_stats=t_rows,
    )
    evaluation_samples = [
        sample for sample in historical_samples if str(sample.get("match_id")) != str(cache_key)
    ]

    return {
        "match_id": cache_key,
        "map_name": map_name,
        "rounds_played": rounds_played,
        "overall": overall,
        "economy_summary": economy_stats.get("economy_summary", pl.DataFrame()),
        "clutch_summary": clutch_stats.get("clutch_summary", pl.DataFrame()),
        "clutch_rounds": clutch_stats.get("clutch_rounds", pl.DataFrame()),
        "round_timeline": round_timeline,
        "timeline_events": round_timeline.get("timeline_events", pl.DataFrame()),
        "player_impact_summary": round_timeline.get("player_impact_summary", pl.DataFrame()),
        "side_rows": side_rows,
        "match_samples": match_samples,
        "evaluation_samples": evaluation_samples,
        "match_ml_event_impact": _match_ml_event_impact(ml_event_impact, cache_key),
    }


def _selected_match_sample(
    match_samples: list[dict[str, Any]],
    steamid: Any,
    side: str,
) -> dict[str, Any]:
    selected_steamid = str(steamid)
    side_upper = side.upper()
    return next(
        (
            sample
            for sample in match_samples
            if str(sample.get("steamid")) == selected_steamid
            and str(sample.get("side", "ALL")).upper() == side_upper
        ),
        {},
    )


def _selected_side_row(
    side_rows: list[dict[str, Any]],
    steamid: Any,
    side: str,
) -> dict[str, Any]:
    selected_steamid = str(steamid)
    side_upper = side.upper()
    return next(
        (
            row
            for row in side_rows
            if str(row.get("steamid")) == selected_steamid and str(row.get("side", "")).upper() == side_upper
        ),
        {},
    )


def _evaluate_for_side(match_context: dict[str, Any], steamid: Any, side: str) -> dict[str, Any]:
    side_upper = side.upper()
    sample = _selected_match_sample(match_context["match_samples"], steamid, side_upper)
    metrics = sample.get("metrics") if isinstance(sample.get("metrics"), dict) else {}
    counts = sample.get("counts") if isinstance(sample.get("counts"), dict) else {}
    rounds = sample.get("rounds_played")

    if not metrics and side_upper != "ALL":
        row = _selected_side_row(match_context["side_rows"], steamid, side_upper)
        metrics = {
            "adr": row.get("adr"),
            "kast": row.get("kast"),
            "hs_percent": row.get("hs_percent"),
            "kpr": row.get("kpr"),
            "opening_duel_win_pct": row.get("opening_duel_win_pct"),
            "full_buy_win_rate": row.get("full_buy_win_rate"),
            "force_win_rate": row.get("force_win_rate"),
            "clutch_win_rate": row.get("clutch_win_rate"),
        }
        counts = {
            "kills": row.get("kills"),
            "opening_duels": row.get("opening_duels"),
            "full_buy_rounds": row.get("full_buy_rounds"),
            "force_rounds": row.get("force_rounds"),
            "clutch_attempts": row.get("clutch_attempts"),
        }
        rounds = row.get("rounds_played")

    return evaluate_player(
        metrics,
        match_context["evaluation_samples"],
        map_name=match_context["map_name"],
        side=side_upper if side_upper in {"ALL", "CT", "T"} else "ALL",
        player_counts=counts,
        rounds_played=rounds,
    )


def _sanitize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, (pl.DataFrame, pl.Series, SimpleNamespace)):
        return None
    return None


def _sanitize_situation(
    situation: dict[str, Any],
    *,
    source_match_index: int,
    source_player_index: int,
) -> dict[str, Any]:
    enriched = dict(situation)
    enriched.setdefault("source_match_index", source_match_index)
    enriched.setdefault("source_player_index", source_player_index)
    enriched.setdefault("build_version", BUILD_VERSION)
    enriched.setdefault("created_from", CREATED_FROM)
    return {str(key): _sanitize_value(value) for key, value in enriched.items()}


def _sort_situations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            _safe_match_sort(row.get("match_id")),
            _safe_int(row.get("source_match_index")) or 0,
            _safe_int(row.get("source_player_index")) or 0,
            _safe_int(row.get("round_num")) or 0,
            _safe_int(row.get("tick")) if row.get("tick") is not None else -1,
            str(row.get("situation_type") or ""),
            str(row.get("steamid") or ""),
            str(row.get("situation_id") or ""),
        ),
    )


def _dedupe_situations_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    situations_before_dedupe = len(rows)
    duplicate_ids = {
        situation_id
        for situation_id, count in Counter(str(row.get("situation_id") or "") for row in rows).items()
        if situation_id and count > 1
    }
    if not duplicate_ids:
        return rows

    deduped_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        situation_id = str(row.get("situation_id") or "")
        if situation_id:
            if situation_id in seen_ids:
                continue
            seen_ids.add(situation_id)
        deduped_rows.append(row)

    LOGGER.info("situations_before_dedupe=%d", situations_before_dedupe)
    LOGGER.info("duplicate_situation_id_count=%d", len(duplicate_ids))
    LOGGER.info("situations_after_dedupe=%d", len(deduped_rows))
    return deduped_rows


def _log_summary(
    rows: list[dict[str, Any]],
    *,
    cached_demos_found: int,
    matches_processed: int,
    players_processed: int,
    errors: list[dict[str, Any]],
    output_path: Path,
) -> None:
    LOGGER.info("cached demos found=%d", cached_demos_found)
    LOGGER.info("matches processed=%d", matches_processed)
    LOGGER.info("players processed=%d", players_processed)
    LOGGER.info("situations_total=%d", len(rows))

    type_counts = Counter(str(row.get("situation_type") or "None") for row in rows)
    for situation_type, count in sorted(type_counts.items()):
        LOGGER.info("situation_type=%s count=%d", situation_type, count)

    map_counts = Counter(str(row.get("map_name")) if row.get("map_name") is not None else "None" for row in rows)
    for map_name, count in sorted(map_counts.items()):
        LOGGER.info("map_name=%s count=%d", map_name, count)

    source_flag_counts = Counter()
    for row in rows:
        for flag in row.get("source_flags") or []:
            source_flag_counts[str(flag)] += 1
    for flag, count in sorted(source_flag_counts.items()):
        LOGGER.info("source_flag=%s count=%d", flag, count)

    LOGGER.info("high_impact_kill_count=%d", sum(1 for row in rows if row.get("high_impact_kill") is True))
    LOGGER.info("low_impact_kill_count=%d", sum(1 for row in rows if row.get("low_impact_kill") is True))
    LOGGER.info("high_cost_death_count=%d", sum(1 for row in rows if row.get("high_cost_death") is True))
    LOGGER.info("low_cost_death_count=%d", sum(1 for row in rows if row.get("low_cost_death") is True))
    LOGGER.info("errors count=%d", len(errors))
    LOGGER.info("output path=%s", output_path)


def _record_error(
    errors: list[dict[str, Any]],
    *,
    cache_key: str,
    player: dict[str, Any] | None,
    exc: Exception,
) -> None:
    player_name = player.get("name") if isinstance(player, dict) else None
    player_steamid = player.get("steamid") if isinstance(player, dict) else None
    errors.append(
        {
            "match_id": cache_key,
            "player_name": player_name,
            "player_steamid": player_steamid,
            "error": str(exc),
        }
    )
    if isinstance(player, dict):
        LOGGER.exception(
            "Failed processing player | match_id=%s | player=%s | steamid=%s",
            cache_key,
            player_name,
            player_steamid,
        )
    else:
        LOGGER.exception("Failed processing match | match_id=%s", cache_key)


def _build_player_analysis(match_context: dict[str, Any], player_row: dict[str, Any]) -> dict[str, Any]:
    steamid = player_row.get("steamid")
    selected_player = match_context["overall"].filter(pl.col("steamid") == steamid).head(1)
    selected_player_stats = selected_player.to_dicts()[0] if not selected_player.is_empty() else dict(player_row)

    selected_impact = _selected_impact_rows(match_context["player_impact_summary"], steamid)
    selected_timeline_events = _selected_rows(match_context["timeline_events"], steamid)
    selected_clutch_rounds = _selected_rows(match_context["clutch_rounds"], steamid)
    economy_summary_row = _summary_row_for_player(match_context["economy_summary"], steamid)
    clutch_summary_row = _summary_row_for_player(match_context["clutch_summary"], steamid)
    player_ml_impact = _player_ml_impact_summary(match_context["match_ml_event_impact"], steamid)

    benchmark_evaluations_all = _evaluate_for_side(match_context, steamid, "ALL")
    benchmark_evaluations_ct = _evaluate_for_side(match_context, steamid, "CT")
    benchmark_evaluations_t = _evaluate_for_side(match_context, steamid, "T")

    feedback = generate_feedback(
        {
            "overall_stats": selected_player_stats,
            "economy_summary": economy_summary_row,
            "clutch_summary": clutch_summary_row,
            "benchmark_evaluations": benchmark_evaluations_all,
            "impact_summary": selected_impact,
            "selected_player_impact": selected_impact.get("ALL", {}),
            "selected_player_timeline_events": selected_timeline_events,
            "selected_player_clutch_rounds": selected_clutch_rounds,
            "player_ml_impact": player_ml_impact,
        }
    )

    return {
        "match_id": match_context["match_id"],
        "map_name": match_context["map_name"],
        "rounds_played": match_context["rounds_played"],
        "overall": match_context["overall"],
        "selected_player": selected_player,
        "selected_player_stats": selected_player_stats,
        "round_timeline": match_context["round_timeline"],
        "economy_summary_selected": economy_summary_row,
        "clutch_summary_selected": clutch_summary_row,
        "benchmark_evaluations": benchmark_evaluations_all,
        "benchmark_evaluations_all": benchmark_evaluations_all,
        "benchmark_evaluations_ct": benchmark_evaluations_ct,
        "benchmark_evaluations_t": benchmark_evaluations_t,
        "benchmark_pool_source": "historical",
        "benchmark_pool_size_before_append": len(match_context["evaluation_samples"]),
        "benchmark_pool_size_after_append": len(match_context["evaluation_samples"]),
        "benchmark_samples_appended": False,
        "benchmark_match_already_analyzed": True,
        "selected_player_impact": selected_impact.get("ALL", {}),
        "selected_player_impact_by_side": selected_impact,
        "selected_player_timeline_events": selected_timeline_events,
        "selected_player_clutch_rounds": selected_clutch_rounds,
        "player_ml_impact": player_ml_impact,
        "feedback": feedback,
    }


def build_all_situations(
    cache_keys: list[str],
    *,
    cache_dir: str,
    limit_players: int | None,
    continue_on_error: bool,
) -> tuple[list[dict[str, Any]], int, int, list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    matches_processed = 0
    players_processed = 0
    historical_samples = load_benchmark_samples()
    ml_event_impact = _load_ml_event_impact()

    for match_index, cache_key in enumerate(cache_keys):
        try:
            match_context = _build_match_context(
                cache_key=cache_key,
                cache_dir=cache_dir,
                historical_samples=historical_samples,
                ml_event_impact=ml_event_impact,
            )
        except Exception as exc:
            _record_error(errors, cache_key=cache_key, player=None, exc=exc)
            if not continue_on_error:
                raise
            continue

        matches_processed += 1
        player_rows = _select_player_rows(match_context["overall"], limit_players)

        for player_index, player_row in enumerate(player_rows):
            try:
                analysis = _build_player_analysis(match_context, player_row)
                build_match_report(analysis)
                situations = build_player_situations(analysis)
                sanitized = [
                    _sanitize_situation(
                        situation,
                        source_match_index=match_index,
                        source_player_index=player_index,
                    )
                    for situation in situations
                ]
                all_rows.extend(sanitized)
                players_processed += 1
            except Exception as exc:
                _record_error(errors, cache_key=cache_key, player=player_row, exc=exc)
                if not continue_on_error:
                    raise

    return _sort_situations(all_rows), matches_processed, players_processed, errors


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.limit_matches is not None and args.limit_matches < 0:
        raise SystemExit("--limit-matches must be >= 0.")
    if args.limit_players is not None and args.limit_players < 0:
        raise SystemExit("--limit-players must be >= 0.")

    output_path = Path(args.output)
    json_path = _json_output_path(output_path)
    if output_path.exists() and not args.force:
        raise SystemExit(f"Output already exists: {output_path}. Use --force to overwrite.")
    if args.json and json_path.exists() and not args.force:
        raise SystemExit(f"JSON output already exists: {json_path}. Use --force to overwrite.")

    cache_dir = Path(args.cache_dir)
    cache_keys_all = [
        cache_key
        for cache_key in discover_cache_keys(cache_dir=cache_dir, cache_key_path=DEFAULT_CACHE_KEY_PATH)
        if (cache_dir / f"{cache_key}.pkl").exists()
    ]
    if args.limit_matches is not None:
        cache_keys = cache_keys_all[: max(args.limit_matches, 0)]
    else:
        cache_keys = cache_keys_all

    rows, matches_processed, players_processed, errors = build_all_situations(
        cache_keys,
        cache_dir=str(cache_dir),
        limit_players=args.limit_players,
        continue_on_error=args.continue_on_error,
    )
    rows = _dedupe_situations_by_id(rows)

    save_situations_parquet(rows, output_path)
    if args.json:
        save_situations_json(rows, json_path)

    _log_summary(
        rows,
        cached_demos_found=len(cache_keys_all),
        matches_processed=matches_processed,
        players_processed=players_processed,
        errors=errors,
        output_path=output_path,
    )
    if args.json:
        LOGGER.info("json output path=%s", json_path)


if __name__ == "__main__":
    main()
