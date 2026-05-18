"""Local benchmark storage and percentile evaluation for CS2 coaching stats."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BENCHMARKS_PATH = "src/benchmarks/local_benchmarks.json"
DEFAULT_ANALYZED_MATCHES_PATH = "src/benchmarks/analyzed_matches.json"
SCHEMA_VERSION = 1
MIN_BENCHMARK_POOL_SIZE = 20

KNOWN_METRICS = {
    "adr",
    "kast",
    "hs_percent",
    "kpr",
    "opening_duel_win_pct",
    "full_buy_win_rate",
    "force_win_rate",
    "clutch_win_rate",
}

KNOWN_COUNTS = {
    "kills",
    "deaths",
    "assists",
    "headshot_kills",
    "full_buy_rounds",
    "full_buy_wins",
    "force_rounds",
    "force_wins",
    "clutch_attempts",
    "clutches_won",
    "opening_duels",
    "opening_duels_won",
}

MIN_DENOMINATORS: dict[str, tuple[str, int]] = {
    "full_buy_win_rate": ("full_buy_rounds", 5),
    "force_win_rate": ("force_rounds", 3),
    "clutch_win_rate": ("clutch_attempts", 3),
    "opening_duel_win_pct": ("opening_duels", 3),
    "hs_percent": ("kills", 5),
    "kpr": ("rounds_played", 5),
    "adr": ("rounds_played", 5),
    "kast": ("rounds_played", 5),
}


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if path == DEFAULT_BENCHMARKS_PATH:
        return Path(__file__).resolve().parent / "benchmarks" / "local_benchmarks.json"
    if path == DEFAULT_ANALYZED_MATCHES_PATH:
        return Path(__file__).resolve().parent / "benchmarks" / "analyzed_matches.json"
    return p


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not (isinstance(value, float) and math.isnan(value))
    return False


def _to_float_or_none(value: Any) -> float | None:
    return float(value) if _is_numeric(value) else None


def _to_int_or_none(value: Any) -> int | None:
    if not _is_numeric(value):
        return None
    return int(value)


def _match_balance(round_count: int | None) -> str:
    if round_count is None:
        return "unknown"
    if round_count >= 28:
        return "close"
    if round_count <= 18:
        return "stomp"
    return "normal"


def load_benchmark_samples(path: str = DEFAULT_BENCHMARKS_PATH) -> list[dict]:
    try:
        p = _resolve_path(path)
        if not p.exists():
            return []
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []
        return [row for row in payload if isinstance(row, dict)]
    except Exception:
        return []


def save_benchmark_samples(samples: list[dict], path: str = DEFAULT_BENCHMARKS_PATH) -> None:
    p = _resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


def load_analyzed_matches(path: str = DEFAULT_ANALYZED_MATCHES_PATH) -> dict:
    try:
        p = _resolve_path(path)
        if not p.exists():
            return {"matches": {}}
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"matches": {}}
        matches = payload.get("matches")
        if not isinstance(matches, dict):
            payload["matches"] = {}
        return payload
    except Exception:
        return {"matches": {}}


def save_analyzed_matches(registry: dict, path: str = DEFAULT_ANALYZED_MATCHES_PATH) -> None:
    p = _resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(registry.get("matches"), dict):
        registry["matches"] = {}
    p.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def is_match_analyzed(match_id: str, registry: dict) -> bool:
    matches = registry.get("matches") if isinstance(registry, dict) else None
    return isinstance(matches, dict) and str(match_id) in matches


def mark_match_analyzed(
    match_id: str,
    metadata: dict,
    path: str = DEFAULT_ANALYZED_MATCHES_PATH,
) -> None:
    registry = load_analyzed_matches(path=path)
    matches = registry.setdefault("matches", {})
    if not isinstance(matches, dict):
        registry["matches"] = {}
        matches = registry["matches"]
    entry = dict(metadata)
    entry["analyzed_at"] = datetime.now(timezone.utc).isoformat()
    matches[str(match_id)] = entry
    save_analyzed_matches(registry, path=path)


def make_match_samples(
    match_id: str,
    map_name: str | None,
    round_count: int | None,
    player_stats: list[dict],
    side: str = "ALL",
) -> list[dict]:
    samples: list[dict] = []
    for player in player_stats:
        if not isinstance(player, dict):
            continue
        steamid = player.get("steamid")
        if steamid is None:
            continue

        metrics: dict[str, float] = {}
        for metric in KNOWN_METRICS:
            value = _to_float_or_none(player.get(metric))
            if value is not None:
                metrics[metric] = value

        counts: dict[str, int] = {}
        for count_name in KNOWN_COUNTS:
            count_value = _to_int_or_none(player.get(count_name))
            if count_value is not None:
                counts[count_name] = count_value

        player_side = str(player.get("side", side) or "ALL").upper()
        rounds_played = _to_int_or_none(player.get("rounds_played"))

        samples.append(
            {
                "schema_version": SCHEMA_VERSION,
                "match_id": str(match_id),
                "map_name": map_name,
                "side": player_side,
                "round_count": _to_int_or_none(round_count),
                "rounds_played": rounds_played,
                "match_balance": _match_balance(_to_int_or_none(round_count)),
                "steamid": str(steamid),
                "name": player.get("name"),
                "metrics": metrics,
                "counts": counts,
            }
        )
    return samples


def _ratio_pct(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return round((numerator / denominator) * 100.0, 2)


def _weighted_by_rounds(
    ct_value: float | None,
    ct_rounds: int | None,
    t_value: float | None,
    t_rounds: int | None,
) -> float | None:
    left_rounds = ct_rounds or 0
    right_rounds = t_rounds or 0
    total_rounds = left_rounds + right_rounds
    if total_rounds <= 0:
        return None
    weighted_sum = 0.0
    has_component = False
    if ct_value is not None and left_rounds > 0:
        weighted_sum += ct_value * left_rounds
        has_component = True
    if t_value is not None and right_rounds > 0:
        weighted_sum += t_value * right_rounds
        has_component = True
    if not has_component:
        return None
    return round(weighted_sum / total_rounds, 2)


def make_all_side_samples(
    match_id: str,
    map_name: str | None,
    round_count: int | None,
    ct_player_stats: list[dict],
    t_player_stats: list[dict],
) -> list[dict]:
    by_id: dict[str, dict[str, dict]] = {}
    for row in ct_player_stats:
        if isinstance(row, dict) and row.get("steamid") is not None:
            by_id.setdefault(str(row["steamid"]), {})["CT"] = row
    for row in t_player_stats:
        if isinstance(row, dict) and row.get("steamid") is not None:
            by_id.setdefault(str(row["steamid"]), {})["T"] = row

    merged_rows: list[dict] = []
    for steamid, sides in by_id.items():
        ct = sides.get("CT", {})
        t = sides.get("T", {})
        ct_rounds = _to_int_or_none(ct.get("rounds_played"))
        t_rounds = _to_int_or_none(t.get("rounds_played"))
        total_rounds = (ct_rounds or 0) + (t_rounds or 0)

        counts: dict[str, int] = {}
        for count_name in KNOWN_COUNTS:
            summed = (_to_int_or_none(ct.get(count_name)) or 0) + (_to_int_or_none(t.get(count_name)) or 0)
            if summed > 0:
                counts[count_name] = summed

        metrics: dict[str, float] = {}
        kills = counts.get("kills")
        hs_kills = counts.get("headshot_kills")
        full_buy_rounds = counts.get("full_buy_rounds")
        full_buy_wins = counts.get("full_buy_wins")
        force_rounds = counts.get("force_rounds")
        force_wins = counts.get("force_wins")
        clutch_attempts = counts.get("clutch_attempts")
        clutches_won = counts.get("clutches_won")
        opening_duels = counts.get("opening_duels")
        opening_duels_won = counts.get("opening_duels_won")

        if total_rounds > 0 and kills is not None:
            metrics["kpr"] = round(kills / total_rounds, 2)
        hs_percent = _ratio_pct(hs_kills, kills)
        if hs_percent is not None:
            metrics["hs_percent"] = hs_percent
        full_buy_wr = _ratio_pct(full_buy_wins, full_buy_rounds)
        if full_buy_wr is not None:
            metrics["full_buy_win_rate"] = full_buy_wr
        force_wr = _ratio_pct(force_wins, force_rounds)
        if force_wr is not None:
            metrics["force_win_rate"] = force_wr
        clutch_wr = _ratio_pct(clutches_won, clutch_attempts)
        if clutch_wr is not None:
            metrics["clutch_win_rate"] = clutch_wr
        opening_wr = _ratio_pct(opening_duels_won, opening_duels)
        if opening_wr is not None:
            metrics["opening_duel_win_pct"] = opening_wr

        ct_adr = _to_float_or_none(ct.get("adr"))
        t_adr = _to_float_or_none(t.get("adr"))
        combined_adr = _weighted_by_rounds(ct_adr, ct_rounds, t_adr, t_rounds)
        if combined_adr is not None:
            metrics["adr"] = combined_adr

        ct_kast = _to_float_or_none(ct.get("kast"))
        t_kast = _to_float_or_none(t.get("kast"))
        combined_kast = _weighted_by_rounds(ct_kast, ct_rounds, t_kast, t_rounds)
        if combined_kast is not None:
            metrics["kast"] = combined_kast

        merged: dict[str, Any] = {
            "steamid": steamid,
            "name": ct.get("name") or t.get("name"),
            "side": "ALL",
            "rounds_played": total_rounds if total_rounds > 0 else None,
        }
        merged.update(metrics)
        merged.update(counts)
        merged_rows.append(merged)

    return make_match_samples(
        match_id=match_id,
        map_name=map_name,
        round_count=round_count,
        player_stats=merged_rows,
        side="ALL",
    )


def make_contextual_match_samples(
    match_id: str,
    map_name: str | None,
    round_count: int | None,
    ct_player_stats: list[dict],
    t_player_stats: list[dict],
) -> list[dict]:
    ct_samples = make_match_samples(match_id, map_name, round_count, ct_player_stats, side="CT")
    t_samples = make_match_samples(match_id, map_name, round_count, t_player_stats, side="T")
    all_samples = make_all_side_samples(match_id, map_name, round_count, ct_player_stats, t_player_stats)
    return ct_samples + t_samples + all_samples


def append_match_samples(
    new_samples: list[dict],
    path: str = DEFAULT_BENCHMARKS_PATH,
    deduplicate: bool = True,
) -> list[dict]:
    all_samples = load_benchmark_samples(path=path)
    all_samples.extend(new_samples)

    if deduplicate:
        unique: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for sample in reversed(all_samples):
            if not isinstance(sample, dict):
                continue
            key = (
                str(sample.get("match_id", "")),
                str(sample.get("steamid", "")),
                str(sample.get("side", "ALL")).upper(),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(sample)
        all_samples = list(reversed(unique))

    save_benchmark_samples(all_samples, path=path)
    return all_samples


def _sample_count(sample: dict, count_name: str) -> int | None:
    if count_name == "rounds_played":
        return _to_int_or_none(sample.get("rounds_played"))
    counts = sample.get("counts")
    if not isinstance(counts, dict):
        return None
    return _to_int_or_none(counts.get(count_name))


def _selected_count(selected_counts: dict | None, selected_rounds_played: int | None, count_name: str) -> int | None:
    if count_name == "rounds_played":
        return _to_int_or_none(selected_rounds_played)
    if not isinstance(selected_counts, dict):
        return None
    return _to_int_or_none(selected_counts.get(count_name))


def _sample_has_reliable_denominator(sample: dict, metric: str) -> bool:
    requirement = MIN_DENOMINATORS.get(metric)
    if requirement is None:
        return True
    denom_name, min_required = requirement
    denom_value = _sample_count(sample, denom_name)
    if denom_value is None:
        return False
    return denom_value >= min_required


def percentile_rank(value: float, population: list[float]) -> float | None:
    clean = [float(v) for v in population if _is_numeric(v)]
    if not clean:
        return None
    le_count = sum(1 for v in clean if v <= value)
    return (le_count / len(clean)) * 100.0


def metric_distribution(samples: list[dict], metric: str) -> list[float]:
    distribution: list[float] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        if not _sample_has_reliable_denominator(sample, metric):
            continue
        metrics = sample.get("metrics")
        if not isinstance(metrics, dict):
            continue
        value = _to_float_or_none(metrics.get(metric))
        if value is not None:
            distribution.append(value)
    return distribution


def filter_samples(
    samples: list[dict],
    map_name: str | None = None,
    side: str | None = None,
    match_balance: str | None = None,
) -> list[dict]:
    side_norm = side.upper() if side is not None else None
    out: list[dict] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        if map_name is not None and sample.get("map_name") != map_name:
            continue
        if side_norm is not None and str(sample.get("side", "ALL")).upper() != side_norm:
            continue
        if match_balance is not None and sample.get("match_balance") != match_balance:
            continue
        out.append(sample)
    return out


def choose_best_context_pool(
    samples: list[dict],
    metric: str,
    map_name: str | None,
    side: str | None,
    min_samples: int = 30,
) -> tuple[list[dict], str]:
    candidates: list[tuple[list[dict], str]] = []
    if side is not None:
        side = side.upper()
        if map_name is not None:
            candidates.append((filter_samples(samples, map_name=map_name, side=side), "map+side"))
        candidates.append((filter_samples(samples, map_name=None, side=side), "side"))
    else:
        if map_name is not None:
            candidates.append((filter_samples(samples, map_name=map_name, side=None), "map"))
        candidates.append((samples, "global"))

    for pool, label in candidates:
        if len(metric_distribution(pool, metric)) >= min_samples:
            return pool, label
    return [], "insufficient"


def evaluate_metric(
    value: float,
    metric: str,
    samples: list[dict],
    map_name: str | None = None,
    side: str | None = None,
    min_samples: int = 30,
    selected_counts: dict | None = None,
    selected_rounds_played: int | None = None,
) -> dict:
    requirement = MIN_DENOMINATORS.get(metric)
    if requirement is not None:
        denom_name, required_min = requirement
        player_denom = _selected_count(selected_counts, selected_rounds_played, denom_name)
        if player_denom is None or player_denom < required_min:
            return {
                "metric": metric,
                "value": float(value),
                "percentile": None,
                "sample_size": 0,
                "context": "unavailable",
                "rating": "unknown",
                "reason": "not_enough_player_samples",
                "count": player_denom,
                "required_count": required_min,
            }

    pool, context = choose_best_context_pool(samples, metric, map_name, side, min_samples=min_samples)
    population = metric_distribution(pool, metric)
    if context == "insufficient":
        return {
            "metric": metric,
            "value": float(value),
            "percentile": None,
            "sample_size": len(population),
            "context": context,
            "rating": "unknown",
            "reason": "insufficient_population",
            "count": None,
            "required_count": None,
        }

    pct = percentile_rank(float(value), population)

    if pct is None:
        return {
            "metric": metric,
            "value": float(value),
            "percentile": None,
            "sample_size": len(population),
            "context": context,
            "rating": "unknown",
            "reason": "insufficient_population",
            "count": None,
            "required_count": None,
        }

    if pct < 20:
        rating = "critical"
    elif pct < 40:
        rating = "warning"
    elif pct < 60:
        rating = "average"
    elif pct < 80:
        rating = "good"
    else:
        rating = "excellent"

    return {
        "metric": metric,
        "value": float(value),
        "percentile": round(pct, 2),
        "sample_size": len(population),
        "context": context,
        "rating": rating,
        "reason": None,
        "count": None,
        "required_count": None,
    }


def evaluate_player(
    player_metrics: dict,
    samples: list[dict],
    map_name: str | None = None,
    side: str | None = None,
    min_samples: int = 30,
    player_counts: dict | None = None,
    rounds_played: int | None = None,
) -> dict[str, dict]:
    evaluations: dict[str, dict] = {}
    for metric_name, raw_value in player_metrics.items():
        metric_value = _to_float_or_none(raw_value)
        if metric_value is None:
            continue
        evaluations[str(metric_name)] = evaluate_metric(
            value=metric_value,
            metric=str(metric_name),
            samples=samples,
            map_name=map_name,
            side=side,
            min_samples=min_samples,
            selected_counts=player_counts,
            selected_rounds_played=rounds_played,
        )
    return evaluations


if __name__ == "__main__":
    # Lightweight sanity checks for local development.
    ct = [{"steamid": "1", "name": "p1", "side": "CT", "rounds_played": 10, "kills": 5, "headshot_kills": 2, "adr": 60.0}]
    tt = [{"steamid": "1", "name": "p1", "side": "T", "rounds_played": 10, "kills": 7, "headshot_kills": 3, "adr": 80.0}]
    contextual = make_contextual_match_samples("m1", "de_mirage", 20, ct, tt)
    assert len([s for s in contextual if s.get("side") == "ALL"]) == 1
    merged = [s for s in contextual if s.get("side") == "ALL"][0]
    assert merged.get("rounds_played") == 20
    assert merged.get("counts", {}).get("kills") == 12
    pool, label = choose_best_context_pool(contextual, "adr", "de_mirage", "ALL", min_samples=30)
    assert label == "insufficient"
    assert pool == []

    synthetic_samples: list[dict] = []
    for idx in range(10):
        player_id = str(idx + 1)
        synthetic_samples.extend(
            [
                {
                    "steamid": player_id,
                    "name": f"p{player_id}",
                    "map_name": "de_mirage",
                    "side": "CT",
                    "rounds_played": 12,
                    "metrics": {"adr": 70.0 + idx},
                    "counts": {},
                },
                {
                    "steamid": player_id,
                    "name": f"p{player_id}",
                    "map_name": "de_mirage",
                    "side": "T",
                    "rounds_played": 12,
                    "metrics": {"adr": 75.0 + idx},
                    "counts": {},
                },
                {
                    "steamid": player_id,
                    "name": f"p{player_id}",
                    "map_name": "de_mirage",
                    "side": "ALL",
                    "rounds_played": 24,
                    "metrics": {"adr": 72.5 + idx},
                    "counts": {},
                },
            ]
        )

    pool, label = choose_best_context_pool(synthetic_samples, "adr", "de_mirage", "CT", min_samples=20)
    assert label == "insufficient"
    assert pool == []

    pool, label = choose_best_context_pool(synthetic_samples, "adr", "de_mirage", "ALL", min_samples=20)
    assert label == "insufficient"
    assert pool == []
