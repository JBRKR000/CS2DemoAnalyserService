from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Analyser import analyse_demo, load_demo_for_analysis
from Parser import compute_file_sha256, get_demo
from report_builder import build_match_report
from sectors.situation_builder import (
    build_player_situations,
    save_situations_json,
    save_situations_parquet,
)

REPO_ROOT = SRC_DIR.parent
DEMO_PATH = REPO_ROOT / "Demos" / "demo.dem"
CACHE_DIR = REPO_ROOT / ".cache"
CACHE_KEY_PATH = REPO_ROOT / "last_cache_key.txt"
SITUATIONS_DIR = REPO_ROOT / "data" / "situations"

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
LOGGER = logging.getLogger(__name__)


def _pick_latest_cache_key() -> str | None:
    if not CACHE_DIR.exists():
        return None
    cache_files = sorted(CACHE_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cache_files[0].stem if cache_files else None


def _looks_like_sha256(value: str | None) -> bool:
    return bool(value and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value))


def _resolve_cache_key() -> tuple[str | None, str | None]:
    """Returns (cache_key, match_id)."""
    if DEMO_PATH.exists():
        match_id = compute_file_sha256(DEMO_PATH)
        result = get_demo(
            demo_path=DEMO_PATH,
            cache_dir=str(CACHE_DIR),
            return_cache_key=True,
            delete_source=False,
        )
        if isinstance(result, tuple):
            _, cache_key = result
        else:
            cache_key = match_id
        CACHE_KEY_PATH.write_text(cache_key, encoding="utf-8")
        return cache_key, match_id

    if CACHE_KEY_PATH.exists():
        cache_key = CACHE_KEY_PATH.read_text(encoding="utf-8").strip()
        if _looks_like_sha256(cache_key) and (CACHE_DIR / f"{cache_key}.pkl").exists():
            return cache_key, cache_key

    latest = _pick_latest_cache_key()
    if latest:
        LOGGER.warning("Falling back to latest cache key: %s", latest)
        CACHE_KEY_PATH.write_text(latest, encoding="utf-8")
        match_id = latest if _looks_like_sha256(latest) else None
        return latest, match_id

    raise FileNotFoundError(
        "No demo file and no valid cache found. Put a .dem in Demos/ or create a cache first."
    )


def main() -> None:
    cache_key, match_id = _resolve_cache_key()

    demo = load_demo_for_analysis(
        str(CACHE_KEY_PATH),
        cache_dir=str(CACHE_DIR),
        verbose=True,
    )
    LOGGER.info("Demo loaded | type=%s", type(demo).__name__)

    analysis = analyse_demo(demo, match_id=match_id)
    analysis["match_id"] = match_id

    # build_match_report populates analysis with vod_review_priority and decision_simulation
    build_match_report(analysis)

    situations = build_player_situations(analysis)

    # --- Log summary ---
    total = len(situations)
    LOGGER.info("situations_total=%d", total)

    type_counts = Counter(s.get("situation_type") for s in situations)
    for stype, count in sorted(type_counts.items()):
        LOGGER.info("situation_type=%s count=%d", stype, count)

    flag_counts: Counter[str] = Counter()
    for s in situations:
        for flag in (s.get("source_flags") or []):
            flag_counts[flag] += 1
    for flag, count in sorted(flag_counts.items()):
        LOGGER.info("source_flag=%s count=%d", flag, count)

    ml_enriched_count = flag_counts.get("ml_enriched", 0)
    ml_ambiguous_count = flag_counts.get("ml_ambiguous_match", 0)
    ml_missing_count = sum(
        1 for s in situations
        if s.get("ml_impact") is None and "ml_ambiguous_match" not in (s.get("source_flags") or [])
        and s.get("situation_type") in ("kill_situation", "death_situation")
    )
    LOGGER.info("ml_enriched_count=%d", ml_enriched_count)
    LOGGER.info("ml_ambiguous_match_count=%d", ml_ambiguous_count)
    LOGGER.info("ml_missing_count=%d", ml_missing_count)

    high_impact_kill_count = sum(1 for s in situations if s.get("high_impact_kill") is True)
    low_impact_kill_count = sum(1 for s in situations if s.get("low_impact_kill") is True)
    high_cost_death_count = sum(1 for s in situations if s.get("high_cost_death") is True)
    low_cost_death_count = sum(1 for s in situations if s.get("low_cost_death") is True)
    LOGGER.info("high_impact_kill_count=%d", high_impact_kill_count)
    LOGGER.info("low_impact_kill_count=%d", low_impact_kill_count)
    LOGGER.info("high_cost_death_count=%d", high_cost_death_count)
    LOGGER.info("low_cost_death_count=%d", low_cost_death_count)

    # Detect rounds where multiple kill_situations share an identical non-None ml_impact
    from collections import defaultdict as _defaultdict
    kill_impacts_by_round: dict[int, list[float]] = _defaultdict(list)
    for s in situations:
        if s.get("situation_type") == "kill_situation" and s.get("ml_impact") is not None:
            kill_impacts_by_round[s["round_num"]].append(s["ml_impact"])
    dup_rounds = sum(
        1 for impacts in kill_impacts_by_round.values()
        if len(impacts) > 1 and len(set(impacts)) < len(impacts)
    )
    LOGGER.info("duplicate_ml_impact_same_round_count=%d", dup_rounds)

    if not situations:
        LOGGER.warning("No situations generated — nothing saved.")
        return

    # --- Resolve output paths ---
    player_steamid = situations[0].get("steamid") or "unknown"
    mid = str(match_id) if match_id else "unknown"
    out_dir = SITUATIONS_DIR / mid
    json_path = out_dir / f"{player_steamid}_situations.json"
    parquet_path = out_dir / f"{player_steamid}_situations.parquet"

    save_situations_json(situations, json_path)
    LOGGER.info("Saved JSON  -> %s", json_path)

    save_situations_parquet(situations, parquet_path)
    LOGGER.info("Saved parquet -> %s", parquet_path)


if __name__ == "__main__":
    main()
