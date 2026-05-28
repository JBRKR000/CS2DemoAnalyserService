from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from Analyser import _build_benchmark_player_rows
from Parser import load_cached_demo
from benchmarks import (
    append_match_samples,
    is_match_analyzed,
    load_analyzed_matches,
    load_benchmark_samples,
    make_contextual_match_samples,
    mark_match_analyzed,
)


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill local benchmark samples from all parsed demo caches.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild samples even when the match is already in analyzed_matches.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of cached demos to inspect.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate samples and log counts without writing benchmark files.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _safe_len(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return None


def _map_name_from_demo(demo: Any) -> str | None:
    header = getattr(demo, "header", None)
    if isinstance(header, dict):
        map_name = header.get("map_name")
        return str(map_name) if map_name is not None else None
    return None


def _cache_files(limit: int | None) -> list[Path]:
    files = sorted(CACHE_DIR.glob("*.pkl"))
    if limit is not None:
        return files[: max(limit, 0)]
    return files


def _deduplicated_count(existing_samples: list[dict], new_samples: list[dict]) -> int:
    all_samples = [*existing_samples, *new_samples]
    seen: set[tuple[str, str, str]] = set()
    unique_count = 0
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
        unique_count += 1
    return unique_count


def _build_samples_for_cache(cache_key: str) -> tuple[list[dict], dict[str, Any]]:
    demo = load_cached_demo(cache_key, cache_dir=str(CACHE_DIR))
    map_name = _map_name_from_demo(demo)
    round_count = _safe_len(getattr(demo, "rounds", None))
    side_rows = _build_benchmark_player_rows(demo)
    ct_rows = [row for row in side_rows if str(row.get("side", "")).upper() == "CT"]
    t_rows = [row for row in side_rows if str(row.get("side", "")).upper() == "T"]
    samples = make_contextual_match_samples(
        match_id=cache_key,
        map_name=map_name,
        round_count=round_count,
        ct_player_stats=ct_rows,
        t_player_stats=t_rows,
    )
    metadata = {
        "map_name": map_name,
        "round_count": round_count,
        "samples_written": len(samples),
    }
    return samples, metadata


def main() -> None:
    configure_logging()
    args = parse_args()

    LOGGER.info(
        "Benchmark KAST definition now includes traded rounds; rerun backfill with --force to refresh existing local_benchmarks.json entries."
    )

    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be >= 0.")

    files = _cache_files(args.limit)
    registry = load_analyzed_matches()
    pool_before = len(load_benchmark_samples())

    skipped = 0
    processed = 0
    generated_samples: list[dict] = []
    processed_metadata: list[tuple[str, dict[str, Any]]] = []

    LOGGER.info("Cached demos found: %d", len(files))
    LOGGER.info("Benchmark pool size before: %d", pool_before)

    for cache_file in files:
        cache_key = cache_file.stem
        if not args.force and is_match_analyzed(cache_key, registry):
            skipped += 1
            continue

        try:
            samples, metadata = _build_samples_for_cache(cache_key)
        except Exception:
            LOGGER.exception("Failed to generate benchmark samples for cache key %s", cache_key)
            continue

        processed += 1
        generated_samples.extend(samples)
        processed_metadata.append((cache_key, metadata))
        LOGGER.info("Processed %s: samples=%d", cache_key, len(samples))

    if args.dry_run:
        pool_after = _deduplicated_count(load_benchmark_samples(), generated_samples)
    else:
        all_samples = append_match_samples(generated_samples)
        pool_after = len(all_samples)
        for cache_key, metadata in processed_metadata:
            mark_match_analyzed(cache_key, metadata)

    LOGGER.info("Matches skipped: %d", skipped)
    LOGGER.info("Matches processed: %d", processed)
    LOGGER.info("Samples generated: %d", len(generated_samples))
    LOGGER.info("Benchmark pool size before: %d", pool_before)
    LOGGER.info("Benchmark pool size after: %d", pool_after)
    if args.dry_run:
        LOGGER.info("Dry run complete: no benchmark files were modified.")


if __name__ == "__main__":
    main()
