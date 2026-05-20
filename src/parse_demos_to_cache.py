from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from Parser import parse_demo_to_cache


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DEMOS_DIR = BASE_DIR / "Demos"
DEFAULT_CACHE_DIR = BASE_DIR / ".cache"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse .dem files into the existing parser cache.",
    )
    parser.add_argument(
        "--demos-dir",
        default=str(DEFAULT_DEMOS_DIR),
        help="Directory containing .dem files (default: repo-root Demos next to src/).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="Number of worker processes to use (default: max(1, os.cpu_count() - 1)).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of demos to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-parse demos even if cache already exists.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Cache directory to use (default: repo .cache directory).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label to store in cache metadata for each processed demo, e.g. 'pro'.",
    )
    return parser.parse_args()


def discover_demo_paths(demos_dir: str | Path, limit: int | None = None) -> list[Path]:
    demo_directory = Path(demos_dir)
    if not demo_directory.is_absolute():
        demo_directory = BASE_DIR / demo_directory
    demo_paths = sorted(demo_directory.rglob("*.dem"))
    if limit is not None:
        return demo_paths[: max(0, limit)]
    return demo_paths


def _worker_parse_demo(demo_path: str, force: bool, cache_dir: str, label: str | None) -> dict:
    return parse_demo_to_cache(demo_path=demo_path, force=force, cache_dir=cache_dir, label=label)


def _log_result(result: dict) -> None:
    status = str(result.get("status", "unknown")).upper()
    demo_name = Path(str(result.get("demo_path", ""))).name or "<unknown>"
    elapsed = float(result.get("elapsed_seconds", 0.0) or 0.0)
    error = result.get("error")
    label = result.get("label")
    suffix = f" | label={label}" if label else ""

    if status == "FAILED" and error:
        LOGGER.error("%s | %s | %.2fs%s | %s", status, demo_name, elapsed, suffix, error)
        return

    LOGGER.info("%s | %s | %.2fs%s", status, demo_name, elapsed, suffix)


def _run_sequential(demo_paths: Iterable[Path], force: bool, cache_dir: str, label: str | None) -> list[dict]:
    results: list[dict] = []
    for demo_path in demo_paths:
        result = _worker_parse_demo(str(demo_path), force, cache_dir, label)
        _log_result(result)
        results.append(result)
    return results


def _run_parallel(
    demo_paths: Iterable[Path],
    workers: int,
    force: bool,
    cache_dir: str,
    label: str | None,
) -> list[dict]:
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(_worker_parse_demo, str(demo_path), force, cache_dir, label): demo_path
            for demo_path in demo_paths
        }
        for future in as_completed(future_to_path):
            demo_path = future_to_path[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "demo_path": str(demo_path),
                    "match_id": None,
                    "status": "failed",
                    "cache_path": None,
                    "error": str(exc),
                    "label": label,
                    "elapsed_seconds": 0.0,
                }
            _log_result(result)
            results.append(result)
    return results


def main() -> None:
    args = parse_args()
    configure_logging()

    cache_dir = str(Path(args.cache_dir))
    demo_paths = discover_demo_paths(args.demos_dir, limit=args.limit)

    LOGGER.info("Demos found: %d", len(demo_paths))
    LOGGER.info("Workers: %d", args.workers)
    LOGGER.info("Cache dir: %s", cache_dir)
    if args.label:
        LOGGER.info("Label: %s", args.label)

    if not demo_paths:
        LOGGER.info("No .dem files found, nothing to do.")
        return

    started_at = time.perf_counter()
    if args.workers <= 1:
        results = _run_sequential(demo_paths, force=args.force, cache_dir=cache_dir, label=args.label)
    else:
        results = _run_parallel(
            demo_paths,
            workers=args.workers,
            force=args.force,
            cache_dir=cache_dir,
            label=args.label,
        )

    parsed_count = sum(1 for result in results if result.get("status") == "parsed")
    skipped_count = sum(1 for result in results if result.get("status") == "skipped")
    failed_count = sum(1 for result in results if result.get("status") == "failed")
    total_elapsed = time.perf_counter() - started_at

    LOGGER.info("Total demos: %d", len(results))
    LOGGER.info("Parsed count: %d", parsed_count)
    LOGGER.info("Skipped count: %d", skipped_count)
    LOGGER.info("Failed count: %d", failed_count)
    LOGGER.info("Total elapsed time: %.2fs", total_elapsed)


if __name__ == "__main__":
    main()
