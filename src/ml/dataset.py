from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import polars as pl


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Parser import load_cached_demo
from features import (
    MAX_PLAYERS_PER_SIDE,
    MAX_REASONABLE_ROUND_SECONDS,
    SNAPSHOT_COLUMNS,
    SNAPSHOT_SCHEMA,
    build_round_snapshot_rows,
    empty_snapshot_dataset,
)


REPO_ROOT = SRC_DIR.parent
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache"
DEFAULT_CACHE_KEY_PATH = REPO_ROOT / "last_cache_key.txt"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "ml" / "round_snapshots.parquet"


def _apply_snapshot_safety(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame.select(SNAPSHOT_COLUMNS)

    return frame.with_columns(
        [
            pl.col("alive_team").clip(lower_bound=0, upper_bound=MAX_PLAYERS_PER_SIDE).cast(pl.Int64),
            pl.col("alive_enemy").clip(lower_bound=0, upper_bound=MAX_PLAYERS_PER_SIDE).cast(pl.Int64),
            pl.col("seconds_remaining").clip(lower_bound=0.0, upper_bound=MAX_REASONABLE_ROUND_SECONDS).cast(pl.Float64),
            pl.col("is_time_anomaly").fill_null(False).cast(pl.Boolean),
        ]
    )


def discover_cache_keys(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    cache_key_path: str | Path = DEFAULT_CACHE_KEY_PATH,
) -> list[str]:
    cache_directory = Path(cache_dir)
    keys = sorted(path.stem for path in cache_directory.glob("*.pkl"))
    if keys:
        return keys

    key_path = Path(cache_key_path)
    if key_path.exists():
        cache_key = key_path.read_text(encoding="utf-8").strip()
        if cache_key:
            return [cache_key]

    return []


def build_match_snapshot_dataset(
    cache_key: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> pl.DataFrame:
    demo = load_cached_demo(cache_key, cache_dir=str(cache_dir))
    return build_round_snapshot_rows(demo, match_id=cache_key)


def build_combined_snapshot_dataset(
    cache_keys: Iterable[str] | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    cache_key_path: str | Path = DEFAULT_CACHE_KEY_PATH,
) -> pl.DataFrame:
    keys = list(cache_keys) if cache_keys is not None else discover_cache_keys(cache_dir, cache_key_path)
    if not keys:
        return empty_snapshot_dataset()

    frames: list[pl.DataFrame] = []
    for cache_key in keys:
        frame = build_match_snapshot_dataset(cache_key=cache_key, cache_dir=cache_dir)
        if not frame.is_empty():
            frames.append(frame)

    if not frames:
        return empty_snapshot_dataset()

    return (
        _apply_snapshot_safety(pl.concat(frames, how="vertical_relaxed"))
        .select(SNAPSHOT_COLUMNS)
        .with_columns([pl.col(column).cast(dtype, strict=False) for column, dtype in SNAPSHOT_SCHEMA.items()])
        .sort(["match_id", "round_num", "tick", "snapshot_type", "side"])
    )


def save_snapshot_dataset(
    dataset: pl.DataFrame,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    (
        _apply_snapshot_safety(dataset)
        .select(SNAPSHOT_COLUMNS)
        .with_columns([pl.col(column).cast(dtype, strict=False) for column, dtype in SNAPSHOT_SCHEMA.items()])
        .write_parquet(destination)
    )
    return destination
