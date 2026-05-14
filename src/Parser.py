from __future__ import annotations

import hashlib
import logging
import pickle
import sys
import threading
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from typing import Any

from awpy import Demo


logger = logging.getLogger(__name__)


class _ParseProgress:
    def __init__(self, label: str = "Parsing demo", expected_seconds: float = 30.0):
        self.label = label
        self.expected_seconds = max(1.0, expected_seconds)
        self._frames = "|/-\\"
        self._bar_width = 28
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._last_percent = 0

    def _line(self, percent: int, frame: str) -> str:
        filled = int((percent / 100) * self._bar_width)
        bar = "#" * filled + "-" * (self._bar_width - filled)
        return f"\r{self.label} {frame} [{bar}] {percent:3d}%"

    def _run(self) -> None:
        frame_idx = 0
        while not self._stop.is_set():
            elapsed = time.perf_counter() - self._start_time
            percent = min(99, int((elapsed / self.expected_seconds) * 100))
            self._last_percent = percent
            frame = self._frames[frame_idx % len(self._frames)]
            sys.stdout.write(self._line(percent, frame))
            sys.stdout.flush()
            frame_idx += 1
            time.sleep(0.12)

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def finish(self, success: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        if success:
            sys.stdout.write(self._line(100, "|"))
            sys.stdout.write("\n")
        else:
            frame = "!"
            sys.stdout.write(self._line(self._last_percent, frame))
            sys.stdout.write(" FAILED\n")
        sys.stdout.flush()


def _normalize_cache_key(cache_key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cache_key)


def _cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{_normalize_cache_key(cache_key)}.pkl"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(path: str | Path) -> str:
    file_path = Path(path)
    hasher = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_file(path: Path) -> str:
    return compute_file_sha256(path)


# awpy v2 exposes kills, damages, etc. as cached properties.
# They only land in __dict__ after first access, so we must touch
# them explicitly before iterating __dict__ for pickling.
_AWPY_V2_PROPERTIES = [
    "kills",
    "damages",
    "shots",
    "bomb",
    "smokes",
    "infernos",
    "grenades",
    "footsteps",
    "ticks",
    "rounds",
]
_REQUESTED_PLAYER_PROPS = [
    "current_equip_value",
    "cash_spent_this_round",
    "armor_value",
    "active_weapon",
]


def _pickleable_state(demo: Demo) -> dict[str, Any]:
    # Force-access all cached properties so they appear in __dict__.
    for prop in _AWPY_V2_PROPERTIES:
        try:
            getattr(demo, prop)
        except Exception:
            pass

    state: dict[str, Any] = {}
    for key, value in demo.__dict__.items():
        if key == "parser":
            continue
        try:
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            state[key] = value
        except Exception:
            pass
    return state


def _save_cache(cache_path: Path, cache_key: str, state: dict[str, Any], source_name: str | None) -> None:
    payload = {
        "cache_key": cache_key,
        "source_name": source_name,
        "state": state,
    }
    with cache_path.open("wb") as file_handle:
        pickle.dump(payload, file_handle, protocol=pickle.HIGHEST_PROTOCOL)


def _load_cache(cache_path: Path) -> dict[str, Any]:
    with cache_path.open("rb") as file_handle:
        payload = pickle.load(file_handle)
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError("Invalid cache format.")
    return payload


def _to_cached_demo(state: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**state)


def _parse_state_from_path(demo_path: Path) -> dict[str, Any]:
    demo = Demo(demo_path)
    progress = _ParseProgress()
    progress.start()
    try:
        demo.parse(player_props=_REQUESTED_PLAYER_PROPS)
    except Exception:
        progress.finish(success=False)
        raise
    progress.finish(success=True)
    return _pickleable_state(demo)


def get_demo(
    demo_path: str | Path | None = None,
    cache_key: str | None = None,
    cache_dir: str = ".cache",
    delete_source: bool = False,
    force_reparse: bool = False,
    return_cache_key: bool = False,
):
    cache_directory = Path(cache_dir)
    cache_directory.mkdir(parents=True, exist_ok=True)

    source_path = Path(demo_path) if demo_path is not None else None

    if cache_key is None:
        if source_path is None or not source_path.exists():
            raise ValueError("Provide cache_key or an existing demo_path.")
        cache_key = _hash_file(source_path)

    path_to_cache = _cache_path(cache_directory, cache_key)

    if path_to_cache.exists() and not force_reparse:
        payload = _load_cache(path_to_cache)
        state = payload["state"]

        # Invalidate stale cache that is missing kills/damages (parsed before
        # the cached-property fix). Force a re-parse automatically.
        ticks_df = state.get("ticks")
        missing_economy_props = (
            ticks_df is None
            or not hasattr(ticks_df, "columns")
            or "current_equip_value" not in ticks_df.columns
        )

        if "kills" not in state or "damages" not in state or missing_economy_props:
            logger.warning(
                "Cache is stale (missing kills/damages or economy tick props). Re-parsing demo."
            )
            if source_path is None or not source_path.exists():
                raise FileNotFoundError(
                    "Stale cache detected but demo file is missing — cannot re-parse."
                )
        else:
            if delete_source and source_path is not None and source_path.exists():
                source_path.unlink(missing_ok=True)
            demo_obj = _to_cached_demo(state)
            logger.info("Loaded parsed demo from cache.")
            return (demo_obj, cache_key) if return_cache_key else demo_obj

    if source_path is None or not source_path.exists():
        raise FileNotFoundError("Demo file does not exist and cache is missing.")

    state = _parse_state_from_path(source_path)
    _save_cache(path_to_cache, cache_key, state, source_path.name)

    if delete_source:
        source_path.unlink(missing_ok=True)

    demo_obj = _to_cached_demo(state)
    logger.info("Parsed demo and saved cache.")
    return (demo_obj, cache_key) if return_cache_key else demo_obj


def load_cached_demo(cache_key: str, cache_dir: str = ".cache", return_cache_key: bool = False):
    cache_directory = Path(cache_dir)
    path_to_cache = _cache_path(cache_directory, cache_key)
    payload = _load_cache(path_to_cache)
    demo_obj = _to_cached_demo(payload["state"])
    return (demo_obj, cache_key) if return_cache_key else demo_obj


def cache_demo_from_bytes(
    demo_bytes: bytes,
    cache_key: str | None = None,
    cache_dir: str = ".cache",
    force_reparse: bool = False,
    return_cache_key: bool = False,
):
    if cache_key is None:
        cache_key = _hash_bytes(demo_bytes)

    cache_directory = Path(cache_dir)
    cache_directory.mkdir(parents=True, exist_ok=True)
    path_to_cache = _cache_path(cache_directory, cache_key)

    if path_to_cache.exists() and not force_reparse:
        payload = _load_cache(path_to_cache)
        state = payload["state"]
        ticks_df = state.get("ticks")
        missing_economy_props = (
            ticks_df is None
            or not hasattr(ticks_df, "columns")
            or "current_equip_value" not in ticks_df.columns
        )
        if "kills" not in state or "damages" not in state or missing_economy_props:
            logger.warning(
                "Cache is stale (missing kills/damages or economy tick props). Re-parsing demo."
            )
        else:
            demo_obj = _to_cached_demo(state)
            logger.info("Loaded parsed demo from cache.")
            return (demo_obj, cache_key) if return_cache_key else demo_obj

    tmp_path: Path | None = None
    try:
        with NamedTemporaryFile(suffix=".dem", delete=False) as tmp:
            tmp.write(demo_bytes)
            tmp_path = Path(tmp.name)

        state = _parse_state_from_path(tmp_path)
        _save_cache(path_to_cache, cache_key, state, source_name=None)
        demo_obj = _to_cached_demo(state)
        logger.info("Parsed demo and saved cache.")
        return (demo_obj, cache_key) if return_cache_key else demo_obj
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
