from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS_PATH = REPO_ROOT / "data" / "ml" / "death_risk_timeseries_5s_predictions.parquet"

_WEAPON_DISPLAY_NAMES: dict[str, str] = {
    "ak47": "AK-47",
    "aug": "AUG",
    "awp": "AWP",
    "bizon": "PP-Bizon",
    "cz75a": "CZ75-Auto",
    "deagle": "Desert Eagle",
    "dual berettas": "Dual Berettas",
    "elite": "Dual Berettas",
    "famas": "FAMAS",
    "fiveseven": "Five-SeveN",
    "galilar": "Galil AR",
    "glock": "Glock-18",
    "g3sg1": "G3SG1",
    "hkp2000": "P2000",
    "incgrenade": "Incendiary Grenade",
    "mac10": "MAC-10",
    "m249": "M249",
    "m4a1": "M4A1",
    "m4a1_silencer": "M4A1-S",
    "mag7": "MAG-7",
    "molotov": "Molotov",
    "mp5sd": "MP5-SD",
    "mp7": "MP7",
    "mp9": "MP9",
    "negev": "Negev",
    "nova": "Nova",
    "p90": "P90",
    "p250": "P250",
    "p2000": "P2000",
    "revolver": "R8 Revolver",
    "sawedoff": "Sawed-Off",
    "scar20": "SCAR-20",
    "sg556": "SG 553",
    "smokegrenade": "Smoke Grenade",
    "ssg08": "SSG 08",
    "tec9": "TEC-9",
    "ump45": "UMP-45",
    "usp": "USP",
    "usp_silencer": "USP-S",
    "xm1014": "XM1014",
}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_weapon_value(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return "-"

    lookup_key = text.strip().lower()
    display_name = _WEAPON_DISPLAY_NAMES.get(lookup_key)
    if display_name is not None:
        return display_name

    normalized = lookup_key.replace("_", " ").replace("-", " ")
    compact_lookup = normalized.replace(" ", "")
    display_name = _WEAPON_DISPLAY_NAMES.get(normalized) or _WEAPON_DISPLAY_NAMES.get(compact_lookup)
    if display_name is not None:
        return display_name

    if lookup_key.isdigit():
        return text

    return text


def _format_weapon_name(value: Any) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None

    lookup_key = text.strip().lower()
    if lookup_key.isdigit():
        return None

    display_name = _WEAPON_DISPLAY_NAMES.get(lookup_key)
    if display_name is not None:
        return display_name

    normalized = lookup_key.replace("_", " ").replace("-", " ")
    compact_lookup = normalized.replace(" ", "")
    display_name = _WEAPON_DISPLAY_NAMES.get(normalized) or _WEAPON_DISPLAY_NAMES.get(compact_lookup)
    if display_name is not None:
        return display_name

    if any(char.isalpha() for char in text):
        return text

    return None


@lru_cache(maxsize=8)
def _load_death_risk_predictions_cached(path_text: str) -> pl.DataFrame | None:
    path = Path(path_text)
    if not path.exists():
        LOGGER.debug("Death risk predictions parquet missing: %s", path)
        return None

    try:
        return pl.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive logging path
        LOGGER.warning("Unable to load death risk predictions from %s: %s", path, exc)
        return None


def load_death_risk_predictions(path: str | Path | None = None) -> pl.DataFrame | None:
    resolved_path = Path(path) if path is not None else DEFAULT_PREDICTIONS_PATH
    return _load_death_risk_predictions_cached(str(resolved_path))


def find_player_pre_event_death_risk(
    predictions: pl.DataFrame | None,
    *,
    match_id: str | int | None,
    steamid: str | int | None,
    round_num: int | str | None,
    event_tick: int | str | None = None,
    tickrate: float | int | None = None,
) -> dict[str, Any] | None:
    if predictions is None or predictions.is_empty():
        return None

    match_id_text = _normalize_text(match_id)
    steamid_text = _normalize_text(steamid)
    round_value = _as_int(round_num)
    event_tick_value = _as_int(event_tick)
    tickrate_value = _as_float(tickrate)

    if not match_id_text or not steamid_text or round_value is None:
        return None

    required_columns = {
        "match_id",
        "steamid",
        "round_num",
        "tick",
        "death_risk_5s",
        "death_risk_bucket_global",
        "risk_label",
    }
    if not required_columns.issubset(set(predictions.columns)):
        return None

    filtered = predictions.filter(
        pl.col("match_id").cast(pl.Utf8, strict=False) == match_id_text,
        pl.col("steamid").cast(pl.Utf8, strict=False) == steamid_text,
        pl.col("round_num").cast(pl.Int64, strict=False) == round_value,
        pl.col("death_risk_5s").is_not_null(),
    )

    if filtered.is_empty():
        return None

    if event_tick_value is not None:
        filtered = filtered.filter(pl.col("tick").cast(pl.Int64, strict=False) < event_tick_value)
        if filtered.is_empty():
            return None

        if tickrate_value is not None and tickrate_value > 0.0:
            window_ticks = max(1, int(round(5.0 * tickrate_value)))
            lower_tick = event_tick_value - window_ticks
            filtered = filtered.filter(pl.col("tick").cast(pl.Int64, strict=False) >= lower_tick)
            if filtered.is_empty():
                return None

    best_row = (
        filtered.select(
            [
                pl.col("death_risk_5s"),
                pl.col("death_risk_bucket_global"),
                pl.col("risk_label"),
                pl.col("tick"),
                pl.col("nearest_enemy_distance"),
                pl.col("nearest_teammate_distance"),
                pl.col("player_hp"),
                pl.col("weapon"),
            ]
        )
        .sort(["death_risk_5s", "tick"], descending=[True, True])
        .head(1)
        .to_dicts()
    )

    if not best_row:
        return None

    row = best_row[0]
    return {
        "max_death_risk_5s": _as_float(row.get("death_risk_5s")),
        "max_risk_label": _normalize_text(row.get("risk_label")) or None,
        "max_risk_bucket": _normalize_text(row.get("death_risk_bucket_global")) or None,
        "risk_snapshot_tick": _as_int(row.get("tick")),
        "nearest_enemy_distance_at_max_risk": _as_float(row.get("nearest_enemy_distance")),
        "nearest_teammate_distance_at_max_risk": _as_float(row.get("nearest_teammate_distance")),
        "player_hp_at_max_risk": _as_float(row.get("player_hp")),
        "weapon_at_max_risk": _format_weapon_value(row.get("weapon")),
        "weapon_name": _format_weapon_name(row.get("weapon")),
    }