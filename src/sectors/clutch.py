"""Clutch analysis for CS2 demos.

Detects clutch situations from round kill timelines and summarizes outcomes
per player.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from coach_metrics import _has_cols, _kills_df, _safe_df


def _empty_clutch_rounds() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "round_num": pl.Int64,
            "steamid": pl.UInt64,
            "side": pl.Utf8,
            "clutch_type": pl.Utf8,
            "won": pl.Boolean,
        }
    )


def _empty_clutch_summary() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "steamid": pl.UInt64,
            "name": pl.Utf8,
            "total_clutches": pl.Int64,
            "clutches_won": pl.Int64,
            "win_rate": pl.Float64,
            "v1_won": pl.Int64,
            "v1_total": pl.Int64,
            "v2_won": pl.Int64,
            "v2_total": pl.Int64,
            "v3plus_won": pl.Int64,
            "v3plus_total": pl.Int64,
        }
    )


def _round_winners(demo: Any) -> dict[int, str]:
    rounds = _safe_df(getattr(demo, "rounds", None))
    if not _has_cols(rounds, ["round_num", "winner"]):
        return {}
    out: dict[int, str] = {}
    for row in rounds.select(["round_num", "winner"]).drop_nulls(["round_num"]).to_dicts():
        round_num = int(row["round_num"])
        winner = str(row.get("winner") or "").lower()
        out[round_num] = winner
    return out


def _player_name_lookup(demo: Any) -> dict[int, str]:
    ticks = _safe_df(getattr(demo, "ticks", None))
    if not _has_cols(ticks, ["steamid", "name"]):
        return {}
    names: dict[int, str] = {}
    rows = (
        ticks.select(["steamid", "name"])
        .drop_nulls(["steamid", "name"])
        .group_by("steamid")
        .agg(pl.first("name").alias("name"))
        .to_dicts()
    )
    for row in rows:
        names[int(row["steamid"])] = str(row["name"])
    return names


def _round_rosters_from_ticks(demo: Any) -> dict[int, dict[str, set[int]]]:
    ticks = _safe_df(getattr(demo, "ticks", None))
    rosters: dict[int, dict[str, set[int]]] = {}
    if not _has_cols(ticks, ["round_num", "steamid", "side"]):
        return rosters
    for row in ticks.select(["round_num", "steamid", "side"]).drop_nulls().unique().to_dicts():
        rnd = int(row["round_num"])
        steamid = int(row["steamid"])
        side = str(row["side"]).lower()
        if side not in {"ct", "t"}:
            continue
        rosters.setdefault(rnd, {"ct": set(), "t": set()})
        rosters[rnd][side].add(steamid)
    return rosters


def _fill_rosters_from_kills(kills: pl.DataFrame, rosters: dict[int, dict[str, set[int]]]) -> None:
    if kills.is_empty():
        return
    for row in kills.select(
        ["round_num", "attacker_steamid", "attacker_side", "victim_steamid", "victim_side"]
    ).to_dicts():
        rnd = row.get("round_num")
        if rnd is None:
            continue
        round_num = int(rnd)
        rosters.setdefault(round_num, {"ct": set(), "t": set()})

        attacker_side = str(row.get("attacker_side") or "").lower()
        victim_side = str(row.get("victim_side") or "").lower()
        attacker = row.get("attacker_steamid")
        victim = row.get("victim_steamid")
        if attacker is not None and attacker_side in {"ct", "t"}:
            rosters[round_num][attacker_side].add(int(attacker))
        if victim is not None and victim_side in {"ct", "t"}:
            rosters[round_num][victim_side].add(int(victim))


def _clutch_label(opponents_alive: int) -> str:
    if opponents_alive <= 1:
        return "1v1"
    if opponents_alive == 2:
        return "1v2"
    return "1v3+"


def build_clutch_stats(demo: Any) -> dict[str, pl.DataFrame]:
    kills = _kills_df(demo)
    if kills.is_empty() or not _has_cols(kills, ["round_num", "tick", "victim_steamid", "victim_side"]):
        return {"clutch_rounds": _empty_clutch_rounds(), "clutch_summary": _empty_clutch_summary()}

    winners = _round_winners(demo)
    name_lookup = _player_name_lookup(demo)
    rosters = _round_rosters_from_ticks(demo)
    _fill_rosters_from_kills(kills, rosters)

    clutch_rows: list[dict[str, Any]] = []

    grouped = (
        kills.select(["round_num", "tick", "victim_steamid", "victim_side"])
        .drop_nulls(["round_num", "tick"])
        .sort(["round_num", "tick"])
        .group_by("round_num", maintain_order=True)
        .agg(
            [
                pl.col("tick"),
                pl.col("victim_steamid"),
                pl.col("victim_side"),
            ]
        )
    )

    for round_row in grouped.to_dicts():
        round_num = int(round_row["round_num"])
        round_roster = rosters.get(round_num, {"ct": set(), "t": set()})

        ct_alive = set(round_roster.get("ct", set()))
        t_alive = set(round_roster.get("t", set()))

        if not ct_alive and not t_alive:
            continue

        triggered: set[tuple[str, int]] = set()
        ticks: list[Any] = round_row.get("tick", [])
        victims: list[Any] = round_row.get("victim_steamid", [])
        victim_sides: list[Any] = round_row.get("victim_side", [])

        for _, victim, victim_side_raw in zip(ticks, victims, victim_sides):
            victim_side = str(victim_side_raw or "").lower()
            if victim is not None and victim_side == "ct":
                ct_alive.discard(int(victim))
            elif victim is not None and victim_side == "t":
                t_alive.discard(int(victim))

            ct_count = len(ct_alive)
            t_count = len(t_alive)

            if ct_count == 1 and t_count >= 1:
                steamid = next(iter(ct_alive))
                key = ("ct", steamid)
                if key not in triggered:
                    triggered.add(key)
                    clutch_rows.append(
                        {
                            "round_num": round_num,
                            "steamid": steamid,
                            "side": "ct",
                            "clutch_type": _clutch_label(t_count),
                            "won": winners.get(round_num, "") == "ct",
                        }
                    )

            if t_count == 1 and ct_count >= 1:
                steamid = next(iter(t_alive))
                key = ("t", steamid)
                if key not in triggered:
                    triggered.add(key)
                    clutch_rows.append(
                        {
                            "round_num": round_num,
                            "steamid": steamid,
                            "side": "t",
                            "clutch_type": _clutch_label(ct_count),
                            "won": winners.get(round_num, "") == "t",
                        }
                    )

    if not clutch_rows:
        return {"clutch_rounds": _empty_clutch_rounds(), "clutch_summary": _empty_clutch_summary()}

    clutch_rounds = pl.DataFrame(clutch_rows).with_columns(
        [
            pl.col("round_num").cast(pl.Int64),
            pl.col("steamid").cast(pl.UInt64),
            pl.col("side").cast(pl.Utf8),
            pl.col("clutch_type").cast(pl.Utf8),
            pl.col("won").cast(pl.Boolean),
        ]
    )

    summary = (
        clutch_rounds.group_by("steamid")
        .agg(
            [
                pl.len().cast(pl.Int64).alias("total_clutches"),
                pl.col("won").cast(pl.Int64).sum().cast(pl.Int64).alias("clutches_won"),
            ]
        )
        .join(
            clutch_rounds.filter(pl.col("clutch_type") == "1v1")
            .group_by("steamid")
            .agg(
                [
                    pl.col("won").cast(pl.Int64).sum().cast(pl.Int64).alias("v1_won"),
                    pl.len().cast(pl.Int64).alias("v1_total"),
                ]
            ),
            on="steamid",
            how="left",
        )
        .join(
            clutch_rounds.filter(pl.col("clutch_type") == "1v2")
            .group_by("steamid")
            .agg(
                [
                    pl.col("won").cast(pl.Int64).sum().cast(pl.Int64).alias("v2_won"),
                    pl.len().cast(pl.Int64).alias("v2_total"),
                ]
            ),
            on="steamid",
            how="left",
        )
        .join(
            clutch_rounds.filter(pl.col("clutch_type") == "1v3+")
            .group_by("steamid")
            .agg(
                [
                    pl.col("won").cast(pl.Int64).sum().cast(pl.Int64).alias("v3plus_won"),
                    pl.len().cast(pl.Int64).alias("v3plus_total"),
                ]
            ),
            on="steamid",
            how="left",
        )
        .with_columns(
            [
                pl.col("v1_won").fill_null(0),
                pl.col("v1_total").fill_null(0),
                pl.col("v2_won").fill_null(0),
                pl.col("v2_total").fill_null(0),
                pl.col("v3plus_won").fill_null(0),
                pl.col("v3plus_total").fill_null(0),
            ]
        )
    )
    summary = summary.with_columns(
        pl.when(pl.col("total_clutches") > 0)
        .then((pl.col("clutches_won") / pl.col("total_clutches") * 100.0).round(2))
        .otherwise(0.0)
        .alias("win_rate")
    )

    if summary.is_empty():
        return {"clutch_rounds": clutch_rounds, "clutch_summary": _empty_clutch_summary()}

    name_rows = [{"steamid": sid, "name": name} for sid, name in name_lookup.items()]
    names_df = (
        pl.DataFrame(name_rows, schema={"steamid": pl.UInt64, "name": pl.Utf8})
        if name_rows
        else pl.DataFrame(schema={"steamid": pl.UInt64, "name": pl.Utf8})
    )
    if not names_df.is_empty():
        summary = summary.join(names_df, on="steamid", how="left")
    summary = summary.with_columns(pl.col("name").fill_null(""))

    clutch_summary = summary.select(
        [
            "steamid",
            "name",
            "total_clutches",
            "clutches_won",
            "win_rate",
            "v1_won",
            "v1_total",
            "v2_won",
            "v2_total",
            "v3plus_won",
            "v3plus_total",
        ]
    )

    return {"clutch_rounds": clutch_rounds, "clutch_summary": clutch_summary}
