from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


REPO_ROOT = SRC_DIR.parent
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "situations" / "all_situations.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the combined situations dataset.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Parquet dataset to inspect.",
    )
    return parser.parse_args()


def _print_counter(title: str, counter: Counter[str]) -> None:
    print(title)
    if not counter:
        print("  none")
        return
    for key, count in sorted(counter.items()):
        print(f"  {key}: {count}")


def _frame_counter(frame: pl.DataFrame, column: str) -> Counter[str]:
    if frame.is_empty() or column not in frame.columns:
        return Counter()
    rows = frame.select(column).to_series().to_list()
    return Counter(str(value) if value is not None else "None" for value in rows)


def _source_flag_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        for flag in row.get("source_flags") or []:
            counter[str(flag)] += 1
    return counter


def _null_count(frame: pl.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return frame.height
    return int(frame.select(pl.col(column).is_null().sum()).item())


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Dataset not found: {input_path}")

    dataset = pl.read_parquet(input_path)
    rows = dataset.to_dicts()

    print(f"total rows: {dataset.height}")
    _print_counter("rows per match", _frame_counter(dataset, "match_id"))
    _print_counter("rows per map", _frame_counter(dataset, "map_name"))
    _print_counter("rows per situation_type", _frame_counter(dataset, "situation_type"))
    _print_counter("rows per source_flag", _source_flag_counter(rows))

    print("null counts")
    for column in ["tick", "side", "weapon", "ml_impact", "damage_before_death"]:
        print(f"  {column}: {_null_count(dataset, column)}")

    print(f"high_impact_kill_count: {sum(1 for row in rows if row.get('high_impact_kill') is True)}")
    print(f"low_impact_kill_count: {sum(1 for row in rows if row.get('low_impact_kill') is True)}")
    print(f"high_cost_death_count: {sum(1 for row in rows if row.get('high_cost_death') is True)}")
    print(f"low_cost_death_count: {sum(1 for row in rows if row.get('low_cost_death') is True)}")


if __name__ == "__main__":
    main()