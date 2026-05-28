from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.dataset import DEFAULT_OUTPUT_PATH, build_combined_snapshot_dataset, discover_cache_keys, save_snapshot_dataset


def main() -> None:
    cache_keys = discover_cache_keys()
    dataset = build_combined_snapshot_dataset(cache_keys=cache_keys)
    output_path = save_snapshot_dataset(dataset, DEFAULT_OUTPUT_PATH)

    print(f"Saved {dataset.height} rows from {len(cache_keys)} cached demo(s) to {output_path}")


if __name__ == "__main__":
    main()
