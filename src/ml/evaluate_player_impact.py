from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sectors.player_ml_impact import build_player_ml_impact_summary, format_player_ml_impact_summary


REPO_ROOT = SRC_DIR.parent
DEFAULT_IMPACT_PATH = REPO_ROOT / "data" / "ml" / "ml_event_impact.parquet"
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate normal-kill ML impact for one player.",
    )
    parser.add_argument(
        "--steamid",
        required=True,
        help="Selected player's SteamID.",
    )
    parser.add_argument(
        "--impact-path",
        type=Path,
        default=DEFAULT_IMPACT_PATH,
        help=f"Path to full ML event impact parquet (default: {DEFAULT_IMPACT_PATH})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="How many top events to show in each section (default: 5).",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.top_n <= 0:
        raise SystemExit("--top-n must be > 0.")
    if not args.impact_path.exists():
        raise SystemExit(f"Impact file not found: {args.impact_path}")

    impact = pl.read_parquet(args.impact_path)
    LOGGER.info("impact_loaded=%s path=%s", impact.height, args.impact_path)
    if impact.is_empty():
        raise SystemExit(f"Impact file is empty: {args.impact_path}")

    summary = build_player_ml_impact_summary(
        ml_event_impact=impact,
        selected_steamid=args.steamid,
        top_n=args.top_n,
    )
    print(format_player_ml_impact_summary(summary))


if __name__ == "__main__":
    main()
