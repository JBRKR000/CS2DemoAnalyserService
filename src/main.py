import logging
from pathlib import Path

import colorlog

from Analyser import load_demo_for_analysis, analyse_demo
from Parser import compute_file_sha256, get_demo


def configure_logging(level: int = logging.INFO) -> None:
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(asctime)s | %(log_color)s%(levelname)-8s%(reset)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


configure_logging()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_PATH = BASE_DIR / "Demos" / "demo.dem"
CACHE_DIR = BASE_DIR / ".cache"
CACHE_KEY_PATH = BASE_DIR / "last_cache_key.txt"


def _format_float(value: object, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "-"


def _log_benchmark_block(title: str, evaluations: dict) -> None:
    if not isinstance(evaluations, dict) or not evaluations:
        logger.info("%s: no data", title)
        return

    metric_order = [
        "adr",
        "kast",
        "kpr",
        "hs_percent",
        "opening_duel_win_pct",
        "full_buy_win_rate",
        "force_win_rate",
        "clutch_win_rate",
    ]

    metric_labels = {
        "adr": "ADR",
        "kast": "KAST",
        "kpr": "KPR",
        "hs_percent": "HS%",
        "opening_duel_win_pct": "OPEN%",
        "full_buy_win_rate": "FULLBUY%",
        "force_win_rate": "FORCE%",
        "clutch_win_rate": "CLUTCH%",
    }

    logger.info("%s", title)
    logger.info("%-10s | %-9s | %-8s | %-7s | %-12s | %-12s", "Metric", "Value", "Percentyl", "Rating", "Kontekst", "Powod")
    logger.info("%s", "-" * 76)

    for metric in metric_order:
        evaluation = evaluations.get(metric)
        if not isinstance(evaluation, dict):
            continue

        value = _format_float(evaluation.get("value"))
        percentile_raw = evaluation.get("percentile")
        percentile = _format_float(percentile_raw, 1) if percentile_raw is not None else "-"
        rating = str(evaluation.get("rating", "-"))
        context = str(evaluation.get("context", "-"))
        reason = str(evaluation.get("reason", "-")) if evaluation.get("reason") is not None else "-"
        logger.info(
            "%-10s | %-9s | %-8s | %-7s | %-12s | %-12s",
            metric_labels.get(metric, metric),
            value,
            percentile,
            rating,
            context,
            reason,
        )


def _pick_latest_cache_key() -> str | None:
    if not CACHE_DIR.exists():
        return None

    cache_files = sorted(CACHE_DIR.glob("*.pkl"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not cache_files:
        return None
    return cache_files[0].stem


def main() -> None:
    cache_key: str | None = None
    match_id: str | None = None

    if DEMO_PATH.exists():
        match_id = compute_file_sha256(DEMO_PATH)
        result = get_demo(
            demo_path=DEMO_PATH,
            cache_dir=str(CACHE_DIR),
            return_cache_key=True,
            delete_source=False,
        )
        if not isinstance(result, tuple):
            raise TypeError("Expected (demo, cache_key) from get_demo when return_cache_key=True")

        _, cache_key = result
        CACHE_KEY_PATH.write_text(cache_key, encoding="utf-8")
        logger.info("Saved cache key to %s", CACHE_KEY_PATH)
    elif CACHE_KEY_PATH.exists():
        cache_key = CACHE_KEY_PATH.read_text(encoding="utf-8").strip()
        logger.info("Using cache key from %s", CACHE_KEY_PATH)
    else:
        cache_key = _pick_latest_cache_key()
        if cache_key is None:
            raise FileNotFoundError(
                "No source demo and no cache found. Put a .dem in Demos/ or create cache first."
            )
        CACHE_KEY_PATH.write_text(cache_key, encoding="utf-8")
        logger.info("last_cache_key.txt missing, using latest cache key: %s", cache_key)

    demo_for_analysis = load_demo_for_analysis(
        str(CACHE_KEY_PATH),
        cache_dir=str(CACHE_DIR),
        verbose=True,
    )
    logger.info("Demo ready for analysis | type=%s", type(demo_for_analysis).__name__)
    if match_id is None:
        logger.info("No source .dem file available; benchmark append will be skipped for this analysis.")
    analysis = analyse_demo(demo_for_analysis, match_id=match_id)

    player = analysis.get("selected_player_stats", {}) or {}
    econ = analysis.get("economy_summary_selected", {}) or {}
    clutch = analysis.get("clutch_summary_selected", {}) or {}
    benchmark_evals = analysis.get("benchmark_evaluations", {}) or {}
    benchmark_all = analysis.get("benchmark_evaluations_all", {}) or {}
    benchmark_ct = analysis.get("benchmark_evaluations_ct", {}) or {}
    benchmark_t = analysis.get("benchmark_evaluations_t", {}) or {}
    feedback = analysis.get("feedback", []) or []

    logger.info(
        "Selected player: %s (%s)",
        player.get("name", "Unknown"),
        player.get("steamid", "Unknown"),
    )
    logger.info("Economy summary (selected): %s", econ)
    logger.info("Clutch summary (selected): %s", clutch)
    logger.info(
        "Benchmark pool: source=%s before=%s after=%s",
        analysis.get("benchmark_pool_source", "-"),
        analysis.get("benchmark_pool_size_before_append", "-"),
        analysis.get("benchmark_pool_size_after_append", "-"),
    )
    _log_benchmark_block("Benchmark evaluations (ALL)", benchmark_all if benchmark_all else benchmark_evals)
    _log_benchmark_block("Benchmark evaluations (CT)", benchmark_ct)
    _log_benchmark_block("Benchmark evaluations (T)", benchmark_t)
    logger.info("Feedback tips count: %d", len(feedback))
    for idx, tip in enumerate(feedback, start=1):
        logger.info(
            "Tip %d | [%s/%s] %s | %s",
            idx,
            tip.get("severity", ""),
            tip.get("category", ""),
            tip.get("title", ""),
            tip.get("message", ""),
        )


if __name__ == "__main__":
    main()
