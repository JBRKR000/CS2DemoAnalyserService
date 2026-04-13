import logging
from pathlib import Path

import colorlog

from Analyser import load_demo_for_analysis, analyse_demo
from Parser import get_demo


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


def _pick_latest_cache_key() -> str | None:
    if not CACHE_DIR.exists():
        return None

    cache_files = sorted(CACHE_DIR.glob("*.pkl"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not cache_files:
        return None
    return cache_files[0].stem


def main() -> None:
    cache_key: str | None = None

    if DEMO_PATH.exists():
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
    analyse_demo(demo_for_analysis)


if __name__ == "__main__":
    main()
