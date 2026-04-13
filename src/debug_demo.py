# debug_demo.py — uruchom w katalogu Analysis
from pathlib import Path
from Parser import load_cached_demo

cache_key = Path("last_cache_key.txt").read_text().strip()
demo = load_cached_demo(cache_key)

print("=== LISTING ALL ATTRIBUTES ===")
for k, v in vars(demo).items():
    import polars as pl
    if isinstance(v, pl.DataFrame):
        print(f"\n[DataFrame] {k}: shape={v.shape}, cols={v.columns}")
        if v.height > 0:
            print(v.head(2))
    elif isinstance(v, dict):
        print(f"\n[dict] {k}: keys={list(v.keys())[:10]}")
    else:
        print(f"\n[{type(v).__name__}] {k}: {str(v)[:100]}")