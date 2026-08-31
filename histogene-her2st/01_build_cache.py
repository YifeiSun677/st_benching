#!/usr/bin/env python
"""Build the 112x112 patch cache and the expression cache. Run once per pod.

    python scripts/01_build_cache.py
    python scripts/01_build_cache.py --force          # rebuild everything
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from htg import cache, config as C, her2st   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--panel", default=str(C.PANEL_FILE))
    ap.add_argument("--sections", nargs="*", default=None)
    args = ap.parse_args()

    panel = her2st.load_panel(args.panel)
    print(f"her2st dir : {C.HER2ST_DIR}")
    print(f"cache dir  : {C.CACHE_DIR}")
    print(f"panel      : {args.panel} ({len(panel)} genes)")
    print(f"sections   : {len(her2st.section_names())}")

    t0 = time.time()
    cache.build_patch_cache(args.sections, force=args.force)
    print(f"patch cache done in {(time.time()-t0)/60:.1f} min")

    t1 = time.time()
    cache.build_expr_cache(panel, args.sections, force=args.force)
    print(f"expression cache done in {(time.time()-t1)/60:.1f} min")

    cov = her2st.panel_coverage(panel)
    cov.to_csv(C.CACHE_DIR / "panel_coverage.csv", index=False)
    print(cov.to_string(index=False))
    print(f"\nmean genes present per section: {cov['present'].mean():.1f} / {len(panel)}")


if __name__ == "__main__":
    main()
