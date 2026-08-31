"""Build the 112x112 patch cache and the expression cache. Run once per pod.

    python -m histogene.build_cache
    python -m histogene.build_cache --force
"""
from __future__ import annotations

import argparse
import time

from . import cache, config as C, her2st


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
