#!/usr/bin/env python
"""Stage 3 -- non-overlapping 7-spot hexagons (centre + 6 neighbours) per Visium section.

Visium array indices are doubled hex coordinates: col-row is even, neighbours of
(row,col) are (row,col+-2) and (row+-1,col+-1).  Axial q=(col-row)/2, r=row.
The 7-hex flower tiling has centres on the sublattice spanned by axial (2,1) and
(-1,3) (squared hex length 7, 60 deg apart), i.e. centre  <=>  (3q + r) mod 7 == 0.
Only complete hexagons (all 7 spots kept by the export) are used.

writes: /workspace/ext/calib/pseudospots_<SEC>.npz  (member_ids [n,7] centre first,
        centre_ids [n]) and pseudospots_<SEC>.png
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common as K

NB = [(0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1)]


def pseudo_groups(rows, cols):
    key = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rows, cols))}
    assert all((c - r) % 2 == 0 for r, c in key), "array_col - array_row must be even (doubled coords)"
    used = np.zeros(len(rows), bool)
    groups = []
    for (r, c), i in key.items():
        q = (c - r) // 2
        if (3 * q + r) % 7:
            continue
        members = [i] + [key.get((r + dr, c + dc)) for dr, dc in NB]
        if any(m is None for m in members):
            continue
        assert not used[members].any(), "overlapping hexagons: tiling rule broken"
        used[members] = True
        groups.append(members)
    return np.array(groups, dtype=int).reshape(-1, 7), used


def main():
    secs = sys.argv[1:] or K.VISIUM_SECTIONS
    for sec in secs:
        sp = K.read_spots(sec, K.VIS_ROOT)
        groups, used = pseudo_groups(sp.array_row.values, sp.array_col.values)
        ids = np.array(sp.index)
        member_ids = ids[groups]
        np.savez(K.CALIB / f"pseudospots_{sec}.npz", member_ids=member_ids,
                 centre_ids=member_ids[:, 0], n_spots=len(sp))
        cover = used.mean()
        print(f"{sec}: {len(sp)} spots -> {len(groups)} hexagons, {100*cover:.1f}% of spots covered "
              f"(ideal ~{len(sp)//7} groups; edge losses expected)")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(sp.x_eq, sp.y_eq, s=3, c="lightgrey")
            gid = np.full(len(sp), -1)
            for g, m in enumerate(groups):
                gid[m] = g
            m = gid >= 0
            ax.scatter(sp.x_eq[m], sp.y_eq[m], s=3, c=(gid[m] * 7919) % 20, cmap="tab20")
            ax.scatter(sp.x_eq.values[groups[:, 0]], sp.y_eq.values[groups[:, 0]], s=1, c="k")
            ax.invert_yaxis(); ax.set_aspect("equal"); ax.set_title(f"{sec}: {len(groups)} pseudo-spots")
            fig.savefig(K.CALIB / f"pseudospots_{sec}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print("  (plot skipped:", e, ")")


if __name__ == "__main__":
    main()
