#!/usr/bin/env python3
"""
inspect_data.py - look at a recorded dataset before training on it.

Worth running whenever training accuracy looks wrong, because most bad
datasets turn out to be one of a few mechanical problems:

  - a class was recorded but every sample was filtered out by --min-energy
    (a slow gesture moves few pixels per frame)
  - the trigger fired before the gesture started, so samples contain mostly
    the hand at rest
  - a label was confused during capture (a mislabelled class shows up as two
    labels with near-identical MHIs)
  - frames are blown out or almost black, meaning the lighting was bad

    python3 pc/inspect_data.py --data data/raw
    python3 pc/inspect_data.py --data data/raw --compare 0 1
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pc"))

import mhi  # noqa: E402


def load_class(data_dir: Path, li: int, verbose: bool = False):
    d = data_dir / f"label{li}"
    if not d.is_dir():
        return [], []
    mhis, energies, bright = [], [], []
    for f in sorted(d.glob("*.bin")):
        raw = f.read_bytes()
        if len(raw) != mhi.MHI_SEQ_LEN * mhi.MHI_PIXELS:
            if verbose:
                print(f"    {f.name}: wrong size {len(raw)}, skipped")
            continue
        seq = np.frombuffer(raw, dtype=np.uint8).reshape(
            mhi.MHI_SEQ_LEN, mhi.MHI_H, mhi.MHI_W)
        mhis.append(mhi.sequence_to_mhi(seq))
        e = mhi.mhi_energy(seq)
        energies.append(int(e.max()) if e.size else 0)
        bright.append(float(seq.mean()))
    return mhis, energies, bright


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--compare", nargs=2, type=int, metavar=("A", "B"),
                    help="report how different two classes are")
    ap.add_argument("--min-energy", type=int, default=25)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    data_dir = Path(a.data)
    if not data_dir.is_dir():
        sys.exit(f"error: {data_dir} not found")

    classes = sorted([int(p.name[5:]) for p in data_dir.glob("label*")
                      if p.is_dir()])
    if not classes:
        sys.exit(f"error: no label* directories in {data_dir}")

    print(f"dataset: {data_dir}\n")
    print(f"{'label':<8}{'files':>7}{'ok':>6}{'drop':>6}{'energy min':>12}"
          f"{'energy max':>12}{'mean lum':>10}")
    print("-" * 61)

    per_class_mhis = {}
    problems = []

    for li in classes:
        mhis, energies, bright = load_class(data_dir, li, a.verbose)
        if not mhis:
            print(f"{li:<8}{0:>7}{0:>6}{0:>6}{'-':>12}{'-':>12}{'-':>10}")
            problems.append(f"label {li}: no readable samples")
            continue

        keep = [e >= a.min_energy for e in energies]
        n_ok = sum(keep)
        per_class_mhis[li] = np.stack(mhis)

        print(f"{li:<8}{len(mhis):>7}{n_ok:>6}{len(mhis) - n_ok:>6}"
              f"{min(energies):>12}{max(energies):>12}"
              f"{np.mean(bright):>10.1f}")

        if n_ok == 0:
            problems.append(
                f"label {li}: every sample below --min-energy {a.min_energy}. "
                f"Peak energy is only {max(energies)}. Lower the threshold, or "
                "re-record with more lighting contrast")
        elif n_ok < 0.5 * len(mhis):
            problems.append(f"label {li}: {len(mhis) - n_ok}/{len(mhis)} "
                            f"samples below --min-energy {a.min_energy}")
        if n_ok and n_ok < 60:
            problems.append(f"label {li}: only {n_ok} usable samples. Below "
                            "~100 expect overfitting")
        mean_lum = float(np.mean(bright))
        if mean_lum < 8:
            problems.append(f"label {li}: mean luminance {mean_lum:.1f} - the "
                            "frames are nearly black")
        elif mean_lum > 200:
            problems.append(f"label {li}: mean luminance {mean_lum:.1f} - the "
                            "frames are blown out")

    # --- cross-class separation ----------------------------------------
    print("\npairwise MHI distance (mean |difference| per pixel, 0-255):")
    print("      " + "".join(f"{c:>8}" for c in classes))
    dist = {}
    for i in classes:
        if i not in per_class_mhis:
            continue
        row = []
        for j in classes:
            if j not in per_class_mhis:
                row.append(float("nan"))
                continue
            a_ = per_class_mhis[i].astype(np.float32).mean(axis=0)
            b_ = per_class_mhis[j].astype(np.float32).mean(axis=0)
            d = float(np.abs(a_ - b_).mean())
            dist[(i, j)] = d
            row.append(d)
        print(f"  {i:<4}" + "".join(f"{v:>8.1f}" for v in row))

    small = [(k, v) for k, v in dist.items()
             if k[0] < k[1] and v < 12.0]
    for (i, j), v in small:
        problems.append(f"labels {i} and {j} look nearly identical "
                        f"(distance {v:.1f}). Either the gestures are too "
                        f"similar at 64x64, or one was mislabelled")

    if a.compare:
        i, j = a.compare
        if i in per_class_mhis and j in per_class_mhis:
            print(f"\nmean MHI for label {i} vs {j}: "
                  f"distance {dist.get((i, j), float('nan')):.1f}")
            for li in (i, j):
                m = per_class_mhis[li].astype(np.float32).mean(axis=0)
                rows = []
                for r in range(0, mhi.MHI_H, 2):
                    rows.append("".join(
                        " .:-=+*#%@"[min(9, int(m[r, c] / 26))] if m[r, c] > 8
                        else " " for c in range(0, mhi.MHI_W, 1)))
                print(f"\nlabel {li}:")
                print("\n".join(rows))

    # --- verdict ---------------------------------------------------------
    print()
    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print(f"  - {p}")
        print("\nFix these before training. A model trained on data with these "
              "issues will look accurate and behave randomly on the device.")
        return 1

    print("no structural problems found. Proceed:")
    print(f"  python3 pc/train.py --data {data_dir} --min-energy {a.min_energy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
