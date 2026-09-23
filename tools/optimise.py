"""
Search the settings space for the best reconstruction of one real slice.

The point of this script is to show that the answer is not obvious. The
defaults are not optimal, the optimum is not at either extreme, and PSNR and
SSIM do not agree about where it is -- so "best" is a judgement, not a formula.

    python tools/optimise.py                         # default file and slice
    python tools/optimise.py <file.h5> <slice>

The reference is the average of every repetition of the same scan found on
disk. That is genuinely a better picture than any single acquisition, and it
was not made from the file being reconstructed, so improving against it means
something.
"""
from __future__ import annotations
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from mri.raw_data import load_raw_kspace, reference_image, find_repetitions
from mri.pipeline import reconstruct

DEFAULT_FILE = "D:/mri_data/raw/2022083101_T201.h5"
DEFAULT_SLICE = 12

# The axes of the search. Every combination of these is tried.
GRID = {
    "window":   ["none", "hamming", "gaussian"],
    "sigma":    [0.35, 0.45, 0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00, 1.20, 1.50],
    "dc_scale": [0.85, 0.95, 1.00, 1.05, 1.15],
    "rate":     [1.00],
}


def combinations(grid: dict) -> list[dict]:
    """Every point of the grid, minus the ones that would be duplicates."""
    keys = list(grid)
    out, seen = [], set()
    for values in itertools.product(*(grid[k] for k in keys)):
        s = dict(zip(keys, values))
        # sigma only means anything to the gaussian window, so collapse the
        # duplicates instead of reconstructing the same image seven times.
        if s["window"] != "gaussian":
            s["sigma"] = GRID["sigma"][0]
        key = tuple(sorted(s.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else DEFAULT_FILE
    z = int(argv[2]) if len(argv) > 2 else DEFAULT_SLICE

    reps = find_repetitions(path)
    k = load_raw_kspace(path, z)
    ref, src = reference_image(path, z)
    print(f"file      {Path(path).name}   slice {z}")
    print(f"reference {src}  ({len(reps)} file(s) on disk)")
    if len(reps) < 2:
        print("\n  WARNING: only one acquisition, so the reference is a reconstruction of")
        print("  this same data. The search will simply find 'change nothing'.\n")

    grid = combinations(GRID)
    print(f"searching {len(grid)} combinations...\n")

    base = reconstruct(None, {}, kspace=k, reference=ref)["metrics"]
    t0 = time.perf_counter()
    results = []
    for i, s in enumerate(grid, 1):
        m = reconstruct(None, s, kspace=k, reference=ref)["metrics"]
        results.append((m["ssim"], m["psnr"], s))
        if i % 10 == 0 or i == len(grid):
            print(f"  {i}/{len(grid)}", end="\r", flush=True)
    print(f"  {len(grid)} combinations in {time.perf_counter()-t0:.1f}s        \n")

    def show(title, rows, key):
        """Print every varying parameter, so a winning row can be reproduced."""
        print(title)
        print(f"  {'window':<10}{'sigma':>7}{'lowpass':>9}{'highp':>7}{'DC':>7}"
              f"{'ssim':>9}{'psnr':>9}{'vs default':>12}")
        for ssim, psnr, s in rows:
            sig = f"{s['sigma']:.2f}" if s["window"] == "gaussian" else "-"
            lp = s["low_pass"] or "-"
            hp = s["high_pass"] or "-"
            delta = (ssim - base["ssim"]) if key == "ssim" else (psnr - base["psnr"])
            print(f"  {s['window']:<10}{sig:>7}{str(lp):>9}{str(hp):>7}"
                  f"{s['dc_scale']:>7.2f}{ssim:>9.4f}{psnr:>9.2f}{delta:>+12.4f}")
        print()

    by_ssim = sorted(results, key=lambda r: -r[0])
    by_psnr = sorted(results, key=lambda r: -r[1])

    print(f"DEFAULT (nothing switched on):  ssim={base['ssim']:.4f}  psnr={base['psnr']:.2f}\n")
    show("BEST 5 BY SSIM", by_ssim[:5], "ssim")
    show("BEST 5 BY PSNR", by_psnr[:5], "psnr")
    show("WORST 3 BY SSIM  (over-filtering costs more than it saves)", by_ssim[-3:], "ssim")

    b_ssim, b_psnr = by_ssim[0], by_psnr[0]
    def recipe(s):
        bits = [f"window={s['window']}"]
        if s["window"] == "gaussian": bits.append(f"sigma={s['sigma']:.2f}")
        if s["low_pass"]:  bits.append(f"low_pass={s['low_pass']}")
        if s["high_pass"]: bits.append(f"high_pass={s['high_pass']}")
        if s["dc_scale"] != 1.0: bits.append(f"DC={s['dc_scale']:.2f}")
        return ", ".join(bits)
    print(f"SET THESE FOR BEST SSIM:  {recipe(b_ssim[2])}")
    print(f"SET THESE FOR BEST PSNR:  {recipe(b_psnr[2])}")
    print()
    print(f"best SSIM: {b_ssim[0]:.4f}  ({b_ssim[0]-base['ssim']:+.4f} vs default)")
    print(f"best PSNR: {b_psnr[1]:.2f}   ({b_psnr[1]-base['psnr']:+.2f} dB vs default)")
    if b_ssim[2] != b_psnr[2]:
        print("\nThe two metrics disagree about which setting is best. SSIM rewards keeping")
        print("structure, PSNR rewards suppressing noise, and filtering trades one for the")
        print("other -- so there is no single answer, only a choice about what matters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
