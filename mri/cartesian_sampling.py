"""Cartesian sampling masks: evenly spaced rows at a fraction of the Nyquist rate, and scan progress."""
from __future__ import annotations
import numpy as np


def nyquist_mask(shape: tuple[int, int], rate: float = 1.0, centre: int | None = None) -> np.ndarray:
    """Keeps round(rate*rows) evenly spaced rows, anchored on the DC row so it is never dropped."""
    rows = shape[0]
    if not 0 < rate <= 1.0:
        raise ValueError("rate must be in (0, 1]")
    n_keep = max(1, int(round(rate*rows)))
    centre = rows//2 if centre is None else int(centre)
    keep = (centre + np.round(np.arange(n_keep)*rows/n_keep).astype(int)) % rows
    mask = np.zeros(shape, dtype=np.uint8)
    mask[np.unique(keep), :] = 1
    return mask


def acquisition_mask(shape: tuple[int, int], fraction: float, order: str = "linear",
                     centre: int | None = None) -> np.ndarray:
    """Samples recorded after `fraction` of the scan: rows top-down (linear) or DC-outward (centric)."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    rows, cols = shape
    if order == "linear":
        row_order = np.arange(rows)
    elif order == "centric":
        c = rows//2 if centre is None else int(centre)
        below, above = np.arange(c, rows), np.arange(c - 1, -1, -1)
        n = min(len(below), len(above))
        row_order = np.empty(rows, dtype=int)
        row_order[0:2*n:2], row_order[1:2*n:2] = below[:n], above[:n]
        row_order[2*n:] = np.concatenate([below[n:], above[n:]])
    else:
        raise ValueError(f"Unknown order: {order!r}. Expected: linear, centric.")
    full, partial = divmod(int(round(fraction*rows*cols)), cols)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[row_order[:full], :] = 1
    if partial and full < rows:
        mask[row_order[full], :partial] = 1  # the line being read right now
    return mask


if __name__ == "__main__":
    shape = (256, 256)
    prev = 0
    for rate in (0.25, 0.5, 0.75, 0.9, 1.0):
        kept = int(nyquist_mask(shape, rate)[:, 0].sum())
        assert nyquist_mask(shape, rate)[128, 0] == 1 and kept >= prev and abs(kept/256 - rate) < 0.02
        prev = kept
    assert nyquist_mask(shape, 1.0).all()
    for bad in (0.0, -0.5, 1.5):
        try:
            nyquist_mask(shape, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    for order in ("linear", "centric"):
        assert acquisition_mask(shape, 0.0, order).sum() == 0 and acquisition_mask(shape, 1.0, order).all()
        assert acquisition_mask(shape, 0.5, order).sum() == 256*128
    lin, cen = acquisition_mask(shape, 0.25, "linear"), acquisition_mask(shape, 0.25, "centric")
    assert lin[:64].all() and not lin[64:].any()
    assert cen[96:160].all() and not cen[:96].any()
    part = acquisition_mask(shape, (3*256 + 10)/256**2, "linear")
    assert part[3, :10].all() and not part[3, 10:].any()
    for rate in (0.9, 0.8, 0.7, 0.6, 0.5):
        assert nyquist_mask(shape, rate, centre=125)[125, 0] == 1
    first = acquisition_mask(shape, 1/256, "centric", centre=125)
    assert first[125].all() and first.sum() == 256
    print("cartesian_sampling: all self-tests passed")
