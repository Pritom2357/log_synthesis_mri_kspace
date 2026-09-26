"""Apodization windows, peaked on the real DC sample."""
from __future__ import annotations
import numpy as np

from mri.kspace_core import radius_grid


def make_window(kind: str, shape: tuple[int, int], centre: tuple[int, int] | None = None,
                sigma: float = 0.35) -> np.ndarray:
    """kind: none | hamming (separable raised cosine) | gaussian (sigma in normalised radius)."""
    rows, cols = shape
    cy, cx = (rows//2, cols//2) if centre is None else centre
    if kind == "none":
        return np.ones(shape)
    if kind == "hamming":
        wu = 0.54 + 0.46*np.cos(np.pi*(np.arange(rows) - cy)/(rows/2))
        wv = 0.54 + 0.46*np.cos(np.pi*(np.arange(cols) - cx)/(cols/2))
        return np.outer(wu, wv)
    if kind == "gaussian":
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        r = radius_grid(shape, (cy, cx))/(min(rows, cols)/2)
        return np.exp(-r**2/(2*sigma**2))
    raise ValueError(f"Unknown window kind: {kind!r}. Expected: none, hamming, gaussian.")


def apply_window(k: np.ndarray, kind: str, **kw) -> np.ndarray:
    from mri.kspace_edit import dc_index
    return k*make_window(kind, k.shape[-2:], centre=dc_index(k), **kw)
