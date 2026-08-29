"""
Non-Cartesian sampling trajectories and their density compensation weights.
"""

from __future__ import annotations
import numpy as np

__all__ = ["radial", "spiral"]

def _ramp_weights(r:np.ndarray, n_arms:int)->np.ndarray:
    """Ramp |r|, floored so the centre sample is not annihilated."""
    w = np.abs(r).astype(np.float64)
    floor = 1.0/(4.0*max(n_arms, 1))
    w[w<floor] = floor
    return w/w.max()

def radial(shape:tuple[int, int], n_spokes:int=64, n_samples:int|None=None):
    """
    Radial spokes through the centre of k-space.
    Returns (coords, weights). coords is (N,2) of (row,col) offsets from the
    centre in pixels; weights is the matching density compensation ramp.
    """

    rows, cols = shape
    if n_samples is None:
        n_samples = max(rows, cols)

    kmax = min(rows, cols)/2.0
    r = np.linspace(-kmax, kmax, n_samples)
    theta = np.arange(n_spokes)*np.pi/n_spokes

    ky = np.outer(np.sin(theta), r).ravel()
    kx = np.outer(np.cos(theta), r).ravel()
    w = _ramp_weights(np.tile(r, n_spokes), n_spokes)
    return np.column_stack([ky,kx]), w

def spiral(shape:tuple[int, int], n_interleaves:int=16, n_turns:int=12, n_samples:int=2048):
    """Archimedean spiral arms, rotated copies of each other."""
    rows, cols = shape
    kmax = min(rows, cols)/2.0
    t = np.linspace(0.0, 1.0, n_samples)
    r = kmax*t
    base = 2*np.pi*n_turns*t

    kys, kxs, ws = [], [], []
    for i in range(n_interleaves):
        th = base + 2*np.pi*i/n_interleaves
        kys.append(r*np.sin(th)); kxs.append(r*np.cos(th))
        ws.append(r)

    w = _ramp_weights(np.concatenate(ws), n_interleaves)
    return np.column_stack([np.concatenate(kys), np.concatenate(kxs)]), w