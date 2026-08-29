"""
Gridding: scattered non-Cartesian samples onto the Cartesian grid.
"""

from __future__ import annotations
import numpy as np

__all__ = ["sample_at", "grid_accumulate", "reconstruct_noncartesian"]

def sample_at(k:np.ndarray, coords:np.ndarray)->np.ndarray:
    """Bilinear sample of centred complex k-space at scattered offsets."""
    from scipy.ndimage import map_coordinates

    rows, cols = k.shape
    idx = np.vstack([coords[:, 0]+rows//2, coords[:, 1]+cols/2])
    re = map_coordinates(k.real, idx, order=1, mode="constant", cval=0.0)
    im = map_coordinates(k.imag, idx, order=1, mode="constant",cval=0.0)

    return re + 1j*im

def grid_accumulate(coords:np.ndarray,values:np.ndarray,shape:tuple[int,int],weights:np.ndarray|None=None):
    """
    True gridding: SPREAD each sample onto its four neighbouring grid points
    and ADD the contributions up.
    """
    rows,cols = shape
    out = np.zeros(shape,dtype=np.complex128)
    y = coords[:,0]+rows//2
    x = coords[:,1]+cols//2
    v = values if weights is None else values*weights
    y0 = np.floor(y).astype(int); x0 = np.floor(x).astype(int)
    fy = y-y0; fx = x-x0
    for dy in (0,1):
        for dx in (0,1):
            yy = y0+dy; xx = x0+dx
            wgt = (fy if dy else 1-fy)*(fx if dx else 1-fx)
            ok = (yy>=0)&(yy<rows)&(xx>=0)&(xx<cols)
            np.add.at(out,(yy[ok],xx[ok]),v[ok]*wgt[ok])
    return out

def reconstruct_noncartesian(k:np.ndarray,coords:np.ndarray,weights:np.ndarray,density_comp:bool=True):
    """Full non-Cartesian path: sample, grid, invert."""
    from mri.kspace_core import from_kspace
    vals = sample_at(k,coords)
    w = weights if density_comp else None
    return from_kspace(grid_accumulate(coords,vals,k.shape,w))