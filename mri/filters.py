"""
Apodization windows and thermal noise.
"""

from __future__ import annotations
import numpy as np

from mri.kspace_core import radius_grid

__all__ = ["make_window", "apply_window", "add_noise"]


def __hamming_window(shape:tuple[int, int], centre:tuple[int, int]|None=None)->np.ndarray:
    """2D Hamming, built as the outer product of two 1D raised cosines."""
    rows, cols = shape
    cy, cx = (rows//2, cols//2) if centre is None else centre
    u = np.arange(rows) - cy
    v = np.arange(cols) - cx
    wu = 0.54 + 0.46*np.cos(np.pi*u/(rows/2))
    wv = 0.54 + 0.46*np.cos(np.pi*v/(cols/2))
    return np.outer(wu, wv)

def _gaussian_window(shape:tuple[int, int], sigma:float=0.35,
                     centre:tuple[int, int]|None=None):
    """2D Gaussian, radially symmetric, sigma in normalised radius."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    rows, cols = shape
    r = radius_grid(shape, centre)/(min(rows, cols)/2)
    return np.exp(-(r**2)/(2*sigma**2))

def make_window(kind:str, shape:tuple[int, int], **kw)->np.ndarray:
    """
    kind: 'none'|'hamming'|'gaussian'. centre= puts the peak somewhere other
    than the array middle, which is where the DC sample of scanner data is.
    """
    kind_norm = kind.strip().lower()
    if kind_norm == "none":
        return np.ones(shape, dtype=np.float64)
    elif kind_norm == "hamming":
        return __hamming_window(shape, kw.get("centre"))
    elif kind_norm == "gaussian":
        return _gaussian_window(shape, **kw)
    else:
        raise ValueError(
            f"Unknown window kind: {kind!r}. Expected: none, hamming, gaussian."
        )

def apply_window(k:np.ndarray, kind:str, **kw)->np.ndarray:
    """
    Multiplies the window element-wise into centred k-space.

    Built on the last two axes only, so multi-coil k-space of shape
    (coils, ky, kx) broadcasts: every channel gets the same window. The peak
    sits on the real DC sample, found rather than assumed.
    """
    from mri.kspace_edit import dc_index
    return k*make_window(kind, k.shape[-2:], centre=dc_index(k), **kw)

def add_noise(k:np.ndarray, sigma:float, seed:int|None=None)->np.ndarray:
    """
    Adds complex Gaussian noise. sigma is a fraction of the mean k-space
    magnitude, so the same slider value means the same visible grain on any
    image. sigma=0 returns k untouched.

    The mean is taken over measured (non-zero) samples only. Averaging in the
    rows that undersampling zeroed out used to make the noise weaker the fewer
    rows were kept -- half the rows, half the noise.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        return k

    rng = np.random.default_rng(seed)
    mag = np.abs(k)
    measured = mag[mag > 0]
    scale = sigma * float(measured.mean() if measured.size else 0.0)
    noise = rng.normal(0.0, scale, k.shape) + 1j*rng.normal(0.0, scale, k.shape)
    return k+noise