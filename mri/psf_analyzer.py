"""
Point spread function analysis for sampling masks.

The PSF is the inverse transform of the mask on its own, so it predicts the
artifact a mask will produce before any image has been reconstructed.
"""
from __future__ import annotations
import numpy as np

from mri.kspace_core import from_kspace, log_magnitude

__all__ = ["psf","psf_display","psf_profile","ghost_offsets","peak_sidelobe",
           "predicted_spacing"] # Only these will be exported

# ---------------------
# Methods to call from
# ---------------------

def psf(mask:np.ndarray)->np.ndarray:
    """
    Spatial response of a mask, with the main lobe centred.

    A mask is just a k-space matrix of ones and zeros, so running it back
    through the inverse transform shows where its energy lands in the image.
    """
    return np.fft.fftshift(from_kspace(np.asarray(mask,dtype=np.float64)))

def psf_display(mask:np.ndarray)->np.ndarray:
    """Log-compressed PSF in [0,1]. Side lobes are invisible on a linear scale."""
    return log_magnitude(psf(mask))

def psf_profile(mask:np.ndarray)->np.ndarray:
    """
    Central column through the PSF.

    Row-skipping spreads energy along the row axis, so this cut is where the
    ghost copies show up.
    """
    p = psf(mask)
    return p[:, p.shape[1]//2]

def ghost_offsets(mask:np.ndarray,threshold:float=0.05)->np.ndarray:
    """
    Row offsets from the main lobe carrying at least `threshold` of its height.

    For a uniform mask of acceleration R this returns exactly R entries, spaced
    `predicted_spacing` apart. Offset 0 is the main lobe itself.
    """
    p = psf_profile(mask)
    p = p/p.max()
    return np.flatnonzero(p >= threshold) - len(p)//2

def peak_sidelobe(mask:np.ndarray)->float:
    """
    Height of the strongest ghost relative to the main lobe.

    Near 1.0 means a ghost as bright as the anatomy, which is what regular
    skipping produces. Small values mean the energy was scattered instead.
    """
    p = psf_profile(mask)
    p = p/p.max()
    c = len(p)//2
    p[max(0,c-2):c+3] = 0.0 # blank the main lobe and its immediate skirt
    return float(p.max())

def predicted_spacing(shape:tuple[int,int],R:int)->float:
    """Rows between successive ghosts for a uniform mask: N/R."""
    return shape[0]/R

# --------------------------------
# Tests for psf_analyzer.py only
# --------------------------------

if __name__ == "__main__":
    from mri.cartesian_sampling import make_mask

    shape = (256,256)

    # --- Full sampling: one spike, nothing else ---
    m_full = make_mask("uniform", shape, R=1)
    off_full = ghost_offsets(m_full)
    assert len(off_full) == 1 and off_full[0] == 0, off_full
    assert peak_sidelobe(m_full) < 1e-9, "full sampling must have no side lobes"
    print(f"[R=1]  ghosts={len(off_full)}  offsets={off_full.tolist()}  sidelobe={peak_sidelobe(m_full):.2e}")

    # --- Uniform masks: exactly R spikes, spaced N/R apart ---
    for R in (2,4,8):
        m = make_mask("uniform", shape, R=R)
        off = ghost_offsets(m)
        gaps = np.diff(off)
        assert len(off) == R, f"R={R} expected {R} spikes, got {len(off)}"
        assert np.all(gaps == shape[0]//R), f"R={R} spacing {gaps} != {shape[0]//R}"
        assert abs(gaps[0] - predicted_spacing(shape,R)) < 1e-9
        print(f"[R={R}]  ghosts={len(off)}  spacing={gaps[0]}  predicted={predicted_spacing(shape,R):.0f}  sidelobe={peak_sidelobe(m):.3f}")

    # --- Row skipping must ghost vertically, not horizontally ---
    # Compare energy away from the main lobe: the ghosts sit in one column only.
    p = psf(make_mask("uniform", shape, R=2))
    cy,cx = shape[0]//2, shape[1]//2
    col = p[:,cx].copy(); col[cy] = 0.0
    row = p[cy,:].copy(); row[cx] = 0.0
    assert col.sum() > 0.1, "no vertical ghost found"
    assert row.sum() < 1e-9, "ghosting is running sideways, the mask is transposed"
    print(f"[direction]  off-lobe column={col.sum():.3f}  off-lobe row={row.sum():.3e}")

    # --- Randomness trades one bright ghost for scattered grain ---
    s_uniform = peak_sidelobe(make_mask("uniform", shape, R=4))
    s_random = peak_sidelobe(make_mask("random", shape, R=4, seed=42))
    assert s_random < s_uniform/2, (s_uniform, s_random)
    print(f"[incoherence]  uniform_sidelobe={s_uniform:.3f}  random_sidelobe={s_random:.3f}")

    # --- Variable density keeps the centre, so its main lobe stays dominant ---
    s_vds = peak_sidelobe(make_mask("variable_density", shape, R=4))
    print(f"[vds]  sidelobe={s_vds:.3f}")

    print("\nAll self-tests passed")
