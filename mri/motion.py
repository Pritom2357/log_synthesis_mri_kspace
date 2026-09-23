"""
Patient motion during the scan, and the ghosting it causes.

A rigid shift of the anatomy is a linear phase ramp in k-space (the Fourier
shift theorem). A scan is not instantaneous: Cartesian k-space is filled one
line at a time, so if the patient moves partway through, some lines carry the
phase of the old position and some the new. That inconsistency between lines
is what shows up as ghosts.
"""
from __future__ import annotations
import numpy as np

__all__ = ["displacement_profile","apply_motion"] # Only these will be exported

# -----------------------------------------------------------------
# Displacement over scan time (private helpers are not exported)
# -----------------------------------------------------------------

def _centred_freqs(n:int)->np.ndarray:
    """Frequency coordinate of each row/column of centred k-space, in cycles per pixel."""
    return np.fft.fftshift(np.fft.fftfreq(n))

# ---------------------
# Methods to call from
# ---------------------

def displacement_profile(n_lines:int,amplitude:float,kind:str="sudden",
                         n_cycles:float=3.0,seed:int|None=None)->np.ndarray:
    """
    How far the patient has moved, in pixels, at the time each line is acquired.

    kind: 'none'|'sudden'|'periodic'|'random'
        sudden   - one jump halfway through the scan (a flinch)
        periodic - smooth oscillation (breathing), n_cycles over the scan
        random   - independent jitter per line (the worst case)

    Return: float64 array of length n_lines.
    """
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative")

    t = np.arange(n_lines)/n_lines # scan progress, 0 to 1
    kind_norm = kind.strip().lower() # input handling

    if kind_norm == "none" or amplitude == 0:
        return np.zeros(n_lines)
    elif kind_norm == "sudden":
        return np.where(t < 0.5, 0.0, amplitude)
    elif kind_norm == "periodic":
        return amplitude*np.sin(2*np.pi*n_cycles*t)
    elif kind_norm == "random":
        return np.random.default_rng(seed).normal(0.0,amplitude,n_lines)
    else:
        raise ValueError(
            f"Unknown motion kind: {kind!r}. Expected: none, sudden, periodic, random."
        )

def apply_motion(k:np.ndarray,amplitude:float,kind:str="sudden",
                 n_cycles:float=3.0,seed:int|None=None)->np.ndarray:
    """
    Corrupts centred Cartesian k-space with motion along the row axis.

    Row r of k-space was acquired at time r/N, by which point the patient had
    moved d[r] pixels. The shift theorem says that multiplies the row by
    exp(-2*pi*i*u_r*d[r]), where u_r is that row's frequency. Because d varies
    from row to row, the phases disagree, and disagreeing phases are ghosts.
    """
    rows,_ = k.shape
    d = displacement_profile(rows,amplitude,kind,n_cycles,seed)
    if not d.any():
        return k

    u = _centred_freqs(rows)
    return k*np.exp(-2j*np.pi*u*d)[:,None] # one phase factor per row

# --------------------------
# Tests for motion.py only
# --------------------------

if __name__ == "__main__":
    from mri.kspace_core import load_phantom, to_kspace, from_kspace
    from mri.metrics import compute_metrics

    img = load_phantom()
    k = to_kspace(img)
    shape = img.shape

    # --- Zero amplitude, or kind 'none', must change nothing at all ---
    assert apply_motion(k,0.0,"periodic") is k
    assert apply_motion(k,5.0,"none") is k
    assert not displacement_profile(256,5.0,"none").any()
    print("[no-op]      amplitude=0 and kind='none' both return k untouched")

    # --- The profiles have the shapes their names promise ---
    sud = displacement_profile(256,4.0,"sudden")
    per = displacement_profile(256,4.0,"periodic",n_cycles=3)
    assert sud[0] == 0.0 and sud[-1] == 4.0 and len(np.unique(sud)) == 2
    assert abs(per).max() <= 4.0 + 1e-9 and abs(per.mean()) < 0.5
    print(f"[profiles]   sudden: 0 -> {sud[-1]:.1f}px   periodic: +/-{abs(per).max():.1f}px")

    # --- A uniform shift of every line is NOT motion: it just moves the image ---
    # Every line agreeing on the same displacement cannot produce a ghost.
    rows = shape[0]
    u = _centred_freqs(rows)
    shifted = from_kspace(k*np.exp(-2j*np.pi*u*8.0)[:,None])
    assert abs(shifted.sum()-img.sum())/img.sum() < 1e-6, "a pure shift must conserve brightness"
    rolled = np.roll(img,8,axis=0)
    assert np.abs(shifted-rolled).max() < 1e-6, "phase ramp must equal a real shift"
    print("[shift thm]  a consistent ramp reproduces np.roll exactly (no ghosting)")

    # --- Inconsistent lines degrade the image; more motion degrades it more ---
    last = float("inf")
    for amp in (0.0,1.0,3.0,6.0):
        rec = from_kspace(apply_motion(k,amp,"periodic",n_cycles=4))
        p = compute_metrics(img,np.clip(rec,0,1))["psnr"]
        assert p <= last+1e-9, f"PSNR rose at amplitude {amp}"
        last = p
        print(f"[ghosting]   amplitude={amp:.1f}px  psnr={p:6.2f}")

    # --- Ghosts land OUTSIDE the anatomy, which is how you recognise them ---
    background = img < 0.01 # the phantom's true background
    clean = from_kspace(k)
    moved = from_kspace(apply_motion(k,5.0,"periodic",n_cycles=4))
    assert moved[background].mean() > 8*clean[background].mean()+1e-6, \
        "motion must put energy into the background"
    print(f"[background] clean={clean[background].mean():.5f} -> moved={moved[background].mean():.5f}")

    print("\nAll self-tests passed")
