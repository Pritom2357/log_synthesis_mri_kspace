"""
The coordinator. Runs one full simulation and returns everything the GUI draws.

Order matters: choose the samples first (mask or trajectory), then apodize,
then add noise, then invert. Noise goes on after the window because a real
scanner's noise enters at the receiver, so it is not itself apodized.
"""
from __future__ import annotations
import numpy as np

from mri.kspace_core import to_kspace, from_kspace, log_magnitude
from mri.cartesian_sampling import make_mask
from mri.filters import apply_window, add_noise
from mri.metrics import compute_metrics, error_map
from mri.trajectories import radial, spiral
from mri.gridding import sample_at, grid_accumulate

__all__ = ["DEFAULTS","reconstruct"] # Only these will be exported

DEFAULTS = {
    "sampling": "cartesian",   # cartesian | radial | spiral
    "mask": "uniform",
    "R": 1,
    "window": "none",          # none | hamming | gaussian
    "sigma": 0.35,             # gaussian window width, normalised radius
    "noise": 0.0,              # fraction of mean |K|
    "seed": None,
    "n_spokes": 64,
    "n_interleaves": 16,
    "density_comp": True,
}

# ------
# Helper
# ------

def _normalise(a:np.ndarray)->np.ndarray:
    """Accumulation gridding changes the absolute scale, so rescale before scoring."""
    lo,hi = float(a.min()),float(a.max())
    return np.zeros_like(a) if hi-lo < 1e-12 else (a-lo)/(hi-lo)

# ---------------------
# Methods to call from
# ---------------------

def reconstruct(img:np.ndarray,settings:dict|None=None)->dict:
    """
    Runs the whole pipeline once. Missing settings fall back to DEFAULTS.

    Returns a dict:
        kspace   - log-magnitude of the sampled k-space, ready to display
        recon    - reconstructed image, normalised to [0,1]
        mask     - the binary mask, or None for non-Cartesian
        error    - absolute difference against the original
        metrics  - mse / psnr / ssim / max_error
    """
    s = {**DEFAULTS,**(settings or {})}
    k = to_kspace(img)

    if s["sampling"] == "cartesian":
        mask = make_mask(s["mask"],img.shape,s["R"])
        ks = k*mask
    else:
        if s["sampling"] == "radial":
            coords,w = radial(img.shape,n_spokes=s["n_spokes"])
        elif s["sampling"] == "spiral":
            coords,w = spiral(img.shape,n_interleaves=s["n_interleaves"])
        else:
            raise ValueError(f"Unknown sampling: {s['sampling']!r}")
        vals = sample_at(k,coords)
        ks = grid_accumulate(coords,vals,img.shape,w if s["density_comp"] else None)
        mask = None

    kw = {"sigma":s["sigma"]} if s["window"] == "gaussian" else {}
    ks = apply_window(ks,s["window"],**kw)
    ks = add_noise(ks,s["noise"],s["seed"])

    recon = _normalise(from_kspace(ks))
    return {
        "kspace": log_magnitude(ks),
        "recon": recon,
        "mask": mask,
        "error": error_map(img,recon),
        "metrics": compute_metrics(img,recon),
    }

# ----------------------------
# Tests for pipeline.py only
# ----------------------------

if __name__ == "__main__":
    from mri.kspace_core import load_phantom

    img = load_phantom()

    # --- Identity: no undersampling, no window, no noise must reproduce the input ---
    r = reconstruct(img)
    assert r["metrics"]["ssim"] > 0.999, r["metrics"]
    assert r["recon"].shape == img.shape and r["mask"].all()
    print(f"[identity]   psnr={r['metrics']['psnr']:.1f}  ssim={r['metrics']['ssim']:.4f}")

    # --- Every combination must run and return the full dict ---
    n = 0
    for samp in ("cartesian","radial","spiral"):
        for win in ("none","hamming","gaussian"):
            for noise in (0.0,0.05):
                out = reconstruct(img,{"sampling":samp,"R":4,"window":win,
                                       "noise":noise,"seed":1})
                assert set(out) == {"kspace","recon","mask","error","metrics"}
                assert out["recon"].shape == img.shape
                assert 0.0 <= out["recon"].min() and out["recon"].max() <= 1.0
                assert np.isfinite(out["metrics"]["ssim"])
                n += 1
    print(f"[combos]     {n} sampling x window x noise combinations ran clean")

    # --- Density compensation must be a large win, not a small one ---
    off = reconstruct(img,{"sampling":"radial","n_spokes":128,"density_comp":False})
    on  = reconstruct(img,{"sampling":"radial","n_spokes":128,"density_comp":True})
    gain = on["metrics"]["psnr"] - off["metrics"]["psnr"]
    assert gain > 6.0, f"DCF gain only {gain:.1f} dB, check the weights"
    print(f"[dcf]        off={off['metrics']['psnr']:.2f}  on={on['metrics']['psnr']:.2f}  gain={gain:+.1f} dB")

    # --- Bad settings are rejected, not silently defaulted ---
    try:
        reconstruct(img,{"sampling":"hexagonal"})
    except ValueError:
        print("[guard]      unknown sampling rejected")
    else:
        raise AssertionError("unknown sampling should have raised")

    print("\nAll self-tests passed")
