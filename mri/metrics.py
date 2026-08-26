"""
Quantitative image quality metrics comparing a reconstruction against the
original, plus the error map the GUI shows as a heatmap.
"""
from __future__ import annotations
import numpy as np

__all__ = ["mse","psnr","ssim","error_map","compute_metrics"] # Only these will be exported

# ------
# Helper
# ------

def _as_pair(a:np.ndarray,b:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    x = np.asarray(a,dtype=np.float64)
    y = np.asarray(b,dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    return x,y

# ---------------------
# Methods to call from
# ---------------------

def mse(original:np.ndarray,recon:np.ndarray)->float:
    """Mean squared error. Zero means the two images are identical."""
    x,y = _as_pair(original,recon)
    return float(np.mean((x-y)**2))

def psnr(original:np.ndarray,recon:np.ndarray,data_range:float=1.0)->float:
    """
    Peak signal-to-noise ratio in dB. Higher is better.

    Returns infinity for a perfect match, since the MSE in the denominator is
    zero there. The GUI has to be ready to print that.
    """
    e = mse(original,recon)
    if e == 0.0:
        return float("inf")
    return float(10.0*np.log10((data_range**2)/e))

def ssim(original:np.ndarray,recon:np.ndarray,data_range:float=1.0)->float:
    """
    Structural similarity in [0,1]. Tracks perceived quality better than MSE
    because it compares local structure rather than pixel-by-pixel differences.
    """
    from skimage.metrics import structural_similarity
    x,y = _as_pair(original,recon)
    return float(structural_similarity(x,y,data_range=data_range))

def error_map(original:np.ndarray,recon:np.ndarray)->np.ndarray:
    """Absolute difference per pixel, for display as a heatmap."""
    x,y = _as_pair(original,recon)
    return np.abs(x-y)

def compute_metrics(original:np.ndarray,recon:np.ndarray,data_range:float=1.0)->dict:
    """
    All three metrics in one call, which is what the GUI reads each update.

    Returns a dict:
        mse         - mean squared error, 0 is perfect
        psnr        - peak signal-to-noise ratio in dB, inf is perfect
        ssim        - structural similarity, 1.0 is perfect
        max_error   - the single worst pixel difference
    """
    x,y = _as_pair(original,recon)
    return {
        "mse": mse(x,y),
        "psnr": psnr(x,y,data_range),
        "ssim": ssim(x,y,data_range),
        "max_error": float(np.abs(x-y).max()),
    }

# ---------------------------
# Tests for metrics.py only
# ---------------------------

if __name__ == "__main__":
    from mri.kspace_core import load_phantom, to_kspace, from_kspace
    from mri.cartesian_sampling import make_mask

    img = load_phantom()

    # --- An image against itself is the perfect score ---
    same = compute_metrics(img,img)
    assert same["mse"] == 0.0 and same["ssim"] > 0.999999
    assert np.isinf(same["psnr"]), "identical images must give infinite PSNR"
    assert error_map(img,img).max() == 0.0
    print(f"[identical]   mse={same['mse']:.1e}  psnr=inf  ssim={same['ssim']:.6f}")

    # --- Quality must fall as the acceleration factor rises ---
    k = to_kspace(img)
    last_psnr, last_ssim = float("inf"), 1.0
    for R in (1,2,4,8):
        rec = from_kspace(k*make_mask("uniform",img.shape,R=R))
        m = compute_metrics(img,rec)
        assert m["psnr"] <= last_psnr+1e-9, f"PSNR rose at R={R}"
        assert m["ssim"] <= last_ssim+1e-9, f"SSIM rose at R={R}"
        last_psnr, last_ssim = m["psnr"], m["ssim"]
        shown = "inf" if np.isinf(m["psnr"]) else f"{m['psnr']:.2f}"
        print(f"[R={R}]        mse={m['mse']:.5f}  psnr={shown:>6s}  ssim={m['ssim']:.4f}")

    # --- The error map must light up where the reconstruction is worst ---
    rec4 = from_kspace(k*make_mask("uniform",img.shape,R=4))
    emap = error_map(img,rec4)
    assert emap.shape == img.shape and emap.min() >= 0.0
    assert emap.max() > 0.0, "an undersampled reconstruction cannot be perfect"
    print(f"[error map]   max={emap.max():.4f}  mean={emap.mean():.4f}")

    # --- Shape mismatches are rejected rather than broadcast ---
    try:
        mse(img, img[:128,:128])
    except ValueError:
        print("[guard]       shape mismatch rejected")
    else:
        raise AssertionError("shape mismatch should have raised")

    print("\nAll self-tests passed")
