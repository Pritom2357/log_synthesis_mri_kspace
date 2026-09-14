"""
The coordinator. Runs one full simulation and returns everything the GUI draws.

Order matters, and it follows the physics:

    to_kspace -> motion -> sample -> window -> noise -> field of view -> reconstruct

Motion happens while the data is being collected, so it comes before sampling.
Noise goes on after the window because a real scanner's noise enters at the
receiver, so it is not itself apodized. The field-of-view step is last because
it changes the sample spacing, and therefore the size of the output.
"""
from __future__ import annotations
import numpy as np

from mri.kspace_core import to_kspace, from_kspace, log_magnitude
from mri.cartesian_sampling import make_mask
from mri.filters import apply_window, add_noise
from mri.metrics import compute_metrics, error_map
from mri.trajectories import radial, spiral
from mri.gridding import sample_at, grid_accumulate
from mri.motion import apply_motion, apply_motion_scattered
from mri.cs_recon import cs_reconstruct, sampling_mask_from_coords
from mri.roi import reduced_fov, crop_kspace, zero_fill_to, true_crop

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
    "order": "sequential",     # sequential | golden, radial acquisition order
    "density_comp": True,
    "motion": "none",          # none | sudden | periodic | random
    "motion_amp": 0.0,         # pixels
    "motion_cycles": 4.0,      # oscillations over the scan, for 'periodic'
    "recon": "zero_filled",    # zero_filled | cs
    "cs_iters": 60,
    "cs_weight": 0.05,
    "fov": 1,                  # >1 shrinks the field of view: fold-over
    "crop": 1,                 # >1 keeps only central k-space: blur, not zoom
}

# ------
# Helper
# ------

def _normalise(a:np.ndarray)->np.ndarray:
    """Accumulation gridding changes the absolute scale, so rescale before scoring."""
    lo,hi = float(a.min()),float(a.max())
    return np.zeros_like(a) if hi-lo < 1e-12 else (a-lo)/(hi-lo)

def _acquire(img:np.ndarray,s:dict)->tuple[np.ndarray,np.ndarray|None,np.ndarray]:
    """
    Everything up to and including sampling. Returns (k-space, mask, sampled_mask).

    mask is the binary display mask, or None for non-Cartesian. sampled_mask is
    boolean and always present, because compressed sensing needs to know which
    entries were genuinely measured.
    """
    k = to_kspace(img)

    if s["sampling"] == "cartesian":
        k = apply_motion(k,s["motion_amp"],s["motion"],s["motion_cycles"],s["seed"])
        # Only the stochastic builders take a seed; without it a random mask
        # would be redrawn on every call and nothing would be comparable.
        kw = {"seed":s["seed"]} if s["mask"] in ("random","variable_density","vds") else {}
        mask = make_mask(s["mask"],img.shape,s["R"],**kw)
        return k*mask, mask, mask.astype(bool)

    if s["sampling"] == "radial":
        coords,w = radial(img.shape,n_spokes=s["n_spokes"],order=s["order"])
        n_groups = s["n_spokes"]
    elif s["sampling"] == "spiral":
        coords,w = spiral(img.shape,n_interleaves=s["n_interleaves"])
        n_groups = s["n_interleaves"]
    else:
        raise ValueError(f"Unknown sampling: {s['sampling']!r}")

    vals = sample_at(k,coords)
    vals = apply_motion_scattered(vals,coords,img.shape,n_groups,
                                  s["motion_amp"],s["motion"],s["motion_cycles"],s["seed"])
    gridded = grid_accumulate(coords,vals,img.shape,w if s["density_comp"] else None)
    return gridded, None, sampling_mask_from_coords(coords,img.shape)

# ---------------------
# Methods to call from
# ---------------------

def reconstruct(img:np.ndarray,settings:dict|None=None)->dict:
    """
    Runs the whole pipeline once. Missing settings fall back to DEFAULTS.

    Returns a dict:
        kspace     - log-magnitude of the sampled k-space, ready to display
        recon      - reconstructed image, normalised to [0,1]
        reference  - what recon should be compared against; usually img itself,
                     but a crop of it when the field of view was reduced
        mask       - the binary mask, or None for non-Cartesian
        error      - absolute difference against the reference
        metrics    - mse / psnr / ssim / max_error
    """
    s = {**DEFAULTS,**(settings or {})}
    ks,mask,sampled = _acquire(img,s)

    kw = {"sigma":s["sigma"]} if s["window"] == "gaussian" else {}
    ks = apply_window(ks,s["window"],**kw)
    ks = add_noise(ks,s["noise"],s["seed"])

    # --- field of view and resolution: the two ways to try to zoom ---
    # These are different knobs and they have different consequences:
    #   fov  changes the sample SPACING, so the field of view really does
    #        shrink. The output covers less anatomy, so the reference has to
    #        be the region we aimed at rather than the whole image.
    #   crop changes the sample EXTENT only. The field of view is untouched,
    #        so after zero-filling back the output still lines up with the
    #        original pixel for pixel and the reference must NOT change.
    reference = img
    roi_active = s["fov"] > 1 or s["crop"] > 1
    if s["fov"] > 1:
        ks = reduced_fov(ks,s["fov"])
        reference = true_crop(img,s["fov"])
    if s["crop"] > 1:
        target = ks.shape
        ks = zero_fill_to(crop_kspace(ks,s["crop"]),target)

    # --- reconstruct ---
    if s["recon"] == "cs" and not roi_active:
        recon = cs_reconstruct(ks,sampled,n_iter=s["cs_iters"],weight=s["cs_weight"])
    elif s["recon"] in ("zero_filled","cs"):
        recon = from_kspace(ks) # cs is skipped under ROI: the mask no longer matches
    else:
        raise ValueError(f"Unknown recon: {s['recon']!r}")

    recon = _normalise(recon)
    reference = _normalise(np.asarray(reference,dtype=np.float64))
    return {
        "kspace": log_magnitude(ks),
        "recon": recon,
        "reference": reference,
        "mask": mask,
        "error": error_map(reference,recon),
        "metrics": compute_metrics(reference,recon),
    }

# ----------------------------
# Tests for pipeline.py only
# ----------------------------

if __name__ == "__main__":
    from mri.kspace_core import load_phantom

    img = load_phantom()

    # --- Identity: no undersampling, no window, no noise reproduces the input ---
    r = reconstruct(img)
    assert r["metrics"]["ssim"] > 0.999, r["metrics"]
    assert r["recon"].shape == img.shape and r["mask"].all()
    assert r["reference"].shape == img.shape
    print(f"[identity]   psnr={r['metrics']['psnr']:.1f}  ssim={r['metrics']['ssim']:.4f}")

    # --- Every combination must run and return the full dict ---
    keys = {"kspace","recon","reference","mask","error","metrics"}
    n = 0
    for samp in ("cartesian","radial","spiral"):
        for win in ("none","hamming","gaussian"):
            for noise in (0.0,0.05):
                out = reconstruct(img,{"sampling":samp,"R":4,"window":win,
                                       "noise":noise,"seed":1})
                assert set(out) == keys, set(out)
                assert out["recon"].shape == out["reference"].shape
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

    # --- Motion: quality must fall as the patient moves more ---
    last = float("inf")
    for amp in (0.0,2.0,5.0):
        m = reconstruct(img,{"motion":"periodic","motion_amp":amp})["metrics"]
        assert m["psnr"] <= last+1e-9, f"PSNR rose at amplitude {amp}"
        last = m["psnr"]
        print(f"[motion]     amp={amp:.1f}px  psnr={m['psnr']:6.2f}  ssim={m['ssim']:.3f}")

    # --- Golden-angle ordering is available to the radial path ---
    for order in ("sequential","golden"):
        m = reconstruct(img,{"sampling":"radial","order":order,"n_spokes":128,
                             "motion":"periodic","motion_amp":4.0})["metrics"]
        print(f"[order]      {order:10s} psnr={m['psnr']:6.2f}")

    # --- Compressed sensing must beat zero-filling on an incoherent mask ---
    base = {"mask":"random","R":4,"seed":7}
    zf = reconstruct(img,{**base,"recon":"zero_filled"})["metrics"]
    cs = reconstruct(img,{**base,"recon":"cs"})["metrics"]
    assert cs["psnr"] > zf["psnr"] and cs["ssim"] > zf["ssim"], (zf,cs)
    print(f"[cs]         zero-filled psnr={zf['psnr']:.2f} -> cs psnr={cs['psnr']:.2f}"
          f"  (+{cs['psnr']-zf['psnr']:.2f} dB)")

    # --- Reduced FOV shrinks the output and compares against the intended crop ---
    for f in (2,4):
        out = reconstruct(img,{"fov":f})
        assert out["recon"].shape == (img.shape[0]//f,img.shape[1]//f), out["recon"].shape
        assert out["reference"].shape == out["recon"].shape
        assert out["metrics"]["ssim"] < 0.9, "fold-over should not match the intended crop"
        print(f"[fov]        factor={f}: output {out['recon'].shape}  ssim vs target={out['metrics']['ssim']:.3f}")

    # --- Cropping k-space keeps the size, the field of view AND the reference ---
    steep = lambda a: float(np.percentile(np.hypot(*np.gradient(a)),99))
    for f in (2,4):
        out = reconstruct(img,{"crop":f})
        assert out["recon"].shape == img.shape, out["recon"].shape
        assert np.array_equal(out["reference"],_normalise(img)),             "cropping k-space must not move the reference: the field of view did not change"
        assert steep(out["recon"]) < 0.9*steep(img), "cropped k-space must blur the edges"
        print(f"[crop]       factor={f}: output {out['recon'].shape} unchanged, reference unchanged,"
              f" edges {steep(out['recon'])/steep(img):.2f}x as steep, ssim={out['metrics']['ssim']:.3f}")

    # --- Bad settings are rejected, not silently defaulted ---
    for bad in ({"sampling":"hexagonal"},{"recon":"magic"},{"fov":3}):
        try:
            reconstruct(img,bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should have raised")
    print("[guard]      unknown sampling, unknown recon and indivisible fov rejected")

    print("\nAll self-tests passed")
