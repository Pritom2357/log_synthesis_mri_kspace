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
from mri.kspace_edit import scale_dc, add_spike, erase_patch, partial_fourier
from mri.interpolation import resize
from mri.kspace_core import low_pass, high_pass
from mri.region import Box, blend

__all__ = ["DEFAULTS","reconstruct"] # Only these will be exported

DEFAULTS = {
    "sampling": "cartesian",   # cartesian | radial | spiral
    "mask": "uniform",         # the GUI asks for "nyquist" and drives it with rate
    "R": 1,
    "rate": 1.0,               # sampling rate as a fraction of the Nyquist rate
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

    # --- direct edits to k-space, all no-ops at their defaults ---
    "dc_scale": 1.0,           # multiply the DC term: sets the mean of the image
    "spike": 0,                # put a bright point this many columns off centre
    "erase": 0,                # zero a square patch of this size at the centre
    "low_pass": 0,             # keep only within this radius (0 = off)
    "high_pass": 0,            # drop everything within this radius (0 = off)
    "partial_fourier": 1.0,    # acquire only this fraction of the lines
    "pf_fill": True,           # rebuild the rest from Hermitian symmetry

    # --- resolution ---
    "upscale": 1,              # enlarge the finished image by this factor
    "interp": "sinc",          # zero_order_hold | linear | sinc

    # --- restrict every setting above to one region of the image ---
    "roi": None,               # a region.Box, or None for the whole image
    "feather": 0.15,           # softness of the region boundary
}

# Everything that counts as an "edit". Reset these and you get the plain
# reconstruction, which is what the region blend needs as its background.
_NO_EDITS = {
    "mask": "nyquist", "R": 1, "rate": 1.0, "window": "none", "noise": 0.0,
    "dc_scale": 1.0, "spike": 0, "erase": 0, "low_pass": 0, "high_pass": 0,
    "partial_fourier": 1.0, "recon": "zero_filled",
}

# ------
# Helper
# ------

def _to_image(ks:np.ndarray)->np.ndarray:
    """
    Inverse transform, combining receiver channels if there are any.

    Simulated k-space is a single matrix. Real scanner k-space has a leading
    coil axis, because several receiver channels listened at once and each has
    its own unknown phase. Each channel is transformed on its own and then the
    magnitudes are added in quadrature -- root sum of squares. Adding the
    complex images instead would let those unknown phases cancel.
    """
    if ks.ndim == 2:
        return from_kspace(ks)

    per_coil = np.fft.ifft2(np.fft.ifftshift(ks,axes=(-2,-1)),axes=(-2,-1))
    return np.sqrt(np.sum(np.abs(per_coil)**2,axis=0))

def _display_kspace(ks:np.ndarray)->np.ndarray:
    """What to draw in the k-space panel: one channel's worth, or all combined."""
    return log_magnitude(ks if ks.ndim == 2 else np.sqrt((np.abs(ks)**2).sum(axis=0)))

def _normalise(a:np.ndarray)->np.ndarray:
    """Accumulation gridding changes the absolute scale, so rescale before scoring."""
    lo,hi = float(a.min()),float(a.max())
    return np.zeros_like(a) if hi-lo < 1e-12 else (a-lo)/(hi-lo)

def _acquire(img:np.ndarray,s:dict,kspace:np.ndarray|None=None)->tuple[np.ndarray,np.ndarray|None,np.ndarray]:
    """
    Everything up to and including sampling. Returns (k-space, mask, sampled_mask).

    kspace: real measured k-space to start from, skipping the simulated forward
    transform. It may carry a leading coil axis. When it is None we compute
    k-space from the image, which is the simulation path.

    mask is the binary display mask, or None for non-Cartesian. sampled_mask is
    boolean and always present, because compressed sensing needs to know which
    entries were genuinely measured.
    """
    k = to_kspace(img) if kspace is None else np.asarray(kspace)
    shape = k.shape[-2:]

    if s["sampling"] == "cartesian":
        if k.ndim == 3: # per channel; the patient moved once, not once per coil
            k = np.stack([apply_motion(kc,s["motion_amp"],s["motion"],
                                       s["motion_cycles"],s["seed"]) for kc in k])
        else:
            k = apply_motion(k,s["motion_amp"],s["motion"],s["motion_cycles"],s["seed"])
        # Only the stochastic builders take a seed; without it a random mask
        # would be redrawn on every call and nothing would be comparable.
        kw = {"seed":s["seed"]} if s["mask"] in ("random","variable_density","vds") else {}
        if s["mask"] == "nyquist":
            kw = {"rate":s["rate"]}
        mask = make_mask(s["mask"],shape,s["R"],**kw) # broadcasts over any coil axis
        return k*mask, mask, mask.astype(bool)

    if s["sampling"] == "radial":
        coords,w = radial(shape,n_spokes=s["n_spokes"],order=s["order"])
        n_groups = s["n_spokes"]
    elif s["sampling"] == "spiral":
        coords,w = spiral(shape,n_interleaves=s["n_interleaves"])
        n_groups = s["n_interleaves"]
    else:
        raise ValueError(f"Unknown sampling: {s['sampling']!r}")

    # Each receiver channel is sampled along the same trajectory, so the whole
    # sample-and-grid step just repeats per channel.
    channels = k if k.ndim == 3 else k[None]
    gridded = []
    for kc in channels:
        vals = sample_at(kc,coords)
        vals = apply_motion_scattered(vals,coords,shape,n_groups,
                                      s["motion_amp"],s["motion"],s["motion_cycles"],s["seed"])
        gridded.append(grid_accumulate(coords,vals,shape,w if s["density_comp"] else None))

    out = np.stack(gridded) if k.ndim == 3 else gridded[0]
    return out, None, sampling_mask_from_coords(coords,shape)

# ---------------------
# Methods to call from
# ---------------------

def reconstruct(img:np.ndarray,settings:dict|None=None,
                kspace:np.ndarray|None=None,reference:np.ndarray|None=None)->dict:
    """
    Runs the whole pipeline once. Missing settings fall back to DEFAULTS.

    Two ways in:
        reconstruct(img, settings)
            the simulation path: compute k-space from a picture, then damage it.
        reconstruct(img, settings, kspace=k, reference=ref)
            the real-data path: start from k-space a scanner actually measured,
            never computing a forward transform at all. k may carry a leading
            coil axis. `reference` is what to score against, and it is only ever
            used for the numbers -- the caller decides whether to display it.

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
    ks,mask,sampled = _acquire(img,s,kspace=kspace)
    multicoil = ks.ndim == 3

    # --- direct edits, before the window and the noise ---
    # These are deliberate changes to the measurement, so they belong with the
    # acquisition rather than after it.
    if s["dc_scale"] != 1.0:
        ks = scale_dc(ks,s["dc_scale"])
    if s["erase"]:
        ks = erase_patch(ks,0,0,int(s["erase"]))
    if s["low_pass"]:
        ks = low_pass(ks,float(s["low_pass"]))
    if s["high_pass"]:
        ks = high_pass(ks,float(s["high_pass"]))
    if s["partial_fourier"] < 1.0:
        ks,_ = partial_fourier(ks,float(s["partial_fourier"]),fill=bool(s["pf_fill"]))
    if s["spike"]:
        ks = add_spike(ks,0,int(s["spike"]),strength=1.0)

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
    reference = img if reference is None else reference
    roi_active = s["fov"] > 1 or s["crop"] > 1
    if s["fov"] > 1:
        ks = (np.stack([reduced_fov(c,s["fov"]) for c in ks]) if multicoil
              else reduced_fov(ks,s["fov"]))
        reference = true_crop(reference,s["fov"])
    if s["crop"] > 1:
        target = ks.shape[-2:]
        ks = (np.stack([zero_fill_to(crop_kspace(c,s["crop"]),target) for c in ks]) if multicoil
              else zero_fill_to(crop_kspace(ks,s["crop"]),target))

    # --- reconstruct ---
    if s["recon"] == "cs" and not roi_active:
        # sampled is (ky,kx) whatever the coil count, and cs_reconstruct
        # broadcasts it, so a coil set goes through in one call.
        recon = cs_reconstruct(ks,sampled,n_iter=s["cs_iters"],weight=s["cs_weight"])
        recon_used = "cs"
    elif s["recon"] in ("zero_filled","cs"):
        # cs is still skipped under ROI, where the mask no longer describes the
        # data. recon_used says so, because a method that silently does nothing
        # is indistinguishable from one that does not work.
        recon = _to_image(ks)
        recon_used = "zero_filled"
    else:
        raise ValueError(f"Unknown recon: {s['recon']!r}")

    recon = _normalise(recon)
    reference = _normalise(np.asarray(reference,dtype=np.float64))

    # --- restrict the edits to one region ---
    # k-space has no "local": every sample touches every pixel, so a region
    # cannot be filtered in k-space. Instead reconstruct the slice a second
    # time with nothing switched on, and mix the two images -- edited inside
    # the region, plain outside. The boundary is feathered, because a hard
    # rectangle in a diagnostic image is worse than no processing at all.
    if s["roi"] is not None:
        plain = reconstruct(img,{**s,**_NO_EDITS,"roi":None},
                            kspace=kspace,reference=reference)["recon"]
        recon = _normalise(blend(plain,recon,s["roi"],s["feather"]))

    # --- resolution, last: enlarge the finished picture ---
    # The reference is enlarged the same way so the two stay comparable, which
    # means the metrics measure the interpolation and nothing else.
    if s["upscale"] > 1:
        recon = _normalise(resize(recon,int(s["upscale"]),s["interp"]))
        reference = _normalise(resize(reference,int(s["upscale"]),"sinc"))
    return {
        "kspace": _display_kspace(ks),
        "recon": recon,
        "reference": reference,
        "mask": mask,
        "error": error_map(reference,recon),
        "metrics": compute_metrics(reference,recon),
        "recon_used": recon_used, # what actually ran, which may not be what was asked
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
    keys = {"kspace","recon","reference","mask","error","metrics","recon_used"}
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
