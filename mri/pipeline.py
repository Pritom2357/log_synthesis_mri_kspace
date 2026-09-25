"""
The coordinator. Runs one full simulation and returns everything the GUI draws.

Order matters, and it follows the physics:

    to_kspace -> average repetitions -> motion -> sample -> noise
              -> k-space edits -> denoise -> field of view -> inverse FFT -> sharpen

Repetitions are averaged first because they are separate acquisitions, before
anything is done to any of them. Motion happens while the data is being
collected, so it comes before sampling. Noise enters at the receiver, so it is
part of the measurement: it goes on before any processing, where the denoising
filters can get at it. The field-of-view step is last in k-space because it
changes the sample spacing, and therefore the size of the output. Sharpening
works on the finished picture.

Only Cartesian sampling is simulated: evenly spaced lines at a chosen fraction
of the Nyquist rate, which is the sampling the course teaches and the kind
every file in this project was acquired with.
"""
from __future__ import annotations
import numpy as np

from mri.kspace_core import to_kspace, from_kspace, low_pass, high_pass
from mri.cartesian_sampling import make_mask, acquisition_mask
from mri.filters import apply_window, add_noise
from mri.metrics import compute_metrics, error_map, snr
from mri.motion import apply_motion
from mri.roi import reduced_fov, crop_kspace, zero_fill_to, true_crop
from mri.kspace_edit import (dc_index, scale_dc, add_spike, erase_patch, partial_fourier, keep_part,
                             place_spikes, erase_patches,
                             phase_axis, measured_lines)
from mri.denoise import average_repetitions, noise_floor_filter, sharpen
from mri.interpolation import resize
from mri.region import blend

__all__ = ["DEFAULTS","reconstruct"] # Only these will be exported

DEFAULTS = {
    "rate": 1.0,               # sampling rate as a fraction of the Nyquist rate
    "noise": 0.0,              # added noise, fraction of mean |K| (simulation only)
    "seed": None,
    "motion": "none",          # none | sudden | periodic | random
    "motion_amp": 0.0,         # pixels
    "motion_cycles": 4.0,      # oscillations over the scan, for 'periodic'
    "fov": 1,                  # >1 shrinks the field of view: fold-over
    "crop": 1,                 # >1 keeps only central k-space: blur, not zoom

    # --- denoising, all no-ops at their defaults ---
    "average": 1,              # how many repetitions to average (1 = this scan)
    "align": True,             # undo shift and phase drift before averaging
    "window": "none",          # none | hamming | gaussian
    "sigma": 0.8,              # gaussian window width, normalised radius (0.8 measured best)
    "noise_filter": 0.0,       # noise-floor filter strength (0 = off)
    "sharpen": 0.0,            # unsharp-mask amount (0 = off)

    # --- direct edits to k-space, all no-ops at their defaults ---
    "dc_scale": 1.0,           # multiply the DC term: sets the mean of the image
    "spike": 0,                # put a bright point this many columns off centre
    "erase": 0,                # zero a square patch of this size at the centre
    "partial_fourier": 1.0,    # acquire only this fraction of the lines
    "pf_fill": True,           # rebuild the rest from Hermitian symmetry
    "keep": "both",            # both | magnitude | phase
    "spikes": (),              # clicked (row, col) points made as bright as DC
    "patches": (),             # clicked (row, col, half_size) squares set to zero
    "low_pass": 0,             # keep only within this radius of DC (0 = off)
    "high_pass": 0,            # drop everything within this radius of DC (0 = off)

    # --- acquisition in progress ---
    "acquired": 1.0,           # fraction of the scan's time that has passed
    "acq_order": "linear",     # linear | centric: the order lines are read in

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
    "rate": 1.0, "window": "none", "noise": 0.0, "average": 1,
    "noise_filter": 0.0, "sharpen": 0.0, "keep": "both",
    "dc_scale": 1.0, "spike": 0, "erase": 0, "partial_fourier": 1.0,
    "spikes": (), "patches": (), "low_pass": 0, "high_pass": 0, "acquired": 1.0,
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

def _log_kspace(ks:np.ndarray)->np.ndarray:
    """log(1+|K|), coils combined by root sum of squares. Not yet scaled."""
    mag = np.abs(ks) if ks.ndim == 2 else np.sqrt((np.abs(ks)**2).sum(axis=0))
    return np.log1p(mag)

def _display_kspace(ks:np.ndarray,top:float)->np.ndarray:
    """
    What to draw in the k-space panel, on a FIXED scale.

    `top` is taken from the k-space as measured, before any edit. Scaling each
    frame to its own maximum instead made every other sample brighten when the
    centre was erased: the brightest sample was gone, so everything else was
    stretched up to fill the range. On a fixed scale an erased patch goes
    black and nothing else moves.
    """
    return np.clip(_log_kspace(ks)/max(top,1e-12),0.0,1.0)

def _normalise(a:np.ndarray)->np.ndarray:
    """Rescale to [0,1] before scoring and display."""
    lo,hi = float(a.min()),float(a.max())
    return np.zeros_like(a) if hi-lo < 1e-12 else (a-lo)/(hi-lo)

def _acquire(k:np.ndarray,s:dict)->tuple[np.ndarray,np.ndarray,dict]:
    """
    Motion, then Cartesian sampling at `rate` times the Nyquist rate. Lines
    are rows: a skipped row is a skipped line.

    Rate 1 is the line spacing the scanner chose for its field of view. Below
    1, fewer lines are measured, evenly spread, and the rest are zero.

    Returns (sampled k-space, mask, sampling facts):
        alias_energy  - fraction of the image's energy that the sampling got
                        wrong. By Parseval the error of a zero-filled
                        reconstruction has exactly the energy of the samples
                        that were left out, so this needs no reference image.
    """
    if k.ndim == 3: # per channel; the patient moved once, not once per coil
        k = np.stack([apply_motion(kc,s["motion_amp"],s["motion"],
                                   s["motion_cycles"],s["seed"]) for kc in k])
    else:
        k = apply_motion(k,s["motion_amp"],s["motion"],s["motion_cycles"],s["seed"])
    rate = float(s["rate"])
    dc_row = dc_index(k)[0]   # the real DC row, which scanner data has off-centre
    mask = make_mask("nyquist",k.shape[-2:],1,rate=rate,centre=dc_row) # broadcasts over coils
    if s["acquired"] < 1.0: # the scan is still running: later lines are not in yet
        mask = mask*acquisition_mask(k.shape[-2:],float(s["acquired"]),s["acq_order"],
                                     centre=dc_row)
    power = np.abs(k)**2
    facts = {"rate": rate,
             "alias_energy": float((power*(1-mask)).sum()/max(power.sum(),1e-300))}
    return k*mask, mask, facts

def _coverage(k:np.ndarray,mask:np.ndarray,s:dict,n_scans:int)->dict:
    """
    How much of one full scan this run acquires.

    Sampling skips rows; partial Fourier skips columns. A row or column only
    counts if the file has data in it, so the scanner's empty columns are
    never counted as saved. Every averaged scan acquires the same again, so

        acquired = (rows kept / rows) x (columns kept / columns) x scans

    and the acceleration is 1 / acquired.
    """
    rows = measured_lines(k,-2)
    cols = measured_lines(k,-1)
    rows_kept = rows & mask.any(axis=1)
    cols_kept = cols.copy()
    if s["partial_fourier"] < 1.0:
        n = cols.size
        cols_kept &= np.arange(n) < int(round(s["partial_fourier"]*n))
    acquired = (rows_kept.sum()/max(rows.sum(),1))*(cols_kept.sum()/max(cols.sum(),1))*n_scans
    return {"rows": (int(rows_kept.sum()),int(rows.sum())),
            "cols": (int(cols_kept.sum()),int(cols.sum())),
            "acquired": float(acquired),
            "acceleration": 1.0/max(float(acquired),1e-12)}

# ---------------------
# Methods to call from
# ---------------------

def reconstruct(img:np.ndarray|None,settings:dict|None=None,
                kspace:np.ndarray|None=None,reference:np.ndarray|None=None,
                repeats:list|None=None)->dict:
    """
    Runs the whole pipeline once. Missing settings fall back to DEFAULTS.

    Two ways in:
        reconstruct(img, settings)
            the simulation path: compute k-space from a picture, then damage it.
        reconstruct(img, settings, kspace=k, reference=ref, repeats=[k2, k3])
            the real-data path: start from k-space a scanner actually measured,
            never computing a forward transform at all. k may carry a leading
            coil axis. `reference` is what to score against, and it is only ever
            used for the numbers -- the caller decides whether to display it.
            `repeats` are other acquisitions of the same slice, used when
            settings["average"] asks for more than one.

    Returns a dict:
        kspace     - log-magnitude of the k-space, on a fixed [0,1] scale
        recon      - reconstructed image, normalised to [0,1]
        reference  - what recon should be compared against; usually img itself,
                     but a crop of it when the field of view was reduced
        mask       - the binary sampling mask
        error      - absolute difference against the reference
        metrics    - mse / psnr / ssim / max_error against the reference, and
                     snr of recon alone (no reference needed)
        averaged   - how many acquisitions went into recon
        alignment  - shifts and phase offsets found, when repetitions were aligned
        sampling   - rate; alias_energy, the fraction of image energy lost to
                     undersampling; rows and cols, (kept, with data); acquired,
                     the share of one full scan this run costs; acceleration,
                     1 / acquired; scanner_lines, the phase-encode lines the
                     scanner itself recorded, for turning the share into seconds
    """
    s = {**DEFAULTS,**(settings or {})}
    k = to_kspace(img) if kspace is None else np.asarray(kspace)
    # The scanner's own phase-encode count sets the real duration of one full
    # scan. On the M4Raw files those are columns, so it is counted separately.
    scanner_lines = int(measured_lines(k,phase_axis(k)).sum())

    # --- repetitions: separate acquisitions, averaged before anything else ---
    n = max(1,min(int(s["average"]),1+len(repeats or [])))
    alignment = {"shifts":[],"phases":[]}
    if n > 1:
        k,alignment = average_repetitions([k]+list(repeats or [])[:n-1],align=bool(s["align"]))

    # The display scale comes from the WHOLE scan, before sampling or the
    # acquisition progress removes anything. Taken after, it followed whatever
    # had been read so far: early in a linear scan only dim outer lines are in,
    # so they were stretched to full brightness and the colours drifted as the
    # scan played. From the whole scan, a sample keeps its colour throughout.
    top = float(_log_kspace(k).max())
    ks,mask,sampling = _acquire(k,s)
    sampling.update(_coverage(k,mask,s,n),scanner_lines=scanner_lines)
    multicoil = ks.ndim == 3
    ks = add_noise(ks,s["noise"],s["seed"])*mask # receiver noise, only where measured

    # --- direct edits, deliberate changes to the measurement ---
    if s["dc_scale"] != 1.0:
        ks = scale_dc(ks,s["dc_scale"])
    if s["erase"]:
        ks = erase_patch(ks,0,0,int(s["erase"]))
    if s["partial_fourier"] < 1.0: # skips columns: turn, cut rows, turn back
        ks = np.swapaxes(partial_fourier(np.swapaxes(ks,-1,-2),float(s["partial_fourier"]),
                                         fill=bool(s["pf_fill"]))[0],-1,-2)
    if s["spike"]:
        ks = add_spike(ks,0,int(s["spike"]),strength=1.0)
    ks = place_spikes(ks,s["spikes"])
    ks = erase_patches(ks,s["patches"])
    if s["low_pass"]:
        ks = low_pass(ks,float(s["low_pass"]))
    if s["high_pass"]:
        ks = high_pass(ks,float(s["high_pass"]))
    ks = keep_part(ks,s["keep"])

    # --- denoising in k-space ---
    ks = noise_floor_filter(ks,float(s["noise_filter"]))
    kw = {"sigma":s["sigma"]} if s["window"] == "gaussian" else {}
    ks = apply_window(ks,s["window"],**kw)

    # --- field of view and resolution: the two ways to try to zoom ---
    # These are different knobs and they have different consequences:
    #   fov  changes the sample SPACING, so the field of view really does
    #        shrink. The output covers less anatomy, so the reference has to
    #        be the region we aimed at rather than the whole image.
    #   crop changes the sample EXTENT only. The field of view is untouched,
    #        so after zero-filling back the output still lines up with the
    #        original pixel for pixel and the reference must NOT change.
    reference = img if reference is None else reference
    if s["fov"] > 1:
        ks = (np.stack([reduced_fov(c,s["fov"]) for c in ks]) if multicoil
              else reduced_fov(ks,s["fov"]))
        reference = true_crop(reference,s["fov"])
    if s["crop"] > 1:
        target = ks.shape[-2:]
        ks = (np.stack([zero_fill_to(crop_kspace(c,s["crop"]),target) for c in ks]) if multicoil
              else zero_fill_to(crop_kspace(ks,s["crop"]),target))

    # --- sinc upscaling = zero-filling, done per coil before combining ---
    # Padding k-space with zeros and transforming back is ideal (sinc)
    # interpolation. Scanners do it on each coil's raw k-space, before the
    # coils are combined; padding the finished magnitude image instead would
    # interpolate a picture whose phase has already been thrown away.
    zero_fill = int(s["upscale"]) if s["upscale"] > 1 and s["interp"] == "sinc" else 1
    if zero_fill > 1:
        big = (ks.shape[-2]*zero_fill,ks.shape[-1]*zero_fill)
        ks_out = np.stack([zero_fill_to(c,big) for c in ks]) if multicoil else zero_fill_to(ks,big)
    else:
        ks_out = ks
    # The sharpening blur is 1 pixel of the scan; zero-filling made pixels
    # smaller, so the blur widens with it and the setting keeps its meaning.
    recon = _normalise(sharpen(_normalise(_to_image(ks_out)),float(s["sharpen"]),
                               sigma=float(zero_fill)))
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
    # Sinc was already done above, in k-space. Linear and zero-order hold are
    # image-domain methods by definition, so they work on the finished picture.
    # The reference is enlarged by sinc so the two stay comparable.
    if s["upscale"] > 1:
        if zero_fill == 1:
            recon = _normalise(resize(recon,int(s["upscale"]),s["interp"]))
        reference = _normalise(resize(reference,int(s["upscale"]),"sinc"))
    return {
        "kspace": _display_kspace(ks,top),
        "recon": recon,
        "reference": reference,
        "mask": mask,
        "error": error_map(reference,recon),
        "metrics": {**compute_metrics(reference,recon),"snr": snr(recon)},
        "averaged": n,
        "alignment": alignment,
        "sampling": sampling,
    }

# ----------------------------
# Tests for pipeline.py only
# ----------------------------

if __name__ == "__main__":
    from mri.kspace_core import load_phantom
    from mri.denoise import shift_kspace

    img = load_phantom()

    # --- Identity: no undersampling, no window, no noise reproduces the input ---
    r = reconstruct(img)
    assert r["metrics"]["ssim"] > 0.999, r["metrics"]
    assert r["recon"].shape == img.shape and r["mask"].all()
    assert r["reference"].shape == img.shape
    print(f"[identity]   psnr={r['metrics']['psnr']:.1f}  ssim={r['metrics']['ssim']:.4f}")

    # --- Every combination must run and return the full dict ---
    keys = {"kspace","recon","reference","mask","error","metrics","averaged","alignment","sampling"}
    n = 0
    for rate in (1.0,0.5):
        for win in ("none","hamming","gaussian"):
            for noise in (0.0,0.05):
                for keep in ("both","phase"):
                    out = reconstruct(img,{"rate":rate,"window":win,"noise":noise,
                                           "keep":keep,"noise_filter":1.0,
                                           "sharpen":0.3,"seed":1})
                    assert set(out) == keys, set(out)
                    assert out["recon"].shape == out["reference"].shape
                    assert 0.0 <= out["recon"].min() and out["recon"].max() <= 1.0
                    assert np.isfinite(out["metrics"]["ssim"])
                    n += 1
    print(f"[combos]     {n} rate x window x noise x keep combinations ran clean")

    # --- The k-space panel keeps its scale when the centre is erased ---
    # Only the erased patch may change; everything else must stay the same
    # brightness. Before the fix every other sample brightened.
    plain = reconstruct(img)["kspace"]
    erased = reconstruct(img,{"erase":40})["kspace"]
    c = img.shape[0]//2
    outside = np.ones(img.shape,dtype=bool); outside[c-21:c+22,c-21:c+22] = False
    assert np.abs(erased[outside]-plain[outside]).max() < 1e-12, "erasing moved the rest"
    assert erased[c,c] == 0.0 and plain[c,c] == 1.0
    print("[display]    erasing the centre blacks the patch and leaves every other sample alone")

    # --- Sampling below the Nyquist rate costs quality, gradually ---
    # The aliasing number needs no reference: by Parseval it must equal the
    # energy of the complex reconstruction error exactly.
    last,last_alias = float("inf"),-1.0
    full = np.fft.ifft2(np.fft.ifftshift(to_kspace(img)))
    for rate in (1.0,0.9,0.7,0.5):
        out = reconstruct(img,{"rate":rate})
        m,fact = out["metrics"],out["sampling"]
        assert fact["rows"] == (int(out["mask"][:,0].sum()),img.shape[0])
        assert abs(fact["acceleration"]-img.shape[0]/fact["rows"][0]) < 1e-12
        assert m["psnr"] <= last+1e-9, f"PSNR rose at rate {rate}"
        assert fact["alias_energy"] >= last_alias, f"aliasing fell at rate {rate}"
        k_rate = to_kspace(img)*make_mask("nyquist",img.shape,1,rate=rate)
        err = np.abs(np.fft.ifft2(np.fft.ifftshift(k_rate))-full)**2
        assert abs(err.sum()/(np.abs(full)**2).sum() - fact["alias_energy"]) < 1e-9, "Parseval"
        last,last_alias = m["psnr"],fact["alias_energy"]
        print(f"[rate]       {rate:.2f} x Nyquist  {fact['rows'][0]}/{fact['rows'][1]} rows"
              f"  aliasing {100*fact['alias_energy']:5.2f}%  psnr={m['psnr']:6.2f}")
    assert reconstruct(img)["sampling"]["alias_energy"] == 0.0

    # --- Added noise has the same strength whatever the sampling rate ---
    from mri.filters import add_noise
    k_full = to_kspace(img)
    for rate in (1.0,0.5):
        m = make_mask("nyquist",img.shape,1,rate=rate).astype(bool)
        grain = (add_noise(k_full*m,0.2,0)-k_full*m)[m]
        if rate == 1.0:
            full_grain = np.abs(grain).std()
    assert abs(np.abs(grain).std()/full_grain-1) < 0.05, "noise must not shrink with the rate"
    print("[noise]      added noise per measured sample is the same at rate 1.0 and 0.5")

    # --- Noise is added before the filters, so they can remove it ---
    noisy = reconstruct(img,{"noise":0.5,"seed":3})["metrics"]["psnr"]
    for name,extra in (("noise floor",{"noise_filter":1.0}),
                       ("gaussian",{"window":"gaussian","sigma":0.5})):
        cleaned = reconstruct(img,{"noise":0.5,"seed":3,**extra})["metrics"]["psnr"]
        assert cleaned > noisy+1.0, (name,noisy,cleaned)
        print(f"[denoise]    {name:11s} {noisy:.2f} dB -> {cleaned:.2f} dB")

    # --- Averaging repetitions: aligned wins, naive loses on a phase drift ---
    k0 = to_kspace(img)
    rng = np.random.default_rng(5)
    scan = lambda: k0 + 0.5*np.abs(k0).mean()*(rng.normal(size=k0.shape)+1j*rng.normal(size=k0.shape))
    first,reps = scan(),[scan()*np.exp(3.0j),shift_kspace(scan(),2,0)*np.exp(-2.0j)]
    one = reconstruct(None,{},kspace=first,reference=img,repeats=reps)
    aligned = reconstruct(None,{"average":3},kspace=first,reference=img,repeats=reps)
    naive = reconstruct(None,{"average":3,"align":False},kspace=first,reference=img,repeats=reps)
    assert one["averaged"] == 1 and aligned["averaged"] == 3
    assert aligned["alignment"]["shifts"] == [(0,0),(2,0)], aligned["alignment"]
    assert aligned["metrics"]["psnr"] > one["metrics"]["psnr"]+3.0
    assert naive["metrics"]["psnr"] < aligned["metrics"]["psnr"]-3.0
    over = reconstruct(None,{"average":9},kspace=first,reference=img,repeats=reps)
    assert over["averaged"] == 3, "cannot average more scans than exist"
    print(f"[average]    1 scan {one['metrics']['psnr']:.2f} dB, 3 naive {naive['metrics']['psnr']:.2f} dB,"
          f" 3 aligned {aligned['metrics']['psnr']:.2f} dB")

    # --- Acceleration follows every acquisition knob; lines are rows ---
    # The M4Raw layout: 196 of 256 columns recorded, every row has data.
    padded = k0.copy(); padded[:,:30] = 0; padded[:,-30:] = 0
    reps2 = [padded*np.exp(1.0j),padded*np.exp(-2.0j)]
    run = lambda **kw: reconstruct(None,kw,kspace=padded,reference=img,repeats=reps2)["sampling"]
    base_s = run()
    assert base_s["rows"] == (256,256) and base_s["cols"] == (196,196)
    assert base_s["scanner_lines"] == 196 and base_s["acceleration"] == 1.0
    half = run(rate=0.5)
    assert half["rows"] == (128,256) and half["acceleration"] == 2.0, half
    pf = run(partial_fourier=0.6)            # keeps columns 0..153: 124 of the 196 with data
    assert pf["rows"] == (256,256) and pf["cols"] == (124,196), pf
    assert abs(pf["acceleration"]-196/124) < 1e-12
    both = run(rate=0.5,partial_fourier=0.6)
    assert abs(both["acceleration"]-2*196/124) < 1e-12, "the two savings multiply"
    three = run(average=3)
    assert abs(three["acceleration"]-1/3) < 1e-12
    print(f"[coverage]   rate 0.5 x{half['acceleration']:.2f}, partial Fourier 0.6 x{pf['acceleration']:.2f},"
          f" both x{both['acceleration']:.2f}, 3 scans x{three['acceleration']:.2f}")

    # --- Partial Fourier cuts columns ---
    cut = reconstruct(img,{"partial_fourier":0.6,"pf_fill":False})["kspace"]
    assert cut[:,200:].max() == 0.0 and cut[200:,:150].max() > 0.0, "right-hand columns are gone"

    # --- Undersampling skips rows ---
    under = reconstruct(None,{"rate":0.5},kspace=padded,reference=img)
    assert under["mask"][0,:].all() and not under["mask"][:,0].all(), "rows are skipped"

    # --- Sinc upscaling zero-fills each coil's k-space before combining ---
    coils = np.stack([k0*np.exp(1.0j),0.5*k0*np.exp(-2.0j)])     # two coils, different phases
    up = reconstruct(None,{"upscale":2,"interp":"sinc"},kspace=coils,reference=img)
    assert up["recon"].shape == (512,512) and up["kspace"].shape == img.shape
    per_coil = np.fft.ifft2(np.fft.ifftshift(np.stack([zero_fill_to(c,(512,512)) for c in coils]),
                                             axes=(-2,-1)),axes=(-2,-1))
    manual = _normalise(np.sqrt((np.abs(per_coil)**2).sum(axis=0)))
    assert np.abs(up["recon"]-manual).max() < 1e-12, "must be RSS of zero-filled coils"
    for it in ("linear","zero_order_hold"):
        assert reconstruct(img,{"upscale":2,"interp":it})["recon"].shape == (512,512)
    print(f"[zero-fill]  sinc x2: each coil padded to 512 in k-space, then combined;"
          f" ssim vs sinc-enlarged reference {up['metrics']['ssim']:.3f}")

    # --- Acquisition in progress, clicked edits, and the band filters ---
    fill = {}
    for order in ("linear","centric"):
        fill[order] = reconstruct(img,{"acquired":0.25,"acq_order":order})["metrics"]["ssim"]
        assert reconstruct(img,{"acquired":1.0,"acq_order":order})["metrics"]["ssim"] > 0.999
    assert fill["centric"] > fill["linear"], "centric has the centre early, so it looks right sooner"
    clicked = reconstruct(img,{"spikes":[(100,140)],"patches":[(128,128,4)]})
    assert clicked["kspace"][100,140] == clicked["kspace"].max() and clicked["kspace"][128,128] == 0
    lp = reconstruct(img,{"low_pass":20})["recon"]; hp = reconstruct(img,{"high_pass":20})["recon"]
    steep = lambda a: float(np.percentile(np.hypot(*np.gradient(a)),99))
    assert steep(lp) < steep(_normalise(img)) and hp.mean() < _normalise(img).mean()
    print(f"[acquire]    a quarter of the scan: linear ssim {fill['linear']:.3f}, centric {fill['centric']:.3f};"
          " clicked spike/patch land; low pass blurs, high pass keeps edges")

    # --- A k-space sample keeps its display colour however much is acquired ---
    full_view = reconstruct(img)["kspace"]
    for order in ("linear","centric"):
        for frac in (0.1,0.4,0.8):
            part = reconstruct(img,{"acquired":frac,"acq_order":order})["kspace"]
            shown = part > 0
            assert np.abs(part[shown]-full_view[shown]).max() < 1e-12, (order,frac)
    half = reconstruct(img,{"rate":0.5})["kspace"]
    assert np.abs(half[half > 0]-full_view[half > 0]).max() < 1e-12, "undersampling too"
    print("[colour]     every acquired sample keeps its colour at any scan progress or rate")

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
        assert np.array_equal(out["reference"],_normalise(img)), \
            "cropping k-space must not move the reference: the field of view did not change"
        assert steep(out["recon"]) < 0.9*steep(img), "cropped k-space must blur the edges"
        print(f"[crop]       factor={f}: edges {steep(out['recon'])/steep(img):.2f}x as steep")

    # --- Bad settings are rejected, not silently defaulted ---
    for bad in ({"rate":0.0},{"rate":1.5},{"keep":"neither"},{"fov":3},{"sharpen":-1.0}):
        try:
            reconstruct(img,bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should have raised")
    print("[guard]      zero rate, rate above 1, unknown keep, indivisible fov and negative sharpen rejected")

    print("\nAll self-tests passed")
