"""
One full reconstruction:
average repetitions -> sample -> k-space edits -> denoise -> zero-fill -> IFFT + RSS -> sharpen -> resize.
"""
from __future__ import annotations
import numpy as np

from mri.kspace_core import to_kspace, from_kspace
from mri.cartesian_sampling import nyquist_mask, acquisition_mask
from mri.filters import apply_window
from mri.metrics import compute_metrics, snr
from mri.kspace_edit import dc_index, scale_dc, add_spike, erase_patch, partial_fourier, keep_part, measured_lines
from mri.denoise import average_repetitions, noise_floor_filter, sharpen
from mri.interpolation import resize, zero_fill_to

DEFAULTS = {
    "rate": 1.0,             # fraction of the Nyquist rate (rows kept)
    "acquired": 1.0,         # fraction of the scan read so far
    "acq_order": "linear",   # linear | centric
    "average": 1,            # repetitions averaged, phase-aligned
    "noise_filter": 0.0,
    "window": "none",        # none | hamming | gaussian
    "sigma": 0.8,            # gaussian width, normalised radius
    "sharpen": 0.0,
    "partial_fourier": 1.0,  # fraction of columns kept
    "pf_fill": True,         # Hermitian rebuild of the rest
    "dc_scale": 1.0,         # multiplies the DC sample: mean brightness
    "spike": 0,              # columns from DC
    "erase": 0,              # square size at DC
    "keep": "both",          # both | magnitude | phase
    "upscale": 1,
    "interp": "sinc",        # sinc | linear | zero_order_hold
}


def _to_image(ks: np.ndarray) -> np.ndarray:
    """Inverse FFT; coils combined by root sum of squares."""
    if ks.ndim == 2:
        return from_kspace(ks)
    per_coil = np.fft.ifft2(np.fft.ifftshift(ks, axes=(-2, -1)), axes=(-2, -1))
    return np.sqrt((np.abs(per_coil)**2).sum(axis=0))


def _log_kspace(ks: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(ks) if ks.ndim == 2 else np.sqrt((np.abs(ks)**2).sum(axis=0)))


def _normalise(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo)/(hi - lo)


def reconstruct(img: np.ndarray | None, settings: dict | None = None, kspace: np.ndarray | None = None,
                reference: np.ndarray | None = None, repeats: list | None = None) -> dict:
    """Simulated (img) or measured (kspace, optional coil axis) data through every setting."""
    s = {**DEFAULTS, **(settings or {})}
    k = to_kspace(img) if kspace is None else np.asarray(kspace)

    n = max(1, min(int(s["average"]), 1 + len(repeats or [])))
    alignment = {"shifts": [], "phases": []}
    if n > 1:
        k, alignment = average_repetitions([k] + list(repeats or [])[:n - 1])

    top = float(_log_kspace(k).max())  # display scale from the whole scan, so colours never drift
    dc_row = dc_index(k)[0]
    mask = nyquist_mask(k.shape[-2:], float(s["rate"]), centre=dc_row)
    if s["acquired"] < 1.0:
        mask = mask*acquisition_mask(k.shape[-2:], float(s["acquired"]), s["acq_order"], centre=dc_row)
    power = np.abs(k)**2
    rows, cols = measured_lines(k, -2), measured_lines(k, -1)
    kept_cols = cols & (np.arange(cols.size) < int(round(s["partial_fourier"]*cols.size)))
    sampling = {"rate": float(s["rate"]),
                "alias_energy": float((power*(1 - mask)).sum()/max(power.sum(), 1e-300)),  # Parseval
                "rows": (int((rows & mask.any(axis=1)).sum()), int(rows.sum())),
                "cols": (int(kept_cols.sum()), int(cols.sum()))}
    ks = k*mask

    if s["dc_scale"] != 1.0:
        ks = scale_dc(ks, s["dc_scale"])
    if s["erase"]:
        ks = erase_patch(ks, 0, 0, int(s["erase"]))
    if s["partial_fourier"] < 1.0:  # cuts columns: transpose, cut rows, transpose back
        cut = partial_fourier(np.swapaxes(ks, -1, -2), float(s["partial_fourier"]), fill=bool(s["pf_fill"]))[0]
        ks = np.swapaxes(cut, -1, -2)
    if s["spike"]:
        ks = add_spike(ks, 0, int(s["spike"]))
    ks = keep_part(ks, s["keep"])

    ks = noise_floor_filter(ks, float(s["noise_filter"]))
    ks = apply_window(ks, s["window"], **({"sigma": s["sigma"]} if s["window"] == "gaussian" else {}))

    up = int(s["upscale"])
    zero_fill = up if up > 1 and s["interp"] == "sinc" else 1  # sinc = zero-fill each coil before RSS
    ks_out = zero_fill_to(ks, (ks.shape[-2]*zero_fill, ks.shape[-1]*zero_fill))
    recon = _normalise(sharpen(_normalise(_to_image(ks_out)), float(s["sharpen"]), sigma=float(zero_fill)))
    if reference is None:  # no reference given: the phantom/picture itself, else score against itself
        reference = img if img is not None else recon
    reference = _normalise(np.asarray(reference, dtype=np.float64))
    if up > 1:
        if zero_fill == 1:
            recon = _normalise(resize(recon, up, s["interp"]))
        reference = _normalise(resize(reference, up, "sinc"))

    return {"kspace": np.clip(_log_kspace(ks)/max(top, 1e-12), 0.0, 1.0),
            "recon": recon, "reference": reference, "mask": mask,
            "metrics": {**compute_metrics(reference, recon), "snr": snr(recon)},
            "averaged": n, "alignment": alignment, "sampling": sampling}


if __name__ == "__main__":
    from mri.denoise import shift_kspace
    from mri.kspace_core import load_phantom
    img = load_phantom()
    k0 = to_kspace(img)
    rng = np.random.default_rng(5)
    noisy = lambda: k0 + 0.5*np.abs(k0).mean()*(rng.normal(size=k0.shape) + 1j*rng.normal(size=k0.shape))

    r = reconstruct(img)
    assert r["metrics"]["ssim"] > 0.999 and r["mask"].all() and r["sampling"]["alias_energy"] == 0.0

    for rate in (1.0, 0.5):
        for win in ("none", "hamming", "gaussian"):
            for keep in ("both", "phase"):
                o = reconstruct(None, {"rate": rate, "window": win, "keep": keep, "noise_filter": 1.0,
                                       "sharpen": 0.3}, kspace=noisy(), reference=img)
                assert 0 <= o["recon"].min() and o["recon"].max() <= 1 and np.isfinite(o["metrics"]["ssim"])

    plain, erased = reconstruct(img)["kspace"], reconstruct(img, {"erase": 40})["kspace"]
    outside = np.ones(img.shape, bool); outside[107:149, 107:149] = False
    assert np.abs(erased[outside] - plain[outside]).max() < 1e-12 and erased[128, 128] == 0, "fixed display scale"

    full = np.fft.ifft2(np.fft.ifftshift(k0))
    last_p, last_a = float("inf"), -1.0
    for rate in (1.0, 0.9, 0.7, 0.5):
        o = reconstruct(img, {"rate": rate})
        a = o["sampling"]["alias_energy"]
        err = np.abs(np.fft.ifft2(np.fft.ifftshift(k0*nyquist_mask(img.shape, rate))) - full)**2
        assert abs(err.sum()/(np.abs(full)**2).sum() - a) < 1e-9, "Parseval"
        assert o["metrics"]["psnr"] <= last_p + 1e-9 and a >= last_a
        last_p, last_a = o["metrics"]["psnr"], a

    base = reconstruct(None, {}, kspace=noisy(), reference=img)["metrics"]["psnr"]
    for extra in ({"noise_filter": 1.0}, {"window": "gaussian", "sigma": 0.5}):
        assert reconstruct(None, extra, kspace=noisy(), reference=img)["metrics"]["psnr"] > base + 1, extra

    first, reps = noisy(), [noisy()*np.exp(3.0j), shift_kspace(noisy(), 2, 0)*np.exp(-2.0j)]
    one = reconstruct(None, {}, kspace=first, reference=img, repeats=reps)
    avg = reconstruct(None, {"average": 3}, kspace=first, reference=img, repeats=reps)
    assert avg["averaged"] == 3 and avg["alignment"]["shifts"] == [(0, 0), (2, 0)]
    assert avg["metrics"]["psnr"] > one["metrics"]["psnr"] + 3
    assert reconstruct(None, {"average": 9}, kspace=first, reference=img, repeats=reps)["averaged"] == 3

    padded = k0.copy(); padded[:, :30] = 0; padded[:, -30:] = 0  # M4Raw layout: 196 of 256 columns
    run = lambda **kw: reconstruct(None, kw, kspace=padded, reference=img)
    assert run()["sampling"]["cols"] == (196, 196) and run(rate=0.5)["sampling"]["rows"] == (128, 256)
    assert run(partial_fourier=0.6)["sampling"]["cols"] == (124, 196)
    under = run(rate=0.5)["mask"]
    assert under[0].all() and not under[:, 0].all(), "rows are skipped"
    cut = reconstruct(img, {"partial_fourier": 0.6, "pf_fill": False})["kspace"]
    assert cut[:, 200:].max() == 0 and cut[200:, :150].max() > 0, "columns are cut"

    coils = np.stack([k0*np.exp(1.0j), 0.5*k0*np.exp(-2.0j)])
    up = reconstruct(None, {"upscale": 2}, kspace=coils, reference=img)
    manual = _normalise(_to_image(zero_fill_to(coils, (512, 512))))
    assert up["recon"].shape == (512, 512) and np.abs(up["recon"] - manual).max() < 1e-12
    for it in ("linear", "zero_order_hold"):
        assert reconstruct(img, {"upscale": 2, "interp": it})["recon"].shape == (512, 512)

    ssim = {o: reconstruct(img, {"acquired": 0.25, "acq_order": o})["metrics"]["ssim"] for o in ("linear", "centric")}
    assert ssim["centric"] > ssim["linear"]
    for order in ("linear", "centric"):
        for keep in ("both", "phase", "magnitude"):
            e = reconstruct(img, {"acquired": 0.0, "acq_order": order, "keep": keep})
            assert np.isfinite(e["recon"]).all() and np.isfinite(e["kspace"]).all()
        for frac in (0.1, 0.4, 0.8):
            part = reconstruct(img, {"acquired": frac, "acq_order": order})["kspace"]
            assert np.abs(part[part > 0] - plain[part > 0]).max() < 1e-12, "colours stay put"

    for bad in ({"rate": 0.0}, {"rate": 1.5}, {"keep": "neither"}, {"sharpen": -1.0}):
        try:
            reconstruct(img, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should have raised")
    print("pipeline: all self-tests passed")
