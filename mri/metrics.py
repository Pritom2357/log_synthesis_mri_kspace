"""Image quality: PSNR and SSIM against a reference, SNR from the image alone."""
from __future__ import annotations
import numpy as np


def _pair(a, b):
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    return x, y


def psnr(original, recon, data_range: float = 1.0) -> float:
    x, y = _pair(original, recon)
    e = float(np.mean((x - y)**2))
    return float("inf") if e == 0.0 else float(10*np.log10(data_range**2/e))


def ssim(original, recon, data_range: float = 1.0) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(*_pair(original, recon), data_range=data_range))


def snr(img, corner: int = 32) -> float:
    """Mean tissue / noise sigma; sigma from the corners, Rayleigh-corrected (/0.655)."""
    a, c = np.asarray(img, dtype=np.float64), corner
    bg = np.concatenate([a[:c, :c].ravel(), a[:c, -c:].ravel(), a[-c:, :c].ravel(), a[-c:, -c:].ravel()])
    sd = float(bg.std())
    if sd <= 1e-9*max(float(np.abs(a).max()), 1e-300):  # flat up to FFT rounding
        return float("inf")
    tissue = a[a > bg.mean() + 3*sd]
    return float(tissue.mean()/(sd/0.655)) if tissue.size else 0.0


def compute_metrics(original, recon, data_range: float = 1.0) -> dict:
    return {"psnr": psnr(original, recon, data_range), "ssim": ssim(original, recon, data_range)}


if __name__ == "__main__":
    from mri.kspace_core import load_phantom, to_kspace, from_kspace
    from mri.cartesian_sampling import nyquist_mask
    img = load_phantom()
    same = compute_metrics(img, img)
    assert np.isinf(same["psnr"]) and same["ssim"] > 0.999999
    last = float("inf")
    for rate in (1.0, 0.75, 0.5, 0.25):
        p = compute_metrics(img, from_kspace(to_kspace(img)*nyquist_mask(img.shape, rate)))["psnr"]
        assert p <= last + 1e-9
        last = p
    rng = np.random.default_rng(0)
    last = float("inf")
    for sigma in (0.02, 0.05, 0.1):
        noisy = np.abs(img + sigma*(rng.normal(size=img.shape) + 1j*rng.normal(size=img.shape)))
        bg = np.concatenate([noisy[:32, :32].ravel(), noisy[:32, -32:].ravel(),
                             noisy[-32:, :32].ravel(), noisy[-32:, -32:].ravel()])
        truth = noisy[noisy > bg.mean() + 3*bg.std()].mean()/sigma
        est = snr(noisy)
        assert abs(est/truth - 1) < 0.15 and est < last, (sigma, est, truth)
        last = est
    assert np.isinf(snr(img))
    try:
        psnr(img, img[:128, :128])
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch")
    print("metrics: all self-tests passed")
