"""Denoising: aligned repetition averaging, noise-floor filter, unsharp mask."""
from __future__ import annotations
import numpy as np

from mri.kspace_core import radius_grid
from mri.kspace_edit import dc_index


def _coils(k):
    return k if k.ndim == 3 else k[None]


def _image(k):
    per_coil = np.fft.ifft2(np.fft.ifftshift(_coils(k), axes=(-2, -1)), axes=(-2, -1))
    return np.sqrt((np.abs(per_coil)**2).sum(axis=0))


def align_phase(ref: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Rotates each coil of k by the angle of its lag-0 cross-correlation with ref."""
    kc, rc = _coils(k), _coils(ref)
    out = np.stack([kc[c]*np.exp(1j*np.angle(np.vdot(kc[c], rc[c]))) for c in range(kc.shape[0])])
    return out if k.ndim == 3 else out[0]


def find_shift(ref: np.ndarray, k: np.ndarray) -> tuple[int, int]:
    """Whole-pixel (rows, cols) shift from ref's image to k's, by phase correlation."""
    cross = np.conj(np.fft.fft2(_image(ref)))*np.fft.fft2(_image(k))
    mag = np.abs(cross)
    surface = np.real(np.fft.ifft2(np.where(mag > 1e-12*mag.max(), cross/np.maximum(mag, 1e-300), 0.0)))
    i, j = np.unravel_index(int(surface.argmax()), surface.shape)
    H, W = surface.shape
    return (int(i - H) if i > H//2 else int(i), int(j - W) if j > W//2 else int(j))


def shift_kspace(k: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Shift theorem: moves the image (dy, dx) px with a phase ramp."""
    u = np.fft.fftshift(np.fft.fftfreq(k.shape[-2]))[:, None]
    v = np.fft.fftshift(np.fft.fftfreq(k.shape[-1]))[None, :]
    return k*np.exp(-2j*np.pi*(u*dy + v*dx))


def average_repetitions(ks: list, align: bool = True) -> tuple[np.ndarray, dict]:
    """Mean of ks, each first undone for shift and phase against ks[0] when align."""
    if not ks:
        raise ValueError("need at least one acquisition")
    ref = ks[0]
    out, shifts, phases = [ref], [], []
    for k in ks[1:]:
        if k.shape != ref.shape:
            raise ValueError(f"repetition shape {k.shape} differs from {ref.shape}")
        if align:
            dy, dx = find_shift(ref, k)
            if dy or dx:
                k = shift_kspace(k, -dy, -dx)
            shifts.append((dy, dx))
            phases.append(float(np.angle(np.vdot(_coils(k)[0], _coils(ref)[0]))))
            k = align_phase(ref, k)
        out.append(k)
    return np.mean(out, axis=0), {"shifts": shifts, "phases": phases}


def noise_power(k: np.ndarray) -> float:
    """Mean power per sample in the corners beyond the inscribed circle, zeros skipped."""
    p = (np.abs(_coils(k))**2).sum(axis=0)
    corner = p[(radius_grid(p.shape, dc_index(k)) > min(p.shape)/2) & (p > 0)]
    return float(corner.mean()) if corner.size else 0.0


def noise_floor_filter(k: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Per ring: gain = (P - strength*N)/P clipped to [0,1], P over measured samples only."""
    if strength < 0:
        raise ValueError("strength must be non-negative")
    N = noise_power(k) if strength else 0.0
    if N == 0.0:
        return k
    p = (np.abs(_coils(k))**2).sum(axis=0)
    ring = radius_grid(p.shape, dc_index(k)).astype(int).ravel()
    mean_p = np.bincount(ring, p.ravel())/np.maximum(np.bincount(ring, (p > 0).ravel().astype(float)), 1)
    gain = np.clip((mean_p - strength*N)/np.maximum(mean_p, 1e-300), 0.0, 1.0)
    return k*gain[ring].reshape(p.shape)


def sharpen(img: np.ndarray, amount: float, sigma: float = 1.0) -> np.ndarray:
    """Unsharp mask in the frequency domain: Y = X((1+a) - a*G)."""
    if amount < 0:
        raise ValueError("amount must be non-negative")
    if amount == 0:
        return img
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.fftfreq(img.shape[1])[None, :]
    G = np.exp(-2*(np.pi*sigma)**2*(fy**2 + fx**2))
    return np.real(np.fft.ifft2(np.fft.fft2(img)*((1 + amount) - amount*G)))


if __name__ == "__main__":
    from mri.kspace_core import load_phantom, to_kspace, from_kspace
    from mri.metrics import compute_metrics
    img = load_phantom()
    k = to_kspace(img)
    nrm = lambda a: (a - a.min())/(a.max() - a.min() + 1e-12)
    psnr = lambda kk: compute_metrics(img, nrm(from_kspace(kk)))["psnr"]
    rng = np.random.default_rng(0)
    noisy = lambda: k + 0.5*np.abs(k).mean()*(rng.normal(size=k.shape) + 1j*rng.normal(size=k.shape))

    moved = shift_kspace(k, 7, -12)
    assert np.abs(from_kspace(moved) - np.roll(img, (7, -12), (0, 1))).max() < 1e-9
    assert find_shift(k, moved) == (7, -12)
    two = np.stack([k, k])
    assert np.abs(align_phase(two, two*np.array([np.exp(2.5j), np.exp(-1.1j)])[:, None, None]) - two).max() < 1e-6*np.abs(k).max()

    reps = [noisy(), noisy()*np.exp(3.0j), shift_kspace(noisy(), 0, 4)*np.exp(-2.0j)]
    naive, _ = average_repetitions(reps, align=False)
    aligned, info = average_repetitions(reps)
    assert info["shifts"] == [(0, 0), (0, 4)]
    assert psnr(aligned) > psnr(reps[0]) + 3 and psnr(naive) < psnr(aligned) - 3

    n = noisy()
    added = noise_power(n) - noise_power(k)
    assert abs(added/(2*(0.5*np.abs(k).mean())**2) - 1) < 0.1, "corner power rises by the noise power"
    assert psnr(noise_floor_filter(n, 1.0)) > psnr(n) + 1
    half = np.zeros(k.shape); half[::2] = 1
    keep = lambda kk: np.abs(noise_floor_filter(kk, 1.0)).sum()/np.abs(kk).sum()
    assert abs(keep(n*half) - keep(n)) < 0.05, "same strength at half sampling"
    silent = np.zeros_like(k)
    assert noise_floor_filter(silent, 1.0) is silent and noise_floor_filter(n, 0.0) is n

    blurred = nrm(from_kspace(k*np.exp(-(radius_grid(k.shape)/60.0)**2)))
    steep = lambda a: float(np.percentile(np.hypot(*np.gradient(a)), 99))
    assert sharpen(blurred, 0.0) is blurred and steep(nrm(sharpen(blurred, 1.0))) > steep(blurred)

    for bad in (lambda: noise_floor_filter(k, -1.0), lambda: sharpen(img, -0.5),
                lambda: average_repetitions([]), lambda: average_repetitions([k, k[:10]])):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("bad input should have raised")
    print("denoise: all self-tests passed")
