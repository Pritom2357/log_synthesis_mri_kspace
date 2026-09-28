"""Deliberate k-space edits: DC, spike, erase, magnitude/phase, partial Fourier. All coil-safe."""
from __future__ import annotations
import numpy as np


def _mag(k):
    return np.abs(k) if k.ndim == 2 else np.abs(k).sum(axis=0)


def dc_index(k: np.ndarray) -> tuple[int, int]:
    """The brightest sample; scanner data has it off the array middle (row 125 on M4Raw)."""
    m = _mag(k)
    return tuple(int(v) for v in np.unravel_index(m.argmax(), m.shape))


def measured_lines(k: np.ndarray, axis: int) -> np.ndarray:
    """Per row (axis=-2) or column (axis=-1): True where anything was recorded."""
    return _mag(k).sum(axis=-1 if axis == -2 else 0) > 0


def scale_dc(k: np.ndarray, scale: float) -> np.ndarray:
    if scale < 0:
        raise ValueError("scale must be non-negative")
    out = np.array(k, copy=True)
    cy, cx = dc_index(k)
    out[..., cy, cx] *= scale
    return out


def add_spike(k: np.ndarray, dy: int, dx: int, strength: float = 1.0) -> np.ndarray:
    """One sample at (dy,dx) from DC set to strength*max|k|: a single grating in the image."""
    out = np.array(k, copy=True)
    cy, cx = dc_index(k)
    y, x = cy + int(dy), cx + int(dx)
    if not (0 <= y < k.shape[-2] and 0 <= x < k.shape[-1]):
        raise ValueError(f"spike at ({dy},{dx}) falls outside k-space")
    out[..., y, x] = strength*np.abs(out).max()
    return out


def erase_patch(k: np.ndarray, dy: int, dx: int, size: int) -> np.ndarray:
    """Zeroes a disc of diameter `size` centred (dy,dx) from DC: an ideal high-pass when dy=dx=0."""
    if size < 1:
        raise ValueError("size must be at least 1")
    from mri.kspace_core import radius_grid
    cy, cx = dc_index(k)
    return np.where(radius_grid(k.shape[-2:], (cy + dy, cx + dx)) <= size/2, 0, k)


def keep_part(k: np.ndarray, part: str) -> np.ndarray:
    """both | magnitude (phase set to 0) | phase (magnitude set to the mean; unmeasured stay 0)."""
    if part == "both":
        return k
    if part == "magnitude":
        return np.abs(k).astype(np.complex128)
    if part == "phase":
        mag = np.abs(k)
        measured = mag > 0
        if not measured.any():
            return np.zeros_like(k, dtype=np.complex128)
        return np.where(measured, np.exp(1j*np.angle(k)), 0)*float(mag[measured].mean())
    raise ValueError(f"Unknown part: {part!r}. Expected: both, magnitude, phase.")


def hermitian_fill(k: np.ndarray, measured: np.ndarray, centre: tuple[int, int] | None = None) -> np.ndarray:
    """Fills unmeasured samples from K[-u,-v] = conj(K[u,v]), mirrored about the DC sample."""
    out = np.array(k, copy=True)
    meas = np.asarray(measured, dtype=bool)
    rows, cols = out.shape[-2:]
    cy, cx = dc_index(out) if centre is None else centre
    shift = (2*cy - rows + 1, 2*cx - cols + 1)  # flip sends i to n-1-i; this roll makes it 2c-i
    flipped = np.roll(np.conj(np.flip(out, axis=(-2, -1))), shift, axis=(-2, -1))
    partner = np.roll(np.flip(meas, axis=(-2, -1)), shift, axis=(-2, -1))
    fillable = ~meas & partner
    out[..., fillable] = flipped[..., fillable]
    return out


def partial_fourier(k: np.ndarray, fraction: float = 0.6, fill: bool = True) -> tuple:
    """Keeps the first `fraction` of rows; optionally rebuilds the rest. Returns (k, measured)."""
    if not 0.5 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0.5 and 1.0")
    keep = int(round(fraction*k.shape[-2]))
    meas = np.zeros(k.shape[-2:], dtype=bool)
    meas[:keep] = True
    centre = dc_index(k)  # before the cut: at 0.5 the cut can remove DC itself
    out = np.array(k, copy=True)
    out[..., keep:, :] = 0
    return (hermitian_fill(out, meas, centre) if fill else out), meas


if __name__ == "__main__":
    from mri.kspace_core import load_phantom, to_kspace, from_kspace
    from mri.metrics import compute_metrics
    img = load_phantom()
    k = to_kspace(img)
    n = img.shape[0]
    nrm = lambda a: (a - a.min())/(a.max() - a.min() + 1e-12)
    signed = lambda kk: np.real(np.fft.ifft2(np.fft.ifftshift(kk)))

    assert dc_index(k) == (n//2, n//2) and dc_index(np.roll(k, (5, -3), (0, 1))) == (n//2 + 5, n//2 - 3)
    assert np.array_equal(scale_dc(k, 1.0), k)
    for s in (0.5, 0.0):
        assert abs(signed(scale_dc(k, s)).mean() - s*signed(k).mean()) < 1e-9
    assert from_kspace(scale_dc(k, 0.0)).mean() > from_kspace(k).mean(), "magnitude folds negatives up"

    flat = np.zeros_like(k); flat[n//2, n//2] = np.abs(k).max()
    row = from_kspace(add_spike(flat, 0, 12))[n//2]
    assert int(np.argmax(np.abs(np.fft.rfft(row - row.mean())))) == 12, "spike 12 -> 12 cycles"

    centre_gone = nrm(from_kspace(erase_patch(k, 0, 0, 40)))
    edge_gone = nrm(from_kspace(erase_patch(k, 90, 90, 40)))
    assert compute_metrics(img, edge_gone)["ssim"] > compute_metrics(img, centre_gone)["ssim"]
    disc = erase_patch(k, 0, 0, 40)
    assert disc[n//2, n//2] == 0 and disc[n//2 + 19, n//2] == 0 and disc[n//2 + 18, n//2 + 18] != 0, "a disc, not a square"

    for frac in (0.6, 0.75):
        filled, zero = partial_fourier(k, frac)[0], partial_fourier(k, frac, fill=False)[0]
        assert compute_metrics(img, nrm(from_kspace(filled)))["ssim"] > compute_metrics(img, nrm(from_kspace(zero)))["ssim"]
    assert compute_metrics(img, nrm(from_kspace(partial_fourier(k, 1.0)[0])))["ssim"] > 0.999

    moved = np.roll(k, -3, axis=0)  # off-centre DC, as on the scanner files
    half = np.zeros(img.shape, bool); half[:160] = True
    ok = slice(160, 251)  # rows 251.. mirror onto unmeasured rows
    assert np.abs(hermitian_fill(moved*half, half, dc_index(moved)) - moved)[ok].max() < 1e-9*np.abs(moved).max()
    assert np.abs(hermitian_fill(moved*half, half, (n//2, n//2)) - moved)[ok].max() > 1e-3*np.abs(moved).max()

    padded = k.copy(); padded[:, :30] = 0; padded[:, -30:] = 0
    assert measured_lines(padded, -1).sum() == n - 60 and measured_lines(padded, -2).all()
    assert keep_part(k, "both") is k
    edges = lambda a: np.corrcoef(np.hypot(*np.gradient(a)).ravel(), np.hypot(*np.gradient(img)).ravel())[0, 1]
    assert edges(nrm(from_kspace(keep_part(k, "phase")))) > edges(nrm(from_kspace(keep_part(k, "magnitude")))) + 0.2
    assert not keep_part(padded, "phase")[:, :30].any() and not keep_part(np.zeros_like(k), "phase").any()

    for bad in (lambda: keep_part(k, "neither"), lambda: scale_dc(k, -1.0), lambda: add_spike(k, 9999, 0),
                lambda: erase_patch(k, 0, 0, 0), lambda: partial_fourier(k, 0.2)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("bad input should have raised")
    print("kspace_edit: all self-tests passed")
