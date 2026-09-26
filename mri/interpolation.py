"""Upscaling three ways: zero-order hold, linear, and ideal sinc (zero-padded spectrum)."""
from __future__ import annotations
import numpy as np


def zero_order_hold(img: np.ndarray, factor: int) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    return np.repeat(np.repeat(a, factor, axis=0), factor, axis=1)


def linear_interp(img: np.ndarray, factor: int) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    if factor == 1:
        return a
    rows, cols = a.shape
    y, x = np.linspace(0, rows - 1, rows*factor), np.linspace(0, cols - 1, cols*factor)
    y0, x0 = np.clip(np.floor(y).astype(int), 0, rows - 2), np.clip(np.floor(x).astype(int), 0, cols - 2)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    top = a[y0][:, x0]*(1 - fx) + a[y0][:, x0 + 1]*fx
    bot = a[y0 + 1][:, x0]*(1 - fx) + a[y0 + 1][:, x0 + 1]*fx
    return top*(1 - fy) + bot*fy


def zero_fill_to(k: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Centres k (last two axes) in a zero array of `shape`."""
    nr, nc = k.shape[-2:]
    if (nr, nc) == tuple(shape):
        return k
    out = np.zeros(k.shape[:-2] + tuple(shape), dtype=np.complex128)
    r0, c0 = shape[0]//2 - nr//2, shape[1]//2 - nc//2
    out[..., r0:r0 + nr, c0:c0 + nc] = k
    return out


def sinc_interp(img: np.ndarray, factor: int) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    if factor == 1:
        return a
    big = zero_fill_to(np.fft.fftshift(np.fft.fft2(a)), (a.shape[0]*factor, a.shape[1]*factor))
    return np.real(np.fft.ifft2(np.fft.ifftshift(big)))*factor**2  # padding dilutes by 1/factor^2


def resize(img: np.ndarray, factor: int, method: str = "sinc") -> np.ndarray:
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("factor must be a positive integer")
    fn = {"zero_order_hold": zero_order_hold, "linear": linear_interp, "sinc": sinc_interp}.get(method)
    if fn is None:
        raise ValueError(f"Unknown method: {method!r}. Expected: zero_order_hold, linear, sinc.")
    return fn(img, factor)


if __name__ == "__main__":
    from mri.kspace_core import load_phantom, to_kspace
    from mri.metrics import compute_metrics
    img = load_phantom()
    nrm = lambda a: (a - a.min())/(a.max() - a.min() + 1e-12)
    for m in ("zero_order_hold", "linear", "sinc"):
        assert resize(img, 1, m).shape == img.shape and resize(img, 2, m).shape == (512, 512)
    held = zero_order_hold(np.array([[0.0, 1.0], [1.0, 0.0]]), 3)
    assert (held[:3, :3] == 0).all() and (held[:3, 3:] == 1).all()
    small = np.real(np.fft.ifft2(np.fft.ifftshift(to_kspace(img)[96:160, 96:160])))/16  # band-limit, 64x64
    psnr = {m: compute_metrics(nrm(img), nrm(resize(small, 4, m)))["psnr"] for m in ("zero_order_hold", "linear", "sinc")}
    assert psnr["sinc"] > psnr["linear"] > psnr["zero_order_hold"], psnr
    for bad in (lambda: resize(img, 0), lambda: resize(img, 2, "cubic")):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("bad input should have raised")
    print("interpolation: all self-tests passed", {m: round(v, 2) for m, v in psnr.items()})
