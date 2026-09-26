"""2D FFT to and from centred k-space, and the input path to a 256x256 [0,1] image."""
from __future__ import annotations
import numpy as np

SIZE = 256


def _normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    return np.zeros_like(a) if hi - lo < 1e-12 else (a - lo)/(hi - lo)


def prepare(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Grayscale -> pad to square -> resize -> [0,1]."""
    from skimage.transform import resize
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        a = a[:, :, :3] @ np.array([0.299, 0.587, 0.114]) if a.shape[2] >= 3 else a[:, :, 0]
    if a.ndim != 2:
        raise ValueError(f"Cannot read shape {a.shape!r} as a 2D image.")
    side = max(a.shape)
    sq = np.zeros((side, side))
    r0, c0 = (side - a.shape[0])//2, (side - a.shape[1])//2
    sq[r0:r0 + a.shape[0], c0:c0 + a.shape[1]] = a
    if sq.shape != (size, size):
        sq = resize(sq, (size, size), anti_aliasing=True, preserve_range=True)
    return _normalize(sq)


def load_image(path: str, size: int = SIZE) -> np.ndarray:
    from skimage.io import imread
    return prepare(imread(path), size)


def load_phantom(size: int = SIZE) -> np.ndarray:
    from skimage.data import shepp_logan_phantom
    return prepare(shepp_logan_phantom(), size)


def to_kspace(img: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.asarray(img, dtype=np.float64)))


def from_kspace(k: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.ifft2(np.fft.ifftshift(np.asarray(k))))


def log_magnitude(k: np.ndarray) -> np.ndarray:
    """log(1+|K|) in [0,1], for display only."""
    return _normalize(np.log1p(np.abs(k)))


def radius_grid(shape: tuple[int, int], centre: tuple[int, int] | None = None) -> np.ndarray:
    rows, cols = shape
    cy, cx = (rows//2, cols//2) if centre is None else centre
    Y, X = np.ogrid[:rows, :cols]
    return np.hypot(Y - cy, X - cx)


if __name__ == "__main__":
    img = load_phantom()
    assert img.shape == (SIZE, SIZE) and 0 <= img.min() and img.max() <= 1
    k = to_kspace(img)
    assert np.abs(img - from_kspace(k)).max() < 1e-12, "round trip"
    assert abs(np.sum(img**2) - np.sum(np.abs(k)**2)/img.size) < 1e-10*np.sum(img**2), "Parseval"
    assert abs(abs(k[SIZE//2, SIZE//2]) - img.sum()) < 1e-10*img.sum(), "DC = pixel sum"
    disp = log_magnitude(k)
    assert disp[SIZE//2, SIZE//2] > disp[0, 0]
    rng = np.random.default_rng(0)
    assert prepare(rng.random((120, 40))).shape == prepare(rng.random((64, 64, 3))).shape == (SIZE, SIZE)
    assert _normalize(np.full((8, 8), 0.5)).max() == 0.0
    print("kspace_core: all self-tests passed")
