"""
Forward and inverse 2D Fourier transforms, plus the input path that turns any
image into the 256x256 float array every other module assumes.
"""
from __future__ import annotations
import numpy as np

__all__ = ["prepare","load_image","load_phantom","to_kspace","from_kspace",
           "log_magnitude","radius_grid","low_pass","high_pass"] # Only these will be exported

SIZE = 256
_LUMA = np.array([0.299,0.587,0.114]) # ITU-R BT.601 luminance weights

# -----------------------------------------------------------------
# Input preparation (private helpers that will NOT be exported)
# -----------------------------------------------------------------

def _to_grayscale(img:np.ndarray)->np.ndarray:
    """Collapses a colour image to one channel; any alpha channel is dropped."""
    arr = np.asarray(img,dtype=np.float64)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:,:,:3] @ _LUMA
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:,:,0]

    raise ValueError(f"Cannot read shape {arr.shape!r} as a 2D image.")

def _pad_to_square(img:np.ndarray)->np.ndarray:
    """Zero-pads the shorter side. Runs before resizing, or the anatomy stretches."""
    rows,cols = img.shape
    if rows == cols:
        return img

    side = max(rows,cols)
    out = np.zeros((side,side),dtype=img.dtype)
    r0 = (side-rows)//2
    c0 = (side-cols)//2
    out[r0:r0+rows, c0:c0+cols] = img
    return out

def _normalize(img:np.ndarray)->np.ndarray:
    """Rescales to [0,1]. A constant image returns zeros rather than dividing by zero."""
    arr = np.asarray(img,dtype=np.float64)
    lo,hi = float(arr.min()),float(arr.max())

    if hi-lo < 1e-12:
        return np.zeros_like(arr)
    return (arr-lo)/(hi-lo)

# ---------------------
# Methods to call from
# ---------------------

def prepare(img:np.ndarray,size:int=SIZE)->np.ndarray:
    """
    Full input path: grayscale -> pad to square -> resize -> normalise.

    Return: float64 array of shape (size,size) with values in [0,1].
    """
    from skimage.transform import resize

    sq = _pad_to_square(_to_grayscale(img))
    if sq.shape != (size,size):
        sq = resize(sq,(size,size),anti_aliasing=True,preserve_range=True)
    return _normalize(sq)

def load_image(path:str,size:int=SIZE)->np.ndarray:
    """Reads an image file from disk and runs it through `prepare`."""
    from skimage.io import imread
    return prepare(imread(path),size)

def load_phantom(size:int=SIZE)->np.ndarray:
    """Shepp-Logan phantom, the standard reconstruction test object."""
    from skimage.data import shepp_logan_phantom
    return prepare(shepp_logan_phantom(),size)

def to_kspace(img:np.ndarray)->np.ndarray:
    """
    Image -> centred complex k-space.

    fft2 does the transform, fftshift then moves the DC term from the corner to
    the middle, which is the layout every mask and filter downstream assumes.
    """
    return np.fft.fftshift(np.fft.fft2(np.asarray(img,dtype=np.float64)))

def from_kspace(k:np.ndarray)->np.ndarray:
    """Centred k-space -> magnitude image. Exact inverse of `to_kspace`."""
    return np.abs(np.fft.ifft2(np.fft.ifftshift(np.asarray(k))))

def log_magnitude(k:np.ndarray)->np.ndarray:
    """
    k-space compressed for display, normalised to [0,1].

    The centre outweighs the edges by orders of magnitude, so a linear display
    is a black square with one white dot. For display only; never reconstruct
    from this.
    """
    return _normalize(np.log1p(np.abs(np.asarray(k))))

def radius_grid(shape:tuple[int,int],centre:tuple[int,int]|None=None)->np.ndarray:
    """
    Distance in pixels of every entry from `centre`, the middle of the matrix
    by default. Pass the real DC position for scanner data, where the two
    differ by a few samples.
    """
    rows,cols = shape
    cy,cx = (rows//2,cols//2) if centre is None else centre
    Y,X = np.ogrid[:rows,:cols]
    return np.hypot(Y-cy, X-cx)

def low_pass(k:np.ndarray,radius:float)->np.ndarray:
    """
    Keeps the disc of k-space around the DC sample, zeroing everything outside.

    Distance is measured from where the DC sample really is (dc_index), not
    the array middle: on the scanner files here they are 3 to 8 rows apart,
    and a small disc around the wrong point cuts the DC term out entirely.
    The grid is built on the last two axes, so (coils, ky, kx) broadcasts.
    """
    from mri.kspace_edit import dc_index # local: kspace_edit's tests import this module
    return np.where(radius_grid(k.shape[-2:],dc_index(k)) <= radius, k, 0.0)

def high_pass(k:np.ndarray,radius:float)->np.ndarray:
    """Deletes the disc around the DC sample and keeps the rest. Coil-safe, as above."""
    from mri.kspace_edit import dc_index
    return np.where(radius_grid(k.shape[-2:],dc_index(k)) > radius, k, 0.0)

# -------------------------------
# Tests for kspace_core.py only
# -------------------------------

if __name__ == "__main__":
    img = load_phantom()
    assert img.shape == (SIZE,SIZE), img.shape
    assert 0.0 <= img.min() and img.max() <= 1.0
    print(f"[phantom]         shape={img.shape}  range=[{img.min():.2f},{img.max():.2f}]")

    # --- Round-trip: the gate everything else depends on ---
    back = from_kspace(to_kspace(img))
    err = float(np.abs(img-back).max())
    assert err < 1e-12, f"round-trip error {err:.3e} too large, check the shifts"
    print(f"[round-trip]      max_error={err:.3e}")

    # --- Parseval: energy is conserved across the two domains ---
    k = to_kspace(img)
    e_img = float(np.sum(img**2))
    e_k = float(np.sum(np.abs(k)**2)/img.size)
    rel = abs(e_img-e_k)/e_img
    assert rel < 1e-10, f"Parseval failed, relative error {rel:.3e}"
    print(f"[parseval]        rel_error={rel:.3e}")

    # --- DC term equals the sum of all pixels ---
    dc = float(np.abs(k[SIZE//2,SIZE//2]))
    total = float(img.sum())
    assert abs(dc-total)/total < 1e-10, (dc,total)
    print(f"[dc]              K_centre={dc:.1f}  pixel_sum={total:.1f}")

    # --- Display helper: centre must outshine the corner ---
    disp = log_magnitude(k)
    assert disp.shape == k.shape and 0.0 <= disp.min() and disp.max() <= 1.0
    assert disp[SIZE//2,SIZE//2] > disp[0,0], "k-space is not centred"
    print(f"[log_magnitude]   centre={disp[SIZE//2,SIZE//2]:.3f}  corner={disp[0,0]:.3f}")

    # --- Awkward inputs must survive the input path ---
    tall = np.random.default_rng(0).random((120,40))
    colour = np.random.default_rng(1).random((64,64,3))
    assert prepare(tall).shape == (SIZE,SIZE)
    assert prepare(colour).shape == (SIZE,SIZE)
    assert _normalize(np.full((8,8),0.5)).max() == 0.0, "constant image must not divide by zero"
    print("[edge cases]      non-square, colour and constant inputs all handled")

    # --- Frequency probing: low-pass blurs, high-pass keeps only edges ---
    lo = from_kspace(low_pass(k,20))
    hi = from_kspace(high_pass(k,20))
    assert lo.std() < img.std(), "low-pass should reduce variation"
    assert hi.mean() < img.mean(), "high-pass should be darker than the original"
    print(f"[band probing]    lo_std={lo.std():.4f} < img_std={img.std():.4f}  hi_mean={hi.mean():.4f}")

    print("\nAll self-tests passed")
