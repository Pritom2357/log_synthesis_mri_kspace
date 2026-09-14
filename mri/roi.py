"""
Scanning a small region instead of the whole head, and the two ways it goes wrong.

k-space has two independent knobs, and confusing them is the classic mistake:

    sample SPACING (dk)  sets the field of view,   FOV = 1/dk
    sample EXTENT (kmax) sets the resolution,      dx  = 1/(2*kmax)

So there are two different things you might mean by "zoom in", and neither one
gives you a magnified picture for free. Sampling more coarsely shrinks the FOV
but folds whatever is outside it back on top of the anatomy. Keeping only the
middle of k-space leaves the FOV alone and just throws away resolution.
"""
from __future__ import annotations
import numpy as np

__all__ = ["reduced_fov","crop_kspace","zero_fill_to","true_crop",
           "expected_aliasing"] # Only these will be exported

# ------
# Helper
# ------

def _check(shape:tuple[int,int],factor:int)->None:
    rows,cols = shape

    if not isinstance(factor,(int,np.integer)) or factor < 1:
        raise ValueError("factor must be a positive integer")
    if rows % factor or cols % factor:
        raise ValueError(f"factor {factor} must divide both {rows} and {cols}")

# ---------------------
# Methods to call from
# ---------------------

def reduced_fov(k:np.ndarray,factor:int)->np.ndarray:
    """
    Acquires a smaller field of view by sampling k-space more coarsely.

    Keeping every factor-th sample multiplies dk by factor, so the field of view
    divides by factor. The scan really is faster and the image really is
    smaller, but Nyquist has been violated for everything outside the new field
    of view, and that anatomy wraps around onto what you wanted.

    Return: complex array of shape (rows/factor, cols/factor), DC still centred.
    """
    _check(k.shape,factor)
    if factor == 1:
        return k

    return k[::factor,::factor] # DC survives because rows/2 is a multiple of factor

def crop_kspace(k:np.ndarray,factor:int)->np.ndarray:
    """
    Keeps only the central patch of k-space.

    This is what people try when they want to zoom, and it does something quite
    different: dk is unchanged so the field of view is unchanged, and the image
    still shows the whole head. All that was lost is kmax, which is to say
    resolution. The picture gets smaller and blurrier, not closer.

    Return: complex array of shape (rows/factor, cols/factor).
    """
    _check(k.shape,factor)
    if factor == 1:
        return k

    rows,cols = k.shape
    nr,nc = rows//factor, cols//factor
    r0,c0 = rows//2 - nr//2, cols//2 - nc//2
    return k[r0:r0+nr, c0:c0+nc]

def zero_fill_to(k:np.ndarray,shape:tuple[int,int])->np.ndarray:
    """
    Pads cropped k-space back out with zeros, centred.

    This is zero-filled interpolation, and it is what a scanner does when it
    measured only the middle of k-space but still displays on the full matrix.
    Because dk is unchanged, the field of view is unchanged, and because the
    output is the original size again it lines up pixel for pixel with the
    original image. What was lost is real resolution, and padding cannot bring
    it back -- the picture is simply blurrier.
    """
    rows,cols = shape
    nr,nc = k.shape
    if (nr,nc) == (rows,cols):
        return k
    if nr > rows or nc > cols:
        raise ValueError(f"cannot zero-fill {k.shape} out to the smaller {shape}")

    out = np.zeros(shape,dtype=np.complex128)
    r0,c0 = rows//2 - nr//2, cols//2 - nc//2
    out[r0:r0+nr, c0:c0+nc] = k
    return out

def true_crop(img:np.ndarray,factor:int)->np.ndarray:
    """The centre region a reduced-FOV scan was aiming at, cut straight from the original."""
    _check(img.shape,factor)
    if factor == 1:
        return img

    rows,cols = img.shape
    nr,nc = rows//factor, cols//factor
    r0,c0 = rows//2 - nr//2, cols//2 - nc//2
    return img[r0:r0+nr, c0:c0+nc]

def expected_aliasing(img:np.ndarray,factor:int)->np.ndarray:
    """
    What a reduced-FOV scan must produce, worked out directly in image space.

    Decimating k-space by factor is periodic summation in image space: the
    picture is cut into factor-by-factor tiles and those tiles are stacked on
    top of each other. Useful as ground truth for the fold-over.
    """
    _check(img.shape,factor)
    if factor == 1:
        return img

    rows,cols = img.shape
    nr,nc = rows//factor, cols//factor
    tiles = np.asarray(img,dtype=np.float64).reshape(factor,nr,factor,nc)
    return tiles.sum(axis=(0,2))

# ------------------------
# Tests for roi.py only
# ------------------------

if __name__ == "__main__":
    from mri.kspace_core import load_phantom, to_kspace, from_kspace
    from mri.metrics import compute_metrics
    from skimage.transform import resize

    img = load_phantom()
    k = to_kspace(img)
    rows,cols = img.shape

    # --- factor 1 is a no-op everywhere ---
    assert reduced_fov(k,1) is k and crop_kspace(k,1) is k and true_crop(img,1) is img
    print("[no-op]      factor=1 returns the inputs untouched")

    # --- Shapes and the surviving DC term ---
    for f in (2,4,8):
        rf, ck = reduced_fov(k,f), crop_kspace(k,f)
        assert rf.shape == ck.shape == (rows//f,cols//f), (rf.shape,ck.shape)
        m = rf.shape[0]//2
        assert abs(rf[m,m]) == abs(k[rows//2,cols//2]), "reduced FOV must keep the DC sample"
        assert abs(ck[m,m]) == abs(k[rows//2,cols//2]), "cropping must keep the DC sample"
    print("[shapes]     factor 2/4/8 give N/f squares, DC centred in both")

    # --- Reduced FOV is EXACTLY periodic summation. This is the fold-over, proven. ---
    for f in (2,4):
        # numpy ifft2 already carries 1/M^2, so the identity needs no rescaling.
        got = from_kspace(reduced_fov(k,f))
        want = expected_aliasing(img,f)
        err = np.abs(got-want).max()
        assert err < 1e-9, f"factor {f}: fold-over does not match tile summation, err={err:.2e}"
        print(f"[fold-over]  factor={f}: matches tile summation to {err:.1e}")

    # --- And that fold-over is NOT what you wanted ---
    f = 2
    got = from_kspace(reduced_fov(k,f))
    want = true_crop(img,f)
    nrm = lambda a: (a-a.min())/(a.max()-a.min()+1e-12)
    m_fold = compute_metrics(want,nrm(got))
    assert m_fold["ssim"] < 0.9, "the wrapped image should not match the intended crop"
    outside = true_crop(img,f) < 0.01 # background inside the reduced FOV
    assert nrm(got)[outside].mean() > 5*nrm(want)[outside].mean(), \
        "folded anatomy must light up the background of the target region"
    print(f"[not a zoom] reduced FOV vs intended crop: ssim={m_fold['ssim']:.3f}"
          f"   background {nrm(want)[outside].mean():.4f} -> {nrm(got)[outside].mean():.4f}")

    # --- Cropping k-space keeps the WHOLE head, just blurrier. Also not a zoom. ---
    for f in (2,4):
        small = from_kspace(crop_kspace(k,f))
        whole = resize(img,small.shape,anti_aliasing=True) # the full head, shrunk
        centre = resize(true_crop(img,f),small.shape,anti_aliasing=True) # the zoom you wanted
        s_whole = compute_metrics(whole,nrm(small))["ssim"]
        s_centre = compute_metrics(centre,nrm(small))["ssim"]
        assert s_whole > s_centre, \
            f"cropped k-space should look like the whole head, not the centre ({s_whole:.3f} vs {s_centre:.3f})"
        print(f"[crop != zoom] factor={f}: ssim vs whole head={s_whole:.3f}"
              f"  vs intended zoom={s_centre:.3f}")

    # --- Zero-filling puts it back on the original grid: same FOV, same size ---
    # This is why cropping needs no change of reference: the picture still lines
    # up with the original pixel for pixel, it is only softer.
    for f in (2,4):
        padded = zero_fill_to(crop_kspace(k,f),img.shape)
        assert padded.shape == img.shape
        assert abs(padded[rows//2,cols//2]) == abs(k[rows//2,cols//2]), "DC must survive padding"
        rec = nrm(from_kspace(padded))
        m = compute_metrics(img,rec)
        # Edge steepness, measured at the 99th percentile of the gradient. The
        # MEAN gradient is no good here: truncation adds ringing everywhere,
        # which raises it even as the real edges get softer.
        steep = lambda a: np.percentile(np.hypot(*np.gradient(a)),99)
        ratio = steep(rec)/steep(img)
        assert m["ssim"] < 0.999, "zero-filling must not magically restore resolution"
        assert ratio < 0.9, f"edges should soften, but steepness ratio was {ratio:.2f}"
        print(f"[zero-fill]  factor={f}: shape {rec.shape}  ssim vs original={m['ssim']:.3f}"
              f"  edge steepness={ratio:.2f}x")

    assert zero_fill_to(k,img.shape) is k, "already-full k-space is returned untouched"
    try:
        zero_fill_to(k,(64,64))
    except ValueError:
        print("[guard]      refusing to zero-fill down to a smaller shape")
    else:
        raise AssertionError("shrinking via zero_fill_to should have raised")

    # --- Guards ---
    for bad in (0,3,-2):
        try:
            reduced_fov(k,bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"factor {bad} should have raised")
    print("[guard]      non-dividing and non-positive factors rejected")

    print("\nAll self-tests passed")
