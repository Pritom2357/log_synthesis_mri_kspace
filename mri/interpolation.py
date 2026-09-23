"""
Changing the size of an image, three ways, and which one the theory says is right.

Lecture 5 names three reconstructions of a sampled signal, and this module is
those three side by side:

    zero_order_hold   hold each sample until the next one. A staircase.
    linear_interp     draw straight lines between samples. A polyline.
    sinc_interp       the ideal band-limited reconstruction, which for a DFT is
                      simply zero-padding the spectrum.

The sampling theorem says the third one is exact for any signal that was
band-limited before sampling, and the other two are approximations. That is a
claim you can measure, and the self-test at the bottom measures it.

Why this matters for MRI: a hospital archive stores k-space, not pictures. From
one stored dataset you can produce a small thumbnail or a full-resolution image
on demand, and which interpolation you use decides how honest the result is.
"""
from __future__ import annotations
import numpy as np

__all__ = ["zero_order_hold","linear_interp","sinc_interp","downscale","resize"]

# ------
# Helper
# ------

def _check(factor:int)->None:
    if not isinstance(factor,(int,np.integer)) or factor < 1:
        raise ValueError("factor must be a positive integer")

# ---------------------
# Methods to call from
# ---------------------

def zero_order_hold(img:np.ndarray,factor:int)->np.ndarray:
    """
    Repeats every sample `factor` times in each direction.

    The simplest reconstruction there is: hold the current sample until the
    next one arrives. In one dimension it draws a staircase; in two it gives
    visible square blocks. It is cheap and it never overshoots, but it invents
    sharp edges at every sample boundary that the original signal did not have.
    """
    _check(factor)
    if factor == 1:
        return np.asarray(img,dtype=np.float64)
    return np.repeat(np.repeat(np.asarray(img,dtype=np.float64),factor,axis=0),factor,axis=1)

def linear_interp(img:np.ndarray,factor:int)->np.ndarray:
    """
    Joins neighbouring samples with straight lines.

    Better than the staircase because the result is continuous, so the blocky
    edges disappear. It still is not the ideal reconstruction: a straight line
    between two samples is not what a band-limited signal does between them,
    and the corners where the lines meet add high frequencies of their own.
    """
    _check(factor)
    a = np.asarray(img,dtype=np.float64)
    if factor == 1:
        return a

    rows,cols = a.shape
    # Sample positions of the output, expressed in input coordinates.
    y = np.linspace(0,rows-1,rows*factor)
    x = np.linspace(0,cols-1,cols*factor)
    y0 = np.clip(np.floor(y).astype(int),0,rows-2)
    x0 = np.clip(np.floor(x).astype(int),0,cols-2)
    fy = (y-y0)[:,None]
    fx = (x-x0)[None,:]

    top = a[y0][:,x0]*(1-fx) + a[y0][:,x0+1]*fx
    bot = a[y0+1][:,x0]*(1-fx) + a[y0+1][:,x0+1]*fx
    return top*(1-fy) + bot*fy

def sinc_interp(img:np.ndarray,factor:int)->np.ndarray:
    """
    The ideal band-limited reconstruction, done by zero-padding the spectrum.

    Interpolating with a sinc kernel in the image is the same thing as padding
    k-space with zeros and transforming back, and the second is far cheaper.
    No new frequency is invented -- the spectrum is unchanged, it is simply
    evaluated on a finer grid. That is precisely what the sampling theorem
    promises, and it is why MRI scanners do this and call it zero-filled
    interpolation.
    """
    _check(factor)
    a = np.asarray(img,dtype=np.float64)
    if factor == 1:
        return a

    rows,cols = a.shape
    k = np.fft.fftshift(np.fft.fft2(a))
    big = np.zeros((rows*factor,cols*factor),dtype=np.complex128)
    y0 = (rows*factor)//2 - rows//2
    x0 = (cols*factor)//2 - cols//2
    big[y0:y0+rows, x0:x0+cols] = k

    out = np.fft.ifft2(np.fft.ifftshift(big))
    # One honest caveat. For an even-sized spectrum the Nyquist row and column
    # are shared between +N/2 and -N/2, so padding cannot preserve Hermitian
    # symmetry perfectly and the result carries a small imaginary part, about
    # 2% of the real part here. Taking the real part discards it, which puts
    # roughly 0.5% of the energy outside the original band. It is a property of
    # even-length DFTs, not a mistake, and it is far smaller than the gap
    # between this method and the other two.
    return np.real(out)*(factor*factor) # padding dilutes by 1/factor^2

def downscale(img:np.ndarray,factor:int)->np.ndarray:
    """
    Shrinks by keeping only the central part of the spectrum.

    This is the matching operation to sinc_interp: band-limit first, then
    sample more coarsely. Doing it in k-space rather than by dropping pixels is
    what stops the discarded detail from aliasing back in.
    """
    _check(factor)
    a = np.asarray(img,dtype=np.float64)
    if factor == 1:
        return a

    rows,cols = a.shape
    if rows % factor or cols % factor:
        raise ValueError(f"factor {factor} must divide {a.shape}")

    k = np.fft.fftshift(np.fft.fft2(a))
    nr,nc = rows//factor, cols//factor
    y0,x0 = rows//2-nr//2, cols//2-nc//2
    small = k[y0:y0+nr, x0:x0+nc]
    return np.real(np.fft.ifft2(np.fft.ifftshift(small)))/(factor*factor)

def resize(img:np.ndarray,factor:int,method:str="sinc")->np.ndarray:
    """
    Upscales by `factor` using one of the three reconstructions.

    method: 'zero_order_hold'/'zoh' | 'linear' | 'sinc'
    """
    m = method.strip().lower()
    if m in ("zero_order_hold","zoh"):
        return zero_order_hold(img,factor)
    elif m == "linear":
        return linear_interp(img,factor)
    elif m in ("sinc","ideal"):
        return sinc_interp(img,factor)
    raise ValueError(
        f"Unknown method: {method!r}. Expected: zero_order_hold/zoh, linear, sinc."
    )

# -------------------------------
# Tests for interpolation.py only
# -------------------------------

if __name__ == "__main__":
    from mri.kspace_core import load_phantom
    from mri.metrics import compute_metrics

    img = load_phantom()
    nrm = lambda a: (a-a.min())/(a.max()-a.min()+1e-12)

    # --- Shapes, and factor 1 is a no-op ---
    for m in ("zoh","linear","sinc"):
        assert resize(img,1,m).shape == img.shape
        assert resize(img,2,m).shape == (512,512), m
    print("[shapes]      all three methods upscale 256 -> 512, factor 1 is a no-op")

    # --- Zero order hold really does hold ---
    tiny = np.array([[0.0,1.0],[1.0,0.0]])
    held = zero_order_hold(tiny,3)
    assert held.shape == (6,6)
    assert np.all(held[:3,:3] == 0.0) and np.all(held[:3,3:] == 1.0), "blocks must be constant"
    print("[zoh]         each sample becomes a constant block, as the name says")

    # --- The real comparison: band-limit, shrink, then rebuild ---
    # The sampling theorem says sinc is exact for a band-limited signal, so it
    # must beat the other two at getting the original back.
    small = downscale(img,4)
    print(f"[downscale]   256x256 -> {small.shape[0]}x{small.shape[1]} by keeping the centre")

    scores = {}
    for m in ("zoh","linear","sinc"):
        back = nrm(resize(small,4,m))
        scores[m] = compute_metrics(nrm(img),back)
        print(f"[{m:6s}]      rebuilt 256x256: psnr={scores[m]['psnr']:6.2f}"
              f"  ssim={scores[m]['ssim']:.4f}")

    assert scores["sinc"]["psnr"] > scores["linear"]["psnr"] > scores["zoh"]["psnr"], \
        f"the theory says sinc > linear > zero order hold, got {scores}"
    print(f"[theory]      sinc beats linear by {scores['sinc']['psnr']-scores['linear']['psnr']:.2f} dB, "
          f"linear beats zoh by {scores['linear']['psnr']-scores['zoh']['psnr']:.2f} dB")

    # --- Sinc barely touches the spectrum: it re-samples rather than invents ---
    up = sinc_interp(small,4)
    k_up = np.abs(np.fft.fftshift(np.fft.fft2(up)))
    n = small.shape[0]
    c = up.shape[0]//2
    inside = k_up[c-n//2:c+n//2, c-n//2:c+n//2].sum()
    leak = 1 - inside/max(k_up.sum(),1e-12)
    print(f"[sinc]        {leak:.2%} of the energy sits outside the original band "
          "(the even-length Nyquist row; see the docstring)")
    assert leak < 0.02, f"zero padding should barely leak, got {leak:.2%}"

    # The other two invent a great deal more, which is the whole point.
    for m in ("zoh","linear"):
        kk = np.abs(np.fft.fftshift(np.fft.fft2(resize(small,4,m))))
        l = 1 - kk[c-n//2:c+n//2, c-n//2:c+n//2].sum()/max(kk.sum(),1e-12)
        print(f"[{m:6s}]      {l:.2%} of the energy is outside the band: "
              "frequencies the original never had")
        assert l > leak, f"{m} should invent more out-of-band content than sinc"

    # --- Guards ---
    for bad in (lambda: resize(img,0,"sinc"), lambda: resize(img,2,"cubic"),
                lambda: downscale(img,3)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("bad input should have raised")
    print("[guard]       factor 0, unknown method and indivisible downscale rejected")

    print("\nAll self-tests passed")
