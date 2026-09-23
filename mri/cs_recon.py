"""
Compressed sensing reconstruction.

Zero-filling assumes every sample you skipped was zero, which is a lie, and the
artifact you see is that lie. Compressed sensing replaces it with a better
assumption: MRI images are mostly smooth with a few sharp edges, so their total
variation is small. It then looks for the image that has small total variation
AND still agrees with every sample you actually measured.

The loop alternates between those two demands until they settle.

Real scans arrive from several receiver channels at once, so everything here
works on a coil axis as well as a single slice. The prior is applied to the
combined magnitude rather than to each channel separately: there is one
patient, and smoothing four coils independently would let them disagree about
where the edges are.
"""
from __future__ import annotations
import numpy as np

__all__ = ["cs_reconstruct","sampling_mask_from_coords"] # Only these will be exported

# -----------------------------------------------------------------
# Total variation prior (private helpers that will NOT be exported)
# -----------------------------------------------------------------

def _fft(x:np.ndarray)->np.ndarray:
    """
    Same centring convention as kspace_core.to_kspace, but keeps the phase.

    The axes are named because a bare fftshift shifts EVERY axis, which would
    roll a coil array halfway round its channel axis and silently scramble the
    coils into each other.
    """
    return np.fft.fftshift(np.fft.fft2(x,axes=(-2,-1)),axes=(-2,-1))

def _ifft(k:np.ndarray)->np.ndarray:
    """Same convention as kspace_core.from_kspace, but keeps the phase."""
    return np.fft.ifft2(np.fft.ifftshift(k,axes=(-2,-1)),axes=(-2,-1))

def _rss(x:np.ndarray)->np.ndarray:
    """Root sum of squares down the coil axis: one magnitude image from many."""
    return np.sqrt((np.abs(x)**2).sum(axis=0))

def _grad(x:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    """Forward differences down and across."""
    return np.roll(x,-1,0)-x, np.roll(x,-1,1)-x

def _div(py:np.ndarray,px:np.ndarray)->np.ndarray:
    """Divergence, the adjoint of _grad, so the pair is a proper gradient step."""
    return (py-np.roll(py,1,0)) + (px-np.roll(px,1,1))

def _tv_denoise(x:np.ndarray,weight:float,n_steps:int,step:float)->np.ndarray:
    """
    A few gradient steps downhill on total variation.

    Total variation is the sum of edge magnitudes, so pushing it down smooths
    flat regions hard while leaving real edges mostly intact. That is exactly
    the trade MRI wants.

    x may be complex, and may carry a coil axis. The edge magnitude is taken
    with the complex modulus and summed over the channels, so all of them share
    one set of edges: there is one patient, and letting four coils each decide
    where a boundary lies smears the boundary. On a real 2-D array this reduces
    to the ordinary isotropic TV step, unchanged.
    """
    for _ in range(n_steps):
        gy,gx = _grad(x)
        e = np.abs(gy)**2 + np.abs(gx)**2
        mag = np.sqrt(e.sum(axis=0) if x.ndim == 3 else e) + 1e-8
        x = x + step*weight*_div(gy/mag, gx/mag)
    return x

# ---------------------
# Methods to call from
# ---------------------

def sampling_mask_from_coords(coords:np.ndarray,shape:tuple[int,int])->np.ndarray:
    """
    Which grid cells a non-Cartesian trajectory actually touched.

    Compressed sensing needs to know which samples are real measurements so it
    can re-impose them. For radial and spiral there is no mask, so we mark every
    cell that at least one sample landed in.
    """
    rows,cols = shape
    y = np.clip(np.round(coords[:,0]+rows//2).astype(int),0,rows-1)
    x = np.clip(np.round(coords[:,1]+cols//2).astype(int),0,cols-1)

    mask = np.zeros(shape,dtype=bool)
    mask[y,x] = True
    return mask

def cs_reconstruct(k_measured:np.ndarray,mask:np.ndarray,n_iter:int=60,
                   weight:float=0.05,tv_steps:int=4,tv_step:float=0.2)->np.ndarray:
    """
    Reconstructs by alternating a sparsity prior with data consistency.

    Each iteration does two things. First it smooths the current estimate by
    stepping downhill on total variation, which removes artifacts because
    artifacts are not smooth. Then it puts back every k-space sample that was
    genuinely measured, so the smoothing can never overwrite real data.

    k_measured: (ky,kx), or (coils,ky,kx) when several channels listened at
        once. Every channel is held to the same prior.
    mask: True where a sample was actually acquired, shape (ky,kx). It
        broadcasts over the coil axis, because all channels read the same lines.
    weight: how hard to push on smoothness. 0.05 was tuned on the phantom;
        raising it to 0.2 already costs more real detail than it removes artifact.

    Return: float64 magnitude image of shape (ky,kx), whatever went in.
    """
    k = np.asarray(k_measured)
    if k.ndim not in (2,3):
        raise ValueError(f"expected (ky,kx) or (coils,ky,kx), got {k.shape}")
    mask = np.asarray(mask).astype(bool)
    if k.shape[-2:] != mask.shape:
        raise ValueError(f"shape mismatch: {k.shape[-2:]} vs {mask.shape}")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")

    k = k if k.ndim == 3 else k[None]  # a single coil is a coil set of one
    x = _ifft(k)                       # complex, per channel: zero-filled start

    for _ in range(n_iter):
        # The smoothing runs on the complex field itself. An earlier version
        # took the real part and clipped it at zero, which is a fair prior for
        # a phantom -- that really is real and non-negative -- and wrong for a
        # scan, where phase carries signal. On this file that choice is worth
        # 6 dB at R=2. Rescaling by a smoothed magnitude instead was worse than
        # doing nothing: in the background the magnitude is near zero, so the
        # ratio explodes and amplifies exactly the noise it was meant to remove.
        x = _tv_denoise(x,weight,tv_steps,tv_step)
        x = _ifft(np.where(mask,k,_fft(x)))  # data consistency, channel by channel

    return _rss(x)

# ----------------------------
# Tests for cs_recon.py only
# ----------------------------

if __name__ == "__main__":
    from mri.kspace_core import load_phantom, to_kspace, from_kspace
    from mri.cartesian_sampling import make_mask
    from mri.metrics import compute_metrics
    from mri.trajectories import radial

    img = load_phantom()
    k = to_kspace(img)
    shape = img.shape
    nrm = lambda a: (a-a.min())/(a.max()-a.min()+1e-12)

    # --- Fully sampled: there is nothing to guess, so it must stay perfect ---
    full = make_mask("uniform",shape,R=1)
    rec = cs_reconstruct(k*full,full.astype(bool),n_iter=10)
    assert compute_metrics(img,nrm(rec))["ssim"] > 0.99, "CS must not damage complete data"
    print(f"[complete]   R=1  ssim={compute_metrics(img,nrm(rec))['ssim']:.4f}  (data consistency holds)")

    # --- The headline: CS must beat zero-filling on an incoherent mask ---
    for R in (2,4):
        mask = make_mask("random",shape,R=R,seed=7).astype(bool)
        zf = nrm(from_kspace(k*mask))
        cs = nrm(cs_reconstruct(k*mask,mask))
        m_zf = compute_metrics(img,zf)
        m_cs = compute_metrics(img,cs)
        assert m_cs["psnr"] > m_zf["psnr"], f"CS lost to zero-filling at R={R}"
        assert m_cs["ssim"] > m_zf["ssim"], f"CS lost on SSIM at R={R}"
        print(f"[random R={R}] zero-filled psnr={m_zf['psnr']:5.2f} ssim={m_zf['ssim']:.3f}"
              f"   ->  CS psnr={m_cs['psnr']:5.2f} ssim={m_cs['ssim']:.3f}"
              f"   (+{m_cs['psnr']-m_zf['psnr']:.2f} dB)")

    # --- CS likes incoherent sampling: it should help random more than uniform ---
    gains = {}
    for kind in ("uniform","random"):
        mask = make_mask(kind,shape,R=4,seed=7).astype(bool)
        zf = nrm(from_kspace(k*mask))
        cs = nrm(cs_reconstruct(k*mask,mask))
        gains[kind] = compute_metrics(img,cs)["psnr"] - compute_metrics(img,zf)["psnr"]
    print(f"[incoherence] CS gain: uniform {gains['uniform']:+.2f} dB   random {gains['random']:+.2f} dB")
    assert gains["random"] > gains["uniform"], \
        f"CS should prefer incoherent sampling: {gains}"

    # --- A single coil handed over with an explicit axis is the same thing ---
    mask = make_mask("random",shape,R=4,seed=7).astype(bool)
    flat = cs_reconstruct(k*mask,mask,n_iter=20)
    boxed = cs_reconstruct((k*mask)[None],mask,n_iter=20)
    assert np.allclose(flat,boxed), "(ky,kx) and (1,ky,kx) must agree"
    print("[coil axis]  (ky,kx) and (1,ky,kx) give identical results")

    # --- Multi-coil: four channels, one patient, one prior ---
    # Stand-in sensitivity maps: each channel is brightest near its own corner,
    # which is roughly what a real receiver array does.
    yy,xx = np.mgrid[0:shape[0],0:shape[1]]
    maps = [np.exp(-((np.sqrt((yy-cy)**2+(xx-cx)**2))/(1.1*shape[0]))**2)
            for cy,cx in ((0,0),(0,shape[1]),(shape[0],0),(shape[0],shape[1]))]
    coils = np.stack([to_kspace(img*m) for m in maps])
    truth = nrm(np.sqrt(sum((img*m)**2 for m in maps)))
    assert coils.shape == (4,)+shape, coils.shape

    zf4 = nrm(_rss(_ifft(coils*mask)))
    cs4 = nrm(cs_reconstruct(coils*mask,mask))
    m_zf = compute_metrics(truth,zf4)
    m_cs = compute_metrics(truth,cs4)
    assert m_cs["psnr"] > m_zf["psnr"], f"multi-coil CS lost to zero-filling: {m_cs} vs {m_zf}"
    assert m_cs["ssim"] > m_zf["ssim"], f"multi-coil CS lost on SSIM: {m_cs} vs {m_zf}"
    print(f"[4 coils R=4] zero-filled psnr={m_zf['psnr']:5.2f} ssim={m_zf['ssim']:.3f}"
          f"   ->  CS psnr={m_cs['psnr']:5.2f} ssim={m_cs['ssim']:.3f}"
          f"   (+{m_cs['psnr']-m_zf['psnr']:.2f} dB)")

    # --- Phase must survive: a complex image is not a damaged real one ---
    # Give the phantom a smooth phase roll, the way real MRI has one, and check
    # the magnitude still comes back. Taking the real part here would gut it.
    phase = np.exp(1j*2*np.pi*(yy/shape[0]*0.7 + xx/shape[1]*0.4))
    kc = _fft(img*phase)
    zf_c = compute_metrics(nrm(img),nrm(_rss(_ifft((kc*mask)[None]))))
    cs_c = compute_metrics(nrm(img),nrm(cs_reconstruct(kc*mask,mask)))
    assert cs_c["psnr"] > zf_c["psnr"] and cs_c["ssim"] > zf_c["ssim"], \
        f"CS must still beat zero-filling once the image has phase: {cs_c} vs {zf_c}"
    print(f"[complex]    phase-rolled phantom: zero-filled psnr={zf_c['psnr']:5.2f} "
          f"->  CS psnr={cs_c['psnr']:5.2f}  (+{cs_c['psnr']-zf_c['psnr']:.2f} dB)")

    # --- Non-Cartesian masks can be derived from the trajectory ---
    coords,_ = radial(shape,n_spokes=64)
    rmask = sampling_mask_from_coords(coords,shape)
    frac = rmask.mean()
    assert 0.05 < frac < 1.0, f"radial mask covers {frac:.2%}, which looks wrong"
    assert rmask[shape[0]//2,shape[1]//2], "the centre of k-space must be marked as sampled"
    print(f"[traj mask]  64 spokes touch {frac:.1%} of the grid, centre included")

    # --- Guards ---
    for bad in (lambda: cs_reconstruct(k,np.ones((8,8),dtype=bool)),
                lambda: cs_reconstruct(k[None,None],np.ones(shape,dtype=bool))):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("bad input should have raised")
    print("[guard]      shape mismatch and a 4-D input both rejected")

    print("\nAll self-tests passed")
