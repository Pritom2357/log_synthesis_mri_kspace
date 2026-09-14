"""
Compressed sensing reconstruction.

Zero-filling assumes every sample you skipped was zero, which is a lie, and the
artifact you see is that lie. Compressed sensing replaces it with a better
assumption: MRI images are mostly smooth with a few sharp edges, so their total
variation is small. It then looks for the image that has small total variation
AND still agrees with every sample you actually measured.

The loop alternates between those two demands until they settle.
"""
from __future__ import annotations
import numpy as np

__all__ = ["cs_reconstruct","sampling_mask_from_coords"] # Only these will be exported

# -----------------------------------------------------------------
# Total variation prior (private helpers that will NOT be exported)
# -----------------------------------------------------------------

def _fft(x:np.ndarray)->np.ndarray:
    """Same centring convention as kspace_core.to_kspace, but keeps the phase."""
    return np.fft.fftshift(np.fft.fft2(x))

def _ifft(k:np.ndarray)->np.ndarray:
    """Same convention as kspace_core.from_kspace, but keeps the phase."""
    return np.fft.ifft2(np.fft.ifftshift(k))

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
    """
    for _ in range(n_steps):
        gy,gx = _grad(x)
        mag = np.sqrt(gy**2 + gx**2) + 1e-8
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

    mask: True where a sample was actually acquired.
    weight: how hard to push on smoothness. 0.05 was tuned on the phantom;
        raising it to 0.2 already costs more real detail than it removes artifact.

    Return: float64 image, same shape as k_measured.
    """
    if k_measured.shape != mask.shape:
        raise ValueError(f"shape mismatch: {k_measured.shape} vs {mask.shape}")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")

    mask = np.asarray(mask).astype(bool)
    x = np.clip(np.real(_ifft(k_measured)),0.0,None) # zero-filled start

    for _ in range(n_iter):
        x = _tv_denoise(x,weight,tv_steps,tv_step)
        x = np.clip(x,0.0,None)                      # our images are real and non-negative
        k = np.where(mask,k_measured,_fft(x))        # data consistency
        x = np.clip(np.real(_ifft(k)),0.0,None)

    return x

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

    # --- Non-Cartesian masks can be derived from the trajectory ---
    coords,_ = radial(shape,n_spokes=64)
    rmask = sampling_mask_from_coords(coords,shape)
    frac = rmask.mean()
    assert 0.05 < frac < 1.0, f"radial mask covers {frac:.2%}, which looks wrong"
    assert rmask[shape[0]//2,shape[1]//2], "the centre of k-space must be marked as sampled"
    print(f"[traj mask]  64 spokes touch {frac:.1%} of the grid, centre included")

    # --- Guards ---
    try:
        cs_reconstruct(k,np.ones((8,8),dtype=bool))
    except ValueError:
        print("[guard]      shape mismatch rejected")
    else:
        raise AssertionError("shape mismatch should have raised")

    print("\nAll self-tests passed")
