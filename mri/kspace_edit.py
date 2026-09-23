"""
Direct edits to k-space, and what each one does to the picture.

Every other module changes k-space as a side effect of simulating something
physical -- a sampling pattern, a filter, a moving patient. This one lets you
reach in and change it on purpose, which is the fastest way to build intuition
for what each part of k-space is actually holding.

Each function here is a one-line demonstration of a property from the course:

    scale_dc        the DFT at u=v=0 is the sum of all samples, so it is the
                    mean brightness and nothing else
    add_spike       one k-space entry is one basis function, so a single point
                    becomes a single grating across the whole image
    erase_patch     the centre carries shape, the periphery carries edges
    partial_fourier a real-valued image has Hermitian-symmetric k-space, so
                    half of it is redundant and can be reconstructed
"""
from __future__ import annotations
import numpy as np

__all__ = ["dc_index","phase_axis","measured_lines",
           "scale_dc","add_spike","erase_patch","partial_fourier",
           "hermitian_fill","keep_part"] # Only these will be exported

# ------
# Helper
# ------

def _centre(shape:tuple[int,int])->tuple[int,int]:
    rows,cols = shape[-2:]
    return rows//2, cols//2

# ---------------------
# Methods to call from
# ---------------------

def dc_index(k:np.ndarray)->tuple[int,int]:
    """
    Where the k-space origin actually sits, which is not always the middle.

    For simulated data the origin is the exact centre of the array, because we
    put it there. Real scanner data is often offset by a few samples: the
    readout and phase-encode centres are set by hardware timing, not by array
    indexing. Measured on the files in this project the offset is 3 rows for
    three of them and 8 for the fourth, and it is identical for every coil and
    every slice -- so it is a property of the acquisition, not noise.

    The origin is found as the brightest cell, because it is the sum of every
    spin in the slice and nothing else can exceed it. Assuming the array centre
    instead means edits like scale_dc quietly act on a near-empty sample.
    """
    mag = np.abs(k) if k.ndim == 2 else np.abs(k).sum(axis=0)
    return tuple(int(v) for v in np.unravel_index(mag.argmax(),mag.shape))

def phase_axis(k:np.ndarray)->int:
    """
    Which array axis holds the phase-encode lines: -2 (rows) or -1 (columns).

    Only phase-encode lines cost scan time, one echo each; a readout line is
    recorded in one go, so its samples are free. Scanners that measure fewer
    phase-encode lines than the matrix has leave the rest as whole zero lines,
    which gives the axis away: on the M4Raw files 60 of 256 COLUMNS are empty,
    so there the columns are the phase-encode lines. With no empty lines at all
    (a simulated phantom) rows are assumed.
    """
    mag = np.abs(k) if k.ndim == 2 else np.abs(k).sum(axis=0)
    empty_rows = int((mag.sum(axis=1) == 0).sum())
    empty_cols = int((mag.sum(axis=0) == 0).sum())
    return -1 if empty_cols > empty_rows else -2

def measured_lines(k:np.ndarray,axis:int)->np.ndarray:
    """Boolean per phase-encode line: True where anything was recorded."""
    mag = np.abs(k) if k.ndim == 2 else np.abs(k).sum(axis=0)
    return mag.sum(axis=-1 if axis == -2 else 0) > 0

def scale_dc(k:np.ndarray,scale:float)->np.ndarray:
    """
    Multiplies the single DC entry, leaving every other frequency alone.

    K[0,0] is the sum of every pixel, so it sets the mean of the image and
    contributes nothing else. Scaling it scales that mean exactly.

    There is a catch worth knowing, because it surprises people. MRI images are
    displayed as MAGNITUDES. Subtracting the mean pushes more than half the
    pixels below zero, and the magnitude flips them back up, so the displayed
    picture does not simply get darker -- measured on the phantom its mean
    actually RISES, from 0.123 to 0.142, while the signed mean falls to zero.
    What you see instead is the contrast inverting around the old average.
    """
    if scale < 0:
        raise ValueError("scale must be non-negative")

    out = np.array(k,copy=True)
    cy,cx = dc_index(k) # the real origin, which may not be the array centre
    out[...,cy,cx] *= scale
    return out

def add_spike(k:np.ndarray,dy:int,dx:int,strength:float=1.0)->np.ndarray:
    """
    Plants one bright point in k-space, offset (dy,dx) from the centre.

    Because the inverse transform sums basis functions, a lone k-space entry
    becomes a lone sinusoid: the image gains a stripe pattern running across
    all of it. How fine the stripes are is set by how far the point sits from
    the centre, and their direction by which way it is offset. This is what a
    hardware spike artifact looks like on a real scanner.

    strength is relative to the current maximum of |k|, so 1.0 is a spike as
    bright as the DC term.
    """
    out = np.array(k,copy=True)
    rows,cols = k.shape[-2:]
    cy,cx = dc_index(k) # offsets are measured from the true origin
    y,x = cy+int(dy), cx+int(dx)

    if not (0 <= y < rows and 0 <= x < cols):
        raise ValueError(f"spike at ({dy},{dx}) falls outside k-space")

    out[...,y,x] = strength*np.abs(out).max()
    return out

def erase_patch(k:np.ndarray,dy:int,dx:int,size:int)->np.ndarray:
    """
    Zeroes a square block of k-space, centred (dy,dx) from the middle.

    Erase near the centre and the image loses its broad shapes and contrast
    while the edges survive. Erase at the periphery and the shapes stay but the
    detail softens. It is the same lesson as the low and high pass filters,
    except you choose exactly which frequencies to throw away.
    """
    if size < 1:
        raise ValueError("size must be at least 1")

    out = np.array(k,copy=True)
    rows,cols = k.shape[-2:]
    cy,cx = dc_index(k) # erase around the true origin, not the array centre
    half = size//2
    y0,y1 = np.clip([cy+dy-half, cy+dy+half+1],0,rows)
    x0,x1 = np.clip([cx+dx-half, cx+dx+half+1],0,cols)

    out[...,y0:y1,x0:x1] = 0
    return out

def keep_part(k:np.ndarray,part:str)->np.ndarray:
    """
    Throws away either the magnitude or the phase of every k-space sample.

    part: 'both' | 'magnitude' | 'phase'
        magnitude - every sample keeps its size, loses its angle. The image
                    loses its structure: where each wave sits is gone.
        phase     - every sample keeps its angle, its size set to the mean
                    size. The outlines survive, because the phase is what says
                    where each edge is. This is the magnitude/phase swap from
                    the course, done inside one scan.
    """
    kind = part.strip().lower()
    if kind == "both":
        return k
    if kind == "magnitude":
        return np.abs(k).astype(np.complex128)
    if kind == "phase":
        return np.exp(1j*np.angle(k))*float(np.abs(k).mean())
    raise ValueError(f"Unknown part: {part!r}. Expected: both, magnitude, phase.")

def hermitian_fill(k:np.ndarray,measured:np.ndarray)->np.ndarray:
    """
    Fills unmeasured k-space using the symmetry a real-valued image guarantees.

    If the picture is real then K[-u,-v] = conj(K[u,v]). So a sample that was
    never acquired can be recovered from its opposite number, provided that one
    WAS acquired. This is the DFT symmetry property from the course being used
    to buy back half a scan.

    measured: True where a sample genuinely exists.

    IMPORTANT: this only works if the image really is real-valued. A synthetic
    phantom is, and the fill is then exact. A real MRI image is NOT: coil phase
    and field inhomogeneity give it an imaginary part about as large as its
    real part, so the symmetry is badly violated and filling from it makes the
    reconstruction worse rather than better. Measured on the data in this
    project, the symmetry violation is 0.00 for the phantom and 1.43 for a real
    scan. Real scanners handle this by estimating the phase from the
    fully-sampled centre and correcting for it first, which this does not do.

    Return: k-space with the reflectable gaps filled in.
    """
    out = np.array(k,copy=True)
    meas = np.asarray(measured,dtype=bool)

    # The conjugate partner of index i is its reflection about the centre. For
    # centred k-space of even size that is a flip of both axes, off by one.
    flipped = np.conj(np.flip(np.flip(out,axis=-1),axis=-2))
    partner = np.flip(np.flip(meas,axis=-1),axis=-2)
    flipped = np.roll(flipped,shift=(1,1),axis=(-2,-1))
    partner = np.roll(partner,shift=(1,1),axis=(-2,-1))

    fillable = (~meas) & partner
    out[...,fillable] = flipped[...,fillable]
    return out

def partial_fourier(k:np.ndarray,fraction:float=0.6,fill:bool=True)->tuple:
    """
    Acquires only the first `fraction` of the phase-encode lines.

    Real scanners do this to halve the scan time: because a real image has
    Hermitian-symmetric k-space, slightly more than half the lines already
    contain all the information. A little over half is taken rather than
    exactly half so that the centre, where the phase behaves worst, is measured
    on both sides.

    fill: reconstruct the missing half from the symmetry. With fill off you see
    what zero-filling those lines costs instead. See hermitian_fill for why the
    fill helps on a phantom and hurts on a real scan.

    Return: (k-space, measured-mask).
    """
    if not 0.5 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0.5 and 1.0")

    rows = k.shape[-2]
    keep = int(round(fraction*rows))
    meas = np.zeros(k.shape[-2:],dtype=bool)
    meas[:keep,:] = True

    out = np.array(k,copy=True)
    out[...,keep:,:] = 0
    return (hermitian_fill(out,meas) if fill else out), meas

# ------------------------------
# Tests for kspace_edit.py only
# ------------------------------

if __name__ == "__main__":
    from mri.kspace_core import load_phantom, to_kspace, from_kspace
    from mri.metrics import compute_metrics

    img = load_phantom()
    k = to_kspace(img)
    shape = img.shape
    nrm = lambda a: (a-a.min())/(a.max()-a.min()+1e-12)

    # --- The origin is found, not assumed ---
    assert dc_index(k) == (shape[0]//2,shape[1]//2), "a synthetic phantom is centred"
    shifted = np.roll(k,(5,-3),axis=(0,1))
    assert dc_index(shifted) == (shape[0]//2+5,shape[1]//2-3), "must track a moved origin"
    print(f"[origin]      found at {dc_index(k)} for the phantom, and tracks a deliberate shift")

    # --- The DC term is the mean of the image, and only that ---
    assert np.array_equal(scale_dc(k,1.0),k), "scale 1.0 must change nothing"
    signed = lambda kk: np.real(np.fft.ifft2(np.fft.ifftshift(kk)))
    base = signed(k).mean()
    for scale in (1.0,0.5,0.0):
        ks = scale_dc(k,scale)
        assert abs(signed(ks).mean() - scale*base) < 1e-9, "signed mean must scale exactly"
        print(f"[dc x{scale:.1f}]     signed mean {signed(ks).mean():.5f}"
              f"   magnitude mean {from_kspace(ks).mean():.5f}")

    # The magnitude display does NOT get darker: removing the mean sends most
    # pixels negative and abs() folds them back up.
    flipped = int((signed(scale_dc(k,0.0)) < 0).sum())
    assert from_kspace(scale_dc(k,0.0)).mean() > from_kspace(k).mean(), \
        "on a magnitude display, removing DC raises the mean rather than lowering it"
    print(f"[dc x0]       {flipped} of {img.size} pixels went negative and were folded "
          "back by the magnitude")

    # Every other frequency survived untouched. On the signed image, removing
    # the DC term is a pure constant offset, so every gradient is unchanged --
    # exactly, not approximately.
    g_before = np.gradient(signed(k))[0]
    g_after = np.gradient(signed(scale_dc(k,0.0)))[0]
    assert np.allclose(g_before,g_after,atol=1e-12), \
        "removing a constant cannot change any gradient"
    print(f"[dc x0]       every gradient identical to "
          f"{np.abs(g_before-g_after).max():.1e}: only the constant went")

    # --- One k-space point is one grating ---
    flat = np.zeros_like(k)
    flat[shape[0]//2,shape[1]//2] = np.abs(k).max() # DC only: a flat grey field
    one = add_spike(flat,0,12,strength=1.0)
    g = from_kspace(one)
    # 12 cycles across the image, running horizontally: check the column profile
    row = g[shape[0]//2,:] - g[shape[0]//2,:].mean()
    cycles = int(np.argmax(np.abs(np.fft.rfft(row))))
    print(f"[spike]       point 12 columns from centre -> {cycles} cycles across the image")
    assert cycles == 12, f"a spike at offset 12 should give 12 cycles, got {cycles}"
    vert = add_spike(flat,7,0,strength=1.0)
    col = from_kspace(vert)[:,shape[1]//2]
    assert int(np.argmax(np.abs(np.fft.rfft(col-col.mean())))) == 7, "vertical spike"
    print("[spike]       offsetting downward gives vertical stripes, as it must")

    # --- Erasing the centre and erasing the edge do opposite damage ---
    centre_gone = nrm(from_kspace(erase_patch(k,0,0,40)))
    edge_gone = nrm(from_kspace(erase_patch(k,90,90,40)))
    sharp = lambda a: float(np.percentile(np.hypot(*np.gradient(a)),99))
    print(f"[erase]       centre removed: edge sharpness {sharp(centre_gone)/sharp(nrm(img)):.2f}x")
    print(f"[erase]       corner removed: edge sharpness {sharp(edge_gone)/sharp(nrm(img)):.2f}x")
    assert compute_metrics(nrm(img),edge_gone)["ssim"] > compute_metrics(nrm(img),centre_gone)["ssim"], \
        "losing the centre must hurt more than losing a far corner"

    # --- Hermitian symmetry: half a scan is enough for a real image ---
    for frac in (0.6,0.75):
        filled,meas = partial_fourier(k,frac,fill=True)
        zero,_ = partial_fourier(k,frac,fill=False)
        m_fill = compute_metrics(img,nrm(from_kspace(filled)))
        m_zero = compute_metrics(img,nrm(from_kspace(zero)))
        assert m_fill["ssim"] > m_zero["ssim"], f"symmetry fill should beat zero-filling at {frac}"
        print(f"[partial {frac:.2f}] zero-filled ssim={m_zero['ssim']:.3f}"
              f"  -> symmetry-filled ssim={m_fill['ssim']:.3f}")

    # The symmetry only exists for a real-valued image. Prove that, because it
    # is exactly why the fill helps here and would hurt on real scanner data.
    def _herm_violation(kk):
        flip = np.conj(np.roll(np.flip(np.flip(kk,-1),-2),(1,1),axis=(-2,-1)))
        return float(np.abs(kk-flip).mean()/max(np.abs(kk).mean(),1e-12))
    v = _herm_violation(k)
    assert v < 1e-6, f"a real-valued phantom must have Hermitian k-space, got {v}"
    print(f"[symmetry]    phantom k-space is Hermitian to {v:.1e}, which is why the fill works")

    full,_ = partial_fourier(k,1.0,fill=True)
    assert compute_metrics(img,nrm(from_kspace(full)))["ssim"] > 0.999, \
        "fraction 1.0 keeps everything and must be lossless"
    print("[partial 1.00] keeping every line is lossless, as expected")

    # --- Line layout: which axis costs time, which lines were measured ---
    assert phase_axis(k) == -2, "no empty lines: rows are assumed"
    padded = np.array(k,copy=True); padded[:,:30] = 0; padded[:,-30:] = 0
    assert phase_axis(padded) == -1, "empty columns mark the columns as phase-encode"
    assert phase_axis(np.stack([padded,padded])) == -1, "coil axis must not matter"
    assert measured_lines(padded,-1).sum() == shape[1]-60
    assert measured_lines(padded,-2).all(), "every row still has data"
    print("[layout]      phase-encode axis found from empty lines; measured lines counted")

    # --- Phase carries the structure, magnitude does not ---
    assert keep_part(k,"both") is k
    phase_only = nrm(from_kspace(keep_part(k,"phase")))
    mag_only = nrm(from_kspace(keep_part(k,"magnitude")))
    edges = lambda a: np.corrcoef(np.hypot(*np.gradient(a)).ravel(),
                                  np.hypot(*np.gradient(img)).ravel())[0,1]
    assert edges(phase_only) > edges(mag_only)+0.2, (edges(phase_only),edges(mag_only))
    print(f"[mag/phase]   edge match with the original: phase only {edges(phase_only):.2f}"
          f", magnitude only {edges(mag_only):.2f}")

    # --- Guards ---
    for bad in (lambda: keep_part(k,"neither"),
                lambda: scale_dc(k,-1.0),
                lambda: add_spike(k,9999,0),
                lambda: erase_patch(k,0,0,0),
                lambda: partial_fourier(k,0.2)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("bad input should have raised")
    print("[guard]       negative scale, off-grid spike, empty patch, tiny fraction rejected")

    print("\nAll self-tests passed")
