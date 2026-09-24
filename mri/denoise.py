"""
Removing noise, using only tools from the signals course.

The scanner files in this project come from a 0.3 T magnet, which is weak, so
every single scan is grainy. Each subject was scanned three times, and the
average of those three is a cleaner picture than any one of them. That average
is the reference; the job here is to get a single scan as close to it as the
signal processing allows. Four tools, each one a course topic:

    average_repetitions  linearity: the transform of a mean is the mean of the
                         transforms, so averaging k-space averages the images.
                         The noise is different every time and partly cancels;
                         the anatomy is the same every time and does not.
    align_phase          cross-correlation at lag 0. The scanner's phase
                         drifts between repetitions, and averaging two copies
                         half a turn apart cancels the ANATOMY instead of the
                         noise. Measured here, naive averaging of subject
                         2022062501 drops SSIM from 0.80 to 0.56; aligned, it
                         rises to 0.90.
    find_shift           phase correlation (shift theorem): finds a rigid
    shift_kspace         shift between repetitions and undoes it with a phase
                         ramp, in case the patient moved between scans.
    noise_floor_filter   Parseval: the energy in each ring of k-space is the
                         signal's energy plus a flat noise floor. The floor is
                         measured in the corners, where there is no anatomy,
                         and each ring keeps only the fraction of its energy
                         that rises above it. It is a low-pass filter whose
                         cutoff is chosen by the data instead of by hand.
    sharpen              unsharp masking, y = x + a(x - x*g), done as one
                         multiplication in the frequency domain. It gives back
                         some edge that the smoothing filters took away.
"""
from __future__ import annotations
import numpy as np

from mri.kspace_core import radius_grid
from mri.kspace_edit import dc_index

__all__ = ["align_phase","find_shift","shift_kspace","average_repetitions",
           "noise_power","noise_floor_filter","sharpen"] # Only these will be exported

# ------
# Helper
# ------

def _coils(k:np.ndarray)->np.ndarray:
    """Always (coils, ky, kx), so single-coil and multi-coil share one code path."""
    return k if k.ndim == 3 else k[None]

def _image(k:np.ndarray)->np.ndarray:
    """Root-sum-of-squares magnitude image, used only for measuring shifts."""
    per_coil = np.fft.ifft2(np.fft.ifftshift(_coils(k),axes=(-2,-1)),axes=(-2,-1))
    return np.sqrt((np.abs(per_coil)**2).sum(axis=0))

def _centred_freqs(n:int)->np.ndarray:
    """Frequency of each row/column of centred k-space, in cycles per pixel."""
    return np.fft.fftshift(np.fft.fftfreq(n))

# ---------------------
# Methods to call from
# ---------------------

def align_phase(ref:np.ndarray,k:np.ndarray)->np.ndarray:
    """
    Rotates k so its phase agrees with ref, one coil at a time.

    The lag-0 cross-correlation sum(conj(k) * ref) is a single complex number
    whose angle is exactly the phase k is missing. Multiplying k by that angle
    changes no magnitude anywhere, so the image of k alone is untouched; only
    how it adds to ref changes.
    """
    kc,rc = _coils(k),_coils(ref)
    out = np.empty_like(kc)
    for c in range(kc.shape[0]):
        out[c] = kc[c]*np.exp(1j*np.angle(np.vdot(kc[c],rc[c])))
    return out if k.ndim == 3 else out[0]

def find_shift(ref:np.ndarray,k:np.ndarray)->tuple[int,int]:
    """
    Whole-pixel shift (rows, cols) that moves ref's image onto k's.

    Phase correlation: divide the cross-power spectrum by its magnitude, and
    what is left is a pure phase ramp, whose inverse transform is one impulse
    sitting at the shift. An index past the half-way point is a negative shift.
    """
    A,B = np.fft.fft2(_image(ref)),np.fft.fft2(_image(k))
    cross = np.conj(A)*B
    mag = np.abs(cross)
    unit = np.where(mag > 1e-12*mag.max(),cross/np.maximum(mag,1e-300),0.0)
    surface = np.real(np.fft.ifft2(unit))
    i,j = np.unravel_index(int(surface.argmax()),surface.shape)
    H,W = surface.shape
    return (int(i-H) if i > H//2 else int(i), int(j-W) if j > W//2 else int(j))

def shift_kspace(k:np.ndarray,dy:float,dx:float)->np.ndarray:
    """
    Moves the image (dy, dx) pixels by multiplying k-space by a phase ramp.

    The shift theorem: x[n - d] <-> X[u] exp(-2 pi i u d). Magnitudes are left
    alone; only where each wave sits changes. Coil-safe.
    """
    rows,cols = k.shape[-2:]
    u = _centred_freqs(rows)[:,None]
    v = _centred_freqs(cols)[None,:]
    return k*np.exp(-2j*np.pi*(u*dy + v*dx))

def average_repetitions(ks:list,align:bool=True)->tuple[np.ndarray,dict]:
    """
    Averages several acquisitions of the same slice.

    ks: k-space arrays, first one is the scan being reconstructed.
    align: undo each repetition's shift and phase against the first before
           averaging. Off is the naive average, kept so the failure it
           prevents can be seen.

    Return: (averaged k-space, {"shifts": [...], "phases": [...]}) with one
    entry per extra repetition; phases in radians, coil 0.
    """
    if not ks:
        raise ValueError("need at least one acquisition")
    ref = ks[0]
    out,shifts,phases = [ref],[],[]
    for k in ks[1:]:
        if k.shape != ref.shape:
            raise ValueError(f"repetition shape {k.shape} differs from {ref.shape}")
        if align:
            dy,dx = find_shift(ref,k)
            if dy or dx:
                k = shift_kspace(k,-dy,-dx)
            shifts.append((dy,dx))
            phases.append(float(np.angle(np.vdot(_coils(k)[0],_coils(ref)[0]))))
            k = align_phase(ref,k)
        out.append(k)
    return np.mean(out,axis=0), {"shifts": shifts, "phases": phases}

def noise_power(k:np.ndarray)->float:
    """
    Noise power per k-space sample, coils summed.

    Measured in the four corners beyond the inscribed circle: on a real scan
    the anatomy's energy has fallen far below the noise out there, so what is
    left is noise. (Not on the simulated phantom: its perfectly hard edges keep
    real signal in the corners, so there this reads a little high.)
    Exact zeros are skipped, because some files zero-pad their edges and a
    padded zero is not a measurement of anything.
    """
    p = (np.abs(_coils(k))**2).sum(axis=0)
    r = radius_grid(p.shape,dc_index(k))
    corner = p[(r > min(p.shape)/2) & (p > 0)]
    return float(corner.mean()) if corner.size else 0.0

def noise_floor_filter(k:np.ndarray,strength:float=1.0)->np.ndarray:
    """
    Keeps, in every ring of k-space, only the energy above the noise floor.

    For the ring at radius r with mean power P(r) and a noise floor N, the gain
    is (P - strength*N) / P, clipped to [0, 1]. Near the centre P is huge and
    the gain is 1; far out P is all noise and the gain falls to 0. strength
    scales the floor: above 1 removes more noise and more detail.
    """
    if strength < 0:
        raise ValueError("strength must be non-negative")
    if strength == 0:
        return k
    N = noise_power(k)
    if N == 0.0: # nothing measured at all, e.g. an empty array
        return k
    p = (np.abs(_coils(k))**2).sum(axis=0)
    ring = radius_grid(p.shape,dc_index(k)).astype(int)
    # Ring power is averaged over measured samples only. Counting the zeros
    # that undersampling or partial Fourier left behind would halve a ring's
    # power at half sampling and silently double the filter's strength.
    measured = (p > 0).ravel()
    mean_p = (np.bincount(ring.ravel(),p.ravel())
              /np.maximum(np.bincount(ring.ravel(),measured.astype(float)),1))
    gain = np.clip((mean_p-strength*N)/np.maximum(mean_p,1e-300),0.0,1.0)
    return k*gain[ring]

def sharpen(img:np.ndarray,amount:float,sigma:float=1.0)->np.ndarray:
    """
    Unsharp masking: y = x + amount*(x - x*g), with g a Gaussian of `sigma` px.

    In the frequency domain the whole thing is one product,
    Y = X((1 + amount) - amount*G), because convolution becomes multiplication
    and the transform is linear. amount=0 returns img unchanged.
    """
    if amount < 0:
        raise ValueError("amount must be non-negative")
    if amount == 0:
        return img
    rows,cols = img.shape
    fy = np.fft.fftfreq(rows)[:,None]
    fx = np.fft.fftfreq(cols)[None,:]
    G = np.exp(-2*(np.pi*sigma)**2*(fy**2 + fx**2)) # transform of the Gaussian
    return np.real(np.fft.ifft2(np.fft.fft2(img)*((1+amount) - amount*G)))

# ---------------------------
# Tests for denoise.py only
# ---------------------------

if __name__ == "__main__":
    from mri.kspace_core import load_phantom, to_kspace, from_kspace
    from mri.metrics import compute_metrics

    img = load_phantom()
    k = to_kspace(img)
    nrm = lambda a: (a-a.min())/(a.max()-a.min()+1e-12)
    psnr = lambda kk: compute_metrics(img,nrm(from_kspace(kk)))["psnr"]
    rng = np.random.default_rng(0)
    noisy = lambda: k + 0.5*np.abs(k).mean()*(rng.normal(size=k.shape)+1j*rng.normal(size=k.shape))

    # --- Shift theorem: the ramp is a real shift, and phase correlation finds it ---
    moved = shift_kspace(k,7,-12)
    assert np.abs(from_kspace(moved)-np.roll(img,(7,-12),axis=(0,1))).max() < 1e-9
    assert find_shift(k,moved) == (7,-12), find_shift(k,moved)
    assert np.abs(from_kspace(shift_kspace(moved,-7,12))-img).max() < 1e-9
    print("[shift]       phase ramp equals np.roll, phase correlation finds (7,-12), undo is exact")

    # --- Phase alignment: a rotated copy comes back exactly ---
    turned = np.stack([k,k])*np.array([np.exp(2.5j),np.exp(-1.1j)])[:,None,None]
    back = align_phase(np.stack([k,k]),turned)
    assert np.abs(back-np.stack([k,k])).max() < 1e-6*np.abs(k).max()
    print("[phase]       per-coil phase offsets of +2.5 and -1.1 rad removed exactly")

    # --- Averaging: naive fails on drifted phase, aligned wins ---
    reps = [noisy(), noisy()*np.exp(3.0j), shift_kspace(noisy(),0,4)*np.exp(-2.0j)]
    single = psnr(reps[0])
    naive,_ = average_repetitions(reps,align=False)
    aligned,info = average_repetitions(reps,align=True)
    print(f"[average]     single {single:.2f} dB, naive average {psnr(naive):.2f} dB,"
          f" aligned average {psnr(aligned):.2f} dB, shifts found {info['shifts']}")
    assert info["shifts"] == [(0,0),(0,4)], info["shifts"]
    assert psnr(aligned) > single+3.0, "three aligned copies should gain several dB"
    assert psnr(naive) < psnr(aligned)-3.0, "naive averaging should visibly lose"
    assert average_repetitions([k])[0] is not None

    # --- Noise floor: signal and noise energies add, because they are unrelated ---
    # The phantom's hard edges leave some real signal in the corners, so the
    # corner power of the clean phantom is not zero. What must hold is that
    # adding noise raises it by exactly the noise power.
    n = noisy()
    expected = 2*(0.5*np.abs(k).mean())**2 # complex noise: variance on both parts
    added = noise_power(n)-noise_power(k)
    assert abs(added/expected - 1) < 0.1, (added,expected)
    print(f"[floor]       corner power rose by the noise power to within {100*abs(added/expected-1):.1f}%")

    # --- The noise-floor filter removes noise; nothing to measure, nothing done ---
    before,after = psnr(n),psnr(noise_floor_filter(n,1.0))
    assert after > before+1.0, (before,after) # measured +1.36 dB
    # Zeroed rows must not count as "no signal": at half sampling the filter
    # has to keep the same share of what WAS measured.
    half_rows = np.zeros(k.shape); half_rows[::2] = 1
    keep = lambda kk: np.abs(noise_floor_filter(kk,1.0)).sum()/np.abs(kk).sum()
    assert abs(keep(n*half_rows)-keep(n)) < 0.05, (keep(n),keep(n*half_rows))
    print(f"[floor]       same strength at half sampling: keeps {keep(n):.3f} vs {keep(n*half_rows):.3f}")
    silent = np.zeros_like(k)
    assert noise_floor_filter(silent,1.0) is silent, "no floor, nothing to remove"
    assert noise_floor_filter(n,0.0) is n
    print(f"[filter]      noisy {before:.2f} dB -> filtered {after:.2f} dB")

    # --- Sharpening steepens edges; zero is a no-op ---
    blurred = nrm(from_kspace(k*np.exp(-(radius_grid(k.shape)/60.0)**2)))
    steep = lambda a: float(np.percentile(np.hypot(*np.gradient(a)),99))
    assert sharpen(blurred,0.0) is blurred
    assert steep(nrm(sharpen(blurred,1.0))) > steep(blurred)
    print(f"[sharpen]     edge steepness x{steep(nrm(sharpen(blurred,1.0)))/steep(blurred):.2f}")

    # --- Guards ---
    for bad in (lambda: noise_floor_filter(k,-1.0),
                lambda: sharpen(img,-0.5),
                lambda: average_repetitions([]),
                lambda: average_repetitions([k,k[:10]])):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("bad input should have raised")
    print("[guard]       negative strength/amount, empty list, mismatched shapes rejected")

    print("\nAll self-tests passed")
