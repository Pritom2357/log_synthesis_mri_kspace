"""
This file generates 2D bin k-space sampling masks for Cartesian undersampling.
A helper returns the perceived *acceleration* for any mask
"""
from __future__ import annotations
import numpy as np

__all__ = ["make_mask","acquisition_mask","achieved_acceleration"] # Only these will be exported from this file

# -----------------------------------------------------------------
# Masks (private helper builder methods that will NOT be exported)
# -----------------------------------------------------------------

def _uniform_mask(shape:tuple[int,int],R:int)->np.ndarray:
    """
    Keeps every R-th row; zero out the remaining.

    The pattern is anchored on the DC row rather than row 0. Counting from row 0
    drops the centre line whenever rows//2 is not divisible by R, which throws
    away the brightest sample in k-space and darkens the whole reconstruction.
    """
    rows,_ = shape
    mask = np.zeros(shape,dtype=np.uint8)
    mask[(rows//2)%R::R,:] = 1
    return mask

def _nyquist_mask(shape:tuple[int,int],R:int,rate:float=1.0,centre:int|None=None)->np.ndarray:
    """
    Keeps a given FRACTION of the phase-encode lines, spaced as evenly as possible.

    This is the sampling theorem stated the way the lecture states it: rate is
    the sampling rate as a fraction of the Nyquist rate. At rate=1.0 every line
    is measured and the reconstruction is exact. Below 1.0 the spacing dk grows,
    the field of view 1/dk shrinks, and whatever no longer fits folds back in.

    Unlike keeping every R-th line, the fraction is continuous, so the aliasing
    builds up gradually instead of jumping straight to full-strength copies. The
    DC line is always kept, because it carries the mean brightness.
    """
    rows,_ = shape
    if not 0 < rate <= 1.0:
        raise ValueError("rate must be in (0, 1]")

    n_keep = max(1,int(round(rate*rows)))
    # The DC row of scanner data is not always the array middle (3 rows off on
    # the M4Raw files). Anchoring on the middle dropped the real DC row at some
    # rates -- 41% of the energy gone at rate 0.8 -- so the caller passes it.
    centre = rows//2 if centre is None else int(centre)
    offsets = np.round(np.arange(n_keep)*rows/n_keep).astype(int)
    keep = (centre + offsets) % rows # anchored on DC, then spread evenly

    mask = np.zeros(shape,dtype=np.uint8)
    mask[np.unique(keep),:] = 1
    return mask

# ------
# Helper
# ------
def _validate_shape_R(shape, R):
    if len(shape) != 2:
        raise ValueError("shape must be (rows, cols)")

    rows, cols = shape

    if rows <= 0 or cols <= 0:
        raise ValueError("shape dimensions must be positive")

    if not isinstance(R, (int, np.integer)) or R < 1:
        raise ValueError("R must be a positive integer")
# ---------------------
# Methods to call from
# ---------------------

def make_mask(kind:str,shape:tuple[int,int],R:int,**kw)->np.ndarray:
    """
    Generates binary k-space sampling mask

    kind: 'nyquist'|'uniform'. Only regular, evenly spaced sampling is kept:
          it is the sampling the course teaches, and the one real Cartesian
          scanners use. Random and variable-density patterns were removed.
    shape: (rows,cols)
    R: acceleration factor (changes due to implementation)
    **kw: arguments forwarded to mask builders above
    
    Return: np.uint8 array of shape = `shape`, values ranging in {0,1}.
    """
    _validate_shape_R(shape,R)

    kind_norm = kind.strip().lower() # input handling
    if kind_norm == "nyquist":
        return _nyquist_mask(shape, R, **kw)
    elif kind_norm == "uniform":
        return _uniform_mask(shape, R)
    else:
        raise ValueError(
            f"Unknown mask kind: {kind!r}. "
            "Expected one of: nyquist, uniform."
        )
    
def acquisition_mask(shape:tuple[int,int],fraction:float,order:str="linear",
                     centre:int|None=None)->np.ndarray:
    """
    Which samples a scan has recorded after `fraction` of its time has passed.

    A Cartesian scanner fills k-space one line (row) at a time, sample by sample
    along the readout. The order the lines are visited in decides what the
    half-finished image looks like:
        linear   top row to bottom row. Halfway through, one half of k-space
                 is there and the other is missing, so the image is only
                 half right in its phase.
        centric  the centre row first, then alternately one above and one
                 below. The centre carries the contrast and the shape, so the
                 image looks right almost at once and only sharpens after.

    Return: np.uint8 mask of `shape`, 1 where recorded.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    rows,cols = shape
    kind = order.strip().lower()
    if kind == "linear":
        row_order = np.arange(rows)
    elif kind == "centric":
        c = rows//2 if centre is None else int(centre)  # centric starts at the DC row
        below,above = np.arange(c,rows),np.arange(c-1,-1,-1)
        row_order = np.empty(rows,dtype=int)
        n = min(len(below),len(above))
        row_order[0:2*n:2],row_order[1:2*n:2] = below[:n],above[:n]
        row_order[2*n:] = np.concatenate([below[n:],above[n:]])
    else:
        raise ValueError(f"Unknown order: {order!r}. Expected: linear, centric.")

    done = int(round(fraction*rows*cols))
    full,partial = divmod(done,cols)
    mask = np.zeros(shape,dtype=np.uint8)
    mask[row_order[:full],:] = 1
    if partial and full < rows:
        mask[row_order[full],:partial] = 1 # the line being read right now
    return mask

def achieved_acceleration(mask:np.ndarray)->dict:
    """
    Given ANY mask, it reports the acceleration achieved (which will not be equal to the R requested with).

    Returns a dict:
        sampled             - number of acquired nonzero samples
        total               - total number of k-space samples
        fraction_sampled    - sampled/total
        R_actual            -total/sampled (true acceleration factor)
    """
    total = mask.size
    sampled = int(np.count_nonzero(mask))

    if sampled == 0:
        raise ValueError("Mask has zero sampled points — acceleration is undefined (infinite).")
    
    fraction_sampled = sampled / total
    R_actual = total / sampled
 
    return {
        "sampled": sampled,
        "total": total,
        "fraction_sampled": fraction_sampled,
        "R_actual": R_actual,
    }

# -------------------------------------
# Tests for cartesian_sampling.py only
# -------------------------------------

if __name__ == "__main__":
    shape = (256,256)

    m = make_mask("uniform", shape, R=2)
    rows_set = np.count_nonzero(m[:, 0])
    assert rows_set == shape[0] // 2, f"Expected {shape[0]//2} rows set, got {rows_set}"
    stats = achieved_acceleration(m)
    assert abs(stats["R_actual"] - 2.0) < 1e-9, stats
    print(f"[uniform R=2]        rows_set={rows_set:3d}  R_actual={stats['R_actual']:.3f}  frac={stats['fraction_sampled']:.3f}")
 
    # --- Sampling rate: at Nyquist it is exact, below it aliases gradually ---
    prev = 0
    for rate in (0.25, 0.5, 0.75, 0.9, 1.0):
        m_n = make_mask("nyquist", shape, R=1, rate=rate)
        kept = int(np.count_nonzero(m_n[:, 0]))
        assert m_n[shape[0] // 2, 0] == 1, f"rate={rate} dropped the DC line"
        assert kept >= prev, "a higher rate must never keep fewer lines"
        assert abs(kept / shape[0] - rate) < 0.02, f"rate={rate} kept {kept}/{shape[0]}"
        prev = kept
        print(f"[nyquist {rate:.2f}]      {kept:3d}/{shape[0]} lines kept, DC included")
    assert np.all(make_mask("nyquist", shape, R=1, rate=1.0) == 1), "rate 1.0 keeps everything"

    for bad in (0.0, -0.5, 1.5):
        try:
            make_mask("nyquist", shape, R=1, rate=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"rate {bad} should have raised")
    print("[nyquist]            rate outside (0,1] rejected")

    # --- The DC row must survive at every R, not just the even ones ---
    for R in range(1, 9):
        m_r = make_mask("uniform", shape, R=R)
        assert m_r[shape[0] // 2, 0] == 1, f"uniform R={R} dropped the DC row"
        assert abs(achieved_acceleration(m_r)["R_actual"] - R) < 0.1, f"R={R} drifted"
    print("[uniform R=1..8]     DC row kept at every R, R_actual within 0.1")

    # --- Acquisition order: how much is recorded, and where, part way through ---
    for order in ("linear","centric"):
        assert acquisition_mask(shape,0.0,order).sum() == 0
        assert acquisition_mask(shape,1.0,order).all()
        half = acquisition_mask(shape,0.5,order)
        assert half.sum() == shape[0]*shape[1]//2, half.sum()
    lin = acquisition_mask(shape,0.25,"linear")
    assert lin[:64].all() and not lin[64:].any(), "linear fills from the top"
    cen = acquisition_mask(shape,0.25,"centric")
    c = shape[0]//2
    assert cen[c-32:c+32].all() and not cen[:c-32].any(), "centric fills outward from DC"
    part = acquisition_mask(shape,(3*shape[1]+10)/(shape[0]*shape[1]),"linear")
    assert part[3,:10].all() and not part[3,10:].any(), "the current line is part-read"
    print("[acquisition]         linear fills top-down, centric fills from the centre out")

    # --- An off-centre DC row is kept at every rate, and centric starts there ---
    for rate in (0.9,0.8,0.7,0.6,0.5):
        assert make_mask("nyquist",shape,1,rate=rate,centre=125)[125,0] == 1, rate
    first = acquisition_mask(shape,1.0/shape[0],"centric",centre=125)
    assert first[125].all() and first.sum() == shape[1], "centric reads the DC row first"
    print("[off-centre DC]       the real DC row is kept at every rate and read first")

    # --- Identity check: full sampling (R=1 uniform) keeps everything ---
    m_full = make_mask("uniform", shape, R=1)
    assert np.all(m_full == 1), "R=1 uniform mask must be all-ones"

    print("\nAll self-tests passed")
 
