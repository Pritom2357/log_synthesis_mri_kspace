"""
This file generates 2D bin k-space sampling masks for Cartesian undersampling.
A helper returns the perceived *acceleration* for any mask
"""
from __future__ import annotations
import numpy as np

__all__ = ["make_mask","achieved_acceleration"] # Only these will be exported from this file

# -----------------------------------------------------------------
# Masks (private helper builder methods that will NOT be exported)
# -----------------------------------------------------------------

def _uniform_mask(shape:tuple[int,int],R:int)->np.ndarray:
    """Keeps every R-th row; zero out the remaining."""

    mask = np.zeros(shape,dtype=np.uint8)
    mask[::R,:] = 1
    return mask

def _variable_density_mask(shape:tuple[int,int],R:int,center_frac:float=0.15,periphery:str = "uniform",seed:int|None=None)->np.ndarray:
    rows,_ = shape
    mask = np.zeros(shape,dtype=np.uint8)

    center_width = max(1,int(round(center_frac*rows)))
    center_start = rows//2 - center_width//2
    center_end = center_start + center_width
    mask[center_start:center_end,:] = 1

    if periphery == "random":
        rng = np.random.default_rng(seed)
        row_keep = rng.random(rows) < (1.0/R)
        mask[row_keep,:] = 1
    elif periphery == "uniform":
        mask[::R,:] = 1
    else:
        raise ValueError(f"Unknown periphery mode:{periphery!r}")
    
    return mask

def _random_mask(shape:tuple[int,int],R:int,seed:int|None=None)->np.ndarray:
    """ Keeping each row with 1/R probability (Bernoulli sampling)"""
    rows,_ = shape
    rng = np.random.default_rng(seed)
    row_keep = rng.random(rows) < (1.0/R)
    mask = np.zeros(shape,dtype=np.uint8)
    mask[row_keep,:] = 1
    return mask

def _corner_cut_mask(shape:tuple[int,int],R:int,r_max:float|None=None)->np.ndarray:
    """
    Circular low pass mask: keeps inside: zeros outside
    
    If r_max is not given, the area fraction is approx. 1/R -> an approx. acceleration factor of R
    r_max is in pixels.
    """
    rows,cols = shape
    cy,cx = rows//2 , cols//2 # center
    
    if r_max is None:
        r_max = np.sqrt((rows * cols) / (np.pi * R))
    
    Y,X = np.ogrid[:rows,:cols]
    dist = np.sqrt((Y-cy)**2 + (X-cx)**2)

    mask = (dist<= r_max).astype(np.uint8)
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

    kind: 'uniform'|'variable_density'/'vds'|'random'|'corner_cut'/'circular'
    shape: (rows,cols)
    R: acceleration factor (changes due to implementation)
    **kw: arguments forwarded to mask builders above
    
    Return: np.uint8 array of shape = `shape`, values ranging in {0,1}.
    """
    _validate_shape_R(shape,R)

    kind_norm = kind.strip().lower() # input handling
    if kind_norm == "uniform":
        return _uniform_mask(shape, R)
    elif kind_norm in ("variable_density", "vds"):
        return _variable_density_mask(shape, R, **kw)
    elif kind_norm == "random":
        return _random_mask(shape, R, **kw)
    elif kind_norm in ("corner_cut", "circular"):
        return _corner_cut_mask(shape, R, **kw)
    else:
        raise ValueError(
            f"Unknown mask kind: {kind!r}. "
            "Expected one of: uniform, variable_density/vds, random, corner_cut/circular."
        )
    
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
 
    # --- Variable density: center should be denser than a plain uniform mask ---
    m_vds = make_mask("variable_density", shape, R=4, center_frac=0.15)
    stats_vds = achieved_acceleration(m_vds)
    center_band = m_vds[shape[0] // 2 - 5: shape[0] // 2 + 5, 0]
    assert np.all(center_band == 1), "Center band of VDS mask must be fully sampled"
    print(f"[variable_density R=4] R_actual={stats_vds['R_actual']:.3f}  frac={stats_vds['fraction_sampled']:.3f}")
 
    # --- Random: reproducible with a seed ---
    m_rand_a = make_mask("random", shape, R=4, seed=42)
    m_rand_b = make_mask("random", shape, R=4, seed=42)
    assert np.array_equal(m_rand_a, m_rand_b), "Same seed must give same random mask"
    stats_rand = achieved_acceleration(m_rand_a)
    print(f"[random R=4, seed=42]  R_actual={stats_rand['R_actual']:.3f}  frac={stats_rand['fraction_sampled']:.3f}")
 
    # --- Corner cut: symmetric circular region, roughly matches requested R ---
    m_corner = make_mask("corner_cut", shape, R=4)
    stats_corner = achieved_acceleration(m_corner)
    print(f"[corner_cut R=4]       R_actual={stats_corner['R_actual']:.3f}  frac={stats_corner['fraction_sampled']:.3f}")
    assert m_corner[shape[0] // 2, shape[1] // 2] == 1, "Center of k-space must always be sampled"
 
    # --- Identity check: full sampling (R=1 uniform) keeps everything ---
    m_full = make_mask("uniform", shape, R=1)
    assert np.all(m_full == 1), "R=1 uniform mask must be all-ones"

    print("\nAll self-tests passed")
 
