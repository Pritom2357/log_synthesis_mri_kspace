"""
Applying settings to one region of the image instead of the whole thing.

There is a catch that has to be stated before anything else here makes sense.
Every k-space sample contributes to EVERY pixel of the image -- that is what a
Fourier transform is. So there is no such thing as "filter only this corner of
k-space for only that corner of the anatomy". A radiologist who wants to
sharpen one suspicious region cannot do it by editing k-space locally, because
k-space has no local.

What can be done is this. Reconstruct the slice twice: once with nothing
switched on, and once with the settings applied. Both are full images, both are
computed the same way, and then they are mixed pixel by pixel -- the edited
version inside the region, the clean version outside. The result is a picture
that is genuinely processed in one place and untouched everywhere else.

The mixing is feathered rather than a hard cut. A hard boundary between two
differently-filtered images shows as a visible rectangle, and a visible
rectangle in a diagnostic image is worse than no processing at all.
"""
from __future__ import annotations
import numpy as np

__all__ = ["Box","feather_mask","blend"] # Only these will be exported

class Box:
    """
    A rectangle on the image, in pixels, stored as fractions of the image size.

    Fractions rather than pixels because the output size changes -- upscaling
    by 2 turns a 256 pixel image into 512 -- and a region the user drew should
    stay on the same anatomy when that happens.
    """

    __slots__ = ("y0","x0","h","w")

    def __init__(self,y0:float,x0:float,h:float,w:float):
        if not (0.0 <= y0 <= 1.0 and 0.0 <= x0 <= 1.0):
            raise ValueError("y0 and x0 must be fractions in [0,1]")
        if not (0.0 < h <= 1.0 and 0.0 < w <= 1.0):
            raise ValueError("h and w must be fractions in (0,1]")
        self.y0,self.x0,self.h,self.w = float(y0),float(x0),float(h),float(w)

    def pixels(self,shape:tuple[int,int])->tuple[int,int,int,int]:
        """(row0, col0, row1, col1) for an image of this shape, clipped to it."""
        rows,cols = shape[-2:]
        r0 = int(round(self.y0*rows)); c0 = int(round(self.x0*cols))
        r1 = min(rows,r0+max(1,int(round(self.h*rows))))
        c1 = min(cols,c0+max(1,int(round(self.w*cols))))
        return max(0,r0),max(0,c0),r1,c1

    def __repr__(self):
        return f"Box(y0={self.y0:.3f}, x0={self.x0:.3f}, h={self.h:.3f}, w={self.w:.3f})"

# ---------------------
# Methods to call from
# ---------------------

def feather_mask(shape:tuple[int,int],box:Box,feather:float=0.15)->np.ndarray:
    """
    A soft rectangle: 1.0 well inside the box, 0.0 well outside, smooth between.

    feather is the width of the transition as a fraction of the shorter side of
    the box. Zero gives a hard edge, which is visible as a rectangle in the
    output and is almost never what you want.

    Return: float64 array in [0,1], same shape as the image.
    """
    if not 0.0 <= feather <= 1.0:
        raise ValueError("feather must be in [0,1]")

    rows,cols = shape[-2:]
    r0,c0,r1,c1 = box.pixels(shape)
    mask = np.zeros((rows,cols),dtype=np.float64)
    mask[r0:r1,c0:c1] = 1.0
    if feather == 0.0:
        return mask

    # A box blur applied twice is a smooth ramp, and it costs one convolution
    # per axis rather than building a distance field.
    width = max(1,int(round(feather*min(r1-r0,c1-c0))))
    kernel = np.ones(width)/width
    for _ in range(2):
        mask = np.apply_along_axis(lambda m: np.convolve(m,kernel,mode="same"),0,mask)
        mask = np.apply_along_axis(lambda m: np.convolve(m,kernel,mode="same"),1,mask)
    return np.clip(mask,0.0,1.0)

def blend(clean:np.ndarray,edited:np.ndarray,box:Box|None,
          feather:float=0.15)->np.ndarray:
    """
    Mixes two reconstructions: edited inside the box, clean outside.

    Both must be the same shape, because they are two reconstructions of the
    same slice differing only in the settings that were applied.

    box=None returns the edited image untouched, which is the whole-image case.
    """
    if box is None:
        return edited
    if clean.shape != edited.shape:
        raise ValueError(f"shape mismatch: {clean.shape} vs {edited.shape}")

    m = feather_mask(clean.shape,box,feather)
    return clean*(1.0-m) + edited*m

# --------------------------
# Tests for region.py only
# --------------------------

if __name__ == "__main__":
    from mri.kspace_core import load_phantom

    img = load_phantom()
    shape = img.shape

    # --- The box maps onto pixels sensibly ---
    b = Box(0.25,0.25,0.5,0.5)
    r0,c0,r1,c1 = b.pixels(shape)
    assert (r0,c0,r1,c1) == (64,64,192,192), (r0,c0,r1,c1)
    # and it follows the anatomy when the image is resized
    assert b.pixels((512,512)) == (128,128,384,384), "a fractional box must scale"
    print(f"[box]         {b} -> pixels {(r0,c0,r1,c1)} at 256, scales to 512 correctly")

    # --- A hard mask is exactly the rectangle ---
    hard = feather_mask(shape,b,feather=0.0)
    assert hard.sum() == (r1-r0)*(c1-c0), "a hard mask must be exactly the box"
    assert hard[128,128] == 1.0 and hard[10,10] == 0.0
    print(f"[hard mask]   covers exactly {int(hard.sum())} pixels, nothing else")

    # --- A feathered mask is still 1 in the middle, 0 far away, smooth between ---
    soft = feather_mask(shape,b,feather=0.15)
    assert soft[128,128] > 0.99, "the centre of the box must be fully edited"
    assert soft[5,5] < 1e-9, "far outside must be fully clean"
    edge_jump = np.abs(np.diff(soft[128,:])).max()
    hard_jump = np.abs(np.diff(hard[128,:])).max()
    assert edge_jump < 0.2*hard_jump, "feathering must soften the boundary"
    print(f"[soft mask]   biggest step across the edge {edge_jump:.3f} "
          f"versus {hard_jump:.3f} for a hard cut")

    # --- Blending leaves the outside untouched and changes the inside ---
    edited = np.clip(img*0.3,0,1) # stand-in for "some setting was applied"
    out = blend(img,edited,b)
    assert np.allclose(out[:40,:40],img[:40,:40]), "outside the box must be untouched"
    assert not np.allclose(out[120:136,120:136],img[120:136,120:136]), "inside must change"
    assert np.allclose(out[120:136,120:136],edited[120:136,120:136],atol=1e-6), \
        "the middle of the box must be fully the edited version"
    print("[blend]       outside identical to the clean image, centre fully edited")

    # --- No box means no blending at all ---
    assert blend(img,edited,None) is edited
    print("[whole image] box=None returns the edited image unchanged")

    # --- The seam must not be visible ---
    # The anatomy has strong edges of its own, so an absolute gradient says
    # nothing: row 128 crosses the skull. What matters is the gradient the
    # blend ADDED at the boundary, against what a hard cut adds in the same
    # place.
    # Measured on a deliberately smooth image, not the phantom. The phantom's
    # own skull edge is far sharper than any seam, and it sits inside the box,
    # so it dominates every gradient statistic and hides exactly what is being
    # tested. On a smooth ramp, any step at all is the blend's doing.
    smooth = np.linspace(0.2,0.8,shape[1])[None,:]*np.ones((shape[0],1))
    dim = smooth*0.4

    def seam_step(f):
        return float(np.abs(np.diff(blend(smooth,dim,b,feather=f),axis=1)).max())

    widths = [0.0,0.05,0.15,0.30]
    vals = [seam_step(f) for f in widths]
    for a,bb in zip(vals,vals[1:]):
        assert bb <= a + 1e-9, f"a wider feather must never leave a sharper seam: {vals}"
    assert vals[0] > 10*vals[2], f"feathering must clearly beat a hard cut: {vals}"
    print("[seam]        sharpest step the blend introduced, on a smooth image:")
    for f,v in zip(widths,vals):
        print(f"                 feather {f:.2f} -> {v:.5f}"
              f"{'   (hard cut: a visible rectangle)' if f == 0 else ''}")

    # --- Guards ---
    for bad in (lambda: Box(0.5,0.5,0.0,0.5), lambda: Box(-0.1,0,0.5,0.5),
                lambda: feather_mask(shape,b,feather=2.0),
                lambda: blend(img,img[:64,:64],b)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("bad input should have raised")
    print("[guard]       zero-size box, out-of-range origin, bad feather, shape mismatch")

    print("\nAll self-tests passed")
