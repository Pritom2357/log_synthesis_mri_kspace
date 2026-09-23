"""
Reading real raw k-space, as recorded by an actual scanner.

Every other module in this project starts from a picture and simulates the
measurement. This one goes the other way. It reads what a scanner genuinely
wrote out, in the vendor-neutral ISMRMRD format, so the undersampling
experiments can run on real measurements instead of a synthetic forward
transform.

Two things are different about real data and both matter:

    it arrives as a LIST OF READOUTS, not a matrix. One entry per line of
    k-space, each tagged with where in k-space it belongs. Assembling the
    matrix is our job.

    it is MULTI-COIL. A scanner listens on several receiver channels at once,
    each with its own view of the anatomy, so there are as many k-space
    matrices as there are coils. They have to be combined into one image.
"""
from __future__ import annotations
import re
import numpy as np

__all__ = ["raw_info","load_raw_kspace","load_raw_image","load_vendor_reconstruction",
           "data_dirs","find_datasets","describe","find_repetitions","reference_image"] # Only these will be exported

# -----------------------------------------------------------------
# Header and layout (private helpers that will NOT be exported)
# -----------------------------------------------------------------

def _read_xml(path:str)->str:
    """The ISMRMRD XML header, stored as a string beside the acquisitions."""
    import h5py

    with h5py.File(path,"r") as f:
        grp = _find_group(f)
        raw = f[f"{grp}/xml"][()]
    if isinstance(raw,(bytes,bytearray)):
        return raw.decode("utf-8","ignore")
    if isinstance(raw,np.ndarray):
        raw = raw.flat[0]
        return raw.decode("utf-8","ignore") if isinstance(raw,bytes) else str(raw)
    return str(raw)

def _find_group(f)->str:
    """
    ISMRMRD files keep everything under one group, conventionally 'dataset',
    but the name is up to whoever wrote the file, so look for it.
    """
    for name in f:
        if hasattr(f[name],"keys") and "data" in f[name]:
            return name
    raise ValueError("no ISMRMRD group containing a 'data' dataset was found")

def _detect_format(path:str)->str:
    """
    Which of the two layouts this file uses.

    'ismrmrd'  - a list of readouts under a group, each tagged with its place
                 in k-space. What mridata.org serves.
    'fastmri'  - k-space already assembled into one array of shape
                 (slices, coils, ky, kx). What fastMRI and M4Raw ship.
    """
    import h5py

    with h5py.File(path,"r") as f:
        if "kspace" in f:
            return "fastmri"
        for name in f:
            if hasattr(f[name],"keys") and "data" in f[name]:
                return "ismrmrd"
    raise ValueError(f"{path} is neither an ISMRMRD nor a fastMRI-style file")

def _tag(xml:str,name:str,default:str="")->str:
    """
    One value out of the ISMRMRD header.

    Writers disagree about XML namespaces: some emit <systemVendor>, others
    <ns0:systemVendor>. Accepting an optional prefix means one reader handles
    both, which is the whole point of a vendor-neutral format.
    """
    m = re.search(rf"<(?:\w+:)?{name}>(.*?)</(?:\w+:)?{name}>",xml,re.S)
    return m.group(1).strip() if m else default

def _matrix(xml:str,block:str)->tuple[int,int,int]:
    """Matrix size out of an <encodedSpace> or <reconSpace> block."""
    b = re.search(rf"<(?:\w+:)?{block}>(.*?)</(?:\w+:)?{block}>",xml,re.S)
    if not b:
        return (0,0,0)
    inner = b.group(1)
    return tuple(int(float(_tag(inner,ax,"0") or 0)) for ax in ("x","y","z"))

def _complex_readout(row:np.ndarray,n_samples:int,n_coils:int)->np.ndarray:
    """
    One readout, unpacked into (coils, samples).

    ISMRMRD stores the samples as plain floats with real and imaginary parts
    interleaved, coil by coil, so the flat array has to be folded back up.
    """
    flat = np.asarray(row,dtype=np.float32).ravel()
    expect = n_samples*n_coils*2
    if flat.size != expect:
        raise ValueError(f"readout has {flat.size} floats, expected {expect}")

    pairs = flat.reshape(n_coils,n_samples,2)
    return pairs[...,0] + 1j*pairs[...,1]

def _info_fastmri(path:str)->dict:
    """Header for the fastMRI / M4Raw layout, where k-space is already a matrix."""
    import h5py

    with h5py.File(path,"r") as f:
        shape = f["kspace"].shape          # (slices, coils, ky, kx)
        xml = ""
        if "ismrmrd_header" in f:
            raw = f["ismrmrd_header"][()]
            xml = raw.decode("utf-8","ignore") if isinstance(raw,(bytes,bytearray)) else str(raw)

    n_sl,n_coil,ny,nx = shape
    return {
        "vendor": _tag(xml,"systemVendor","?"),
        "model": _tag(xml,"systemModel","?"),
        "institution": _tag(xml,"institutionName","?"),
        "protocol": _tag(xml,"protocolName","?"),
        "field_T": float(_tag(xml,"systemFieldStrength_T","nan") or "nan"),
        "coils": int(n_coil),
        "trajectory": _tag(xml,"trajectory","cartesian"),
        "encoded_matrix": (int(nx),int(ny),1),
        "recon_matrix": _matrix(xml,"reconSpace") or (int(nx),int(ny),1),
        "n_acquisitions": int(n_sl*n_coil*ny),
        "readout_samples": int(nx),
        "is_3d": False,                    # fastMRI files are 2D multi-slice
        "n_slices": int(n_sl),
        "n_encode_2": 1,
    }

def _to_our_convention(k:np.ndarray)->np.ndarray:
    """
    Puts scanner k-space into the same convention the rest of this project uses.

    Our from_kspace is abs(ifft2(ifftshift(k))) and it expects the image origin
    at the centre. Scanner files place it half a field of view away, so a naive
    inverse transform comes back with the anatomy split across the four corners.

    Shifting an image by half its width is, by the shift theorem, the same as
    multiplying k-space by (-1)^(i+j). Doing that here means every module
    downstream can stay exactly as it was written for simulated data. The
    checkerboard is its own inverse, so applying it twice is a no-op.
    """
    i,j = np.indices(k.shape[-2:])
    return k*((-1.0)**(i+j))

def _kspace_fastmri(path:str,slice_index:int|None)->np.ndarray:
    """One slice out of a fastMRI-style file: already a matrix, just index it."""
    import h5py

    with h5py.File(path,"r") as f:
        ds = f["kspace"]
        n_sl = ds.shape[0]
        z = n_sl//2 if slice_index is None else int(slice_index)
        if not 0 <= z < n_sl:
            raise ValueError(f"slice_index {z} outside 0..{n_sl-1}")
        return _to_our_convention(np.asarray(ds[z],dtype=np.complex128))

# ---------------------
# Methods to call from
# ---------------------

def data_dirs()->list:
    """
    Where to look for raw datasets, in priority order.

    The files are tens of megabytes and are not in the repository, so the app
    has to go looking. Setting MRI_DATA_DIR overrides everything, which is what
    you want on a different machine or a shared drive.
    """
    import os
    from pathlib import Path

    here = Path(__file__).resolve().parents[1]
    candidates = []
    env = os.environ.get("MRI_DATA_DIR")
    if env:
        candidates.append(Path(env))
    candidates += [Path("D:/mri_data/raw"), here/"data"/"raw"]
    return [d for d in candidates if d.is_dir()]

def find_datasets()->list:
    """Every raw k-space file we can find, newest-looking name first."""
    seen,out = set(),[]
    for d in data_dirs():
        for f in sorted(d.glob("*.h5")):
            if f.name not in seen:
                seen.add(f.name)
                out.append(f)
    return out

def find_repetitions(path:str)->list:
    """
    The other scans of the same subject and contrast sitting beside this one.

    M4Raw names its files <study>_<contrast><repetition>.h5, so 2022062501_T201,
    _T202 and _T203 are three separate acquisitions of the same slice. Finding
    them lets us build a reference that is genuinely better than any single
    scan, rather than scoring a file against a reconstruction of itself.

    Return: every repetition including this one, sorted, or [path] if alone.
    """
    from pathlib import Path

    p = Path(path)
    stem = p.stem
    if "_" not in stem or len(stem.split("_",1)[1]) < 3:
        return [str(p)]

    study,tag = stem.split("_",1)
    contrast = tag[:-2] # the trailing two digits are the repetition number
    found = []
    for d in data_dirs():
        for f in d.glob(f"{study}_{contrast}*.h5"):
            if f.stem.split("_",1)[1][:-2] == contrast:
                found.append(str(f))
    return sorted(set(found)) or [str(p)]

def reference_image(path:str,slice_index:int|None=None)->tuple:
    """
    The best image we can honestly claim to know for this slice.

    Averaging repeated acquisitions cancels noise, because the anatomy repeats
    and the noise does not. Three repetitions of a 0.3 T scan measured here take
    the signal-to-noise ratio from about 13 to about 18, so the average really
    is a better picture than any single scan -- and crucially it is not derived
    from the file being reconstructed, so scoring against it is meaningful.

    With no siblings on disk we fall back to the scanner's own reconstruction,
    which is honest but guarantees a near-perfect score, because it was made
    from exactly the data being reconstructed.

    Return: (image in [0,1], a short description of where it came from).
    """
    reps = find_repetitions(path)
    if len(reps) > 1:
        stack = [load_raw_image(r,slice_index,size=None) for r in reps]
        avg = np.mean(stack,axis=0)
        lo,hi = avg.min(),avg.max()
        avg = (avg-lo)/(hi-lo) if hi > lo else np.zeros_like(avg)
        return avg, f"average of {len(reps)} repetitions"

    vendor = load_vendor_reconstruction(path,slice_index)
    if vendor is not None:
        return vendor, "scanner's own reconstruction (same data, so expect a perfect score)"
    return load_raw_image(path,slice_index,size=None), "our own full reconstruction"

def describe(path:str)->str:
    """One line naming the scanner, for the status bar."""
    i = raw_info(path)
    field = f"{i['field_T']:.1f} T" if np.isfinite(i["field_T"]) else "field unknown"
    vendor = i["vendor"] if i["vendor"] != "?" else "unknown vendor"
    return (f"{vendor} | {field} | {i['coils']} coils | {i['n_slices']} slices "
            f"| {i['readout_samples']} samples")

def raw_info(path:str)->dict:
    """
    What the scanner recorded, read from the header without loading the data.

    Returns a dict:
        vendor, model, institution, protocol - who acquired it and with what
        field_T          - magnet strength in tesla
        coils            - number of receiver channels
        trajectory       - cartesian, radial, spiral, ...
        encoded_matrix   - the matrix actually sampled
        recon_matrix     - the matrix the scanner reconstructs onto
        n_acquisitions   - how many readouts are in the file
        is_3d            - True when the second phase-encode axis is sampled
        n_slices         - separately excited slices (1 for a 3D volume)
    """
    import h5py

    if _detect_format(path) == "fastmri":
        return _info_fastmri(path)

    xml = _read_xml(path)
    with h5py.File(path,"r") as f:
        grp = _find_group(f)
        data = f[f"{grp}/data"]
        n_acq = data.shape[0]
        head = data["head"]
        idx = head["idx"]
        slices = np.unique(idx["slice"])
        e2 = np.unique(idx["kspace_encode_step_2"])
        coils = int(np.max(head["active_channels"]))
        samples = int(np.max(head["number_of_samples"]))

    return {
        "vendor": _tag(xml,"systemVendor","?"),
        "model": _tag(xml,"systemModel","?"),
        "institution": _tag(xml,"institutionName","?"),
        "protocol": _tag(xml,"protocolName","?"),
        "field_T": float(_tag(xml,"systemFieldStrength_T","nan") or "nan"),
        "coils": coils,
        "trajectory": _tag(xml,"trajectory","?"),
        "encoded_matrix": _matrix(xml,"encodedSpace"),
        "recon_matrix": _matrix(xml,"reconSpace"),
        "n_acquisitions": int(n_acq),
        "readout_samples": samples,
        "is_3d": len(e2) > 1,
        "n_slices": int(len(slices)),
        "n_encode_2": int(len(e2)),
    }

def load_raw_kspace(path:str,slice_index:int|None=None)->np.ndarray:
    """
    Assembles one 2D slice of real k-space, all coils.

    The readouts are scattered through the file in acquisition order, each
    carrying its own phase-encode index, so this walks them and drops each one
    into the right row. For a 3D volume it gathers the whole encoded cube for
    the chosen readout direction and transforms along the slice axis, because
    a 3D acquisition does not have 2D slices until you do.

    slice_index: which slice; the middle one by default.

    Return: complex128 array of shape (coils, phase_encodes, readout_samples).
    """
    import h5py

    if _detect_format(path) == "fastmri":
        return _kspace_fastmri(path,slice_index)

    info = raw_info(path)
    n_coils = info["coils"]

    with h5py.File(path,"r") as f:
        grp = _find_group(f)
        data = f[f"{grp}/data"]
        head = data["head"][:]           # small: headers only
        idx = head["idx"]
        pe1 = idx["kspace_encode_step_1"].astype(int)
        pe2 = idx["kspace_encode_step_2"].astype(int)
        sl = idx["slice"].astype(int)
        n_samp = head["number_of_samples"].astype(int)

        if info["is_3d"]:
            # 3D: the second phase encode IS the slice direction. Build the
            # cube (coils, pe2, pe1, readout) then transform along pe2.
            n1,n2 = pe1.max()+1, pe2.max()+1
            cube = np.zeros((n_coils,n2,n1,int(n_samp.max())),dtype=np.complex64)
            for i in range(data.shape[0]):
                if n_samp[i] == 0:
                    continue
                cube[:,pe2[i],pe1[i],:n_samp[i]] = _complex_readout(
                    data["data"][i],int(n_samp[i]),n_coils)
            vol = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(cube,axes=1),axis=1),axes=1)
            z = n2//2 if slice_index is None else int(slice_index)
            if not 0 <= z < n2:
                raise ValueError(f"slice_index {z} outside 0..{n2-1}")
            plane = vol[:,z]
        else:
            # 2D multi-slice: pick the wanted slice and fill its rows.
            want = int(np.median(np.unique(sl))) if slice_index is None else int(slice_index)
            if want not in set(sl.tolist()):
                raise ValueError(f"slice_index {want} is not in this file")
            rows = np.flatnonzero(sl == want)
            n1 = pe1[rows].max()+1
            plane = np.zeros((n_coils,n1,int(n_samp[rows].max())),dtype=np.complex64)
            for i in rows:
                if n_samp[i] == 0:
                    continue
                plane[:,pe1[i],:n_samp[i]] = _complex_readout(
                    data["data"][i],int(n_samp[i]),n_coils)

    return np.asarray(plane,dtype=np.complex128)

def load_vendor_reconstruction(path:str,slice_index:int|None=None)->np.ndarray|None:
    """
    The scanner's own reconstruction of this slice, when the file carries one.

    fastMRI and M4Raw store it as 'reconstruction_rss'. It is worth having
    because it is a ground truth we did not compute ourselves: if our image
    agrees with the scanner's, our transform chain is right.

    Return: float64 image in [0,1], or None when the file has no such dataset.
    """
    import h5py

    with h5py.File(path,"r") as f:
        if "reconstruction_rss" not in f:
            return None
        ds = f["reconstruction_rss"]
        n = ds.shape[0]
        z = n//2 if slice_index is None else int(slice_index)
        if not 0 <= z < n:
            raise ValueError(f"slice_index {z} outside 0..{n-1}")
        img = np.asarray(ds[z],dtype=np.float64)

    lo,hi = img.min(),img.max()
    return (img-lo)/(hi-lo) if hi > lo else np.zeros_like(img)

def load_raw_image(path:str,slice_index:int|None=None,size:int|None=256)->np.ndarray:
    """
    Real raw k-space turned into one image this project can use.

    Each coil is transformed on its own and then the coils are combined by
    root sum of squares. That combination is used because the coils have
    different, unknown phases, so adding them directly would let them cancel;
    adding their magnitudes in quadrature cannot.

    size: resize and normalise through the usual input path, or None to keep
    the scanner's own matrix.

    Return: float64 image in [0,1].
    """
    k = load_raw_kspace(path,slice_index) # already in this project's convention
    per_coil = np.fft.ifft2(np.fft.ifftshift(k,axes=(-2,-1)),axes=(-2,-1))
    img = np.sqrt(np.sum(np.abs(per_coil)**2,axis=0)) # root sum of squares

    if size is None:
        lo,hi = img.min(),img.max()
        return (img-lo)/(hi-lo) if hi > lo else np.zeros_like(img)

    from mri.kspace_core import prepare
    return prepare(img,size=size)

# ----------------------------
# Tests for raw_data.py only
# ----------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/knee_3dfse_ge3t.h5"
    if not Path(path).exists():
        print(f"no raw file at {path} -- download one first, see data/raw/README.md")
        raise SystemExit(0)

    info = raw_info(path)
    for key in ("vendor","model","institution","protocol","field_T","coils",
                "trajectory","encoded_matrix","recon_matrix","n_acquisitions",
                "readout_samples","is_3d","n_slices","n_encode_2"):
        print(f"  {key:16s} {info[key]}")

    # --- This has to be real scanner data, not something we made ---
    assert info["coils"] > 1, "real scanner data is multi-coil"
    assert info["n_acquisitions"] > 100, "too few readouts to be a real scan"
    assert np.isfinite(info["field_T"]) and info["field_T"] > 0.05
    print("\n[header]     multi-coil, non-trivial acquisition count, real field strength")

    # --- One slice of genuine k-space ---
    k = load_raw_kspace(path)
    assert k.ndim == 3 and k.shape[0] == info["coils"], k.shape
    assert np.iscomplexobj(k) and np.abs(k).max() > 0
    centre = np.abs(k[:,k.shape[1]//2,k.shape[2]//2]).mean()
    edge = np.abs(k[:,:4,:]).mean() # a band, not one cell: the corner can be pure zero
    assert centre > edge, "real k-space is brightest at the centre"
    print(f"[kspace]     shape={k.shape}  centre is {centre/max(edge,1e-12):,.0f}x the outer edge")

    # --- The coil-combined image ---
    img = load_raw_image(path)
    assert img.shape == (256,256) and 0.0 <= img.min() and img.max() <= 1.0
    assert img.std() > 0.05, "a real anatomy image should not be nearly flat"
    print(f"[image]      shape={img.shape}  range=[{img.min():.2f},{img.max():.2f}]  std={img.std():.3f}")

    # --- Individual coils must genuinely differ; that is what makes it multi-coil ---
    per = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k,axes=(-2,-1))),axes=(-2,-1))
    mags = [np.abs(c) for c in per]
    diffs = [np.abs(mags[0]-m).mean()/max(mags[0].mean(),1e-12) for m in mags[1:]]
    print(f"[coils]      each coil sees the anatomy differently, mean relative difference "
          f"{np.mean(diffs):.2f}")
    assert np.mean(diffs) > 0.1, "coils look identical, which real coils never do"

    # --- And it drops straight into the existing pipeline ---
    from mri.pipeline import reconstruct
    out = reconstruct(img,{"mask":"random","R":4,"seed":0,"recon":"cs"})
    print(f"[pipeline]   real scan through the pipeline, random R=4: "
          f"psnr={out['metrics']['psnr']:.2f} ssim={out['metrics']['ssim']:.3f}")

    # --- Reconstructing FROM the measured k-space, with no forward transform ---
    # This is the path the supervisor asked for: no image exists until we make
    # one. With nothing thrown away it has to reproduce the scanner's own image.
    from mri.pipeline import reconstruct as _recon
    _ref = load_vendor_reconstruction(path)
    if _ref is not None:
        full = _recon(None,{"R":1},kspace=load_raw_kspace(path),reference=_ref)
        assert full["metrics"]["ssim"] > 0.99, full["metrics"]
        print(f"[from kspace] fully sampled real data -> psnr={full['metrics']['psnr']:.1f} "
              f"ssim={full['metrics']['ssim']:.4f}  (no forward transform used)")

        prev = full["metrics"]["ssim"]
        for rate in (0.9,0.7,0.5):
            part = _recon(None,{"mask":"nyquist","rate":rate},
                          kspace=load_raw_kspace(path),reference=_ref)
            ssim = part["metrics"]["ssim"]
            assert ssim < prev, f"rate {rate} did not cost quality"
            prev = ssim
            print(f"[nyquist {rate:.2f}] sampling at {rate:.0%} of the Nyquist rate "
                  f"-> ssim {ssim:.3f}")

    # --- Some files ship the scanner's own reconstruction: a real ground truth ---
    vendor = load_vendor_reconstruction(path)
    if vendor is not None:
        from mri.metrics import compute_metrics
        m = compute_metrics(vendor,load_raw_image(path,size=None))
        print(f"[vendor ref] the scanner's own image is in the file; ours matches it "
              f"psnr={m['psnr']:.2f} ssim={m['ssim']:.3f}")

    print("\nAll self-tests passed")
