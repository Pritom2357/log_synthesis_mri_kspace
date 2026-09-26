"""Reading real multi-coil k-space in the fastMRI / M4Raw HDF5 layout."""
from __future__ import annotations
import os
import re
from pathlib import Path
import numpy as np


def _tag(xml: str, name: str, default: str = "") -> str:
    m = re.search(rf"<(?:\w+:)?{name}>(.*?)</(?:\w+:)?{name}>", xml, re.S)
    return m.group(1).strip() if m else default


def _slice(ds, z):
    n = ds.shape[0]
    z = n//2 if z is None else int(z)
    if not 0 <= z < n:
        raise ValueError(f"slice_index {z} outside 0..{n - 1}")
    return ds[z]


def _norm(a):
    lo, hi = a.min(), a.max()
    return (a - lo)/(hi - lo) if hi > lo else np.zeros_like(a)


def raw_info(path: str) -> dict:
    """Header facts, without loading the data."""
    import h5py
    with h5py.File(path, "r") as f:
        if "kspace" not in f:
            raise ValueError(f"{path} is not a fastMRI-style file (no 'kspace' dataset)")
        n_sl, n_coil, _, nx = f["kspace"].shape
        raw = f["ismrmrd_header"][()] if "ismrmrd_header" in f else b""
    xml = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
    tr, etl = _tag(xml, "TR"), _tag(xml, "echo_train_length")
    return {"vendor": _tag(xml, "systemVendor", "?"),
            "field_T": float(_tag(xml, "systemFieldStrength_T", "nan") or "nan"),
            "coils": int(n_coil), "n_slices": int(n_sl), "readout_samples": int(nx),
            "TR_ms": float(tr) if tr else float("nan"),
            "echo_train_length": int(float(etl)) if etl else 1}


def load_raw_kspace(path: str, slice_index: int | None = None) -> np.ndarray:
    """One slice, all coils: (coils, ky, kx) complex, image origin moved to the centre."""
    import h5py
    with h5py.File(path, "r") as f:
        k = np.asarray(_slice(f["kspace"], slice_index), dtype=np.complex128)
    i, j = np.indices(k.shape[-2:])
    return k*(-1.0)**(i + j)  # shift theorem: half-FOV image shift


def load_raw_image(path: str, slice_index: int | None = None, size: int | None = 256) -> np.ndarray:
    """Root-sum-of-squares image in [0,1]; resized through prepare unless size is None."""
    per_coil = np.fft.ifft2(np.fft.ifftshift(load_raw_kspace(path, slice_index), axes=(-2, -1)), axes=(-2, -1))
    img = np.sqrt((np.abs(per_coil)**2).sum(axis=0))
    if size is None:
        return _norm(img)
    from mri.kspace_core import prepare
    return prepare(img, size=size)


def load_vendor_reconstruction(path: str, slice_index: int | None = None) -> np.ndarray | None:
    import h5py
    with h5py.File(path, "r") as f:
        if "reconstruction_rss" not in f:
            return None
        return _norm(np.asarray(_slice(f["reconstruction_rss"], slice_index), dtype=np.float64))


def data_dirs() -> list:
    """$MRI_DATA_DIR, D:/mri_data/raw, then ./data/raw; whichever exist."""
    env = os.environ.get("MRI_DATA_DIR")
    candidates = ([Path(env)] if env else []) + [Path("D:/mri_data/raw"),
                                                  Path(__file__).resolve().parents[1]/"data"/"raw"]
    return [d for d in candidates if d.is_dir()]


def find_datasets() -> list:
    seen, out = set(), []
    for d in data_dirs():
        for f in sorted(d.glob("*.h5")):
            if f.name not in seen:
                seen.add(f.name)
                out.append(f)
    return out


def find_repetitions(path: str) -> list:
    """Sibling scans <study>_<contrast>NN.h5 of the same subject and contrast, sorted."""
    stem = Path(path).stem
    if "_" not in stem or len(stem.split("_", 1)[1]) < 3:
        return [str(path)]
    study, tag = stem.split("_", 1)
    contrast = tag[:-2]
    found = {str(f) for d in data_dirs() for f in d.glob(f"{study}_{contrast}*.h5")
             if f.stem.split("_", 1)[1][:-2] == contrast}
    return sorted(found) or [str(path)]


def reference_image(path: str, slice_index: int | None = None) -> tuple:
    """(image, note): average of all repetitions, else the scanner's own recon, else ours."""
    reps = find_repetitions(path)
    if len(reps) > 1:
        return _norm(np.mean([load_raw_image(r, slice_index, size=None) for r in reps], axis=0)), \
            f"average of {len(reps)} repetitions"
    vendor = load_vendor_reconstruction(path, slice_index)
    if vendor is not None:
        return vendor, "scanner's own reconstruction (same data, so expect a perfect score)"
    return load_raw_image(path, slice_index, size=None), "our own full reconstruction"


def describe(path: str) -> str:
    i = raw_info(path)
    field = f"{i['field_T']:.1f} T" if np.isfinite(i["field_T"]) else "field unknown"
    vendor = i["vendor"] if i["vendor"] != "?" else "unknown vendor"
    return f"{vendor} | {field} | {i['coils']} coils | {i['n_slices']} slices | {i['readout_samples']} samples"


if __name__ == "__main__":
    import sys
    files = [Path(sys.argv[1])] if len(sys.argv) > 1 else find_datasets()
    if not files:
        print("no raw files found; see data/raw/README.md")
        raise SystemExit(0)
    path = str(files[0])
    info = raw_info(path)
    assert info["coils"] > 1 and np.isfinite(info["field_T"]) and info["field_T"] > 0.05, info
    k = load_raw_kspace(path)
    assert k.ndim == 3 and k.shape[0] == info["coils"] and np.iscomplexobj(k)
    c = np.abs(k).sum(0)
    assert c[64:192, 64:192].max() > 10*c[:4].mean(), "brightest near the centre"
    img = load_raw_image(path)
    assert img.shape == (256, 256) and 0 <= img.min() and img.max() <= 1 and img.std() > 0.05
    vendor = load_vendor_reconstruction(path)
    if vendor is not None:
        from mri.pipeline import reconstruct
        full = reconstruct(None, {}, kspace=k, reference=vendor)
        assert full["metrics"]["ssim"] > 0.99, full["metrics"]
        prev = 1.0
        for rate in (0.9, 0.7, 0.5):
            s = reconstruct(None, {"rate": rate}, kspace=k, reference=vendor)["metrics"]["ssim"]
            assert s < prev
            prev = s
    ref, note = reference_image(path)
    print(f"raw_data: all self-tests passed on {Path(path).name} ({describe(path)}; reference: {note})")
