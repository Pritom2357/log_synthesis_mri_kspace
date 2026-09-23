# Real raw k-space data

Everything in `mri/` other than `raw_data.py` simulates the measurement: it
starts from a picture and computes what a scanner *would* have recorded. The
files here are the opposite — what a scanner **actually** recorded, before any
image existed.

`data/` is in `.gitignore`. These files are gigabytes; never commit them.

## What is downloaded

**`knee_3dfse_ge3t.h5`** — 1.7 GB

| | |
|---|---|
| Source | [mridata.org](http://mridata.org), dataset `52c2fd53-d233-4444-8bfd-7c454240d314` |
| Project | Stanford Fully-Sampled 3D FSE Knees |
| Scanner | GE Medical Systems, **3.0 T** |
| Institution | Lucas Center for Imaging, Stanford |
| Protocol | `Sawyer_Knee_CompSensing_1` |
| Receiver coils | 8 |
| Encoded matrix | 320 x 320 x 256 |
| Recon matrix | 512 x 512 x 256 |
| Trajectory | Cartesian, 3D, fully sampled |
| Format | ISMRMRD (HDF5) |

Fully sampled matters: because nothing was skipped at acquisition time, you can
undersample it yourself with any mask in `cartesian_sampling.py` and compare
against a true reference. Data that was already accelerated on the scanner
gives you nothing to compare to.

Re-download with:

```bash
curl -L -o data/raw/knee_3dfse_ge3t.h5 \
  https://mridata-org-assets.s3.amazonaws.com/media/52c2fd53-d233-4444-8bfd-7c454240d314.h5
```

## How to use it

```python
from mri.raw_data import raw_info, load_raw_kspace, load_raw_image

raw_info("data/raw/knee_3dfse_ge3t.h5")          # header only, instant
k    = load_raw_kspace("data/raw/knee_3dfse_ge3t.h5")   # (8, 320, 320) complex, one slice
img  = load_raw_image("data/raw/knee_3dfse_ge3t.h5")    # 256x256 in [0,1], ready for the app
```

Check it with `python -m mri.raw_data`.

## Two things real data does that synthetic data does not

1. **It is a list of readouts, not a matrix.** ISMRMRD stores one entry per
   line of k-space, each tagged with the phase-encode index and slice it
   belongs to, in the order the scanner happened to acquire them. Assembling
   the matrix is the reader's job.

2. **It is multi-coil.** Eight receiver channels listened at once, so there
   are eight k-space matrices, each with its own view of the knee and its own
   unknown phase. `load_raw_image` combines them by root sum of squares —
   transform each coil separately, then add the magnitudes in quadrature.
   Adding the complex images directly would let the unknown phases cancel.

Also worth knowing: this is a **3D** acquisition, so it has no 2D slices until
you make them. `load_raw_kspace` transforms along the slice-encode axis first,
which is why asking for a slice costs a moment.

## Other sources of real raw k-space

| Dataset | Registration | Notes |
|---|---|---|
| [mridata.org](http://mridata.org) | none | ISMRMRD, direct S3 download, several anatomies. What is used here. |
| [fastMRI](https://fastmri.med.nyu.edu/) (NYU) | **required** | The big one: ~1500 knee and brain acquisitions, Siemens. Links are emailed after you accept the data-use agreement. |
| [M4Raw](https://zenodo.org/records/7523691) | none | Real 0.3 T low-field brain, multi-coil, fastMRI-style HDF5, CC-BY-4.0. Only distributed as 2.4-12.5 GB zips. |
| [OpenNeuro](https://openneuro.org/) | none | Mostly reconstructed NIfTI, not raw k-space. |

A note on fastMRI, since it is the one everybody cites: it cannot be scripted
here because the download links are personal and emailed. If you want it, sign
up yourself and drop the files in this folder — `raw_data.py` reads the
fastMRI layout too, since M4Raw and fastMRI share the
`kspace` / `reconstruction_rss` HDF5 convention.
