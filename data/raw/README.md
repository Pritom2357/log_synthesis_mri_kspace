# Real raw k-space data

`data/` is in `.gitignore`: never commit these files.

The app reads multi-coil k-space in the fastMRI HDF5 layout (`kspace`, optional
`reconstruction_rss` and `ismrmrd_header`). It looks in `$MRI_DATA_DIR`,
`D:/mri_data/raw`, then this folder.

## Data used

[M4Raw](https://zenodo.org/records/7523691): real 0.3 T brain scans, 4 coils,
18 slices of 256 x 256, CC-BY-4.0. Each subject is scanned three times
(`<study>_T201/T202/T203.h5`), and the app averages those repetitions to build
the reference.

Pull single files without downloading the whole archive:

```bash
python tools/fetch_raw.py list          # smallest files in the archive
python tools/fetch_raw.py get <name>    # one file, into data/raw/
python tools/fetch_raw.py auto 3        # the 3 smallest
```

## Use

```python
from mri.raw_data import raw_info, load_raw_kspace, load_raw_image

raw_info(path)          # header only
k = load_raw_kspace(path)   # (coils, 256, 256) complex, middle slice
img = load_raw_image(path)  # root-sum-of-squares image, 256x256 in [0,1]
```

Check with `python -m mri.raw_data`.
