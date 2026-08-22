# The 3-Day, 2-Person Build Plan
### Interactive MRI k-Space Reconstruction Simulator

**Scope:** Full — all six milestones including radial/spiral gridding
**Team:** 2 people, both comfortable with Python
**Working style:** Sequential (both on the engine, then both on the GUI)
**Companion:** `docs/MRI_kSpace_Simulator_Guide.pdf` — every task below cites the chapter that explains it

---

## Read this before Day 1

### One honest warning about the sequential choice

You chose "both on engine, then both on GUI." That is the simpler plan to coordinate, and with two comfortable developers it will work. But you should know the cost you are accepting, because there is one specific way it can hurt you.

**The risk:** the GUI is a full day of work that has never been started when Day 3 begins. If anything on Day 3 goes wrong — a threading bug, a matplotlib embedding problem, one stubborn layout issue — you have no slack, and you finish with a working engine and a half-finished window. That is the single most likely way this project fails to deliver.

**The mitigation, which is built into the plan below and is not optional:** on the evening of Day 1, one person spends **90 minutes** building a throwaway GUI spike. Not the real GUI — just one window, one slider, one matplotlib canvas, proving that PySide6 opens, that the canvas embeds, and that a slider signal reaches a function. Ninety minutes on Day 1 removes the biggest Day 3 unknown. If that spike fails, you learn it with two days left instead of six hours left.

So the plan is *mostly* sequential with one deliberate parallel spike. That is the best of both.

### The rule that makes the whole plan work

**`mri/` never imports Qt. `app.py` never does math.**

This is from Chapter 7 of the guide, and on a 3-day schedule it stops being an architectural nicety and becomes the thing that lets two people work without destroying each other's files. Every function in `mri/` takes arrays and returns arrays. If you follow this, the Day 3 GUI work is *wiring*, not *rewriting*.

### The gate discipline

**Never start the next task while the current task's test is failing.** Each block below ends with a **GATE**. A bug that slips past its gate costs roughly three times more to find later, because by then three other modules are suspects. On a 3-day build you cannot afford that.

### Before you write any code (30 minutes, both people, together)

Agree on and write down the function signatures in `mri/pipeline.py`. Do not skip this. It is the contract that lets you split files without merge conflicts.

- `load_image(path) -> ndarray` — returns 256×256 float in [0,1]
- `to_kspace(img) -> ndarray` — complex, centered
- `from_kspace(k) -> ndarray` — magnitude image
- `make_mask(kind, shape, R, **kw) -> ndarray` — binary
- `apply_window(k, kind, sigma) -> ndarray`
- `add_noise(k, sigma) -> ndarray`
- `reconstruct(img, settings: dict) -> dict` — the one function the GUI calls
- `compute_metrics(orig, recon) -> dict` — MSE, PSNR, SSIM

The last two matter most. `reconstruct()` taking a settings dict and returning a results dict means the GUI never needs to know the order of operations — and on Day 3 that is what saves you.

---

# DAY 1 — The Engine Core

**Goal by end of day:** k-space round-trips perfectly, all four masks work, PSF analysis works, and you have proven PySide6 runs on both machines.

### Morning — Split the files, work in parallel

Both of you are in `mri/`, but in **different files**, so there are no conflicts.

**Person A — `kspace_core.py`** *(Guide Ch. 2)*
- Image loading: grayscale conversion via the luminance formula, resize/pad to 256×256, normalize to [0,1]
- Forward transform: 2D FFT plus the centering shift
- Inverse transform: undo the shift, inverse FFT, take magnitude
- Log-magnitude display helper for showing k-space on screen
- Shepp–Logan phantom loader from `skimage.data`

**Person B — `cartesian_sampling.py`** *(Guide Ch. 3)*
- Uniform mask: keep every R-th row
- Variable-density mask: full center band, sparse periphery
- Random mask: keep each row with probability 1/R, accept a seed
- Corner-cut / circular mask: keep everything inside a radius
- Reporting helper: given any mask, return its *actual* achieved acceleration factor

> **GATE 1** *(both, together, ~20 min)*
> - Round-trip test: image → forward → inverse must match the original to within 1e-12
> - Parseval test: energy in image space equals energy in k-space
> - Visual test: log-magnitude of the phantom shows a bright center star
> - Identity test: an all-ones mask reconstructs the original exactly
> - Mask test: uniform R=2 has exactly half its rows set

### Afternoon — PSF and the first real artifact

**Both, same file, pair on it** — `psf_analyzer.py` is short and it is where the concepts click.

- Take any mask, treat it as a k-space matrix, inverse-transform it to get the PSF
- Display PSF magnitude on a log scale (side lobes are invisible otherwise)
- Extract and plot the PSF's central row as a 1D curve
- Verify the prediction: at acceleration R, count exactly R spikes spaced N/R apart

Then run the experiment that becomes a slide in your demo: fix R=4, reconstruct the phantom through uniform, variable-density, and random masks. Save one figure, three columns — mask, PSF, reconstruction.

> **GATE 2**
> - Full mask's PSF is a single central spike, no energy elsewhere
> - Uniform R=2 reconstruction shows exactly one ghost, shifted half the image height
> - Direction check: skipping *rows* must produce *vertical* ghosting. If it is horizontal, your mask is transposed.

### Evening — The GUI spike (90 min, ONE person, non-negotiable)

Throwaway code in `scratch/`. You will delete it. The point is to de-risk Day 3.

- One `QMainWindow` that opens and closes cleanly
- One `FigureCanvasQTAgg` embedded in a layout, showing the phantom
- One `QSlider` whose `valueChanged` signal calls a function that prints the value
- Confirm the matplotlib **QtAgg** backend is selected before importing pyplot

> **GATE 3** — the window opens, the image displays, and dragging the slider prints numbers. If any of this fails, you have two full days to fix it instead of six hours. That is the entire point.

**Meanwhile, the other person** starts `metrics.py` *(Guide Ch. 6)* — MSE, PSNR with a guard for the MSE=0 case, SSIM via `skimage.metrics`, and an absolute-difference error map. It is small and self-contained.

---

# DAY 2 — Filters, Noise, and the Hard Milestone

**Goal by end of day:** every piece of mathematics in the project is written and tested. No GUI work today except what is already done.

### Morning — Filters and noise (parallel, different files)

**Person A — `filters.py`** *(Guide Ch. 4)*
- 2D Hamming window built as the product of two 1D raised cosines
- 2D Gaussian window with σ as the tunable width parameter
- Both must return matrices the same shape as k-space, valued 1.0 at center fading to 0.0 at the edges
- Complex noise injection: independent Gaussian noise added to real and imaginary parts separately

**Person B — `trajectories.py`** *(Guide Ch. 5)*
- Radial spoke coordinates: for each of N spokes at angle θ, sample points along `(r·cosθ, r·sinθ)`
- Spiral coordinates: radius growing with angle
- Density compensation weights: a ramp filter, `|r|`, because spokes crowd at the center and over-represent low frequencies
- Return coordinates *and* weights together — the gridding step needs both

> **GATE 4**
> - Apodization visibly reduces ringing at sharp edges, at the cost of slight blur
> - Noise at σ=0 must be a no-op — the reconstruction is unchanged
> - Radial coordinates plotted as a scatter must look like a wheel, dense at the hub
> - Weights are largest at the periphery, smallest at the center

### Afternoon — Gridding, the one that can eat your day

**Both people, pair on this.** This is the hardest milestone and the most likely source of a lost afternoon.

**`gridding.py`** *(Guide Ch. 5)*
- Sample the true k-space at your trajectory coordinates to get the "measured" values
- Multiply those values by the density compensation weights
- Interpolate the scattered points onto the Cartesian grid using `scipy.interpolate.griddata` with `method='linear'` and `fill_value=0`
- Inverse-transform the gridded result

> **Use `scipy.griddata`. Do not hand-write a convolution kernel.** A hand-rolled gridding kernel with a Kaiser–Bessel window is the mathematically proper approach and it is a multi-day project by itself. On a 3-day build, `griddata` gets you a correct, demonstrable result this afternoon. Note the simplification in your report — that is a strength, not a weakness, because it shows you understood the trade-off.

> **GATE 5**
> - Without density compensation: the image is blurry with heavy low-frequency glare
> - With density compensation: contrast is visibly restored. **This before/after pair is the single most impressive thing in your demo — save both images.**
> - Increasing the spoke count visibly reduces streak artifacts

### Evening — Wire the pipeline together

**`pipeline.py`** — the coordinator, and the only file the GUI will ever call.

- Implement `reconstruct(img, settings)` following the agreed contract
- Order of operations matters: mask or trajectory first, then filter, then noise, then inverse transform
- Return a dict with everything the GUI needs: k-space display, reconstructed image, error map, metrics

> **GATE 6** — the full pipeline runs headless from a terminal script, with no GUI, for every combination: each mask type, each window, radial and spiral, noise on and off. Save a contact sheet of outputs. **When this gate passes, your project is functionally complete.** Everything on Day 3 is presentation.

---

# DAY 3 — The GUI and Ship

**Goal by end of day:** a running desktop application, tested, documented, demo recorded.

### Morning — Layout and wiring (parallel, and here is how to split one file)

Two people editing `app.py` is the one real conflict risk in this plan. The fix: **split it into two files for the morning, merge at lunch.**

**Person A — `app.py`**, the window shell
- `QMainWindow` with a horizontal layout: control column on the left, 2×2 image grid on the right
- The four panels: original image, current k-space, reconstruction, error heatmap
- Stretch factors so resizing grows the images and leaves the controls fixed
- Status bar

**Person B — `controls.py`**, a self-contained `QWidget` holding every control
- Sliders: acceleration R, noise σ, apodization σ, spoke count
- Dropdowns: mask type, trajectory type, window type
- Buttons: load image, reset, save screenshot
- Metric readouts: MSE, PSNR, SSIM as labels
- Expose **one** signal: `settingsChanged(dict)` carrying the full settings dictionary

That single signal is the whole integration. Person A connects it to a slot that calls `pipeline.reconstruct()` and repaints. Merging is trivial because the two files touch nothing in common.

> **GATE 7** — the window opens, controls are laid out, moving a slider emits a settings dict with correct values. Images need not update yet.

### Midday — Make it live

- Connect `settingsChanged` to the reconstruction call
- Create each matplotlib figure **once** at startup; on update call `set_data` on the existing image object rather than clearing and rebuilding the axes
- Repaint with `draw_idle()`, not `draw()`
- Turn axes off, set figure face color to match the window

**Responsiveness** *(Guide Ch. 7 and Ch. 11)*
- Move `reconstruct()` onto a `QThread` worker; return results via a custom signal
- Debounce sliders with a single-shot `QTimer` at ~150 ms so a drag fires one job, not fifty
- Only the main thread touches widgets — the worker computes and emits, the window paints

> **GATE 8** — drag the R slider continuously. The window must not freeze, images update smoothly, and PSNR/SSIM fall as R rises.

### Afternoon — Edge cases and tests *(Guide Ch. 8)*

Split these; they are independent.

**Person A — input robustness**
- Non-square images: zero-pad to square *before* resizing, or the anatomy stretches
- Color images: route everything through luminance conversion
- All-black or constant images: guard the divide-by-max in the normalizer, and the MSE=0 case in PSNR
- Slider extremes: R=1 must reproduce a perfect round-trip; max R, min spokes, σ=0 must all behave

**Person B — the test file** (`test_pipeline.py` at the project root, plain asserts, run with `python test_pipeline.py`)
- Round-trip accuracy under 1e-12
- Parseval's theorem holds
- Identity mask reproduces the original
- Ghost count matches R
- PSNR decreases monotonically as R increases
- SSIM of an image against itself equals 1.0

### Evening — Ship it

- **Styling** *(Guide Ch. 11)*: `Fusion` style plus a custom `QPalette`, then light QSS for rounded panels and a monospace font on the metric readouts. Twenty lines, and it transforms how the app reads.
- **README**: what it does, how to install, how to run, one screenshot
- **Demo video**: load an image → sweep R and watch ghosts appear → toggle apodization on ringing → switch to radial and show density compensation on/off → point at the falling metrics. Five minutes.
- **Final commit** with the Day-2 contact sheet and the gridding before/after images included

---

## Fast reference: what happens when

| | Person A | Person B | Gate |
|---|---|---|---|
| **D1 AM** | `kspace_core.py` | `cartesian_sampling.py` | Round-trip + masks |
| **D1 PM** | `psf_analyzer.py` (pair) | `psf_analyzer.py` (pair) | PSF spikes, ghost direction |
| **D1 EVE** | GUI spike (90 min) | `metrics.py` | **Window opens + slider works** |
| **D2 AM** | `filters.py` | `trajectories.py` | Ringing reduced, wheel plot |
| **D2 PM** | `gridding.py` (pair) | `gridding.py` (pair) | Density comp before/after |
| **D2 EVE** | `pipeline.py` (pair) | `pipeline.py` (pair) | **Headless full pipeline** |
| **D3 AM** | `app.py` shell | `controls.py` | Settings dict emitted |
| **D3 MID** | Wiring + threading (pair) | Wiring + threading (pair) | **No freeze on drag** |
| **D3 PM** | Edge cases | `test_pipeline.py` | All tests green |
| **D3 EVE** | Styling + README | Demo video | Shipped |

---

## If you fall behind — cut in this order

Cut from the bottom of this list first. Everything above a cut line still makes a coherent, defensible project.

1. **Spiral trajectory** — radial alone demonstrates non-Cartesian sampling completely. Cutting spiral costs you nothing conceptually.
2. **Variable-density and random masks** — uniform plus corner-cut still shows aliasing and truncation.
3. **The error heatmap panel** — go to three panels instead of four.
4. **Worker threading** — accept a brief freeze on slider release, and use `valueChanged` on release only rather than continuously. Note it as known-future-work.
5. **Styling** — a default-looking app that works beats a beautiful one that does not.

**Never cut:** the round-trip test, the metrics, or the density compensation before/after. Those three are what make it a signals project rather than a picture viewer.

---

## The four things most likely to go wrong

1. **Ghosts appear in the wrong direction.** Your mask is transposed. NumPy's row index is the *vertical* axis. Skipping rows must produce vertical wrap-around.
2. **The GUI freezes on every slider move.** Heavy math is running in the event handler on the main thread. This is the problem Chapter 7 exists to solve — move it to the worker.
3. **Gridding produces a washed-out, glaring image.** You forgot density compensation, or applied the weights after interpolating instead of before. The spokes crowd at the hub, so the center is over-counted.
4. **`ModuleNotFoundError` after installing a package.** Somebody opened a new terminal and forgot to activate the venv. Every new terminal needs `.\.venv\Scripts\Activate.ps1`.
