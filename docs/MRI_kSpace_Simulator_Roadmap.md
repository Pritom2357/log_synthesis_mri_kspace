# Deep-Dive Structural Analysis & 6-Week Technical Roadmap
## Project: Interactive MRI k-Space Reconstruction Simulator
**Course Alignment:** BUET CSE 219 (Signals and Linear Systems) & CSE 220 (Sessional)  
**Target Duration:** 6 Weeks  
**Primary Language & Frameworks:** Python 3.10+, NumPy, SciPy, PyQt6 / CustomTkinter, Matplotlib, scikit-image

---

## 1. Executive Summary & Project Purpose

In Magnetic Resonance Imaging (MRI), physical scanner hardware does not directly capture spatial images (e.g., cross-sections of tissue). Instead, radio-frequency (RF) signals emitted by protons inside a magnetic field gradient are sampled in the **spatial frequency domain**, known universally as **$k$-space**. The spatial image $f(x,y)$ is mathematically reconstructed by taking the Inverse 2D Fourier Transform of the sampled $k$-space matrix $K(k_x, k_y)$.

The **Interactive MRI $k$-Space Reconstruction Simulator** is a software engineering and signal processing platform designed to model the entire MRI signal processing pipeline. It allows students, researchers, and instructors to interactively explore:
1. How spatial image details map to specific $k$-space frequency regions.
2. The mathematical consequences of **undersampling $k$-space** (aliasing, resolution degradation, streak artifacts).
3. Reconstructing non-Cartesian trajectories (radial and spiral) via **density compensation** and **gridding algorithms**.
4. The application of **apodization (windowing) filters** to eliminate Gibbs ringing.
5. Quantitative image quality metrics ($PSNR$, $SSIM$, $MSE$) under varying acceleration factors ($R$) and noise levels.

---

## 2. Course Syllabus Mapping (BUET CSE 219 / 220)

| CSE 219 / 220 Syllabus Topic | Mathematical Concept | Project Implementation |
| :--- | :--- | :--- |
| **2D Fourier Transforms** | $\mathcal{F}_2\{f(x,y)\}$, $2D	ext{ IFFT}$, Shift operations | Converting images into complex $k$-space, manipulating phase/magnitude, and spatial reconstruction. |
| **Sampling & Nyquist Rate** | $\Delta k_x \le rac{1}{FOVx}$, Sub-Nyquist sampling | Skipping $k$-space phase-encoding lines ($R > 1$) to demonstrate wrap-around aliasing. |
| **Point Spread Function (PSF)** | $h(x,y) = \mathcal{F}_2^{-1}\{S(k_x, k_y)\}$ | Analyzing the impulse response of sampling masks $S(k_x, k_y)$ to predict artifact patterns. |
| **Non-Uniform Sampling & Interpolation**| Bilinear interpolation, Kernel convolution gridding | Mapping non-Cartesian radial spokes $(k_x, k_y) = (r\cos	heta, r\sin	heta)$ to a Cartesian grid. |
| **LTI Systems & Windowing** | Spatial convolution via frequency multiplication | Applying 2D Hamming/Gaussian masks in $k$-space to suppress edge truncation ringing. |

---

## 3. Structural & Architectural Analysis

### 3.1 Data Flow Pipeline
```
 [Input Spatial Image / Phantom f(x,y)]
                   │
                   ▼
       [2D Forward FFT + fftshift]
                   │
                   ▼
        [Complex k-Space K(kx, ky)]
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
 [Cartesian Mask S]     [Non-Cartesian Trajectory]
       │                       │
       │                       ▼
       │               [Density Compensation]
       │                       │
       │                       ▼
       │            [Gridding / Interpolation]
       └───────────┬───────────┘
                   ▼
      [Undersampled k-Space K_sub]
                   │
                   ▼
     [Optional Apodization Filtering]
                   │
                   ▼
        [2D Inverse FFT + ifftshift]
                   │
                   ▼
      [Reconstructed Image f_hat(x,y)]
                   │
                   ▼
  [Quantitative Metrics & Error Map Analysis]
```

---

### 3.2 Core Mathematical Foundations

#### 1. Forward and Inverse 2D Continuous Fourier Transform
$$K(k_x, k_y) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) \, e^{-i 2\pi (k_x x + k_y y)} \, dx \, dy$$

$$f(x, y) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} K(k_x, k_y) \, e^{+i 2\pi (k_x x + k_y y)} \, dk_x \, dk_y$$

#### 2. Discrete 2D Fourier Transform (DFT) for Discrete Grids ($N 	imes M$)
$$K[u, v] = \sum_{m=0}^{N-1} \sum_{n=0}^{M-1} f[m, n] \, e^{-i 2\pi \left( rac{u m}{N} + rac{v n}{M} ight)}$$

$$f[m, n] = rac{1}{N M} \sum_{u=0}^{N-1} \sum_{v=0}^{M-1} K[u, v] \, e^{+i 2\pi \left( rac{u m}{N} + rac{v n}{M} ight)}$$

#### 3. Logarithmic Scaling for $k$-Space Visualization
Because central DC components in $k$-space dominate by several orders of magnitude, dynamic range compression is required:
$$V(u,v) = \log\left(1 + |K[u,v]|ight)$$

#### 4. Point Spread Function (PSF) and Sampling Mask
Let $S[u,v] \in \{0, 1\}$ be a sampling mask. The undersampled $k$-space is $K_{	ext{sub}}[u,v] = K[u,v] \cdot S[u,v]$.  
By the convolution theorem, the reconstructed image $\hat{f}[m,n]$ is the spatial convolution of the true image with the PSF:
$$\hat{f}[m,n] = f[m,n] ** 	ext{PSF}[m,n], \quad 	ext{where } 	ext{PSF}[m,n] = \mathcal{F}_2^{-1}\{S[u,v]\}$$

#### 5. Radial Density Compensation Function (DCF)
In radial sampling, samples near the center of $k$-space are much denser than at the periphery. To prevent low-frequency over-amplification, a ramp filter DCF $w(r)$ is applied:
$$w(r) = |r| = \sqrt{k_x^2 + k_y^2}$$

#### 6. Quantitative Error Metrics
* **Mean Squared Error (MSE):**
  $$MSE = rac{1}{N M} \sum_{m=0}^{N-1} \sum_{n=0}^{M-1} |f[m,n] - \hat{f}[m,n]|^2$$
* **Peak Signal-to-Noise Ratio (PSNR):**
  $$PSNR = 10 \cdot \log_{10} \left( rac{\max(f)^2}{MSE} ight) \quad 	ext{(dB)}$$
* **Structural Similarity Index (SSIM):**
  $$SSIM(f, \hat{f}) = rac{(2\mu_f \mu_{\hat{f}} + C_1)(2\sigma_{f\hat{f}} + C_2)}{(\mu_f^2 + \mu_{\hat{f}}^2 + C_1)(\sigma_f^2 + \sigma_{\hat{f}}^2 + C_2)}$$

---

### 3.3 Object-Oriented Class Architecture

```
         ┌──────────────────────────────┐
         │     GUIController / View     │
         └──────────────┬───────────────┘
                        │ (Triggers updates)
                        ▼
         ┌──────────────────────────────┐
         │        PipelineEngine        │
         └──────────────┬───────────────┘
    ┌───────────────────┼───────────────────┐
    ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  KSpaceCore  │  │ SamplingMask │  │ GriddingEng  │
└──────────────┘  └──────────────┘  └──────────────┘
    │                   │                   │
    └───────────────────┼───────────────────┘
                        ▼
         ┌──────────────────────────────┐
         │       MetricsCalculator      │
         └──────────────────────────────┘
```

#### Class Descriptions:
1. `KSpaceCore`: Handles image loading, normalization, forward 2D FFT, $k$-space spectrum generation, phase/magnitude extraction, apodization filtering, and inverse 2D FFT reconstruction.
2. `SamplingMaskGenerator`: Generates 2D boolean masks for Cartesian undersampling (Uniform, Variable Density, Random, Corner Cutting) and computes the Point Spread Function (PSF).
3. `NonCartesianTrajectory`: Computes coordinates for Radial spokes and Spiral trajectories. Generates Density Compensation Functions (DCF).
4. `GriddingEngine`: Performs nearest-neighbor and bilinear convolution interpolation from non-uniform trajectories to Cartesian grids.
5. `MetricsCalculator`: Computes spatial difference images, MSE, PSNR, SSIM, and signal-to-noise ratios (SNR).
6. `MainWindow (GUI)`: Qt/CustomTkinter dashboard displaying interactive controls (sliders, drop-downs, canvas views, real-time plotting).

---

### 3.4 Software Stack Matrix

| Layer | Recommended Library | Purpose |
| :--- | :--- | :--- |
| **Mathematical Computations** | `numpy` | Fast multidimensional vector/matrix operations and 2D FFT computations (`np.fft.fft2`, `np.fft.ifft2`, `np.fft.fftshift`). |
| **Interpolation & Filtering** | `scipy.interpolate`, `scipy.ndimage` | Non-Cartesian grid interpolation (`griddata`) and 2D spatial filtering. |
| **Metrics & Phantom Generation** | `scikit-image` (`skimage`) | Shepp-Logan phantom generation (`skimage.data.shepp_logan_phantom`), SSIM computation (`skimage.metrics.structural_similarity`). |
| **GUI Framework** | `PyQt6` or `PySide6` | High-performance desktop UI with native Qt multi-threading (`QThread`) for zero UI freeze during computations. |
| **Plotting Engine** | `matplotlib` (embedded in Qt) | Interactive rendering of image, $k$-space spectrum, sampling mask, and error heatmaps. |

---

## 4. In-Depth 6-Week Technical Roadmap

---

### Week 1: Core Math Engine & 2D Fourier Transformation Foundation

#### Primary Objectives:
Establish the mathematical baseline for forward and inverse 2D Fourier Transforms, dynamic logarithmic scaling, complex matrix manipulations, and benchmark test data ingestion.

#### Technical Tasks:
1. **Data Ingestion Module (`kspace_core.py`):**
   * Implement loader functions for standard benchmark images (Grayscale PNG/JPEG, DICOM files via `pydicom`, and synthetic Shepp-Logan phantoms).
   * Force image array resizing to $2^N 	imes 2^N$ (e.g., $256 	imes 256$ or $512 	imes 512$) for optimized Radix-2 FFT performance.
   * Normalize input values to floating-point range $[0.0, 1.0]$.
2. **Forward 2D FFT Engine:**
   * Compute complex $k$-space: $K = 	ext{fftshift}(	ext{fft2}(f))$.
   * Compute magnitude spectrum $|K|$ and phase spectrum $ngle K = rctan2(	ext{Im}(K), 	ext{Re}(K))$.
   * Implement log-transformed magnitude visualization array $V = \log(1 + |K|)$.
3. **Inverse 2D FFT Engine:**
   * Reconstruct spatial image: $\hat{f} = 	ext{real}(	ext{ifft2}(	ext{ifftshift}(K)))$.
   * Verify Parseval's energy conservation theorem: $\sum |f[m,n]|^2 = rac{1}{NM} \sum |K[u,v]|^2$.
4. **Interactive Component Probing:**
   * Implement isolated low-frequency vs. high-frequency bandpass box filters to prove that low frequencies carry image contrast and structural topology, while high frequencies encode fine boundaries and edges.

#### Key APIs & Function Signatures:
```python
import numpy as np

class KSpaceCore:
    def __init__(self, image_array: np.ndarray):
        self.spatial_image = image_array.astype(np.float64)
        self.k_space = None
        self.compute_forward_fft()

    def compute_forward_fft(self) -> np.ndarray:
        self.k_space = np.fft.fftshift(np.fft.fft2(self.spatial_image))
        return self.k_space

    def get_magnitude_spectrum(self, log_scale: bool = True) -> np.ndarray:
        mag = np.abs(self.k_space)
        return np.log1p(mag) if log_scale else mag

    def reconstruct_image(self, k_space_data: np.ndarray = None) -> np.ndarray:
        data = self.k_space if k_space_data is None else k_space_data
        unshifted = np.fft.ifftshift(data)
        recon = np.fft.ifft2(unshifted)
        return np.abs(recon)
```

#### Deliverables for Week 1:
* `kspace_core.py`: Validated 2D FFT/IFFT class.
* `test_week1_math.py`: Unit test suite verifying Parseval's identity and perfect reconstruction (reconstruction error $< 10^{-12}$).
* Visual plots showing spatial image $ightarrow$ magnitude $k$-space $ightarrow$ reconstructed image.

---

### Week 2: Cartesian Undersampling Strategies & Aliasing Artifact Analysis

#### Primary Objectives:
Implement dynamic 2D sampling masks, quantify acceleration factor $R$, compute Point Spread Functions (PSFs), and model phase-encoding wrap-around aliasing.

#### Technical Tasks:
1. **Sampling Mask Engine (`cartesian_sampling.py`):**
   * **Uniform 1D Subsampling (Phase-encoding skip):** Retain every $R$-th row in $k$-space. Set un-acquired rows to $0$.
   * **Variable Density Sampling (VDS):** Fully sample a central low-frequency region ($10\% - 20\%$ of $k$-space) and uniformly or randomly subsample peripheral high frequencies.
   * **Random Cartesian Subsampling:** Bernoulli random selection of rows with variable probability distributions.
   * **Corner Cutting (Circular Mask):** Zero out high-frequency corners where $u^2 + v^2 > r_{	ext{max}}^2$.
2. **Point Spread Function (PSF) Computation:**
   * Compute $	ext{PSF} = 	ext{fftshift}(	ext{ifft2}(	ext{ifftshift}(S)))$, where $S$ is the binary sampling matrix.
   * Plot 1D cross-sections of the PSF to demonstrate side-lobe energy that causes spatial aliasing artifacts.
3. **Aliasing Simulation & Quantification:**
   * Demonstrate how skipping lines along the phase-encoding axis ($k_y$) reduces the Effective Field of View ($	ext{FOV}_{	ext{eff}} = rac{	ext{FOV}}{R}$), producing ghosting/wrap-around overlap in spatial domain.
   * Calculate Acceleration Factor $R = rac{N_{	ext{total}}}{N_{	ext{sampled}}}$.

#### Mathematical Formulation for Sampling Masks:
$$S_{	ext{uniform}}[u, v] = egin{cases} 1 & 	ext{if } v \pmod R = 0 \ 0 & 	ext{otherwise} \end{cases}$$

$$S_{	ext{VDS}}[u, v] = egin{cases} 1 & 	ext{if } |v - v_0| < W_{	ext{center}} 	ext{ or } 	ext{rand}() < p(v) \ 0 & 	ext{otherwise} \end{cases}$$

#### Deliverables for Week 2:
* `cartesian_sampling.py`: Generator class for standard and custom 2D sampling masks.
* `psf_analyzer.py`: PSF generation and 1D cross-section plotter.
* Experimental report comparing ghosting artifacts from Uniform Subsampling vs. incoherent noise-like artifacts from Random Subsampling.

---

### Week 3: Non-Cartesian Trajectories & Gridding Reconstruction

#### Primary Objectives:
Develop non-Cartesian sampling trajectory generators (Radial spokes and Spirals), implement Density Compensation Functions (DCFs), and construct a 2D Gridding Reconstruction Engine.

#### Technical Tasks:
1. **Non-Cartesian Trajectory Generator (`trajectories.py`):**
   * **Radial Sampling:** Generate $N_{	ext{spokes}}$ radial lines intersecting the $k$-space center at angles $	heta_m = rac{m \pi}{N_{	ext{spokes}}}$. Calculate Continuous $k$-space coordinates:
     $$k_x(r, 	heta_m) = r \cos 	heta_m, \quad k_y(r, 	heta_m) = r \sin 	heta_m, \quad r \in [-k_{	ext{max}}, k_{	ext{max}}]$$
   * **Archimedean Spiral Trajectory:** Generate continuous spiral coordinates:
     $$k_x(t) = a t \cos(\omega t), \quad k_y(t) = a t \sin(\omega t)$$
2. **Density Compensation Function (DCF):**
   * Compute linear ramp filter weights $w_i = |r_i| = \sqrt{k_{x,i}^2 + k_{y,i}^2}$ for radial trajectories to counteract dense oversampling at the origin.
3. **Gridding Algorithm Engine (`gridding.py`):**
   * **Nearest-Neighbor Interpolation:** Map non-uniform trajectory points $(k_{x,i}, k_{y,i})$ to the nearest integer grid cell $(u, v)$ on a Cartesian $N 	imes N$ matrix.
   * **Bilinear / Kernel Convolution Interpolation:** Convolve non-uniform data points with a Kaiser-Bessel or bilinear kernel to distribute continuous values across adjacent grid points.
   * Apply 2D IFFT to the gridded Cartesian matrix to produce spatial reconstructions.

#### Deliverables for Week 3:
* `trajectories.py`: Mathematical module generating coordinates for radial and spiral sampling.
* `gridding.py`: Interpolation engine mapping non-Cartesian points to Cartesian grids.
* Comparative study illustrating streak artifacts in radial undersampling versus coherent aliasing in Cartesian undersampling.

---

### Week 4: Interactive GUI Architecture & Asynchronous Event Processing

#### Primary Objectives:
Build a multi-pane interactive Desktop Application (using PyQt6 or CustomTkinter) with asynchronous background processing threads (`QThread`) to ensure smooth visual performance.

#### Technical Tasks:
1. **Layout & Dashboard Wireframe Construction:**
   * **Pane 1 (Top-Left):** Input Spatial Image / Ground Truth Phantom.
   * **Pane 2 (Top-Right):** Interactive $k$-Space Spectrum View with sampling mask overlays.
   * **Pane 3 (Bottom-Left):** Reconstructed Spatial Image.
   * **Pane 4 (Bottom-Right):** Spatial Error Map $|f - \hat{f}|$ or Point Spread Function (PSF).
2. **Control Panel & Interactivity Controls:**
   * **Dropdown:** Trajectory Type (Full Cartesian, Uniform Cartesian, Variable Density, Radial Spokes, Spiral).
   * **Slider 1:** Acceleration Factor $R$ ($1.0	imes$ to $8.0	imes$) or Spoke Count.
   * **Slider 2:** Low-Pass Cutoff Radius $r_c$.
   * **Checkboxes:** Enable Density Compensation, Enable Log Scaling, Show Error Heatmap.
3. **Multithreaded Pipeline Execution:**
   * Implement worker threads (`QThread` / `WorkerSignal`) so heavy matrix computations (gridding, 2D FFTs) run off the main GUI event loop, preventing UI freezing during slider drag events.

#### GUI Control Flow Architecture:
```
 [User Adjusts Slider / Dropdown]
                │
                ▼
      (GUI Event Triggered)
                │
                ▼
  [WorkerThread.run() Started]
                │
   ├── 1. Compute Mask / Trajectory
   ├── 2. Apply k-Space Subsampling
   ├── 3. Execute Gridding / 2D IFFT
   └── 4. Compute Image Metrics
                │
                ▼
  (Emit Qt Signal: data_ready)
                │
                ▼
 [GUI Main Thread Updates Canvases]
```

#### Deliverables for Week 4:
* `gui_app.py`: Fully functional interactive GUI application.
* Smooth dynamic updates ($> 20	ext{ FPS}$) during slider adjustments.
* Integrated multi-pane Matplotlib/Qt widget layout.

---

### Week 5: Apodization Filtering, Noise Simulation & Quantitative Metrics

#### Primary Objectives:
Implement $k$-space windowing filters to mitigate Gibbs ringing, model complex thermal scanner noise, and build an automated quantitative assessment engine.

#### Technical Tasks:
1. **Apodization (Windowing) Filter Module (`filters.py`):**
   * Implement 2D smoothing windows applied to $k$-space prior to IFFT reconstruction:
     * **2D Hamming Window:** $W(u,v) = \left(0.54 + 0.46 \cosrac{\pi u}{N}ight) \left(0.54 + 0.46 \cosrac{\pi v}{N}ight)$
     * **2D Gaussian Window:** $W(u,v) = \exp\left(-rac{u^2 + v^2}{2 \sigma^2}ight)$
   * Demonstrate trade-off between suppressing Gibbs edge ringing artifacts vs. introducing spatial blurring.
2. **Complex Thermal Noise Generator:**
   * Add Additive White Gaussian Noise (AWGN) directly to complex $k$-space channels:
     $$K_{	ext{noisy}}[u,v] = K[u,v] + \mathcal{N}(0, \sigma_n^2) + i \mathcal{N}(0, \sigma_n^2)$$
   * Evaluate SNR degradation in spatial reconstructions.
3. **Quantitative Metrics Engine (`metrics.py`):**
   * Implement real-time calculations for **MSE**, **PSNR** (in dB), and **SSIM**.
   * Display quantitative scores live in a dedicated GUI panel overlay.

#### Mathematical Formulation for Metrics:
```python
import numpy as np
from skimage.metrics import structural_similarity as ssim

class MetricsCalculator:
    @staticmethod
    def compute_mse(orig: np.ndarray, recon: np.ndarray) -> float:
        return np.mean(np.abs(orig - recon) ** 2)

    @staticmethod
    def compute_psnr(orig: np.ndarray, recon: np.ndarray) -> float:
        mse = MetricsCalculator.compute_mse(orig, recon)
        if mse == 0:
            return float('inf')
        max_pixel = np.max(orig)
        return 20 * np.log10(max_pixel / np.sqrt(mse))

    @staticmethod
    def compute_ssim(orig: np.ndarray, recon: np.ndarray) -> float:
        return ssim(orig, recon, data_range=orig.max() - orig.min())
```

#### Deliverables for Week 5:
* `filters.py`: 2D windowing filter library.
* `metrics.py`: Metrics calculation module integrated into GUI dashboard.
* Experimental evaluation chart plotting $PSNR$ vs. Acceleration Factor $R$ across different sampling trajectories.

---

### Week 6: System Profiling, Optimization, Testing & Final Delivery

#### Primary Objectives:
Optimize performance using vectorization, complete integration tests, conduct user acceptance testing, and compile project documentation for final BUET CSE 220 submission.

#### Technical Tasks:
1. **Code Optimization & Profiling:**
   * Profile execution bottlenecks using `cProfile` and `line_profiler`.
   * Replace explicit Python loops with NumPy vectorization or Numba JIT compilation for gridding loops.
2. **Robustness & Edge Case Handling:**
   * Handle non-square input images through zero-padding to nearest power-of-two square dimensions.
   * Ensure proper scaling for single-channel grayscale vs. multi-channel RGB inputs (convert RGB to luminance $Y = 0.299R + 0.587G + 0.114B$).
3. **Comprehensive Software Package & User Manual:**
   * Prepare complete clean directory repository structure.
   * Write comprehensive `README.md` containing setup instructions, mathematical theory, and usage guides.
   * Record a demonstration video showcasing interactive features.

#### Deliverables for Week 6:
* Complete, modular source code package.
* Final Project Report matching BUET CSE 219/220 academic submission criteria.
* Release executable / runnable script package.

---

## 5. Comprehensive Summary Matrix

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     WEEKLY MILESTONES & DELIVERABLES                    │
├──────┬───────────────────────────────┬──────────────────────────────────┤
│ Week │ Primary Focus                 │ Key Technical Deliverables       │
├──────┼───────────────────────────────┼──────────────────────────────────┤
│ W1   │ Core Fourier Math Engine      │ kspace_core.py, FFT/IFFT tests   │
│ W2   │ Cartesian Undersampling & PSF │ cartesian_sampling.py, PSF viewer│
│ W3   │ Non-Cartesian & Gridding      │ trajectories.py, gridding.py     │
│ W4   │ Interactive GUI Dashboard     │ gui_app.py (PyQt6 multi-threaded)│
│ W5   │ Filtering, Noise & Metrics    │ filters.py, metrics.py           │
│ W6   │ Optimization & Documentation  │ Final Release Package & Report   │
└──────┴───────────────────────────────┴──────────────────────────────────┘
```
