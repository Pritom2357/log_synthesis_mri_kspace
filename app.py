"""
MRI k-space simulator - the desktop app (Day 3).

    python app.py               # opens the window
    python app.py --selftest    # headless wiring check, exits 0 if sound

Layout: four panels -- the k-space profile and the measured k-space side by
side on top, and below them the base reconstruction next to yours -- with the
control panel beside them. The base is the same measured k-space put through
the pipeline with every setting at its default, so it is not the hidden answer:
it is what this data looks like before you touch anything, which is the only
honest thing to judge an edit against. The reference image, the error map and
the difference metrics are all computed but never displayed: the point of the app is that the picture
comes out of the measurements, so showing the answer next to it would defeat
the exercise.
Reconstruction runs in a worker thread behind a short debounce, so dragging
a slider stays smooth.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.dates  # noqa: F401  MUST precede the Qt backend import:
# PySide6's import hook crashes on six.moves (pulled in via dateutil) unless
# that chain is fully loaded first. See scratch/gui_spike.py, Day 1.
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QMainWindow, QSplitter, QScrollArea,
                               QFileDialog, QMessageBox)

from mri.kspace_core import load_phantom, load_image
from mri.pipeline import reconstruct, DEFAULTS
from mri.raw_data import (find_datasets, describe, load_raw_kspace,
                          reference_image, raw_info, find_repetitions)
from controls import Controls, RangeDialog

PHANTOM = "Shepp-Logan phantom (simulated)"

DEBOUNCE_MS = 120


class Worker(QThread):
    """Runs one reconstruct() off the GUI thread and hands the dict back."""
    done = Signal(dict)

    def __init__(self, img, settings, kspace=None, reference=None, base_key=None):
        super().__init__()
        self.img, self.settings = img, settings
        self.kspace, self.reference = kspace, reference
        self.base_key = base_key   # not None means "also build the base image"

    def run(self):
        import time
        try:
            t0 = time.perf_counter()
            out = reconstruct(self.img, self.settings,
                              kspace=self.kspace, reference=self.reference)
            if self.base_key is not None:
                # The same measurement with every knob at its default. Computed
                # here rather than on the GUI thread, and only when the slice or
                # the file changes -- it cannot move when a slider does, so
                # rebuilding it on every drag would double the work for nothing.
                out["base"] = reconstruct(self.img, dict(DEFAULTS),
                                          kspace=self.kspace,
                                          reference=self.reference)["recon"]
                out["base_key"] = self.base_key
            out["_ms"] = int((time.perf_counter() - t0) * 1000)
            self.done.emit(out)
        except Exception as e:  # surface, never crash the GUI thread
            self.done.emit({"exception": str(e)})


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI k-space simulator")
        self.image = load_phantom()
        self.worker = None
        self.pending = None  # latest settings that arrived while busy
        self.elapsed_ms = 0
        self.raw_path = None   # set when a real scan is selected
        self.raw_note = ""     # scanner description for the status bar
        self.ref_note = ""     # where the comparison image came from
        self.base = None       # this slice with nothing switched on
        self._base_key = None  # what self.base was built from
        self._image_gen = 0    # bumped whenever a non-raw image is swapped in

    # --- panels ---
        # Four panels, in the order the data actually flows. The reference
        # image and the error map are both deliberately absent: the whole point
        # is that the picture is built from the measurements, and an error map
        # is the answer in disguise -- it says where the reconstruction is
        # wrong, which is only knowable from the image we are not allowed to
        # show. Both are still computed and still scored in the engine -- see
        # out["reference"], out["error"] and out["metrics"].
        # The base panel is a different thing entirely and is allowed: it is
        # this same measured k-space with every setting left alone, so it gives
        # an edit something to be judged against without revealing the answer.
        # The profile and k-space share the top row; the two reconstructions
        # share the bottom one, which is the taller of the two because they are
        # what the rest of the window exists to produce.
        # Constrained layout, not tight_layout: the profile panel carries axis
        # labels that the image panels do not, and its title is rewritten on
        # every reconstruction. tight_layout is computed once at build time, so
        # it knew nothing about those labels and let them collide with the
        # panel below. Constrained layout re-solves on every draw.
        fig = Figure(figsize=(9.0, 8.4), facecolor="#1e1e1e", layout="constrained")
        self.canvas = FigureCanvasQTAgg(fig)
        # Panel 0 is a 1D cut through k-space rather than another picture. A
        # profile shows things a 2D view hides: where the peak really sits, how
        # fast the signal falls away from it, and exactly which samples are
        # missing. The blank white square it replaced said none of that.
        grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.6])
        self.axes, self.artists = [], []
        ax0 = fig.add_subplot(grid[0, 0])
        ax0.set_facecolor("#111111")
        ax0.tick_params(colors="#999999", labelsize=7)
        for spine in ax0.spines.values():
            spine.set_color("#555555")
        ax0.set_title("k-space profile", color="white", fontsize=9)
        ax0.set_xlabel("k-space column, relative to the array centre",
                       color="#999999", fontsize=7)
        self.profile_line, = ax0.plot([], [], color="#f0a030", lw=1.1)
        self.centre_mark = ax0.axvline(0, color="#4da6ff", lw=1.0, ls="--")
        self.peak_mark = ax0.axvline(0, color="#2ecc71", lw=1.0)
        self.axes.append(ax0)
        self.artists.append(None)   # panel 0 is a line, not an image

        for t, cm, cell in (("measured k-space", "magma", grid[0, 1]),
                            ("base  (nothing switched on)", "gray", grid[1, 0]),
                            ("my reconstruction", "gray", grid[1, 1])):
            ax = fig.add_subplot(cell)
            ax.set_title(t, color="white", fontsize=9)
            ax.axis("off")
            self.axes.append(ax)
            self.artists.append(ax.imshow(self.image, cmap=cm, vmin=0, vmax=1))

        self.controls = Controls()
        self._fill_sources()
        panel = QScrollArea()  # the panel is taller than most windows now
        panel.setWidget(self.controls)
        panel.setWidgetResizable(True)
        panel.setMinimumWidth(360)

        split = QSplitter(Qt.Horizontal)  # drag the handle to resize either side
        split.addWidget(self.canvas)
        split.addWidget(panel)
        split.setStretchFactor(0, 1)
        split.setSizes([900, 400])
        self.setCentralWidget(split)

    # --- debounce: many slider ticks, one reconstruction ---
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(DEBOUNCE_MS)
        self.timer.timeout.connect(self._run)

        self.controls.source.currentTextChanged.connect(self._source_changed)
        self.controls.settingsChanged.connect(lambda _s: self.timer.start())
        self.controls.displayChanged.connect(self._apply_display_range)
        self.controls.wl_auto.clicked.connect(self._fit_window)
        self.controls.loadImageClicked.connect(self._load_image)
        self.controls.phantom_btn.clicked.connect(self._load_phantom)

        act = QAction("Slider ranges...", self)
        act.setShortcut("Ctrl+,")
        act.triggered.connect(self._edit_ranges)
        self.menuBar().addMenu("Settings").addAction(act)
        self._run()

    def _edit_ranges(self):
        """Let the user retune how far each slider travels."""
        dlg = RangeDialog(self.controls, self)
        if dlg.exec():
            dlg.apply()

    def _fit_window(self):
        """
        Set the display window from the picture instead of by hand.

        Everything brighter than the brightest empty-corner pixel is taken as
        tissue, and the window is its 1st to 99th percentile: the whole
        greyscale is spent on anatomy, nothing is clipped, and the empty
        background gets none of it. This is the auto-window every viewer has,
        and it exists here because the right numbers change from slice to
        slice -- a window that suits one cut clips the cortex on the next.
        """
        import numpy as np
        r = np.asarray(self.artists[3].get_array(), dtype=np.float64)
        ceiling = float(r[:32, :32].max())
        tissue = r[r > ceiling]
        if tissue.size < 100:            # nothing to fit to, leave it alone
            return
        lo, hi = np.percentile(tissue, [1, 99])
        c = self.controls
        c.wl_width.setValue(int(round((hi - lo)*100)))
        c.wl_centre.setValue(int(round((lo + hi)/2*100)))

    def _apply_display_range(self, vmin: float, vmax: float):
        """
        Brightness and contrast on the reconstruction, and nothing else.

        This is set_clim on the artist, which is what a radiologist's
        window/level control does: the pixels are untouched and the engine is
        not re-run, so the metrics do not move. Narrowing the window around a
        thin band of grey is how soft tissue is read, and that is a viewing
        decision rather than a reconstruction one.
        """
        # Yours only. The base panel keeps the full range on purpose, so that
        # window/level is itself one of the changes you can see the effect of.
        self.artists[3].set_clim(vmin, vmax)
        full = abs(vmin) < 1e-9 and abs(vmax - 1.0) < 1e-9
        self.axes[3].set_title(
            "augmented reconstruction" if full else
            f"augmented reconstruction    display window {vmin:.2f} to {vmax:.2f}",
            color="white", fontsize=9)
        self.canvas.draw_idle()

    def _fill_sources(self):
        """Put the phantom plus every raw file we can find into the dropdown."""
        self.datasets = find_datasets()
        box = self.controls.source
        box.blockSignals(True)
        box.clear()
        box.addItem(PHANTOM)
        for f in self.datasets:
            box.addItem(f.name)
        box.blockSignals(False)

    def _source_changed(self, name: str):
        """Switch between the simulated phantom and a real measured scan."""
        if name == PHANTOM:
            self.raw_path, self.raw_note, self.ref_note = None, "", ""
            self.image = load_phantom()
        else:
            match = [f for f in self.datasets if f.name == name]
            if not match:
                return
            self.raw_path = str(match[0])
            try:
                info = raw_info(self.raw_path)
                n_reps = len(find_repetitions(self.raw_path))
                self.raw_note = describe(self.raw_path) + (
                    f" | {n_reps} repetitions" if n_reps > 1 else " | single acquisition")
            except Exception as e:
                QMessageBox.warning(self, "Could not read that file", str(e))
                self.raw_path = None
                return
            # a real file decides its own slice range
            self.controls.slice_idx.blockSignals(True)
            self.controls.slice_idx.setRange(0, max(0, info["n_slices"] - 1))
            self.controls.slice_idx.setValue(info["n_slices"] // 2)
            self.controls.slice_idx.blockSignals(False)
        self.controls._update_visibility()
        self._run()

    def _run(self):
        if self.worker is not None and self.worker.isRunning():
            self.pending = self.controls.settings()  # remember, rerun on finish
            return
        self.statusBar().showMessage("reconstructing...")
        settings = self.controls.settings()
        # The base only depends on which slice of which file we are looking at,
        # so it is rebuilt when that changes and reused every other time.
        key = ((self.raw_path, self.controls.slice_idx.value()) if self.raw_path
               else ("image", self._image_gen))
        base_key = key if key != self._base_key else None

        if self.raw_path is None:
            # simulation: the forward transform invents k-space from a picture
            self.worker = Worker(self.image, settings, base_key=base_key)
        else:
            # real data: start from what the scanner measured, no forward transform
            z = self.controls.slice_idx.value()
            try:
                k = load_raw_kspace(self.raw_path, z)
                # The reference is the best image we can honestly claim to know.
                # Where repeated acquisitions exist it is their average, which
                # is genuinely cleaner than any single scan and is NOT derived
                # from the file being reconstructed -- so the starting point is
                # no longer perfect and there is something to improve.
                ref, self.ref_note = reference_image(self.raw_path, z)
            except Exception as e:
                self.statusBar().showMessage(f"error reading slice {z}: {e}")
                return
            self.worker = Worker(None, settings, kspace=k, reference=ref,
                                 base_key=base_key)

        self.worker.done.connect(self._show)
        self.worker.start()

    def _show(self, out: dict):
        """
        Paint one result. Wrapped because Qt swallows exceptions raised inside a
        slot: a typo in here used to leave the window stuck on "reconstructing"
        with nothing in the console to say why. Surfacing it in the status bar
        costs four lines and turns a silent hang into a readable message.
        """
        try:
            self._paint(out)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.statusBar().showMessage(f"display error: {type(e).__name__}: {e}")
            if self.pending is not None:
                self.pending = None

    def _paint(self, out: dict):
        self.elapsed_ms = out.pop("_ms", 0)
        if "exception" in out:
            self.statusBar().showMessage(f"error: {out['exception']}")
        else:
            # Reduced FOV and k-space cropping change the output size, so the
            # panels get a new extent rather than just new data when it changes.
            if "base" in out:
                self.base = out["base"]
                self._base_key = out["base_key"]

            self._draw_profile(out["kspace"], out["mask"])

            panels = [out["kspace"],
                      self.base if self.base is not None else out["recon"],
                      out["recon"]]
            for art, data in zip(self.artists[1:], panels):
                if art.get_array().shape != data.shape:
                    art.set_extent((-0.5, data.shape[1] - 0.5, data.shape[0] - 0.5, -0.5))
                art.set_data(data)
            self.canvas.draw_idle()
            # Difference metrics are computed and kept in out["metrics"], but
            # they are not shown: they only exist by comparing against the
            # hidden reference, which is exactly what this view is not allowed
            # to reveal.
            mask = out["mask"]
            kept = (f"{100*float(mask.mean()):.0f}% of k-space measured"
                    if mask is not None else "non-Cartesian trajectory")
            where = self.raw_note if self.raw_note else "simulated phantom"
            if self.ref_note:
                where += f"  |  ref: {self.ref_note}"
            edits = self._active_edits()
            if out.get("recon_used") == "zero_filled" and                     self.controls.settings()["recon"] == "cs":
                edits += "    [compressed sensing needs single-coil data: zero-filled used]"
            self.statusBar().showMessage(
                f"{kept}    [{out['recon'].shape[0]}x{out['recon'].shape[1]},"
                f" {self.elapsed_ms} ms]{edits}    |  {where}")

        # A setting changed while this reconstruction was still running, so the
        # picture on screen is already out of date. Run once more with the
        # latest values. Without this the last drag of a slider is silently
        # dropped and the display quietly disagrees with the controls.
        if self.pending is not None:
            self.pending = None
            self._run()

    def _draw_profile(self, kdisp, mask):
        """
        One horizontal cut through k-space, taken along the row that holds the
        peak. Two vertical lines say where the peak is and where the middle of
        the array is; on real scanner data those are several samples apart.
        """
        import numpy as np

        peak_row, peak_col = np.unravel_index(int(np.argmax(kdisp)), kdisp.shape)
        profile = kdisp[peak_row]
        x = np.arange(len(profile)) - kdisp.shape[1] // 2

        self.profile_line.set_data(x, profile)
        self.centre_mark.set_xdata([0, 0])
        offset = peak_col - kdisp.shape[1] // 2
        self.peak_mark.set_xdata([offset, offset])

        ax = self.axes[0]
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(0, 1.05)
        # Kept short on purpose: a long title overflows the axes on a narrow
        # window and clips at the figure edge. The sampled percentage already
        # appears in the status bar, so it does not need repeating here.
        row_off = peak_row - kdisp.shape[0] // 2
        ax.set_title(f"k-space profile  |  peak {row_off:+d},{offset:+d} from centre",
                     color="white", fontsize=9)

    def _active_edits(self) -> str:
        """Names whatever is switched on, so a surprising picture is explainable."""
        s = self.controls.settings()
        on = []
        if s["dc_scale"] != 1.0:        on.append(f"DC x{s['dc_scale']:.2f}")
        if s["spike"]:                  on.append(f"spike +{s['spike']}")
        if s["erase"]:                  on.append(f"erased {s['erase']}")
        if s["low_pass"]:               on.append(f"low pass {s['low_pass']}")
        if s["high_pass"]:              on.append(f"high pass {s['high_pass']}")
        if s["partial_fourier"] < 1.0:  on.append(f"partial Fourier {s['partial_fourier']:.2f}")
        if s["window"] != "none":       on.append(s["window"])
        if s["noise"]:                  on.append(f"noise {s['noise']:.2f}")
        if s["upscale"] > 1:            on.append(f"x{s['upscale']} {s['interp']}")
        return "    " + ", ".join(on) if on else ""

    def _load_image(self):
        """Open either a raw k-space file or an ordinary picture."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load raw k-space or image", "",
            "All supported (*.h5 *.png *.jpg *.jpeg *.bmp *.tif);;"
            "Raw k-space (*.h5);;Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if not path:
            return

        if path.lower().endswith(".h5"):
            try:
                raw_info(path) # fails fast if it is not readable k-space
            except Exception as e:
                QMessageBox.warning(self, "Not a readable k-space file", str(e))
                return
            p = Path(path)
            if p not in self.datasets:
                self.datasets.append(p)
                self.controls.source.addItem(p.name)
            self.controls.source.setCurrentText(p.name) # triggers _source_changed
            return

        try:
            self.image = load_image(path)
        except Exception as e:
            QMessageBox.warning(self, "Could not load image", str(e))
            return
        self.raw_path, self.raw_note = None, ""
        self._image_gen += 1
        self.controls.source.setCurrentIndex(0)
        self._run()

    def _load_phantom(self):
        self.controls.source.setCurrentIndex(0)
        self.raw_path, self.raw_note = None, ""
        self.image = load_phantom()
        self._image_gen += 1
        self._run()


if __name__ == "__main__":
    selftest = "--selftest" in sys.argv
    if selftest:
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = QApplication(sys.argv)
    win = Main()
    win.resize(1240, 760)
    win.show()

    if selftest:
        # Drive the controls without a human; the worker is async, so pump the
        # event loop until each reconstruction lands before asserting on it.
        def wait_done():
            # done: the queued signal has landed and _show has painted metrics
            for _ in range(400):
                app.processEvents()
                msg = win.statusBar().currentMessage()
                if win.pending is None and msg and not msg.startswith("reconstructing"):
                    return
                win.worker.wait(50)
            raise AssertionError(win.statusBar().currentMessage())

        wait_done()
        # Full sampling: every line measured, and no quality numbers on screen.
        msg = win.statusBar().currentMessage()
        assert "100% of k-space measured" in msg, msg
        for leaked in ("PSNR", "SSIM", "MSE"):
            assert leaked not in msg, f"{leaked} must not appear in the GUI: {msg}"

        before = win.artists[3].get_array().copy()   # augmented reconstruction
        win.controls.rate.setValue(50)
        win.timer.stop(); win._run()  # skip the debounce in the test
        wait_done()
        assert not (before == win.artists[3].get_array()).all(), "rate slider changed nothing"

        win.controls.sampling.setCurrentText("radial")
        win.timer.stop(); win._run(); wait_done()
        assert "trajectory" in win.statusBar().currentMessage()

        win.controls.sampling.setCurrentText("spiral")
        win.timer.stop(); win._run(); wait_done()

        # compressed sensing still runs, it is just not the default
        win.controls.sampling.setCurrentText("cartesian")
        win.controls.mask.setCurrentText("random")
        win.controls.recon.setCurrentText("compressed sensing")
        win.timer.stop(); win._run(); wait_done()
        win.controls.recon.setCurrentText("zero-filled")
        win.controls.mask.setCurrentText("nyquist")
        win.timer.stop(); win._run(); wait_done()

        # four panels, and none of them leaks the hidden reference: neither the
        # reference itself nor the error map derived from it
        assert len(win.artists) == 4, f"expected 4 panels, got {len(win.artists)}"
        assert win.artists[0] is None, "panel 0 is a profile plot, not an image"
        xs, ys = win.profile_line.get_data()
        assert len(xs) > 0 and float(max(ys)) > 0.5, "the profile should have been drawn"

        # every k-space edit runs and visibly changes the reconstruction
        untouched = win.artists[2].get_array().copy()   # the base panel
        base = win.artists[3].get_array().copy()
        for name, widget, value in (("dc", win.controls.dc, 40),
                                    ("spike", win.controls.spike, 20),
                                    ("erase", win.controls.erase, 40),
                                    ("low pass", win.controls.low_pass, 40),
                                    ("high pass", win.controls.high_pass, 20),
                                    ("partial fourier", win.controls.pf, 60)):
            widget.setValue(value)
            win.timer.stop(); win._run(); wait_done()
            assert not (base == win.artists[3].get_array()).all(), f"{name} changed nothing"
            assert (untouched == win.artists[2].get_array()).all(), \
                f"{name} moved the base panel, which must stay put"
            widget.setValue(100 if widget in (win.controls.dc, win.controls.pf) else 0)
            win.timer.stop(); win._run(); wait_done()

        # upscaling enlarges the output, and the method row appears with it
        c = win.controls
        assert not c._forms["interp"].isRowVisible(c.rows["interp"])
        c.upscale.setCurrentText("2")
        assert c._forms["interp"].isRowVisible(c.rows["interp"])
        win.timer.stop(); win._run(); wait_done()
        assert win.artists[3].get_array().shape == (512, 512), "upscale x2 should give 512"
        assert win.artists[2].get_array().shape == (256, 256), "the base does not upscale"
        c.upscale.setCurrentText("1")
        win.timer.stop(); win._run(); wait_done()

        # rows hide themselves when the mode cannot use them
        vis = lambda n: c._forms[n].isRowVisible(c.rows[n])
        assert vis("rate") and not vis("spokes")
        c.sampling.setCurrentText("radial")
        assert vis("spokes") and not vis("rate")
        c.sampling.setCurrentText("spiral")
        assert vis("arms") and not vis("spokes")
        assert not vis("sigma")
        c.window.setCurrentText("gaussian")
        assert vis("sigma"), "sigma must appear for the gaussian window"
        c.window.setCurrentText("none")
        c.sampling.setCurrentText("cartesian")
        win.timer.stop(); win._run(); wait_done()

        # the range dialog retunes how far a slider travels, within the caps
        from controls import LIMITS
        dlg = RangeDialog(c, win)
        assert set(dlg.spins) == set(LIMITS), "every retunable slider needs a row"
        assert c.sigma.maximum() == 100
        lo_spin, hi_spin = dlg.spins["sigma"]
        assert hi_spin.maximum() == LIMITS["sigma"][2], "the spin box must cap at the limit"
        hi_spin.setValue(10_000)                    # far past the cap
        assert hi_spin.value() == LIMITS["sigma"][2], "Qt should have clamped it"
        hi_spin.setValue(300); dlg.apply()
        assert c.sigma.maximum() == 300, "the dialog should have widened sigma"
        c.sigma.setValue(280)                       # only reachable once widened
        win.timer.stop(); win._run(); wait_done()
        assert c.settings()["sigma"] == 2.8, c.settings()["sigma"]

        # an inverted range is ignored rather than applied: a slider that cannot
        # move looks exactly like a broken one
        lo_spin.setValue(300); hi_spin.setValue(300); dlg.apply()
        assert c.sigma.minimum() < c.sigma.maximum(), "an empty range must be refused"

        dlg.restore()
        assert (c.sigma.minimum(), c.sigma.maximum()) == c.default_ranges["sigma"]
        win.timer.stop(); win._run(); wait_done()

        # window/level moves the display only: the clim changes, the pixels do not
        pixels = win.artists[3].get_array().copy()
        c.wl_centre.setValue(30); c.wl_width.setValue(40)
        app.processEvents()
        lo, hi = win.artists[3].get_clim()
        assert abs(lo - 0.10) < 1e-9 and abs(hi - 0.50) < 1e-9, (lo, hi)
        assert (win.artists[3].get_array() == pixels).all(), \
            "window/level must not touch a single pixel"
        assert "display window" in win.axes[3].get_title()
        assert win.artists[2].get_clim() == (0.0, 1.0), "the base keeps the full range"
        c.wl_reset.click(); app.processEvents()
        assert win.artists[3].get_clim() == (0.0, 1.0), win.artists[3].get_clim()
        assert win.axes[3].get_title() == "augmented reconstruction"

        # fit-to-tissue measures the picture: the window must land inside the
        # tissue range and must not clip it
        import numpy as np
        c.wl_auto.click(); app.processEvents()
        lo, hi = win.artists[3].get_clim()
        r = np.asarray(win.artists[3].get_array()); tissue = r[r > r[:32, :32].max()]
        assert 0.0 <= lo < hi < 1.0, (lo, hi)   # narrower than full, never inverted
        assert (tissue > hi + 0.01).mean() < 0.02, "auto window must not clip tissue"
        assert (win.artists[3].get_array() == pixels).all()
        c.wl_reset.click(); app.processEvents()

        # the base is cached per slice, not rebuilt on every knob: it must not
        # move for a setting change, and must move for a slice change
        assert win._base_key is not None and win.base is not None
        seen = win._base_key
        c.noise.setValue(10)
        win.timer.stop(); win._run(); wait_done()
        assert win._base_key == seen, "a slider must not rebuild the base"
        c.noise.setValue(0)
        win.timer.stop(); win._run(); wait_done()

        print("SELFTEST PASSED - 4 panels, no leaked metrics, k-space edits, "
              "base panel, slider ranges and window/level wired")
        QTimer.singleShot(0, app.quit)

    sys.exit(app.exec())
