"""
MRI k-space simulator - the desktop app (Day 3).

    python app.py               # opens the window
    python app.py --selftest    # headless wiring check, exits 0 if sound

Layout: four panels -- the scan and sampling facts and the measured k-space side by
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

    def __init__(self, img, settings, kspace=None, reference=None, base_key=None,
                 repeats=None, gen=0):
        super().__init__()
        self.gen = gen             # which run this is; stale results are dropped
        self.img, self.settings = img, settings
        self.kspace, self.reference, self.repeats = kspace, reference, repeats
        self.base_key = base_key   # not None means "also build the base image"

    def run(self):
        import time
        try:
            t0 = time.perf_counter()
            out = reconstruct(self.img, self.settings, kspace=self.kspace,
                              reference=self.reference, repeats=self.repeats)
            if self.base_key is not None:
                # The same measurement with every knob at its default. Computed
                # here rather than on the GUI thread, and only when the slice or
                # the file changes -- it cannot move when a slider does, so
                # rebuilding it on every drag would double the work for nothing.
                base = reconstruct(self.img, dict(DEFAULTS), kspace=self.kspace,
                                   reference=self.reference)
                out["base"] = base["recon"]
                out["base_scores"] = (base["metrics"], base["sampling"])
                out["base_key"] = self.base_key
            out["_ms"] = int((time.perf_counter() - t0) * 1000)
            out["_gen"] = self.gen
            self.done.emit(out)
        except Exception as e:  # surface, never crash the GUI thread
            self.done.emit({"exception": str(e), "_gen": self.gen})


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI k-space simulator")
        self.image = load_phantom()
        self.worker = None
        self.pending = None  # latest settings that arrived while busy
        self._gen = 0          # number of the newest run started
        self.elapsed_ms = 0
        self.raw_path = None   # set when a real scan is selected
        self.raw_note = ""     # scanner description for the status bar
        self.ref_note = ""     # where the comparison image came from
        self.base = None       # this slice with nothing switched on
        self.base_scores = None  # (metrics, sampling) of the base, for the deltas
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
        # The facts and the k-space share the top row; the two reconstructions
        # share the bottom one, which is the taller of the two because they are
        # what the rest of the window exists to produce.
        # Constrained layout re-solves on every draw, so titles rewritten on
        # every reconstruction never collide with the panel below.
        fig = Figure(figsize=(9.0, 8.4), facecolor="#1e1e1e", layout="constrained")
        self.canvas = FigureCanvasQTAgg(fig)
        # Panel 0 is the scorecard: five numbers, each next to how it moved
        # against the base, so every knob can be judged the moment it moves.
        # Top row: scorecard | k-space | scan details. Bottom row: base | yours.
        # Six columns so three top cells and two bottom cells share one grid.
        grid = fig.add_gridspec(2, 6, height_ratios=[1.35, 1.6])
        self.axes, self.artists = [], []
        ax0 = fig.add_subplot(grid[0, 0:2])
        self._build_scorecard(ax0)
        self.axes.append(ax0)
        self.artists.append(None)   # panel 0 is a table, not an image

        for t, cm, cell in (("measured k-space", "magma", grid[0, 2:4]),
                            ("base  (nothing switched on)", "gray", grid[1, 0:3]),
                            ("my reconstruction", "gray", grid[1, 3:6])):
            ax = fig.add_subplot(cell)
            ax.set_title(t, color="white", fontsize=9)
            ax.axis("off")
            self.axes.append(ax)
            self.artists.append(ax.imshow(self.image, cmap=cm, vmin=0, vmax=1))

        # The scan details sit right of the k-space. They replace the status
        # bar, which ran off the bottom of the window where nobody read it.
        self._build_details(fig.add_subplot(grid[0, 4:6]))

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
            self.controls.set_repetitions(1)
        else:
            match = [f for f in self.datasets if f.name == name]
            if not match:
                return
            self.raw_path = str(match[0])
            try:
                info = raw_info(self.raw_path)
                n_reps = len(find_repetitions(self.raw_path))
                self.controls.set_repetitions(n_reps)
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
        self._set_state("reconstructing...")
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
                self._set_state(f"error reading slice {z}: {e}")
                return
            # Other scans of the same slice, loaded only when averaging asks
            # for them: this scan first, then its siblings in file order.
            repeats = None
            if settings["average"] > 1:
                others = [p for p in find_repetitions(self.raw_path)
                          if Path(p).resolve() != Path(self.raw_path).resolve()]
                try:
                    repeats = [load_raw_kspace(p, z)
                               for p in others[:settings["average"] - 1]]
                except Exception as e:
                    self._set_state(f"error reading a repetition: {e}")
                    return
            self.worker = Worker(None, settings, kspace=k, reference=ref,
                                 base_key=base_key, repeats=repeats)

        self._gen += 1
        self.worker.gen = self._gen
        self.worker.done.connect(self._show)
        self.worker.finished.connect(self._next)
        self.worker.start()

    def _next(self):
        """
        A setting changed while this reconstruction was running, so the picture
        on screen is already out of date: run once more with the latest values.

        This hangs off QThread.finished, not off the result arriving. The
        result is emitted from inside run(), so at that moment the thread is
        still alive; restarting from there found it "busy", re-queued the
        settings, and nothing ever started them -- the last slider move was
        silently dropped and the picture disagreed with the controls.
        """
        if self.pending is not None:
            self.pending = None
            self._run()

    def _show(self, out: dict):
        """
        Paint one result. Wrapped because Qt swallows exceptions raised inside a
        slot: a typo in here used to leave the window stuck on "reconstructing"
        with nothing in the console to say why. Surfacing it in the details panel
        costs four lines and turns a silent hang into a readable message.

        A result from an older run is dropped. Two runs can overlap (a source
        change starts one directly and one through the debounce timer), and the
        older result used to land after the newer run started: it painted an
        out-of-date picture and marked the window "ready" while work was still
        going on.
        """
        if out.get("_gen") != self._gen:
            return
        try:
            self._paint(out)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._set_state(f"display error: {type(e).__name__}: {e}")

    def _paint(self, out: dict):
        self.elapsed_ms = out.pop("_ms", 0)
        if "exception" in out:
            self._set_state(f"error: {out['exception']}")
        else:
            # Reduced FOV and k-space cropping change the output size, so the
            # panels get a new extent rather than just new data when it changes.
            if "base" in out:
                self.base = out["base"]
                self._base_key = out["base_key"]
                self.base_scores = out["base_scores"]

            panels = [out["kspace"],
                      self.base if self.base is not None else out["recon"],
                      out["recon"]]
            for art, data in zip(self.artists[1:], panels):
                if art.get_array().shape != data.shape:
                    art.set_extent((-0.5, data.shape[1] - 0.5, data.shape[0] - 0.5, -0.5))
                art.set_data(data)
            self.canvas.draw_idle()
            self._update_scorecard(out)
            self.canvas.draw_idle()
            self._details = self._detail_rows(out)
            self._set_state(f"ready  ({self.elapsed_ms} ms)")

    # ------
    # Scorecard
    # ------

    # name, how to print it, and whether a bigger number is the better one
    _SCORES = (
        ("Aliasing loss", lambda v: f"{v:.1f} %", False),
        ("SNR", lambda v: f"{v:.1f}", True),
        ("SSIM", lambda v: f"{v:.3f}", True),
        ("PSNR", lambda v: f"{v:.2f} dB", True),
    )
    _DELTA = {"Aliasing loss": "{:+.1f} %",
              "SNR": "{:+.1f}", "SSIM": "{:+.3f}", "PSNR": "{:+.2f} dB"}

    def _build_scorecard(self, ax):
        """A rounded box, a header, and one row per score at fixed positions."""
        from matplotlib.patches import FancyBboxPatch
        ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title("metrics", color="white", fontsize=9)
        ax.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.9,
                                    boxstyle="round,pad=0.0,rounding_size=0.04",
                                    facecolor="#262626", edgecolor="#444444", lw=1.0))
        head = dict(color="#8a8a8a", fontsize=9, va="center")
        ax.text(0.10, 0.85, "metric", ha="left", **head)
        ax.text(0.62, 0.85, "now", ha="right", **head)
        ax.text(0.90, 0.85, "vs base", ha="right", **head)
        ax.plot([0.08, 0.92], [0.785, 0.785], color="#444444", lw=0.8)
        self.score_cells = {}
        for i, (name, _, _) in enumerate(self._SCORES):
            y = 0.66 - i*0.16
            ax.text(0.10, y, name, ha="left", va="center", color="#cfcfcf", fontsize=11)
            now = ax.text(0.62, y, "", ha="right", va="center", color="white",
                          fontsize=12, family="monospace")
            delta = ax.text(0.90, y, "", ha="right", va="center", color="#8a8a8a",
                            fontsize=10, family="monospace")
            self.score_cells[name] = (now, delta)
        self.info_text = ""

    def _update_scorecard(self, out: dict):
        """
        Fills the rows. The deltas are against the base panel, so green and
        red say directly whether a change helped.

            Aliasing loss energy of the skipped samples / total (Parseval)
            SNR           tissue / background noise, from the picture alone
            SSIM, PSNR    against the reference: the average of the other
                          repetitions for a scan, the image itself for the phantom
        """
        import math

        def scores(metrics, sampling):
            return {"Aliasing loss": 100*sampling["alias_energy"],
                    "SNR": metrics["snr"], "SSIM": metrics["ssim"], "PSNR": metrics["psnr"]}

        now = scores(out["metrics"], out["sampling"])
        base = scores(*self.base_scores) if self.base_scores else now
        lines = []
        for name, fmt, higher_better in self._SCORES:
            v, b = now[name], base[name]
            shown = "\u221e" if math.isinf(v) else fmt(v)   # a perfect match
            if math.isinf(v) or math.isinf(b):
                d, colour, dtext = 0.0, "#8a8a8a", "\u2014" if math.isinf(v) != math.isinf(b) else "\u00b10"
            else:
                d = v - b
                tiny = abs(d) < 5e-4*max(1.0, abs(b))
                better = (d > 0) == higher_better
                colour = "#8a8a8a" if tiny else ("#4cd07d" if better else "#ff6b6b")
                dtext = "\u00b10" if tiny else self._DELTA[name].format(d)
            cell_now, cell_delta = self.score_cells[name]
            cell_now.set_text(shown)
            cell_delta.set_text(dtext); cell_delta.set_color(colour)
            lines.append(f"{name}  {shown}  {dtext}")
        self.info_text = "\n".join(lines)

    # ------
    # Scan details (right of the k-space)
    # ------

    def _build_details(self, ax):
        """A box like the scorecard's: grey keys on the left, values beside them."""
        from matplotlib.patches import FancyBboxPatch
        ax.axis("off")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title("scan details", color="white", fontsize=9)
        ax.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.9,
                                    boxstyle="round,pad=0.0,rounding_size=0.04",
                                    facecolor="#262626", edgecolor="#444444", lw=1.0))
        self.detail_ax = ax
        self.detail_artists = []   # one key and one value text per line, redrawn each time
        self._details = []
        self.state = ""
        self.details_text = ""

    def _detail_rows(self, out: dict) -> list:
        """
        (key, value) pairs the controls do not already show: what was measured,
        what alignment found, which scanner, and what the scores compare with.
        """
        rows = [("measured", "{} of {} rows\n{} of {} columns".format(
            *out["sampling"]["rows"], *out["sampling"]["cols"]))]
        if out["averaged"] > 1:
            # A phase offset near pi is exactly what cancels a naive average,
            # so it is shown: it explains why alignment matters.
            al = out["alignment"]
            rows.append(("averaged", f"{out['averaged']} scans"))
            if al["phases"]:
                rows.append(("phase offsets", ", ".join(f"{p:+.2f}" for p in al["phases"]) + " rad"))
                rows.append(("shifts", "  ".join(f"({d[0]},{d[1]})" for d in al["shifts"]) + " px"))
        if self.raw_note:
            parts = [p.strip() for p in self.raw_note.split("|")]
            rows.append(("scanner", parts[0]))
            rows.append(("", " \u00b7 ".join(parts[1:4])))
            rows.append(("", " \u00b7 ".join(parts[4:])))
        else:
            rows.append(("source", "simulated phantom"))
        if self.ref_note:
            rows.append(("reference", self.ref_note))
        return rows

    def _set_state(self, state: str):
        """Shows reconstructing / ready / an error as the first detail row."""
        import textwrap
        self.state = state
        keys, vals = [], []
        for key, value in [("status", state)] + self._details:
            wrapped = []
            for part in str(value).split("\n"):
                wrapped += textwrap.wrap(part, 34) or [""]
            keys += [key] + [""]*(len(wrapped)-1)
            vals += wrapped
        # Each line at a fixed height, so keys and values can never drift apart.
        for artist in self.detail_artists:
            artist.remove()
        self.detail_artists = []
        step = min(0.075, 0.8/max(len(vals), 1))
        for i, (key, value) in enumerate(zip(keys, vals)):
            y = 0.86 - i*step
            for x, text, colour in ((0.09, key, "#8a8a8a"), (0.36, value, "#dddddd")):
                self.detail_artists.append(self.detail_ax.text(
                    x, y, text, ha="left", va="top", color=colour, fontsize=9))
        self.details_text = "\n".join(f"{k} {v}" for k, v in zip(keys, vals))
        self.canvas.draw_idle()

    def _active_edits(self) -> str:
        """Names whatever is switched on, so a surprising picture is explainable."""
        s = self.controls.settings()
        on = []
        if s["dc_scale"] != 1.0:        on.append(f"DC x{s['dc_scale']:.2f}")
        if s["spike"]:                  on.append(f"spike +{s['spike']}")
        if s["erase"]:                  on.append(f"erased {s['erase']}")
        if s["partial_fourier"] < 1.0:  on.append(f"partial Fourier {s['partial_fourier']:.2f}")
        if s["window"] != "none":       on.append(s["window"])
        if s["noise"]:                  on.append(f"noise {s['noise']:.2f}")
        if s["noise_filter"]:           on.append(f"noise floor x{s['noise_filter']:.2f}")
        if s["sharpen"]:                on.append(f"sharpen {s['sharpen']:.2f}")
        if s["keep"] != "both":         on.append(f"{s['keep']} only")
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
                msg = win.state
                if win.pending is None and msg and not msg.startswith("reconstructing"):
                    return
                win.worker.wait(50)
            raise AssertionError(win.state)

        wait_done()
        # Full sampling: every line measured, and all five scores on the card.
        assert "256 of 256 rows" in win.details_text and "256 of 256 columns" in win.details_text, \
            win.details_text
        assert win.state.startswith("ready"), win.state
        assert win.statusBar().isHidden() or not win.statusBar().currentMessage(), "no status bar"
        for name in ("Aliasing loss", "SNR", "SSIM", "PSNR"):
            assert name in win.info_text, win.info_text
        for gone in ("Acceleration", "Scan time"):
            assert gone not in win.info_text, win.info_text
        assert "0.0 %" in win.info_text

        before = win.artists[3].get_array().copy()   # augmented reconstruction
        win.controls.rate.setValue(50)
        win.timer.stop(); win._run()  # skip the debounce in the test
        wait_done()
        assert not (before == win.artists[3].get_array()).all(), "rate slider changed nothing"

        win.timer.stop(); win._run(); wait_done()

        # four panels, and none of them leaks the hidden reference: neither the
        # reference itself nor the error map derived from it
        assert len(win.artists) == 4, f"expected 4 panels, got {len(win.artists)}"
        assert win.artists[0] is None, "panel 0 is the scorecard, not an image"

        # the card moves with the knob: half the lines aliases, and it shows
        # as a change against the base
        win.controls.rate.setValue(50)
        win.timer.stop(); win._run(); wait_done()
        now, delta = win.score_cells["Aliasing loss"]
        assert float(now.get_text().split()[0]) > 20 and delta.get_color() == "#ff6b6b"
        now, delta = win.score_cells["SSIM"]
        assert delta.get_text().startswith("-") and delta.get_color() == "#ff6b6b"
        win.controls.rate.setValue(100)
        win.timer.stop(); win._run(); wait_done()
        assert win.score_cells["Aliasing loss"][1].get_text() == "\u00b10"
        win.controls.rate.setValue(50)
        win.timer.stop(); win._run(); wait_done()

        # every k-space edit runs and visibly changes the reconstruction
        untouched = win.artists[2].get_array().copy()   # the base panel
        base = win.artists[3].get_array().copy()
        for name, widget, value in (("dc", win.controls.dc, 40),
                                    ("spike", win.controls.spike, 20),
                                    ("erase", win.controls.erase, 40),
                                    ("partial fourier", win.controls.pf, 60),
                                    ("add noise", win.controls.noise, 20),
                                    ("sharpen", win.controls.sharpen, 50)):
            widget.setValue(value)
            win.timer.stop(); win._run(); wait_done()
            assert not (base == win.artists[3].get_array()).all(), f"{name} changed nothing"
            assert (untouched == win.artists[2].get_array()).all(), \
                f"{name} moved the base panel, which must stay put"
            widget.setValue(100 if widget in (win.controls.dc, win.controls.pf) else 0)
            win.timer.stop(); win._run(); wait_done()

        # phase only keeps the outlines; the dropdown changes the picture
        win.controls.keep.setCurrentText("phase only")
        win.timer.stop(); win._run(); wait_done()
        assert not (base == win.artists[3].get_array()).all(), "phase only changed nothing"
        win.controls.keep.setCurrentText("magnitude and phase")
        win.timer.stop(); win._run(); wait_done()

        # erasing the centre must not brighten the rest of the k-space panel
        import numpy as np
        kbefore = np.asarray(win.artists[1].get_array()).copy()
        win.controls.erase.setValue(40)
        win.timer.stop(); win._run(); wait_done()
        kafter = np.asarray(win.artists[1].get_array())
        assert np.abs(kafter[:40, :40] - kbefore[:40, :40]).max() < 1e-12, \
            "the k-space display scale moved when the centre was erased"
        win.controls.erase.setValue(0)
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

        # every section is open at once: no accordion left to click through
        from PySide6.QtWidgets import QGroupBox
        groups = c.findChildren(QGroupBox)
        assert len(groups) == 6 and all(g.isVisible() for g in groups), len(groups)

        # rows hide themselves when the mode cannot use them
        vis = lambda n: c._forms[n].isRowVisible(c.rows[n])
        assert vis("rate") and vis("noise"), "the phantom offers rate and added noise"
        assert not vis("average"), "one phantom, nothing to average"
        assert not vis("sigma")
        c.window.setCurrentText("gaussian")
        assert vis("sigma"), "sigma must appear for the gaussian window"
        c.window.setCurrentText("none")
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

        # real data with repetitions: averaging appears, runs, and says what it found
        from mri.raw_data import find_repetitions
        multi = [f for f in win.datasets if len(find_repetitions(str(f))) > 1]
        if multi:
            c.rate.setValue(100)
            c.source.setCurrentText(multi[0].name); wait_done()
            assert vis("average") and not vis("noise"), "real scans average, never add noise"
            single = win.artists[3].get_array().copy()
            c.average.setCurrentIndex(c.average.count() - 1)
            win.timer.stop(); win._run(); wait_done()
            msg = win.details_text
            assert f"averaged {c.average.count()} scans" in msg and "phase offsets" in msg, msg
            assert "0.3 T" in msg and "reference" in msg, msg
            # averaging is a real denoiser: the card must say SNR went up
            now, delta = win.score_cells["SNR"]
            assert delta.get_color() == "#4cd07d", (now.get_text(), delta.get_text())
            assert not (single == win.artists[3].get_array()).all()
            assert (win.base == win.artists[2].get_array()).all(), "the base stays one scan"
            c.average.setCurrentIndex(0)
            c.source.setCurrentIndex(0); wait_done()
            print(f"  averaging checked on {multi[0].name}")
        else:
            print("  no repeated scans on disk, averaging check skipped")

        print("SELFTEST PASSED - 4 panels, scorecard, k-space edits, denoising, "
              "fixed k-space scale, base panel, slider ranges and window/level wired")
        QTimer.singleShot(0, app.quit)

    sys.exit(app.exec())
