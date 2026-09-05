"""
MRI k-space simulator - the desktop app (Day 3).

    python app.py               # opens the window
    python app.py --selftest    # headless wiring check, exits 0 if sound

Layout: four panels (original | sampled k-space / reconstruction | error
heatmap), the control panel on the right, metrics in the status bar.
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
from PySide6.QtWidgets import (QApplication, QMainWindow, QSplitter,
                               QFileDialog, QMessageBox)

from mri.kspace_core import load_phantom, load_image
from mri.pipeline import reconstruct
from controls import Controls

DEBOUNCE_MS = 120


class Worker(QThread):
    """Runs one reconstruct() off the GUI thread and hands the dict back."""
    done = Signal(dict)

    def __init__(self, img, settings):
        super().__init__()
        self.img, self.settings = img, settings

    def run(self):
        try:
            self.done.emit(reconstruct(self.img, self.settings))
        except Exception as e:  # surface, never crash the GUI thread
            self.done.emit({"exception": str(e)})


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI k-space simulator")
        self.image = load_phantom()
        self.worker = None
        self.pending = None  # latest settings that arrived while busy

    # --- panels ---
        fig = Figure(figsize=(7.6, 7.2), facecolor="#1e1e1e")
        self.canvas = FigureCanvasQTAgg(fig)
        titles = ["original", "sampled k-space", "reconstruction", "|error|"]
        cmaps = ["gray", "magma", "gray", "magma"]
        self.artists = []
        for i, (t, cm) in enumerate(zip(titles, cmaps)):
            ax = fig.add_subplot(2, 2, i + 1)
            ax.set_title(t, color="white", fontsize=9)
            ax.axis("off")
            self.artists.append(ax.imshow(self.image, cmap=cm, vmin=0, vmax=1))
        fig.tight_layout()

        self.controls = Controls()
        self.controls.setMinimumWidth(280)

        split = QSplitter(Qt.Horizontal)  # drag the handle to resize either side
        split.addWidget(self.canvas)
        split.addWidget(self.controls)
        split.setStretchFactor(0, 1)
        split.setSizes([880, 340])
        self.setCentralWidget(split)

    # --- debounce: many slider ticks, one reconstruction ---
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(DEBOUNCE_MS)
        self.timer.timeout.connect(self._run)

        self.controls.settingsChanged.connect(lambda _s: self.timer.start())
        self.controls.loadImageClicked.connect(self._load_image)
        self.controls.phantom_btn.clicked.connect(self._load_phantom)
        self._run()

    def _run(self):
        if self.worker is not None and self.worker.isRunning():
            self.pending = self.controls.settings()  # remember, rerun on finish
            return
        self.statusBar().showMessage("reconstructing...")
        self.worker = Worker(self.image, self.controls.settings())
        self.worker.done.connect(self._show)
        self.worker.start()

    def _show(self, out: dict):
        if "exception" in out:
            self.statusBar().showMessage(f"error: {out['exception']}")
        else:
            self.artists[0].set_data(self.image)
            self.artists[1].set_data(out["kspace"])
            self.artists[2].set_data(out["recon"])
            self.artists[3].set_data(out["error"] / max(out["error"].max(), 1e-12))
            self.canvas.draw_idle()
            m = out["metrics"]
            psnr = "inf" if m["psnr"] == float("inf") else f"{m['psnr']:.2f} dB"
            self.statusBar().showMessage(
                f"PSNR {psnr}    SSIM {m['ssim']:.4f}    MSE {m['mse']:.5f}")
        if self.pending is not None:  # a slider moved while we were busy
            self.pending = None
            self._run()

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if not path:
            return
        try:
            self.image = load_image(path)
        except Exception as e:
            QMessageBox.warning(self, "Could not load image", str(e))
            return
        self._run()

    def _load_phantom(self):
        self.image = load_phantom()
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
                if win.pending is None and (msg.startswith("PSNR") or msg.startswith("error")):
                    return
                win.worker.wait(50)
            raise AssertionError(win.statusBar().currentMessage())

        wait_done()
        # R=1 is the identity: near-perfect PSNR (finite, ~319 dB) and SSIM 1
        assert "SSIM 1.0000" in win.statusBar().currentMessage(), \
            win.statusBar().currentMessage()

        before = win.artists[2].get_array().copy()
        win.controls.R.setValue(4)
        win.timer.stop(); win._run()  # skip the debounce in the test
        wait_done()
        assert not (before == win.artists[2].get_array()).all(), "R slider changed nothing"

        win.controls.sampling.setCurrentText("radial")
        win.timer.stop(); win._run()
        wait_done()
        assert "PSNR" in win.statusBar().currentMessage()

        print("SELFTEST PASSED - panels update, worker runs, metrics shown")
        QTimer.singleShot(0, app.quit)

    sys.exit(app.exec())
