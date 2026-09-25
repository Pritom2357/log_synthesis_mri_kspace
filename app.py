"""
MRI k-space simulator - the desktop app.

    python app.py               # opens the window
    python app.py --selftest    # headless wiring check, exits 0 if sound

Layout: the scorecard and the scan details on top; the measured k-space and
your reconstruction below, taking most of the height; the control panel on
the right. Ticking "compare" splits your reconstruction with a draggable
divider: the base on the left, yours on the right. The base is the same
measured k-space with every setting at its default, so it is not the hidden
answer: it is what this data looks like before you touch anything, which is
the only honest thing to judge an edit against.

Why it is fast enough to animate. A slider tick used to cost about 700 ms:
re-reading the scan and three repetitions from disk (420 ms), a 120 ms
debounce, and 130 ms of matplotlib drawing, around 40 ms of actual work. Now
the scan is read once per file and slice and kept, there is no debounce (a
newer setting simply supersedes an older one), and images are drawn as plain
8-bit Qt images, which takes about a millisecond. What is left is the
reconstruction itself.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.dates  # noqa: F401  MUST precede any PySide6 import:
# PySide6's import hook crashes on six.moves (pulled in via dateutil) unless
# that chain is fully loaded first. See scratch/gui_spike.py, Day 1.
from matplotlib import colormaps

from PySide6.QtCore import Qt, QTimer, QThread, Signal, QRect, QPoint
from PySide6.QtGui import QAction, QImage, QPixmap, QPainter, QColor, QPen, QFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QSplitter, QScrollArea,
                               QFileDialog, QMessageBox, QWidget, QLabel, QFrame,
                               QGridLayout, QVBoxLayout, QHBoxLayout)

from mri.kspace_core import load_phantom, load_image
from mri.pipeline import reconstruct, DEFAULTS
from mri.denoise import average_repetitions
from mri.raw_data import (find_datasets, describe, load_raw_kspace,
                          reference_image, raw_info, find_repetitions)
from controls import Controls, RangeDialog

PHANTOM = "Shepp-Logan phantom (simulated)"

BG, CARD, EDGE = "#1e1e1e", "#262626", "#444444"
GREEN, RED, GREY = "#4cd07d", "#ff6b6b", "#8a8a8a"


def _lut(name: str) -> list[int]:
    """A 256-entry Qt colour table from a matplotlib colormap."""
    rgb = (colormaps[name](np.linspace(0, 1, 256))[:, :3]*255).astype(int)
    return [0xFF000000 | (r << 16) | (g << 8) | b for r, g, b in rgb]


# ------
# Display widgets
# ------

class ImageView(QWidget):
    """
    One picture: a title and an array drawn to fit, aspect kept.

    The array (floats in 0..1) is turned into an 8-bit indexed QImage once per
    update and scaled by Qt when painted -- no plotting library in the loop.
    The method names mirror matplotlib's (set_data, set_clim, get_title...),
    so the rest of the app reads the same as before.
    """
    clicked = Signal(int, int)   # (row, col) of the sample under the mouse

    def __init__(self, title: str, cmap: str = "gray", smooth: bool = True):
        super().__init__()
        self._title = QLabel(title)
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setStyleSheet("color: white; font-size: 12px;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 4)
        self._head = QHBoxLayout()           # the title, plus any header controls
        self._head.addWidget(self._title, 1)
        lay.addLayout(self._head)
        lay.addStretch(1)
        self._lut, self._smooth = _lut(cmap), smooth
        self._data = np.zeros((2, 2))
        self._clim = (0.0, 1.0)
        self._pix = QPixmap()
        self._target = QRect()
        self._compare = None     # another ImageView drawn left of the divider
        self._split = 0.5        # divider position, fraction of the image width
        self.setMinimumSize(160, 160)

    # --- matplotlib-style accessors ---
    def set_data(self, a: np.ndarray):
        self._data = np.asarray(a, dtype=np.float64)
        self._render()

    def get_array(self) -> np.ndarray:
        return self._data

    def set_clim(self, lo: float, hi: float):
        self._clim = (float(lo), float(hi))
        self._render()

    def get_clim(self) -> tuple[float, float]:
        return self._clim

    def set_title(self, text: str, **_):
        self._title.setText(text)

    def get_title(self) -> str:
        return self._title.text()

    # --- drawing ---
    def _render(self):
        lo, hi = self._clim
        u8 = np.clip((self._data - lo)/max(hi - lo, 1e-12), 0, 1)*255
        u8 = np.ascontiguousarray(u8.astype(np.uint8))
        h, w = u8.shape
        img = QImage(u8.data, w, h, w, QImage.Format_Indexed8)
        img.setColorTable(self._lut)
        self._pix = QPixmap.fromImage(img)   # copies, so u8 may go
        self.update()

    def _fit(self) -> QRect:
        top = max(self._title.geometry().bottom(),
                  self._head.geometry().bottom()) + 4
        area = QRect(0, top, self.width(), self.height() - top)
        h, w = self._data.shape
        scale = min(area.width()/w, area.height()/h)
        pw, ph = int(w*scale), int(h*scale)
        return QRect(area.x() + (area.width() - pw)//2, area.y() + (area.height() - ph)//2, pw, ph)

    def add_header(self, widget: QWidget):
        """Put a control on the title line, to the right of the title."""
        self._head.addWidget(widget)

    # --- compare: another picture left of a draggable divider ---
    def set_compare(self, other: "ImageView | None"):
        """Show `other` left of the divider and this one right of it; None to stop."""
        self._compare = other
        self.update()

    def set_split(self, fraction: float):
        self._split = min(1.0, max(0.0, float(fraction)))
        self.update()

    def get_split(self) -> float:
        return self._split

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(BG))
        if not self._pix.isNull():
            t = self._target = self._fit()
            p.setRenderHint(QPainter.SmoothPixmapTransform, self._smooth)
            other = self._compare._pix if self._compare is not None else QPixmap()
            if other.isNull():
                p.drawPixmap(t, self._pix)
            else:
                # Both pictures are scaled into the same rectangle, so they
                # line up pixel for pixel even when yours was upscaled.
                x = t.x() + int(round(self._split*t.width()))
                p.setClipRect(QRect(t.x(), t.y(), x - t.x(), t.height()))
                p.drawPixmap(t, other)
                p.setClipRect(QRect(x, t.y(), t.right() - x + 1, t.height()))
                p.drawPixmap(t, self._pix)
                p.setClipping(False)
                self._draw_divider(p, x, t)
        p.end()

    def _draw_divider(self, p: QPainter, x: int, t: QRect):
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(QPen(QColor("white"), 2))
        p.drawLine(x, t.top(), x, t.bottom())
        r = 20
        centre = QPoint(x, t.center().y())
        p.setBrush(QColor("white")); p.setPen(Qt.NoPen)
        p.drawEllipse(centre, r, r)
        p.setPen(QColor("#222222"))
        f = QFont(); f.setPixelSize(16); f.setBold(True); p.setFont(f)
        p.drawText(QRect(x - r, centre.y() - r, 2*r, 2*r), Qt.AlignCenter, "\u2039 \u203a")
        for label, dx, align in (("base", -8, Qt.AlignRight), ("yours", 8, Qt.AlignLeft)):
            p.setPen(QColor("white"))
            box = QRect(x + dx - (120 if dx < 0 else 0), t.top() + 6, 120, 20)
            p.drawText(box, align | Qt.AlignVCenter, label)

    def index_at(self, x: float, y: float) -> tuple[int, int] | None:
        """The (row, col) of the sample drawn at widget position (x, y)."""
        t = self._target if not self._target.isNull() else self._fit()
        if not t.contains(int(x), int(y)):
            return None
        h, w = self._data.shape
        return (min(h - 1, int((y - t.y())/t.height()*h)),
                min(w - 1, int((x - t.x())/t.width()*w)))

    def mousePressEvent(self, e):
        if self._compare is not None:          # comparing: a press moves the divider
            self._drag(e.position().x())
            return
        hit = self.index_at(e.position().x(), e.position().y())
        if hit is not None and e.button() == Qt.LeftButton:
            self.clicked.emit(*hit)

    def mouseMoveEvent(self, e):
        if self._compare is not None and e.buttons() & Qt.LeftButton:
            self._drag(e.position().x())

    def _drag(self, x: float):
        t = self._target if not self._target.isNull() else self._fit()
        if t.width() > 0:
            self.set_split((x - t.x())/t.width())


class Cell(QLabel):
    """A scorecard entry that remembers its colour, for the self-test."""
    def __init__(self, size: int, colour: str = "white", align=Qt.AlignRight):
        super().__init__("")
        self._size = size
        self.setAlignment(align | Qt.AlignVCenter)
        self.set_color(colour)

    def set_color(self, colour: str):
        self._colour = colour
        self.setStyleSheet(f"color: {colour}; font-size: {self._size}px; "
                           "font-family: Consolas, monospace; background: transparent;")

    def get_color(self) -> str:
        return self._colour

    def get_text(self) -> str:
        return self.text()


def _card(title: str) -> tuple[QWidget, QFrame]:
    """A titled, rounded panel like the image views, for text content."""
    outer = QWidget()
    lay = QVBoxLayout(outer); lay.setContentsMargins(4, 2, 4, 4)
    head = QLabel(title); head.setAlignment(Qt.AlignCenter)
    head.setStyleSheet("color: white; font-size: 12px;")
    frame = QFrame(); frame.setObjectName("card")
    frame.setStyleSheet(f"QFrame#card {{ background: {CARD}; border: 1px solid {EDGE};"
                        " border-radius: 10px; }")
    lay.addWidget(head); lay.addWidget(frame, 1)
    return outer, frame


# ------
# The reconstruction worker
# ------

class Worker(QThread):
    """Runs one reconstruct() off the GUI thread and hands the dict back."""
    done = Signal(dict)

    def __init__(self, img, settings, kspace=None, reference=None, base_key=None,
                 repeats=None, gen=0, extra=None):
        super().__init__()
        self.extra = extra or {}   # facts computed before the run, copied into the result
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
            out.update(self.extra)
            if self.base_key is not None:
                # The same measurement with every knob at its default. Only
                # rebuilt when the slice or the file changes -- it cannot move
                # when a slider does.
                base = reconstruct(self.img, dict(DEFAULTS),
                                   kspace=getattr(self, "base_kspace", self.kspace),
                                   reference=self.reference)
                out["base"] = base["recon"]
                out["base_scores"] = (base["metrics"], base["sampling"])
                out["base_key"] = self.base_key
            out["_ms"] = int((time.perf_counter() - t0) * 1000)
            out["_gen"] = self.gen
            self.done.emit(out)
        except Exception as e:  # surface, never crash the GUI thread
            self.done.emit({"exception": str(e), "_gen": self.gen})


# ------
# The window
# ------

class Main(QMainWindow):
    # name, how to print it, and whether a bigger number is the better one
    _SCORES = (
        ("Aliasing loss", lambda v: f"{v:.1f} %", False),
        ("SNR", lambda v: f"{v:.1f}", True),
        ("SSIM", lambda v: f"{v:.3f}", True),
        ("PSNR", lambda v: f"{v:.2f} dB", True),
    )
    _DELTA = {"Aliasing loss": "{:+.1f} %",
              "SNR": "{:+.1f}", "SSIM": "{:+.3f}", "PSNR": "{:+.2f} dB"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MRI k-space simulator")
        self.image = load_phantom()
        self.worker = None
        self.pending = None      # latest settings that arrived while busy
        self._gen = 0            # number of the newest run started
        self.elapsed_ms = 0
        self.raw_path = None     # set when a real scan is selected
        self.raw_note = ""       # scanner description
        self.ref_note = ""       # where the comparison image came from
        self.base = None         # this slice with nothing switched on
        self.base_scores = None  # (metrics, sampling) of the base, for the deltas
        self._base_key = None    # what self.base was built from
        self._image_gen = 0      # bumped whenever a non-raw image is swapped in
        self._scans = {}         # (path, slice) -> (k-space, reference, note)
        self._repeats = {}       # (path, slice) -> k-space of the other repetitions
        self._averages = {}      # (path, slice, n) -> (aligned average, what alignment found)
        self._details = []
        self.state = ""
        self.details_text = ""
        self.info_text = ""

        # --- panels ---
        # The reference image and the error map are deliberately absent: the
        # picture has to come out of the measurements. Both are still computed
        # in the engine, and the scorecard compares against the reference.
        score_box, score_frame = _card("metrics")
        self._build_scorecard(score_frame)
        detail_box, detail_frame = _card("scan details")
        self._build_details(detail_frame)
        kview = ImageView("measured k-space", "magma", smooth=False)
        # The base is no longer a panel of its own: it is drawn left of the
        # divider when "compare" is on. The view still holds its picture.
        bview = ImageView("base  (nothing switched on)")
        rview = ImageView("my reconstruction")
        kview.clicked.connect(self._kspace_clicked)
        for v, data in ((kview, np.zeros((256, 256))), (bview, self.image), (rview, self.image)):
            v.set_data(data)
        self.axes = [score_box, kview, bview, rview]
        self.artists = [None, kview, bview, rview]

        top, bottom = QHBoxLayout(), QHBoxLayout()
        top.addWidget(score_box, 1); top.addWidget(detail_box, 1)
        bottom.addWidget(kview, 1); bottom.addWidget(rview, 1)
        left = QWidget(); left.setStyleSheet(f"background: {BG};")
        grid = QVBoxLayout(left); grid.setContentsMargins(6, 6, 6, 6)
        grid.addLayout(top, 28); grid.addLayout(bottom, 72)

        self.controls = Controls()
        self._fill_sources()
        # The compare switch sits on your reconstruction's title line, next to
        # the picture it changes, rather than down in the control panel.
        self.controls.compare.setStyleSheet("color: white;")
        rview.add_header(self.controls.compare)
        panel = QScrollArea()
        panel.setWidget(self.controls)
        panel.setWidgetResizable(True)
        panel.setMinimumWidth(380)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(panel)
        # About 62% pictures, 38% controls: the controls need room for their
        # labels and values, the pictures get the rest when the window grows.
        split.setStretchFactor(0, 62)
        split.setStretchFactor(1, 38)
        self._split, self._split_set = split, False
        self.setCentralWidget(split)

        # No debounce: every change runs at once, and a change that arrives
        # while a run is busy is kept as `pending` and run when it finishes, so
        # only the newest setting ever waits.
        self.controls.source.currentTextChanged.connect(self._source_changed)
        self.controls.settingsChanged.connect(lambda _s: self._run())
        self.controls.displayChanged.connect(self._apply_display_range)
        self.controls.compareToggled.connect(self._compare)
        self.controls.wl_auto.clicked.connect(self._fit_window)
        self.controls.loadImageClicked.connect(self._load_image)
        self.controls.phantom_btn.clicked.connect(self._load_phantom)

        act = QAction("Slider ranges...", self)
        act.setShortcut("Ctrl+,")
        act.triggered.connect(self._edit_ranges)
        self.menuBar().addMenu("Settings").addAction(act)
        self._run()

    # ------
    # Scorecard
    # ------

    def showEvent(self, e):
        """Split 62 : 38 from the real width, once the window has one."""
        super().showEvent(e)
        if not self._split_set:
            w = max(self._split.width(), 1)
            self._split.setSizes([int(w*0.62), w - int(w*0.62)])
            self._split_set = True

    def _build_scorecard(self, frame: QFrame):
        g = QGridLayout(frame)
        g.setContentsMargins(18, 12, 18, 12)
        for col, (text, align) in enumerate((("metric", Qt.AlignLeft), ("now", Qt.AlignRight),
                                              ("vs base", Qt.AlignRight))):
            g.addWidget(Cell(12, GREY, align), 0, col)
            g.itemAtPosition(0, col).widget().setText(text)
        line = QFrame(); line.setFixedHeight(1); line.setStyleSheet(f"background: {EDGE};")
        g.addWidget(line, 1, 0, 1, 3)
        self.score_cells = {}
        for i, (name, _, _) in enumerate(self._SCORES):
            label = Cell(15, "#cfcfcf", Qt.AlignLeft); label.setText(name)
            now, delta = Cell(17), Cell(13, GREY)
            g.addWidget(label, i + 2, 0); g.addWidget(now, i + 2, 1); g.addWidget(delta, i + 2, 2)
            self.score_cells[name] = (now, delta)

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
            shown = "∞" if math.isinf(v) else fmt(v)   # a perfect match
            if math.isinf(v) or math.isinf(b):
                colour, dtext = GREY, "—" if math.isinf(v) != math.isinf(b) else "±0"
            else:
                d = v - b
                tiny = abs(d) < 5e-4*max(1.0, abs(b))
                better = (d > 0) == higher_better
                colour = GREY if tiny else (GREEN if better else RED)
                dtext = "±0" if tiny else self._DELTA[name].format(d)
            cell_now, cell_delta = self.score_cells[name]
            cell_now.setText(shown)
            cell_delta.setText(dtext); cell_delta.set_color(colour)
            lines.append(f"{name}  {shown}  {dtext}")
        self.info_text = "\n".join(lines)

    # ------
    # Scan details
    # ------

    def _build_details(self, frame: QFrame):
        lay = QVBoxLayout(frame); lay.setContentsMargins(16, 12, 16, 12)
        self.details_label = QLabel(); self.details_label.setWordWrap(True)
        self.details_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.details_label.setStyleSheet("color: #dddddd; font-size: 12px; background: transparent;")
        lay.addWidget(self.details_label)

    def _detail_rows(self, out: dict) -> list:
        """
        (key, value) pairs the controls do not already show: what was measured,
        what alignment found, which scanner, and what the scores compare with.
        """
        rows = [("measured", "{} of {} rows<br>{} of {} columns".format(
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
            rows.append(("scanner", parts[0] + "<br>" + " · ".join(parts[1:4])
                         + "<br>" + " · ".join(parts[4:])))
        else:
            rows.append(("source", "simulated phantom"))
        if self.ref_note:
            rows.append(("reference", self.ref_note))
        return rows

    def _set_state(self, state: str):
        """Shows reconstructing / ready / an error as the first detail row."""
        self.state = state
        rows = [("status", state)] + self._details
        cells = "".join(f"<tr><td style='color:{GREY}; padding:3px 14px 3px 0'>{k}</td>"
                        f"<td style='padding:3px 0'>{v}</td></tr>" for k, v in rows)
        self.details_label.setText(f"<table>{cells}</table>")
        self.details_text = "\n".join(f"{k} {v.replace('<br>', ' ')}" for k, v in rows)

    # ------
    # Clicks, display window, sources
    # ------

    def _kspace_clicked(self, row: int, col: int):
        """A click on the k-space panel places a spike or a patch, if a mode is on."""
        self.controls.add_point(row, col)

    def _edit_ranges(self):
        """Let the user retune how far each slider travels."""
        dlg = RangeDialog(self.controls, self)
        if dlg.exec():
            dlg.apply()

    def _compare(self, on: bool):
        """Split your reconstruction with the base, or go back to yours alone."""
        self.artists[3].set_compare(self.artists[2] if on else None)
        self._apply_display_range(*self.controls.display_range())

    def _fit_window(self):
        """
        Set the display window from the picture instead of by hand.

        Everything brighter than the brightest empty-corner pixel is taken as
        tissue, and the window is its 1st to 99th percentile: the whole
        greyscale is spent on anatomy, nothing is clipped, and the empty
        background gets none of it.
        """
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
        Brightness and contrast on your reconstruction, and nothing else: the
        pixels are untouched and the engine is not re-run. The base keeps the
        full range, so window/level is itself a change you can see.
        """
        self.artists[3].set_clim(vmin, vmax)
        full = abs(vmin) < 1e-9 and abs(vmax - 1.0) < 1e-9
        title = "augmented reconstruction" if full else \
            f"augmented reconstruction    display window {vmin:.2f} to {vmax:.2f}"
        if self.controls.compare.isChecked():
            title = "base  \u2039 \u203a  " + title
        self.axes[3].set_title(title)

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

    # ------
    # Running a reconstruction
    # ------

    def _scan(self, z: int):
        """
        This slice's k-space and reference, read from disk once and kept.

        Reading them on every run was most of the lag: 96 ms for the scan and
        327 ms for the reference, which reads all three repetitions.
        """
        key = (self.raw_path, z)
        if key not in self._scans:
            k = load_raw_kspace(self.raw_path, z)
            # The reference is the best image we can honestly claim to know:
            # the average of repeated acquisitions where they exist.
            ref, note = reference_image(self.raw_path, z)
            self._scans[key] = (k, ref, note)
        return self._scans[key]

    def _other_repetitions(self, z: int) -> list:
        """The other scans of this slice, in file order; read once and kept."""
        key = (self.raw_path, z)
        if key not in self._repeats:
            others = [p for p in find_repetitions(self.raw_path)
                      if Path(p).resolve() != Path(self.raw_path).resolve()]
            self._repeats[key] = [load_raw_kspace(p, z) for p in others]
        return self._repeats[key]

    def _averaged(self, z: int, n: int):
        """
        The aligned average of n scans of this slice, computed once and kept.
        It depends only on the file, the slice and n -- never on a slider --
        so redoing the alignment on every tick (65 ms) was pure waste.
        """
        key = (self.raw_path, z, n)
        if key not in self._averages:
            k, _, _ = self._scan(z)
            self._averages[key] = average_repetitions([k] + self._other_repetitions(z)[:n - 1])
        return self._averages[key]

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
                k, ref, self.ref_note = self._scan(z)
                n, extra = settings["average"], None
                if n > 1:
                    # Averaging is the pipeline's first step, so handing it the
                    # cached average with average=1 gives the identical result.
                    avg, alignment = self._averaged(z, n)
                    extra = {"averaged": n, "alignment": alignment}
                    settings = {**settings, "average": 1}
            except Exception as e:
                self._set_state(f"error reading slice {z}: {e}")
                return
            base_k = k
            self.worker = Worker(None, settings, kspace=avg if n > 1 else k, reference=ref,
                                 base_key=base_key, extra=extra)
            if base_key is not None and n > 1:
                self.worker.base_kspace = base_k

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
        settings, and nothing ever started them.
        """
        if self.pending is not None:
            self.pending = None
            self._run()

    def _show(self, out: dict):
        """
        Paint one result. Wrapped because Qt swallows exceptions raised inside a
        slot. A result from an older run is dropped: runs can overlap, and an
        older result landing late would paint an out-of-date picture.
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
            return
        if "base" in out:
            self.base = out["base"]
            self._base_key = out["base_key"]
            self.base_scores = out["base_scores"]
        panels = [out["kspace"],
                  self.base if self.base is not None else out["recon"],
                  out["recon"]]
        for view, data in zip(self.artists[1:], panels):
            view.set_data(data)
        self._update_scorecard(out)
        self._details = self._detail_rows(out)
        self._set_state(f"ready  ({self.elapsed_ms} ms)")

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
    win.resize(1400, 820)
    win.show()

    if selftest:
        import time
        # Drive the controls without a human; the worker is async, so pump the
        # event loop until each reconstruction lands before asserting on it.
        def wait_done():
            for _ in range(600):
                app.processEvents()
                msg = win.state
                if win.pending is None and msg and not msg.startswith("reconstructing"):
                    return
                if win.worker is not None:
                    win.worker.wait(50)
            raise AssertionError(win.state)

        def go():
            win._run(); wait_done()

        c = win.controls
        wait_done()
        # Full sampling: every line measured, and all the scores on the card.
        assert "256 of 256 rows" in win.details_text and "256 of 256 columns" in win.details_text, \
            win.details_text
        assert win.state.startswith("ready"), win.state
        assert win.statusBar().isHidden() or not win.statusBar().currentMessage(), "no status bar"
        for name in ("Aliasing loss", "SNR", "SSIM", "PSNR"):
            assert name in win.info_text, win.info_text
        assert "0.0 %" in win.info_text
        assert not hasattr(win, "canvas"), "no matplotlib canvas in the display any more"

        before = win.artists[3].get_array().copy()   # your reconstruction
        c.rate.setValue(50); wait_done()             # no debounce: runs by itself
        assert not (before == win.artists[3].get_array()).all(), "rate slider changed nothing"

        # four panels, and none of them leaks the hidden reference
        assert len(win.artists) == 4 and win.artists[0] is None

        # the card moves with the knob: half the lines aliases, shown against the base
        now, delta = win.score_cells["Aliasing loss"]
        assert float(now.get_text().split()[0]) > 20 and delta.get_color() == RED
        now, delta = win.score_cells["SSIM"]
        assert delta.get_text().startswith("-") and delta.get_color() == RED
        c.rate.setValue(100); wait_done()
        assert win.score_cells["Aliasing loss"][1].get_text() == "±0"
        c.rate.setValue(50); wait_done()

        # every k-space edit runs and visibly changes the reconstruction
        untouched = win.artists[2].get_array().copy()   # the base panel
        base = win.artists[3].get_array().copy()
        for name, widget, value in (("dc", c.dc, 40), ("spike", c.spike, 20),
                                    ("erase", c.erase, 40), ("partial fourier", c.pf, 60),
                                    ("add noise", c.noise, 20), ("sharpen", c.sharpen, 50),
                                    ("low pass", c.low_pass, 30), ("high pass", c.high_pass, 10)):
            widget.setValue(value); wait_done()
            assert not (base == win.artists[3].get_array()).all(), f"{name} changed nothing"
            assert (untouched == win.artists[2].get_array()).all(), \
                f"{name} moved the base panel, which must stay put"
            widget.setValue(100 if widget in (c.dc, c.pf) else 0); wait_done()

        # phase only keeps the outlines; the dropdown changes the picture
        c.keep.setCurrentText("phase only"); wait_done()
        assert not (base == win.artists[3].get_array()).all(), "phase only changed nothing"
        c.keep.setCurrentText("magnitude and phase"); wait_done()

        # erasing the centre must not brighten the rest of the k-space panel
        kbefore = np.asarray(win.artists[1].get_array()).copy()
        c.erase.setValue(40); wait_done()
        kafter = np.asarray(win.artists[1].get_array())
        assert np.abs(kafter[:40, :40] - kbefore[:40, :40]).max() < 1e-12, \
            "the k-space display scale moved when the centre was erased"
        c.erase.setValue(0); wait_done()
        c.rate.setValue(100); wait_done()

        # --- the acquisition animation ---
        # Scan progress reveals k-space line by line in the chosen order, and
        # Play sweeps it on a timer until the scan is complete.
        kv = win.artists[1]
        for order in ("linear", "centric"):
            c.acq_order.setCurrentText(order)
            c.acq.setValue(250); wait_done()
            shown = kv.get_array() > 0
            if order == "linear":
                assert shown[:60].any() and not shown[70:].any(), "linear reads from the top"
            else:
                mid = shown.shape[0]//2
                assert shown[mid-20:mid+20].any() and not shown[:90].any(), "centric reads from DC out"
        c.acq_order.setCurrentText("linear")
        c.acq.setValue(0); wait_done()
        c.play_btn.setChecked(True)
        assert c.play_btn.text() == "Pause"
        seen, t0 = set(), time.perf_counter()
        while c.play_btn.isChecked() and time.perf_counter() - t0 < 30:
            app.processEvents(); seen.add(round(c.settings()["acquired"], 1))
            time.sleep(0.005)
        wait_done()
        assert not c.play_btn.isChecked() and c.acq.value() == c.acq.maximum(), "play runs to the end"
        assert len(seen) >= 8, f"play should pass through many stages, saw {sorted(seen)}"
        assert c.play_btn.text() == "Play"
        c.rewind_btn.click(); wait_done()
        assert c.acq.value() == 0
        c.acq.setValue(c.acq.maximum()); wait_done()

        # --- spikes and patches by clicking on the k-space panel ---
        assert c.click_mode() is None
        before = win.artists[3].get_array().copy()
        kv.clicked.emit(60, 90)                      # no mode: a click does nothing
        wait_done()
        assert c.spike_points == []
        c.spike_btn.setChecked(True)
        kv.clicked.emit(60, 90); wait_done()
        kv.clicked.emit(200, 40); wait_done()
        assert c.spike_points == [(60, 90), (200, 40)], c.spike_points
        assert kv.get_array()[60, 90] == kv.get_array().max(), "the spike is the brightest sample"
        assert not (before == win.artists[3].get_array()).all(), "a spike must stripe the image"
        c.patch_btn.setChecked(True)
        assert not c.spike_btn.isChecked(), "the two click modes exclude each other"
        kv.clicked.emit(128, 128); wait_done()
        assert c.patch_points == [(128, 128, c.patch.value())]
        assert kv.get_array()[128, 128] == 0, "the patch zeroes the centre"
        c.spike_undo.click(); wait_done()
        assert c.spike_points == [(60, 90)]
        c.spike_clear.click(); c.patch_clear.click(); wait_done()
        assert c.spike_points == [] and c.patch_points == []
        c.patch_btn.setChecked(False)
        assert (before == win.artists[3].get_array()).all(), "cleared edits restore the picture"

        # a widget position maps to the sample drawn there
        app.processEvents()
        t = kv._fit()
        r, cidx = kv.index_at(t.x() + t.width()*0.5, t.y() + t.height()*0.25)
        assert abs(r - 64) <= 1 and abs(cidx - 128) <= 1, (r, cidx)
        assert kv.index_at(t.x() - 5, t.y() - 5) is None

        # --- compare: base left of a draggable divider, yours right ---
        rv, bv = win.artists[3], win.artists[2]
        assert not bv.isVisible(), "the base is no longer a panel of its own"
        c.rate.setValue(50); wait_done()             # make yours differ from the base
        app.processEvents()
        alone = rv.grab().toImage()
        c.compare.setChecked(True); app.processEvents()
        assert rv._compare is bv and win.axes[3].get_title().startswith("base")
        split = rv.grab().toImage()
        t = rv._fit()
        left_px = QPoint(t.x() + t.width()//4, t.y() + t.height()//2)
        right_px = QPoint(t.x() + 3*t.width()//4, t.y() + t.height()//2)
        assert split.pixel(right_px) == alone.pixel(right_px), "right of the divider is yours"
        assert split.pixel(left_px) != alone.pixel(left_px), "left of the divider is the base"
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QMouseEvent
        press = QMouseEvent(QMouseEvent.MouseButtonPress, QPointF(t.x() + t.width()*0.2, t.center().y()),
                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        app.sendEvent(rv, press)
        assert abs(rv.get_split() - 0.2) < 0.02, rv.get_split()
        before_spikes = list(c.spike_points)
        c.spike_btn.setChecked(True); app.sendEvent(rv, press); c.spike_btn.setChecked(False)
        assert c.spike_points == before_spikes, "dragging the divider never places a spike"
        c.compare.setChecked(False); app.processEvents()
        assert rv._compare is None and not win.axes[3].get_title().startswith("base")
        c.rate.setValue(100); wait_done()

        # --- k-space colours stay put while a scan plays ---
        full_k = kv.get_array().copy()
        for frac in (100, 400, 800):
            c.acq.setValue(frac); wait_done()
            part = kv.get_array(); shown = part > 0
            assert np.abs(part[shown] - full_k[shown]).max() < 1e-12, "a sample changed colour"
        c.acq.setValue(c.acq.maximum()); wait_done()

        # upscaling enlarges the output, and the method row appears with it
        assert not c._forms["interp"].isRowVisible(c.rows["interp"])
        c.upscale.setCurrentText("2")
        assert c._forms["interp"].isRowVisible(c.rows["interp"])
        wait_done()
        assert win.artists[3].get_array().shape == (512, 512), "upscale x2 should give 512"
        assert win.artists[2].get_array().shape == (256, 256), "the base does not upscale"
        c.upscale.setCurrentText("1"); wait_done()

        # every section is open at once: no accordion left to click through
        from PySide6.QtWidgets import QGroupBox
        groups = c.findChildren(QGroupBox)
        assert len(groups) == 6 and all(g.isVisible() for g in groups), len(groups)
        # the compare switch sits above the reconstruction, not in the control panel
        assert c.compare.parent() is not c and rv.isAncestorOf(c.compare), "compare above yours"
        sizes = win.centralWidget().sizes()
        assert abs(sizes[0]/sum(sizes) - 0.62) < 0.05, sizes
        # layout: metrics on top, k-space and reconstruction side by side below and taller
        assert kv.geometry().top() == rv.geometry().top(), "k-space and yours share the second row"
        assert kv.height() > win.axes[0].height(), "the second row gets more space"

        # rows hide themselves when the mode cannot use them
        vis = lambda n: c._forms[n].isRowVisible(c.rows[n])
        assert vis("rate") and vis("noise"), "the phantom offers rate and added noise"
        assert not vis("average"), "one phantom, nothing to average"
        assert not vis("sigma")
        c.window.setCurrentText("gaussian")
        assert vis("sigma"), "sigma must appear for the gaussian window"
        c.window.setCurrentText("none"); wait_done()

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
        c.sigma.setValue(280); wait_done()          # only reachable once widened
        assert c.settings()["sigma"] == 2.8, c.settings()["sigma"]
        lo_spin.setValue(300); hi_spin.setValue(300); dlg.apply()
        assert c.sigma.minimum() < c.sigma.maximum(), "an empty range must be refused"
        dlg.restore(); wait_done()
        assert (c.sigma.minimum(), c.sigma.maximum()) == c.default_ranges["sigma"]

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

        # fit-to-tissue: the window lands inside the tissue range and does not clip it
        c.rate.setValue(50); wait_done()
        c.wl_auto.click(); app.processEvents()
        lo, hi = win.artists[3].get_clim()
        r = np.asarray(win.artists[3].get_array()); tissue = r[r > r[:32, :32].max()]
        assert 0.0 <= lo < hi < 1.0, (lo, hi)
        assert (tissue > hi + 0.01).mean() < 0.02, "auto window must not clip tissue"
        c.wl_reset.click(); app.processEvents()
        c.rate.setValue(100); wait_done()

        # the base is cached per slice, not rebuilt on every knob
        assert win._base_key is not None and win.base is not None
        seen_key = win._base_key
        c.noise.setValue(10); wait_done()
        assert win._base_key == seen_key, "a slider must not rebuild the base"
        c.noise.setValue(0); wait_done()

        # real data: averaging, and the scan is read from disk only once
        multi = [f for f in win.datasets if len(find_repetitions(str(f))) > 1]
        if multi:
            c.source.setCurrentText(multi[0].name); wait_done()
            assert vis("average") and not vis("noise"), "real scans average, never add noise"
            single = win.artists[3].get_array().copy()
            c.average.setCurrentIndex(c.average.count() - 1); wait_done()
            msg = win.details_text
            assert f"averaged {c.average.count()} scans" in msg and "phase offsets" in msg, msg
            assert "0.3 T" in msg and "reference" in msg, msg
            now, delta = win.score_cells["SNR"]
            assert delta.get_color() == GREEN, (now.get_text(), delta.get_text())
            z = c.slice_idx.value()
            k1, ref1, _ = win._scan(z)
            direct = reconstruct(None, c.settings(), kspace=k1, reference=ref1,
                                 repeats=win._other_repetitions(z))["recon"]
            assert np.abs(direct - win.artists[3].get_array()).max() < 1e-9,                 "the cached average must give exactly the pipeline's result"
            assert not (single == win.artists[3].get_array()).all()
            assert (win.base == win.artists[2].get_array()).all(), "the base stays one scan"

            # smoothness: with the slice cached, a slider tick is only the reconstruction
            import mri.raw_data as rd
            reads = []
            real_load = rd.load_raw_kspace
            globals()["load_raw_kspace"] = lambda *a, **k: (reads.append(a), real_load(*a, **k))[1]
            times = []
            for v in (0, 30, 50, 30, 0):
                t0 = time.perf_counter(); c.nfloor.setValue(v); wait_done()
                times.append((time.perf_counter() - t0)*1000)
            globals()["load_raw_kspace"] = real_load
            assert reads == [], "a slider tick must not read the scan from disk again"
            print(f"  slider tick on real data, 3 scans averaged: "
                  f"{np.median(times):.0f} ms (median of {len(times)})")
            c.average.setCurrentIndex(0)
            c.source.setCurrentIndex(0); wait_done()
            print(f"  averaging checked on {multi[0].name}")
        else:
            print("  no repeated scans on disk, averaging check skipped")

        print("SELFTEST PASSED - scorecard, k-space edits, denoising, acquisition animation, "
              "steady k-space colours, compare divider, clicked spikes/patches, low/high pass, "
              "cached scans, window/level wired")
        QTimer.singleShot(0, app.quit)

    sys.exit(app.exec())
