"""
kSight: MRI k-space simulator.

    python app.py               # opens the window
    python app.py --selftest    # headless wiring check
"""
from __future__ import annotations
import sys
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.dates  # noqa: F401  must precede PySide6, whose import hook breaks on six.moves
from matplotlib import colormaps

from PySide6.QtCore import Qt, QTimer, QThread, Signal, QRect, QPoint
from PySide6.QtGui import QAction, QIcon, QImage, QPixmap, QPainter, QColor, QPen, QFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QSplitter, QScrollArea, QFileDialog,
                               QMessageBox, QWidget, QLabel, QFrame, QGridLayout, QVBoxLayout,
                               QHBoxLayout, QDialog, QPushButton)

from mri.kspace_core import load_phantom, load_image, radius_grid
from mri.pipeline import reconstruct, DEFAULTS
from mri.denoise import average_repetitions
from mri.kspace_edit import dc_index
from mri.raw_data import find_datasets, describe, load_raw_kspace, reference_image, raw_info, find_repetitions
from controls import Controls, RangeDialog

PHANTOM = "Shepp-Logan phantom (simulated)"
BG, CARD, EDGE = "#1e1e1e", "#262626", "#444444"
GREEN, RED, GREY = "#4cd07d", "#ff6b6b", "#8a8a8a"


def _lut(name: str) -> list[int]:
    rgb = (colormaps[name](np.linspace(0, 1, 256))[:, :3]*255).astype(int)
    return [0xFF000000 | (r << 16) | (g << 8) | b for r, g, b in rgb]


def _rss(k: np.ndarray) -> np.ndarray:
    return np.sqrt((np.abs(np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), axes=(-2, -1)))**2).sum(axis=0))


class ImageView(QWidget):
    """A titled [0,1] array drawn as an 8-bit indexed QImage; optional compare divider."""
    TITLE_H = 26

    def __init__(self, title: str, cmap: str = "gray", smooth: bool = True):
        super().__init__()
        self._title = QLabel(title)
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setStyleSheet("color: white; font-size: 12px;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 4)
        self._head = QHBoxLayout()
        self._head.setContentsMargins(0, 0, 0, 0)
        self._head.addWidget(self._title, 1)
        head = QWidget(); head.setFixedHeight(self.TITLE_H); head.setLayout(self._head)
        lay.addWidget(head)
        lay.addStretch(1)
        self._lut, self._smooth = _lut(cmap), smooth
        self._data, self._clim = np.zeros((2, 2)), (0.0, 1.0)
        self._pix, self._target = QPixmap(), QRect()
        self._compare, self._split = None, 0.5  # base view drawn left of the divider
        self.setMinimumSize(160, 160)

    def set_data(self, a):
        self._data = np.asarray(a, dtype=np.float64)
        self._render()

    def get_array(self):
        return self._data

    def set_clim(self, lo, hi):
        self._clim = (float(lo), float(hi))
        self._render()

    def get_clim(self):
        return self._clim

    def set_title(self, text: str):
        self._title.setText(text)

    def get_title(self) -> str:
        return self._title.text()

    def add_header(self, widget: QWidget):
        self._head.addWidget(widget)

    def set_compare(self, other: "ImageView | None"):
        self._compare = other
        self.update()

    def set_split(self, fraction: float):
        self._split = min(1.0, max(0.0, float(fraction)))
        self.update()

    def get_split(self) -> float:
        return self._split

    def _render(self):
        lo, hi = self._clim
        u8 = np.ascontiguousarray((np.nan_to_num(np.clip((self._data - lo)/max(hi - lo, 1e-12), 0, 1))*255).astype(np.uint8))
        h, w = u8.shape
        img = QImage(u8.data, w, h, w, QImage.Format_Indexed8)
        img.setColorTable(self._lut)
        self._pix = QPixmap.fromImage(img)
        self.update()

    def _fit(self) -> QRect:
        top = max(self._title.geometry().bottom(), self._head.geometry().bottom()) + 4
        aw, ah = self.width(), self.height() - top
        h, w = self._data.shape
        scale = min(aw/w, ah/h)
        pw, ph = int(w*scale), int(h*scale)
        return QRect((aw - pw)//2, top + (ah - ph)//2, pw, ph)

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
        r, centre = 20, QPoint(x, t.center().y())
        p.setBrush(QColor("white")); p.setPen(Qt.NoPen)
        p.drawEllipse(centre, r, r)
        p.setPen(QColor("#222222"))
        f = QFont(); f.setPixelSize(16); f.setBold(True); p.setFont(f)
        p.drawText(QRect(x - r, centre.y() - r, 2*r, 2*r), Qt.AlignCenter, "‹ ›")
        p.setPen(QColor("white"))
        p.drawText(QRect(x - 128, t.top() + 6, 120, 20), Qt.AlignRight | Qt.AlignVCenter, "base")
        p.drawText(QRect(x + 8, t.top() + 6, 120, 20), Qt.AlignLeft | Qt.AlignVCenter, "yours")

    def mousePressEvent(self, e):
        if self._compare is not None:
            self._drag(e.position().x())

    def mouseMoveEvent(self, e):
        if self._compare is not None and e.buttons() & Qt.LeftButton:
            self._drag(e.position().x())

    def _drag(self, x: float):
        t = self._target if not self._target.isNull() else self._fit()
        if t.width() > 0:
            self.set_split((x - t.x())/t.width())


class ColorBar(QWidget):
    """The k-space colour key, level with the k-space picture."""

    def __init__(self, view: ImageView, cmap: str = "magma"):
        super().__init__()
        self._view, self._lut = view, _lut(cmap)
        self.setFixedWidth(64)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(BG))
        t = self._view._target if not self._view._target.isNull() else self._view._fit()
        top, h = t.top() + 18, max(t.height() - 36, 20)
        bar = QRect(22, top, 16, h)
        for i in range(h):
            p.setPen(QColor.fromRgba(self._lut[255 - int(255*i/max(h - 1, 1))]))
            p.drawLine(bar.left(), top + i, bar.right(), top + i)
        p.setPen(QColor("#888888")); p.drawRect(bar)
        p.setPen(QColor("white"))
        f = QFont(); f.setPixelSize(11); p.setFont(f)
        p.drawText(QRect(0, top - 18, self.width(), 16), Qt.AlignCenter, "high")
        p.drawText(QRect(0, top + h + 2, self.width(), 16), Qt.AlignCenter, "low")
        p.save(); p.translate(12, top + h//2); p.rotate(-90)
        p.drawText(QRect(-h//2, -8, h, 16), Qt.AlignCenter, "|k| magnitude (log)")
        p.restore(); p.end()


class Prolog(QDialog):
    """Page 1: each coil's k-space revealed from DC outward, with its image. Page 2: the RSS combination."""
    REVEAL_SECONDS, TICK_MS = 4.0, 30

    def __init__(self, k: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prolog: from receiver coils to one image   (F11: full screen)")
        self.setStyleSheet(f"background: {BG}; color: white;")
        self.resize(1200, 760)
        self.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.WindowMinimizeButtonHint |
                            Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        full = QAction(self); full.setShortcut("F11")
        full.triggered.connect(lambda: self.showNormal() if self.isFullScreen() else self.showFullScreen())
        self.addAction(full)
        self.k = np.asarray(k)
        self.n = self.k.shape[0]
        self._radius_grid = radius_grid(self.k.shape[-2:], dc_index(self.k))
        self.max_radius = float(self._radius_grid.max())
        self.radius, self.revealed = 0.0, False
        per_coil = np.fft.ifft2(np.fft.ifftshift(self.k, axes=(-2, -1)), axes=(-2, -1))
        self._k_top = [max(float(np.log1p(np.abs(c)).max()), 1e-12) for c in self.k]  # fixed scales
        self._img_top = [max(float(np.abs(c).max()), 1e-12) for c in per_coil]

        self._layout = QVBoxLayout(self)
        self.caption = QLabel(); self.caption.setWordWrap(True)
        self.caption.setStyleSheet("font-size: 13px;")
        self.stage = QWidget()
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.replay_btn, self.next_btn, self.close_btn = QPushButton("Replay"), QPushButton("Next"), QPushButton("Close")
        for b in (self.replay_btn, self.next_btn, self.close_btn):
            b.setStyleSheet("background: #3a3a3a; padding: 6px 18px;")
            buttons.addWidget(b)
        self.close_btn.hide()
        self.replay_btn.clicked.connect(self._start)
        self.next_btn.clicked.connect(self._combine)
        self.close_btn.clicked.connect(self.accept)
        self._layout.addWidget(self.stage, 1)
        self._layout.addWidget(self.caption)
        self._layout.addLayout(buttons)

        grid = QGridLayout(self.stage)
        per_row = min(self.n, 4)
        self.coil_k_views, self.coil_img_views = [], []
        for c in range(self.n):
            kv, iv = ImageView(f"coil {c + 1}  k-space", "magma", smooth=False), ImageView(f"coil {c + 1}  image")
            r, col = divmod(c, per_row)
            grid.addWidget(kv, 2*r, col); grid.addWidget(iv, 2*r + 1, col)
            self.coil_k_views.append(kv); self.coil_img_views.append(iv)

        self._timer = QTimer(self); self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._start()

    def _start(self):
        self.radius, self.revealed = 0.0, False
        self.next_btn.setEnabled(False)
        self._draw()
        self._timer.start()

    def _tick(self):
        self.radius = min(self.max_radius, self.radius + self.max_radius*self.TICK_MS/(self.REVEAL_SECONDS*1000))
        self._draw()
        if self.radius >= self.max_radius:
            self._timer.stop()
            self.revealed = True
            self.next_btn.setEnabled(True)

    def _draw(self):
        part = self.k*(self._radius_grid <= self.radius)
        imgs = np.abs(np.fft.ifft2(np.fft.ifftshift(part, axes=(-2, -1)), axes=(-2, -1)))
        for c in range(self.n):
            self.coil_k_views[c].set_data(np.log1p(np.abs(part[c]))/self._k_top[c])
            self.coil_img_views[c].set_data(imgs[c]/self._img_top[c])
        self.caption.setText(
            f"<b>{self.n} receiver coils</b>, each recording its own k-space of the same slice at "
            f"the same time. Revealed from the centre outward, radius {self.radius:.0f} of "
            f"{self.max_radius:.0f} ({100*self.radius/self.max_radius:.0f}%): the centre carries each coil's "
            "shape and contrast, the edge carries its fine detail. Each coil sees its own side of the head "
            "brightly and the far side dimly.")

    def _combine(self):
        self._timer.stop()
        old, self.stage = self.stage, QWidget()
        self._layout.replaceWidget(old, self.stage)
        old.hide()  # deleteLater only runs on a later loop pass
        old.deleteLater()
        row = QHBoxLayout(self.stage)
        combined_k = np.log1p(np.sqrt((np.abs(self.k)**2).sum(axis=0)))
        self.final_k_view = ImageView("combined k-space", "magma", smooth=False)
        self.final_img_view = ImageView("final image  (root sum of squares)")
        self.final_k_view.set_data(combined_k/max(float(combined_k.max()), 1e-12))
        final = _rss(self.k)
        lo, hi = float(final.min()), float(final.max())
        self.final_img_view.set_data((final - lo)/max(hi - lo, 1e-12))
        row.addWidget(self.final_k_view, 1); row.addWidget(self.final_img_view, 1)
        terms = " + ".join(f"|x<sub>{c + 1}</sub>|²" for c in range(min(self.n, 4)))
        if self.n > 4:
            terms += f" + … + |x<sub>{self.n}</sub>|²"
        self.caption.setText(
            "<b>Method: root sum of squares.</b> Each coil's k-space K<sub>c</sub> is turned into "
            "its own image x<sub>c</sub> = IFFT2(K<sub>c</sub>), then the magnitudes are added in "
            f"quadrature:<br><br>&nbsp;&nbsp;&nbsp;&nbsp;image = √( {terms} )<br><br>"
            "Every coil has its own unknown phase, so adding the complex images would let them "
            "cancel; squaring the magnitudes cannot cancel, and it leans each pixel toward the coil "
            "that sees it best. The combined k-space shown is √(Σ|K<sub>c</sub>|²), "
            "for display only: the combination itself happens in the image, after each coil's "
            "inverse FFT. This final image is the starting point for every augmentation.")
        self.replay_btn.hide(); self.next_btn.hide(); self.close_btn.show()


class Cell(QLabel):
    """A scorecard entry that remembers its colour."""

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


def _card(title: str) -> tuple[QWidget, QFrame]:
    outer = QWidget()
    lay = QVBoxLayout(outer); lay.setContentsMargins(4, 2, 4, 4)
    head = QLabel(title); head.setAlignment(Qt.AlignCenter)
    head.setStyleSheet("color: white; font-size: 12px;")
    frame = QFrame(); frame.setObjectName("card")
    frame.setStyleSheet(f"QFrame#card {{ background: {CARD}; border: 1px solid {EDGE}; border-radius: 10px; }}")
    lay.addWidget(head); lay.addWidget(frame, 1)
    return outer, frame


class Worker(QThread):
    """One reconstruct() off the GUI thread; also builds the base when base_key is set."""
    done = Signal(dict)

    def __init__(self, img, settings, kspace=None, reference=None, base_key=None, extra=None, base_kspace=None):
        super().__init__()
        self.img, self.settings, self.kspace, self.reference = img, settings, kspace, reference
        self.base_key, self.extra, self.gen = base_key, extra or {}, 0
        self.base_kspace = kspace if base_kspace is None else base_kspace

    def run(self):
        try:
            t0 = time.perf_counter()
            out = reconstruct(self.img, self.settings, kspace=self.kspace, reference=self.reference)
            out.update(self.extra)
            if self.base_key is not None:
                base = reconstruct(self.img, dict(DEFAULTS), kspace=self.base_kspace, reference=self.reference)
                out.update(base=base["recon"], base_scores=(base["metrics"], base["sampling"]), base_key=self.base_key)
            out["_ms"] = int((time.perf_counter() - t0)*1000)
            out["_gen"] = self.gen
            self.done.emit(out)
        except Exception as e:  # surface, never crash the GUI thread
            self.done.emit({"exception": str(e), "_gen": self.gen})


class Main(QMainWindow):
    # name, format, higher is better
    _SCORES = (("Aliasing loss", "{:.1f} %", "{:+.1f} %", False), ("SNR", "{:.1f}", "{:+.1f}", True),
               ("SSIM", "{:.3f}", "{:+.3f}", True), ("PSNR", "{:.2f} dB", "{:+.2f} dB", True))

    def __init__(self):
        super().__init__()
        self.setWindowTitle("kSight — MRI k-space simulator")
        self.image = load_phantom()
        self.worker, self.pending, self._gen = None, None, 0
        self.raw_path, self.raw_note, self.ref_note = None, "", ""
        self.base, self.base_scores, self._base_key, self._image_gen = None, None, None, 0
        self._scans, self._repeats, self._averages = {}, {}, {}  # per (path, slice[, n]) caches
        self._details, self.state, self.details_text, self.info_text = [], "", "", ""

        self.score_box, score_frame = _card("metrics")
        self._build_scorecard(score_frame)
        detail_box, detail_frame = _card("scan details")
        self._build_details(detail_frame)
        kview = ImageView("measured k-space", "magma", smooth=False)
        bview = ImageView("base  (nothing switched on)")  # never shown; drawn left of the divider
        rview = ImageView("my reconstruction")
        for v, data in ((kview, np.zeros((256, 256))), (bview, self.image), (rview, self.image)):
            v.set_data(data)
        self.kview, self.bview, self.rview = kview, bview, rview

        top_row = QWidget()
        top = QHBoxLayout(top_row); top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.score_box, 1); top.addWidget(detail_box, 1)
        top_row.setFixedHeight(self.score_box.sizeHint().height())
        self.colorbar = ColorBar(kview)
        bottom = QHBoxLayout()
        bottom.addWidget(kview, 1); bottom.addWidget(self.colorbar); bottom.addWidget(rview, 1)
        left = QWidget(); left.setStyleSheet(f"background: {BG};")
        grid = QVBoxLayout(left); grid.setContentsMargins(6, 6, 6, 6)
        grid.addWidget(top_row); grid.addLayout(bottom, 1)

        self.controls = Controls()
        self._fill_sources()
        self.controls.compare.setStyleSheet("color: white;")
        rview.add_header(self.controls.compare)
        panel = QScrollArea()
        panel.setWidget(self.controls)
        panel.setWidgetResizable(True)
        panel.setMinimumWidth(380)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left); split.addWidget(panel)
        split.setStretchFactor(0, 62); split.setStretchFactor(1, 38)
        self._split, self._split_set = split, False
        self.setCentralWidget(split)

        c = self.controls
        c.source.currentTextChanged.connect(self._source_changed)
        c.settingsChanged.connect(lambda _s: self._run())
        c.displayChanged.connect(self._apply_display_range)
        c.compareToggled.connect(self._compare)
        c.wl_auto.clicked.connect(self._fit_window)
        c.loadImageClicked.connect(self._load_image)
        c.phantom_btn.clicked.connect(self._load_phantom)
        c.prologClicked.connect(self._open_prolog)
        c.saveClicked.connect(self._save_image)
        self._prolog = None
        self._update_prolog_button()

        act = QAction("Slider ranges...", self)
        act.setShortcut("Ctrl+,")
        act.triggered.connect(self._edit_ranges)
        self.menuBar().addMenu("Settings").addAction(act)
        self._run()

    def showEvent(self, e):
        super().showEvent(e)
        if not self._split_set:  # 62 : 38 once the window has a real width
            w = max(self._split.width(), 1)
            self._split.setSizes([int(w*0.62), w - int(w*0.62)])
            self._split_set = True

    # --- scorecard and details ---

    def _build_scorecard(self, frame: QFrame):
        g = QGridLayout(frame)
        g.setContentsMargins(18, 12, 18, 12)
        for col, (text, align) in enumerate((("metric", Qt.AlignLeft), ("now", Qt.AlignRight), ("vs base", Qt.AlignRight))):
            head = Cell(12, GREY, align); head.setText(text)
            g.addWidget(head, 0, col)
        line = QFrame(); line.setFixedHeight(1); line.setStyleSheet(f"background: {EDGE};")
        g.addWidget(line, 1, 0, 1, 3)
        self.score_cells = {}
        for i, (name, *_) in enumerate(self._SCORES):
            label = Cell(15, "#cfcfcf", Qt.AlignLeft); label.setText(name)
            now, delta = Cell(17), Cell(13, GREY)
            g.addWidget(label, i + 2, 0); g.addWidget(now, i + 2, 1); g.addWidget(delta, i + 2, 2)
            self.score_cells[name] = (now, delta)

    def _update_scorecard(self, out: dict):
        """Now, and the change against the base: green helped, red hurt."""
        scores = lambda m, s: {"Aliasing loss": 100*s["alias_energy"], "SNR": m["snr"], "SSIM": m["ssim"], "PSNR": m["psnr"]}
        now = scores(out["metrics"], out["sampling"])
        base = scores(*self.base_scores) if self.base_scores else now
        lines = []
        for name, fmt, dfmt, higher_better in self._SCORES:
            v, b = now[name], base[name]
            shown = "∞" if math.isinf(v) else fmt.format(v)
            if math.isinf(v) or math.isinf(b):
                colour, dtext = GREY, "—" if math.isinf(v) != math.isinf(b) else "±0"
            else:
                d = v - b
                tiny = abs(d) < 5e-4*max(1.0, abs(b))
                colour = GREY if tiny else (GREEN if (d > 0) == higher_better else RED)
                dtext = "±0" if tiny else dfmt.format(d)
            cell_now, cell_delta = self.score_cells[name]
            cell_now.setText(shown)
            cell_delta.setText(dtext); cell_delta.set_color(colour)
            lines.append(f"{name}  {shown}  {dtext}")
        self.info_text = "\n".join(lines)

    def _build_details(self, frame: QFrame):
        lay = QVBoxLayout(frame); lay.setContentsMargins(16, 12, 6, 12)
        self.details_label = QLabel(); self.details_label.setWordWrap(True)
        self.details_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.details_label.setStyleSheet("color: #dddddd; font-size: 12px; background: transparent;")
        self.details_scroll = QScrollArea()  # scrolls instead of growing the top row
        self.details_scroll.setWidget(self.details_label)
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setFrameShape(QFrame.NoFrame)
        self.details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.details_scroll.setStyleSheet(
            "QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }"
            "QScrollBar:vertical { background: #262626; width: 8px; margin: 0; }"
            "QScrollBar::handle:vertical { background: #5a5a5a; border-radius: 4px; min-height: 24px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }")
        lay.addWidget(self.details_scroll)

    def _detail_rows(self, out: dict) -> list:
        rows = [("measured", "{} of {} rows<br>{} of {} columns".format(*out["sampling"]["rows"], *out["sampling"]["cols"]))]
        (h, w), (kh, kw) = out["recon"].shape, out["kspace"].shape
        if (h, w) != (kh, kw):
            s = self.controls.settings()
            rows.append(("image size", f"{h} × {w}  (scan {kh} × {kw}, "
                         f"×{s['upscale']} {s['interp'].replace('_', ' ')})"))
        if out["averaged"] > 1:
            al = out["alignment"]
            rows.append(("averaged", f"{out['averaged']} scans"))
            if al["phases"]:
                rows.append(("phase offsets", ", ".join(f"{p:+.2f}" for p in al["phases"]) + " rad"))
                rows.append(("shifts", "  ".join(f"({d[0]},{d[1]})" for d in al["shifts"]) + " px"))
        if self.raw_note:
            parts = [p.strip() for p in self.raw_note.split("|")]
            rows.append(("scanner", parts[0] + "<br>" + " · ".join(parts[1:4]) + "<br>" + " · ".join(parts[4:])))
        else:
            rows.append(("source", "simulated phantom"))
        if self.ref_note:
            rows.append(("reference", self.ref_note))
        return rows

    def _set_state(self, state: str):
        self.state = state
        rows = [("status", state)] + self._details
        cells = "".join(f"<tr><td style='color:{GREY}; padding:3px 14px 3px 0'>{k}</td>"
                        f"<td style='padding:3px 0'>{v}</td></tr>" for k, v in rows)
        self.details_label.setText(f"<table>{cells}</table>")
        self.details_text = "\n".join(f"{k} {v.replace('<br>', ' ')}" for k, v in rows)

    # --- actions ---

    def _open_prolog(self):
        if self.raw_path is None:
            return
        k = self._scan(self.controls.slice_idx.value())[0]
        if k.ndim == 3 and k.shape[0] > 1:
            self._prolog = Prolog(k, self)
            self._prolog.open()

    def _update_prolog_button(self):
        ok, why = False, "the phantom is a single simulated image, with no coils"
        if self.raw_path is not None:
            try:
                n = raw_info(self.raw_path)["coils"]
                ok, why = n > 1, f"{n} coils" if n > 1 else "a single-coil scan"
            except Exception as e:
                why = str(e)
        self.controls.prolog_btn.setEnabled(ok)
        self.controls.prolog_btn.setToolTip("Show how the coils combine into the image" if ok
                                            else f"Needs a multi-coil scan: {why}")

    def _save_image(self):
        """Your reconstruction as displayed (window/level applied), at its full resolution."""
        name = f"{Path(self.raw_path).stem}_slice{self.controls.slice_idx.value()}" if self.raw_path else "image"
        path, _ = QFileDialog.getSaveFileName(self, "Save reconstruction", f"{name}_recon.png",
                                              "PNG image (*.png);;TIFF image (*.tif *.tiff);;JPEG image (*.jpg)")
        if path and not self.rview._pix.save(path):
            QMessageBox.warning(self, "Could not save", f"Could not write {path}")

    def _edit_ranges(self):
        dlg = RangeDialog(self.controls, self)
        if dlg.exec():
            dlg.apply()

    def _compare(self, on: bool):
        self.rview.set_compare(self.bview if on else None)
        self._apply_display_range(*self.controls.display_range())

    def _fit_window(self):
        """Window = 1st..99th percentile of pixels brighter than the brightest corner pixel."""
        r = self.rview.get_array()
        tissue = r[r > float(r[:32, :32].max())]
        if tissue.size < 100:
            return
        lo, hi = np.percentile(tissue, [1, 99])
        self.controls.wl_width.setValue(int(round((hi - lo)*100)))
        self.controls.wl_centre.setValue(int(round((lo + hi)/2*100)))

    def _apply_display_range(self, vmin: float, vmax: float):
        """Display only: the pixels are untouched and nothing re-runs."""
        self.rview.set_clim(vmin, vmax)
        full = abs(vmin) < 1e-9 and abs(vmax - 1.0) < 1e-9
        title = "augmented reconstruction" if full else f"augmented reconstruction    display window {vmin:.2f} to {vmax:.2f}"
        if self.controls.compare.isChecked():
            title = "base  ‹ ›  " + title
        self.rview.set_title(title)

    def _fill_sources(self):
        self.datasets = find_datasets()
        box = self.controls.source
        box.blockSignals(True)
        box.clear()
        box.addItem(PHANTOM)
        box.addItems([f.name for f in self.datasets])
        box.blockSignals(False)

    def _source_changed(self, name: str):
        c = self.controls
        if name == PHANTOM:
            self.raw_path, self.raw_note, self.ref_note = None, "", ""
            self.image = load_phantom()
            c.set_repetitions(1)
        else:
            match = [f for f in self.datasets if f.name == name]
            if not match:
                return
            self.raw_path = str(match[0])
            try:
                info = raw_info(self.raw_path)
                n_reps = len(find_repetitions(self.raw_path))
                c.set_repetitions(n_reps)
                self.raw_note = describe(self.raw_path) + (f" | {n_reps} repetitions" if n_reps > 1 else " | single acquisition")
            except Exception as e:
                QMessageBox.warning(self, "Could not read that file", str(e))
                self.raw_path = None
                return
            c.slice_idx.blockSignals(True)
            c.slice_idx.setRange(0, max(0, info["n_slices"] - 1))
            c.slice_idx.setValue(info["n_slices"]//2)
            c.slice_idx.blockSignals(False)
        c._update_visibility()
        self._update_prolog_button()
        self._run()

    # --- running a reconstruction ---

    def _scan(self, z: int):
        """(k-space, reference, note) for this slice, read from disk once."""
        key = (self.raw_path, z)
        if key not in self._scans:
            self._scans[key] = (load_raw_kspace(self.raw_path, z), *reference_image(self.raw_path, z))
        return self._scans[key]

    def _other_repetitions(self, z: int) -> list:
        key = (self.raw_path, z)
        if key not in self._repeats:
            me = Path(self.raw_path).resolve()
            self._repeats[key] = [load_raw_kspace(p, z) for p in find_repetitions(self.raw_path) if Path(p).resolve() != me]
        return self._repeats[key]

    def _averaged(self, z: int, n: int):
        """Aligned average of n scans; depends only on file, slice and n, so kept."""
        key = (self.raw_path, z, n)
        if key not in self._averages:
            self._averages[key] = average_repetitions([self._scan(z)[0]] + self._other_repetitions(z)[:n - 1])
        return self._averages[key]

    def _run(self):
        if self.worker is not None and self.worker.isRunning():
            self.pending = True  # rerun with the latest settings on finish
            return
        self._set_state("reconstructing...")
        settings = self.controls.settings()
        key = (self.raw_path, self.controls.slice_idx.value()) if self.raw_path else ("image", self._image_gen)
        base_key = key if key != self._base_key else None  # base rebuilt only for a new slice or file

        if self.raw_path is None:
            self.worker = Worker(self.image, settings, base_key=base_key)
        else:
            z = self.controls.slice_idx.value()
            try:
                k, ref, self.ref_note = self._scan(z)
                n, extra, kin = settings["average"], None, k
                if n > 1:  # the cached average with average=1 is identical to the pipeline's own
                    kin, alignment = self._averaged(z, n)
                    extra = {"averaged": n, "alignment": alignment}
                    settings = {**settings, "average": 1}
            except Exception as e:
                self._set_state(f"error reading slice {z}: {e}")
                return
            self.worker = Worker(None, settings, kspace=kin, reference=ref, base_key=base_key, extra=extra, base_kspace=k)

        self._gen += 1
        self.worker.gen = self._gen
        self.worker.done.connect(self._show)
        self.worker.finished.connect(self._next)  # finished, not done: done fires while the thread still runs
        self.worker.start()

    def _next(self):
        if self.pending is not None:
            self.pending = None
            self._run()

    def _show(self, out: dict):
        if out.get("_gen") != self._gen:  # stale result from an older run
            return
        try:
            self._paint(out)
        except Exception as e:  # Qt swallows slot exceptions
            import traceback; traceback.print_exc()
            self._set_state(f"display error: {type(e).__name__}: {e}")

    def _paint(self, out: dict):
        ms = out.pop("_ms", 0)
        if "exception" in out:
            self._set_state(f"error: {out['exception']}")
            return
        if "base" in out:
            self.base, self._base_key, self.base_scores = out["base"], out["base_key"], out["base_scores"]
        self.kview.set_data(out["kspace"])
        self.bview.set_data(self.base if self.base is not None else out["recon"])
        self.rview.set_data(out["recon"])
        self._update_scorecard(out)
        self._details = self._detail_rows(out)
        self._set_state(f"ready  ({ms} ms)")

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load raw k-space or image", "",
            "All supported (*.h5 *.png *.jpg *.jpeg *.bmp *.tif);;Raw k-space (*.h5);;Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if not path:
            return
        if path.lower().endswith(".h5"):
            try:
                raw_info(path)
            except Exception as e:
                QMessageBox.warning(self, "Not a readable k-space file", str(e))
                return
            p = Path(path)
            if p not in self.datasets:
                self.datasets.append(p)
                self.controls.source.addItem(p.name)
            self.controls.source.setCurrentText(p.name)
            return
        try:
            self.image = load_image(path)
        except Exception as e:
            QMessageBox.warning(self, "Could not load image", str(e))
            return
        self._show_image()

    def _load_phantom(self):
        self.image = load_phantom()
        self._show_image()

    def _show_image(self):
        self.raw_path, self.raw_note, self.ref_note = None, "", ""
        self._image_gen += 1
        c = self.controls
        c.source.blockSignals(True); c.source.setCurrentIndex(0); c.source.blockSignals(False)  # keep self.image
        c.set_repetitions(1)
        self._update_prolog_button()
        self._run()


if __name__ == "__main__":
    selftest = "--selftest" in sys.argv
    if selftest:
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    if sys.platform == "win32":  # own taskbar entry, so Windows shows our icon instead of Python's
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("log_synthesis.kSight")
    app = QApplication(sys.argv)
    app.setApplicationName("kSight")
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent/"assets"/"mark.png")))
    win = Main()
    win.resize(1400, 820)
    win.show()
    if selftest:
        from selftest import run
        run(app, win)
        QTimer.singleShot(0, app.quit)
    sys.exit(app.exec())
