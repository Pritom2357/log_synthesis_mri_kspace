"""The control panel; settings() returns exactly what pipeline.reconstruct expects."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QTimer, QEvent
from PySide6.QtWidgets import (QWidget, QFormLayout, QComboBox, QSlider, QLabel, QCheckBox, QPushButton,
                               QHBoxLayout, QToolButton, QVBoxLayout, QGroupBox, QDialog, QSpinBox,
                               QDialogButtonBox, QApplication, QAbstractScrollArea)

KEEP_LABELS = {"magnitude and phase": "both", "magnitude only": "magnitude", "phase only": "phase"}
INTERP_LABELS = {"sinc (ideal)": "sinc", "linear": "linear", "zero order hold": "zero_order_hold"}

# slider -> (label, lowest, highest) the engine accepts; the range dialog stays inside these
LIMITS = {
    "rate":    ("sampling rate",      1,  100),
    "sigma":   ("window sigma",       1,  500),
    "nfloor":  ("noise-floor filter", 0,  500),
    "sharpen": ("sharpen",            0,  500),
    "dc":      ("DC term",            0, 1000),
    "spike":   ("spike offset",       0,  255),
    "erase":   ("erase centre",       0,  512),
    "pf":      ("partial Fourier",   50,  100),
}
PLAY_SECONDS, PLAY_TICK_MS = 8.0, 30


def _row(*widgets) -> QWidget:
    w = QWidget()
    box = QHBoxLayout(w)
    box.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        box.addWidget(x)
    return w


class RangeDialog(QDialog):
    """Retunes each slider's travel within LIMITS; an empty range is ignored."""

    def __init__(self, controls, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Slider ranges")
        self.controls, self.spins = controls, {}
        form = QFormLayout(self)
        for name, slider in controls.sliders.items():
            label, cap_lo, cap_hi = LIMITS[name]
            lo, hi = QSpinBox(), QSpinBox()
            for spin, val in ((lo, slider.minimum()), (hi, slider.maximum())):
                spin.setRange(cap_lo, cap_hi)
                spin.setValue(val)
            self.spins[name] = (lo, hi)
            form.addRow(f"{label}   (allowed {cap_lo} .. {cap_hi})", _row(QLabel("min"), lo, QLabel("max"), hi))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore)
        form.addRow(buttons)

    def apply(self) -> None:
        for name, (lo, hi) in self.spins.items():
            if lo.value() < hi.value():
                self.controls.sliders[name].setRange(lo.value(), hi.value())
        self.controls._emit()

    def restore(self) -> None:
        for name, (lo, hi) in self.spins.items():
            a, b = self.controls.default_ranges[name]
            lo.setValue(a); hi.setValue(b)
        self.apply()


class Controls(QWidget):
    settingsChanged = Signal(dict)
    loadImageClicked = Signal()
    displayChanged = Signal(float, float)
    compareToggled = Signal(bool)
    prologClicked = Signal()

    def __init__(self):
        super().__init__()
        self.rows: dict[str, QWidget] = {}
        self._forms: dict[str, QFormLayout] = {}

        self.source = QComboBox()  # filled by app.py
        self.slice_idx, self.slice_idx_lbl = self._slider(0, 17, 9)
        self.load_btn, self.phantom_btn = QPushButton("Load raw k-space..."), QPushButton("Phantom")
        self.prolog_btn = QPushButton("Prolog")

        self.rate, self.rate_lbl = self._slider(20, 100, 100)  # /100 x Nyquist
        self.acq, self.acq_lbl = self._slider(0, 1000, 1000)   # /1000 of the scan read
        self.acq_order = QComboBox(); self.acq_order.addItems(["linear", "centric"])
        self.play_btn = QPushButton("Play"); self.play_btn.setCheckable(True)
        self.rewind_btn = QPushButton("Rewind")
        self._play = QTimer(self); self._play.setInterval(PLAY_TICK_MS)
        self._play.timeout.connect(self._play_tick)

        self.average = QComboBox(); self.average.addItem("1  (this scan only)")
        self.nfloor, self.nfloor_lbl = self._slider(0, 100, 0)
        self.sharpen, self.sharpen_lbl = self._slider(0, 100, 0)
        self.window = QComboBox(); self.window.addItems(["none", "hamming", "gaussian"])
        self.sigma, self.sigma_lbl = self._slider(5, 100, 80)
        self.pf, self.pf_lbl = self._slider(50, 100, 100)
        self.pf_fill = QCheckBox("rebuild the rest from Hermitian symmetry"); self.pf_fill.setChecked(True)

        self.dc, self.dc_lbl = self._slider(0, 200, 100)
        self.spike, self.spike_lbl = self._slider(0, 60, 0)
        self.erase, self.erase_lbl = self._slider(0, 120, 0)
        self.keep = QComboBox(); self.keep.addItems(list(KEEP_LABELS))

        self.upscale = QComboBox(); self.upscale.addItems(["1", "2", "4"])
        self.interp = QComboBox(); self.interp.addItems(list(INTERP_LABELS))

        self.wl_centre, self.wl_centre_lbl = self._slider(0, 100, 50)
        self.wl_width, self.wl_width_lbl = self._slider(1, 100, 100)  # recon is [0,1]: wider shows nothing more
        self.wl_reset, self.wl_auto = QPushButton("Reset to the full range"), QPushButton("Fit to tissue")
        self.compare = QCheckBox("compare with base")  # placed above the reconstruction by app.py

        self.sliders = {n: getattr(self, n) for n in LIMITS}
        self.default_ranges = {n: (s.minimum(), s.maximum()) for n, s in self.sliders.items()}

        sl = lambda name: self._slider_row(getattr(self, name), getattr(self, name + "_lbl"))
        sections = [
            ("1 · Data source", [("source", "dataset", self.source), ("slice_idx", "slice", sl("slice_idx")),
                                 ("load_row", None, _row(self.load_btn, self.phantom_btn)),
                                 ("prolog", None, self.prolog_btn)]),
            ("2 · Sampling", [("rate", "sampling rate  (x Nyquist rate)", sl("rate")),
                              ("acq", "scan progress", sl("acq")),
                              ("acq_row", "line order", _row(self.acq_order, self.play_btn, self.rewind_btn))]),
            ("3 · Denoising", [("average", "repetitions averaged", self.average),
                               ("nfloor", "noise-floor filter", sl("nfloor")),
                               ("window", "apodization", self.window), ("sigma", "window sigma", sl("sigma")),
                               ("sharpen", "sharpen (unsharp mask)", sl("sharpen")),
                               ("pf", "partial Fourier", sl("pf")), ("pf_fill", None, self.pf_fill)]),
            ("4 · k-space edits", [("dc", "DC term", sl("dc")), ("spike", "spike offset", sl("spike")),
                                   ("erase", "erase centre", sl("erase")), ("keep", "k-space keeps", self.keep)]),
            ("5 · Resolution", [("upscale", "upscale by", self.upscale), ("interp", "interpolation", self.interp)]),
            ("6 · Display  (window / level)", [("wl_centre", "level (centre)", sl("wl_centre")),
                                                ("wl_width", "window (width)", sl("wl_width")),
                                                ("wl_buttons", None, _row(self.wl_auto, self.wl_reset))]),
        ]
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        for title, entries in sections:
            group = QGroupBox(title)
            group.setStyleSheet("QGroupBox { font-weight: bold; }")
            form = QFormLayout(group)
            form.setContentsMargins(8, 8, 8, 6)
            for name, label, field in entries:
                form.addRow(field) if label is None else form.addRow(label, field)
                self.rows[name], self._forms[name] = field, form
            outer.addWidget(group)
        outer.addStretch(1)

        for combo in (self.source, self.average, self.window, self.keep, self.upscale, self.interp, self.acq_order):
            combo.currentTextChanged.connect(self._emit)
        for s in (self.rate, self.sigma, self.nfloor, self.sharpen, self.slice_idx,
                  self.dc, self.spike, self.erase, self.pf, self.acq):
            s.valueChanged.connect(self._emit)
        self.pf_fill.toggled.connect(self._emit)
        self.play_btn.toggled.connect(self._toggle_play)
        self.rewind_btn.clicked.connect(lambda: self.acq.setValue(0))
        for s in (self.wl_centre, self.wl_width):  # display only: no reconstruction
            s.valueChanged.connect(self._emit_display)
        self.wl_reset.clicked.connect(self.reset_display)
        self.compare.toggled.connect(self.compareToggled)
        self.load_btn.clicked.connect(self.loadImageClicked)
        self.prolog_btn.clicked.connect(self.prologClicked)

        for kind in (QSlider, QComboBox):  # the wheel scrolls the panel, never edits a value
            for w in self.findChildren(kind):
                w.setFocusPolicy(Qt.StrongFocus)
                w.installEventFilter(self)
        self._update_visibility()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and isinstance(obj, (QSlider, QComboBox)):
            area = self.parentWidget()
            while area is not None and not isinstance(area, QAbstractScrollArea):
                area = area.parentWidget()
            if area is not None:
                QApplication.sendEvent(area.viewport(), event)
            return True
        return super().eventFilter(obj, event)

    def _slider(self, lo, hi, val):
        s = QSlider(Qt.Horizontal); s.setRange(lo, hi); s.setValue(val)
        lbl = QLabel(str(val)); lbl.setMinimumWidth(38)
        return s, lbl

    def _step_btn(self, slider, delta):
        b = QToolButton(); b.setText("-" if delta < 0 else "+")
        b.setAutoRepeat(True)
        b.clicked.connect(lambda: slider.setValue(slider.value() + delta*max(1, (slider.maximum() - slider.minimum())//100)))
        return b

    def _slider_row(self, slider, lbl):
        return _row(self._step_btn(slider, -1), slider, self._step_btn(slider, +1), lbl)

    def _update_visibility(self) -> None:
        """Hide rows the current mode ignores."""
        real = self.source.currentIndex() > 0
        for name, shown in {"slice_idx": real,
                            "average": real and self.average.count() > 1,
                            "sigma": self.window.currentText() == "gaussian",
                            "pf_fill": self.pf.value() < 100,
                            "interp": self.upscale.currentText() != "1"}.items():
            self._forms[name].setRowVisible(self.rows[name], shown)

    def _toggle_play(self, on: bool) -> None:
        if on:
            if self.acq.value() >= self.acq.maximum():
                self.acq.setValue(0)
            self._play.start()
        else:
            self._play.stop()
        self.play_btn.setText("Pause" if on else "Play")

    def _play_tick(self) -> None:
        step = max(1, round(self.acq.maximum()*PLAY_TICK_MS/(PLAY_SECONDS*1000)))
        self.acq.setValue(min(self.acq.maximum(), self.acq.value() + step))
        if self.acq.value() >= self.acq.maximum():
            self.play_btn.setChecked(False)

    def set_repetitions(self, n: int) -> None:
        self.average.blockSignals(True)
        self.average.clear()
        self.average.addItem("1  (this scan only)")
        self.average.addItems([f"{i}  (this scan + {i - 1} more)" for i in range(2, n + 1)])
        self.average.blockSignals(False)
        self._update_visibility()

    def display_range(self) -> tuple[float, float]:
        centre, width = self.wl_centre.value()/100.0, max(0.01, self.wl_width.value()/100.0)
        return centre - width/2.0, centre + width/2.0

    def reset_display(self) -> None:
        self.wl_centre.setValue(50)
        self.wl_width.setValue(100)

    def _emit_display(self, *_):
        self.wl_centre_lbl.setText(f"{self.wl_centre.value()/100:.2f}")
        self.wl_width_lbl.setText(f"{self.wl_width.value()/100:.2f}")
        self.displayChanged.emit(*self.display_range())

    def settings(self) -> dict:
        return {
            "rate": self.rate.value()/100.0,
            "average": self.average.currentIndex() + 1,
            "noise_filter": self.nfloor.value()/100.0,
            "sharpen": self.sharpen.value()/100.0,
            "window": self.window.currentText(),
            "sigma": self.sigma.value()/100.0,
            "keep": KEEP_LABELS[self.keep.currentText()],
            "dc_scale": self.dc.value()/100.0,
            "spike": self.spike.value(),
            "erase": self.erase.value(),
            "partial_fourier": self.pf.value()/100.0,
            "acquired": self.acq.value()/self.acq.maximum(),
            "acq_order": self.acq_order.currentText(),
            "pf_fill": self.pf_fill.isChecked(),
            "upscale": int(self.upscale.currentText()),
            "interp": INTERP_LABELS[self.interp.currentText()],
        }

    def _emit(self, *_):
        for s, lbl, fmt in ((self.rate, self.rate_lbl, "{:.2f}"), (self.sigma, self.sigma_lbl, "{:.2f}"),
                            (self.dc, self.dc_lbl, "{:.2f}"), (self.pf, self.pf_lbl, "{:.2f}")):
            lbl.setText(fmt.format(s.value()/100))
        for s, lbl in ((self.nfloor, self.nfloor_lbl), (self.sharpen, self.sharpen_lbl)):
            lbl.setText("off" if s.value() == 0 else f"{s.value()/100:.2f}")
        for s, lbl in ((self.spike, self.spike_lbl), (self.erase, self.erase_lbl)):
            lbl.setText("off" if s.value() == 0 else str(s.value()))
        self.slice_idx_lbl.setText(str(self.slice_idx.value()))
        self.acq_lbl.setText(f"{100*self.acq.value()/self.acq.maximum():.0f}%")
        self._update_visibility()
        self.settingsChanged.emit(self.settings())
