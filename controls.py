"""
The control panel. Every knob the simulator has, collected into one widget
that emits a single signal: settingsChanged(dict).

The dict it emits is exactly what pipeline.reconstruct expects, so app.py
never translates anything.

The knobs are grouped into an accordion because there are now too many to read
as one flat list. Within a group, rows hide themselves when they cannot do
anything: a slider the current mode ignores is worse than useless, because it
invites you to drag it and conclude the engine is broken.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QFormLayout, QComboBox, QSlider,
                               QLabel, QCheckBox, QPushButton, QHBoxLayout,
                               QToolButton, QVBoxLayout, QGroupBox, QDialog,
                               QSpinBox, QDialogButtonBox)

# The dropdowns say it in full; the pipeline wants the short key.
KEEP_LABELS = {"magnitude and phase": "both", "magnitude only": "magnitude",
               "phase only": "phase"}
INTERP_LABELS = {
    "sinc (ideal)": "sinc",
    "linear": "linear",
    "zero order hold": "zero_order_hold",
}

# How far each slider is ALLOWED to travel, as opposed to how far it travels by
# default. The defaults are the ranges that are useful most of the time; these
# are the widest the engine will accept before it raises. Hunting for something
# faint is exactly when you want to push a setting past its everyday range, so
# the ceiling here is the real constraint and nothing tighter.
#   name -> (row label, lowest allowed, highest allowed)
LIMITS = {
    "rate":      ("sampling rate",      1,  100),   # rate must be in (0, 1]
    "sigma":     ("window sigma",       1,  500),   # sigma must be positive
    "noise":     ("add noise",          0,  500),
    "nfloor":    ("noise-floor filter", 0,  500),   # strength must be non-negative
    "sharpen":   ("sharpen",            0,  500),
    "dc":        ("DC term",            0, 1000),   # scale must be non-negative
    "spike":     ("spike offset",       0,  255),   # must land inside k-space
    "erase":     ("erase centre",       0,  512),
    "pf":        ("partial Fourier",   50,  100),   # fraction must be 0.5 .. 1.0
}


class RangeDialog(QDialog):
    """
    Widens or narrows the travel of any slider on the control panel.

    Every row is capped at what LIMITS says the engine accepts, so no setting
    chosen here can produce a ValueError from the pipeline. A range with its
    floor at or above its ceiling is ignored rather than applied, because a
    slider that cannot move is indistinguishable from a broken one.
    """

    def __init__(self, controls, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Slider ranges")
        self.controls = controls
        self.spins: dict[str, tuple[QSpinBox, QSpinBox]] = {}

        form = QFormLayout(self)
        for name, slider in controls.sliders.items():
            label, cap_lo, cap_hi = LIMITS[name]
            lo, hi = QSpinBox(), QSpinBox()
            for spin, val in ((lo, slider.minimum()), (hi, slider.maximum())):
                spin.setRange(cap_lo, cap_hi)
                spin.setValue(val)
            self.spins[name] = (lo, hi)
            form.addRow(f"{label}   (allowed {cap_lo} .. {cap_hi})",
                        self._pair(lo, hi))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel |
                                   QDialogButtonBox.RestoreDefaults)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore)
        form.addRow(buttons)

    def _pair(self, lo, hi):
        box = QHBoxLayout(); w = QWidget()
        box.addWidget(QLabel("min")); box.addWidget(lo)
        box.addWidget(QLabel("max")); box.addWidget(hi)
        box.setContentsMargins(0, 0, 0, 0); w.setLayout(box)
        return w

    def apply(self) -> None:
        """Push the chosen ranges onto the sliders. Qt clamps the values itself."""
        for name, (lo, hi) in self.spins.items():
            a, b = lo.value(), hi.value()
            if a >= b:
                continue
            self.controls.sliders[name].setRange(a, b)
        self.controls._emit()

    def restore(self) -> None:
        """Back to the ranges the app started with."""
        for name, (lo, hi) in self.spins.items():
            a, b = self.controls.default_ranges[name]
            lo.setValue(a); hi.setValue(b)
        self.apply()


class Controls(QWidget):
    settingsChanged = Signal(dict)
    loadImageClicked = Signal()
    displayChanged = Signal(float, float)   # vmin, vmax for the reconstruction

    def __init__(self):
        super().__init__()
        self.rows: dict[str, QWidget] = {}        # row name -> field widget
        self._forms: dict[str, QFormLayout] = {}  # row name -> the form it lives in

        # ---------------- widgets ----------------
        self.source = QComboBox()            # filled by app.py after scanning disk
        self.slice_idx, self.slice_lbl = self._slider(0, 17, 9)
        self.load_btn = QPushButton("Load raw k-space...")
        self.phantom_btn = QPushButton("Phantom")

        self.rate, self.rate_lbl = self._slider(20, 100, 100)     # /100 -> x Nyquist rate

        # Denoising. The repetition count is filled in by app.py once it knows
        # how many scans of this slice exist on disk.
        self.average = QComboBox(); self.average.addItem("1  (this scan only)")
        # Alignment is always on: without it, averaging can cancel the anatomy
        # instead of the noise (see mri/denoise.py).
        self.nfloor, self.nfloor_lbl = self._slider(0, 100, 0)    # /100 -> strength; best near 0.5, above 1 only blurs
        self.sharpen, self.sharpen_lbl = self._slider(0, 100, 0)  # /100 -> amount
        self.window = QComboBox(); self.window.addItems(["none", "hamming", "gaussian"])
        self.sigma, self.sigma_lbl = self._slider(5, 100, 80)     # /100; 0.8 measured best, 0.35 was worse than no window
        self.noise, self.noise_lbl = self._slider(0, 30, 0)       # /100

        self.dc, self.dc_lbl = self._slider(0, 200, 100)          # /100 -> DC multiplier
        self.spike, self.spike_lbl = self._slider(0, 60, 0)       # columns off centre
        self.erase, self.erase_lbl = self._slider(0, 120, 0)      # patch size
        self.pf, self.pf_lbl = self._slider(50, 100, 100)         # /100 -> fraction kept
        self.pf_fill = QCheckBox("rebuild the rest from Hermitian symmetry")
        self.pf_fill.setChecked(True)
        self.keep = QComboBox(); self.keep.addItems(list(KEEP_LABELS))

        self.upscale = QComboBox(); self.upscale.addItems(["1", "2", "4"])
        self.interp = QComboBox(); self.interp.addItems(list(INTERP_LABELS))

        # Window/level: brightness and contrast on the picture that is already
        # reconstructed. Nothing is recomputed and no k-space is touched -- this
        # is the control a radiologist actually reaches for to read soft tissue,
        # because grey matter and white matter differ by a few percent of the
        # full range and the eye cannot see that spread across black-to-white.
        self.wl_centre, self.wl_centre_lbl = self._slider(0, 100, 50)   # /100
        # Width stops at 1.00 because the reconstruction is normalised to 0..1
        # before it reaches the screen: a wider window cannot show anything
        # more, it can only map black and white to two greys and lose contrast.
        self.wl_width, self.wl_width_lbl = self._slider(1, 100, 100)    # /100
        self.wl_reset = QPushButton("Reset to the full range")
        self.wl_auto = QPushButton("Fit to tissue")   # app.py measures it

        # Everything the range dialog is allowed to retune, and where it started.
        # The slice slider is deliberately absent: the file decides how many
        # slices it has, so a hand-set range there could only be wrong.
        self.sliders = {n: getattr(self, n) for n in LIMITS}
        self.default_ranges = {n: (s.minimum(), s.maximum())
                               for n, s in self.sliders.items()}

        # ---------------- sections, all open ----------------
        # Plain titled sections instead of an accordion: each section has only
        # a few rows, so hiding them behind a click cost more than it saved.
        sections = []
        sections.append((self._page([
            ("source", "dataset", self.source),
            ("slice_idx", "slice", self._slider_row(self.slice_idx, self.slice_lbl)),
            ("load_row", None, self._row(self.load_btn, self.phantom_btn)),
        ]), "1 · Data source"))

        sections.append((self._page([
            ("rate", "sampling rate  (x Nyquist rate)", self._slider_row(self.rate, self.rate_lbl)),
        ]), "2 · Sampling"))

        sections.append((self._page([
            ("average", "repetitions averaged", self.average),
            ("nfloor", "noise-floor filter", self._slider_row(self.nfloor, self.nfloor_lbl)),
            ("window", "apodization", self.window),
            ("sigma", "window sigma", self._slider_row(self.sigma, self.sigma_lbl)),
            ("sharpen", "sharpen (unsharp mask)", self._slider_row(self.sharpen, self.sharpen_lbl)),
            ("pf", "partial Fourier", self._slider_row(self.pf, self.pf_lbl)),
            ("pf_fill", None, self.pf_fill),
            ("noise", "add noise", self._slider_row(self.noise, self.noise_lbl)),
        ]), "3 · Denoising"))

        sections.append((self._page([
            ("dc", "DC term", self._slider_row(self.dc, self.dc_lbl)),
            ("spike", "spike offset", self._slider_row(self.spike, self.spike_lbl)),
            ("erase", "erase centre", self._slider_row(self.erase, self.erase_lbl)),
            ("keep", "k-space keeps", self.keep),
        ]), "4 · k-space edits"))

        sections.append((self._page([
            ("upscale", "upscale by", self.upscale),
            ("interp", "interpolation", self.interp),
        ]), "5 · Resolution"))

        sections.append((self._page([
            ("wl_centre", "level (centre)", self._slider_row(self.wl_centre, self.wl_centre_lbl)),
            ("wl_width", "window (width)", self._slider_row(self.wl_width, self.wl_width_lbl)),
            ("wl_buttons", None, self._row(self.wl_auto, self.wl_reset)),
        ]), "6 · Display  (window / level)"))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        for page, title in sections:
            group = QGroupBox(title)
            group.setStyleSheet("QGroupBox { font-weight: bold; }")
            lay = QVBoxLayout(group)
            lay.setContentsMargins(0, 4, 0, 0)
            lay.addWidget(page)
            outer.addWidget(group)
        outer.addStretch(1)

        # ---------------- wiring ----------------
        for combo in (self.source, self.average, self.window, self.keep,
                      self.upscale, self.interp):
            combo.currentTextChanged.connect(self._emit)
        for slider in (self.rate, self.sigma, self.noise, self.nfloor, self.sharpen,
                       self.slice_idx,
                       self.dc, self.spike, self.erase, self.pf):
            slider.valueChanged.connect(self._emit)
        self.pf_fill.toggled.connect(self._emit)
        # Window/level does not go through _emit: it changes nothing the engine
        # computes, so triggering a reconstruction for it would be wasted work.
        for slider in (self.wl_centre, self.wl_width):
            slider.valueChanged.connect(self._emit_display)
        self.wl_reset.clicked.connect(self.reset_display)
        self.load_btn.clicked.connect(self.loadImageClicked)

        self._update_visibility()

    # ------
    # Layout helpers
    # ------

    def _page(self, entries) -> QWidget:
        """One accordion page: a form built from (name, label, widget) rows."""
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(8, 4, 8, 6)
        for name, label, field in entries:
            if label is None:
                form.addRow(field)
            else:
                form.addRow(label, field)
            self.rows[name] = field
            self._forms[name] = form
        return page

    def _slider(self, lo, hi, val):
        s = QSlider(Qt.Horizontal); s.setRange(lo, hi); s.setValue(val)
        lbl = QLabel(str(val)); lbl.setMinimumWidth(38)
        return s, lbl

    def _step_btn(self, slider, delta):
        b = QToolButton(); b.setText("-" if delta < 0 else "+")
        b.setAutoRepeat(True)  # hold to keep stepping
        # The step is read at click time, not fixed at build time: widening a
        # slider to 0..1000 in the range dialog would otherwise leave these
        # buttons nudging it one unit at a time, which is useless.
        def step():
            span = slider.maximum() - slider.minimum()
            slider.setValue(slider.value() + delta*max(1, span//100))
        b.clicked.connect(step)
        return b

    def _slider_row(self, slider, lbl):
        return self._row(self._step_btn(slider, -1), slider,
                         self._step_btn(slider, +1), lbl)

    def _row(self, *widgets):
        box = QHBoxLayout(); w = QWidget()
        for x in widgets:
            if x is not None:
                box.addWidget(x)
        box.setContentsMargins(0, 0, 0, 0); w.setLayout(box)
        return w

    def _set_visible(self, name: str, shown: bool) -> None:
        self._forms[name].setRowVisible(self.rows[name], shown)

    # ------
    # Which rows apply right now
    # ------

    def _update_visibility(self) -> None:
        """
        Hides every row the current mode would ignore.

        Averaging needs more than one scan of the slice. Adding noise is for the phantom only: a real scan already
        carries its own. Window sigma is read only by the gaussian window, the
        Hermitian checkbox only matters once partial Fourier is actually on,
        and the interpolation method only matters once you are upscaling.
        """
        real = self.source.currentIndex() > 0              # index 0 is the phantom
        for name, shown in {
            "slice_idx": real,
            "average":   real and self.average.count() > 1,
            "noise":     not real,
            "sigma":     self.window.currentText() == "gaussian",
            "pf_fill":   self.pf.value() < 100,
            "interp":    self.upscale.currentText() != "1",
        }.items():
            self._set_visible(name, shown)

    # ------
    # State
    # ------

    def set_repetitions(self, n: int) -> None:
        """How many scans of this slice exist; offers 1..n to average."""
        self.average.blockSignals(True)
        self.average.clear()
        self.average.addItem("1  (this scan only)")
        for i in range(2, n + 1):
            self.average.addItem(f"{i}  (this scan + {i-1} more)")
        self.average.blockSignals(False)
        self._update_visibility()

    def display_range(self) -> tuple[float, float]:
        """(vmin, vmax) for the reconstruction panel, from the centre and width."""
        centre = self.wl_centre.value()/100.0
        width = max(0.01, self.wl_width.value()/100.0)
        return centre - width/2.0, centre + width/2.0

    def reset_display(self) -> None:
        self.wl_centre.setValue(50)
        self.wl_width.setValue(100)

    def _emit_display(self, *_):
        lo, hi = self.display_range()
        self.wl_centre_lbl.setText(f"{self.wl_centre.value()/100:.2f}")
        self.wl_width_lbl.setText(f"{self.wl_width.value()/100:.2f}")
        self.displayChanged.emit(lo, hi)

    def settings(self) -> dict:
        """Current state of every knob, in pipeline.reconstruct's vocabulary."""
        real = self.source.currentIndex() > 0
        return {
            "rate": self.rate.value() / 100.0,
            "average": self.average.currentIndex() + 1,
            "noise_filter": self.nfloor.value() / 100.0,
            "sharpen": self.sharpen.value() / 100.0,
            "window": self.window.currentText(),
            "sigma": self.sigma.value() / 100.0,
            # A real scan brings its own noise; the slider is hidden then, and a
            # value left over from the phantom must not leak into it.
            "noise": 0.0 if real else self.noise.value() / 100.0,
            "seed": 0,           # fixed, so the added noise stays comparable
            "keep": KEEP_LABELS[self.keep.currentText()],
            "dc_scale": self.dc.value() / 100.0,
            "spike": self.spike.value(),
            "erase": self.erase.value(),
            "partial_fourier": self.pf.value() / 100.0,
            "pf_fill": self.pf_fill.isChecked(),
            "upscale": int(self.upscale.currentText()),
            "interp": INTERP_LABELS[self.interp.currentText()],
        }

    def _emit(self, *_):
        self.rate_lbl.setText(f"{self.rate.value()/100:.2f}")
        self.sigma_lbl.setText(f"{self.sigma.value()/100:.2f}")
        self.noise_lbl.setText(f"{self.noise.value()/100:.2f}")
        for sl, lbl in ((self.nfloor, self.nfloor_lbl), (self.sharpen, self.sharpen_lbl)):
            lbl.setText("off" if sl.value() == 0 else f"{sl.value()/100:.2f}")
        self.slice_lbl.setText(str(self.slice_idx.value()))
        self.dc_lbl.setText(f"{self.dc.value()/100:.2f}")
        self.pf_lbl.setText(f"{self.pf.value()/100:.2f}")
        for sl, lbl in ((self.spike, self.spike_lbl),
                        (self.erase, self.erase_lbl)):
            lbl.setText("off" if sl.value() == 0 else str(sl.value()))
        self._update_visibility()
        self.settingsChanged.emit(self.settings())
