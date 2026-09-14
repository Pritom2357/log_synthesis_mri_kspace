"""
The control panel. Every knob the simulator has, collected into one widget
that emits a single signal: settingsChanged(dict).

The dict it emits is exactly what pipeline.reconstruct expects, so app.py
never translates anything.

Rows hide themselves when they cannot do anything. A slider that the current
sampling mode ignores is worse than useless: it invites you to drag it and
conclude the engine is broken.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QFormLayout, QComboBox, QSlider,
                               QLabel, QCheckBox, QPushButton, QHBoxLayout,
                               QToolButton, QFrame)

# The dropdown says it in full; the pipeline wants the short key.
RECON_LABELS = {"zero-filled": "zero_filled", "compressed sensing": "cs"}


class Controls(QWidget):
    settingsChanged = Signal(dict)
    loadImageClicked = Signal()

    def __init__(self):
        super().__init__()
        self.form = QFormLayout(self)

        # --- what we sample ---
        self.sampling = QComboBox(); self.sampling.addItems(["cartesian","radial","spiral"])
        self.mask = QComboBox(); self.mask.addItems(["uniform","variable_density","random","corner_cut"])
        self.order = QComboBox(); self.order.addItems(["sequential","golden"])
        self.window = QComboBox(); self.window.addItems(["none","hamming","gaussian"])

        self.R, self.R_lbl = self._slider(1, 8, 1)
        self.sigma, self.sigma_lbl = self._slider(5, 100, 35)          # /100 -> 0.05..1.0
        self.noise, self.noise_lbl = self._slider(0, 30, 0)            # /100 -> 0.0..0.3
        self.spokes, self.spokes_lbl = self._slider(16, 256, 64)
        self.arms, self.arms_lbl = self._slider(4, 32, 16)
        self.density = QCheckBox("density compensation"); self.density.setChecked(True)

        # --- patient motion ---
        self.motion = QComboBox(); self.motion.addItems(["none","sudden","periodic","random"])
        self.motion_amp, self.motion_amp_lbl = self._slider(0, 10, 0)  # pixels
        self.motion_cyc, self.motion_cyc_lbl = self._slider(1, 10, 4)

        # --- how we rebuild, and how much of the head we scan ---
        self.recon = QComboBox(); self.recon.addItems(list(RECON_LABELS))
        self.fov = QComboBox(); self.fov.addItems(["1","2","4","8"])
        self.crop = QComboBox(); self.crop.addItems(["1","2","4","8"])

        self.load_btn = QPushButton("Load image..."); self.phantom_btn = QPushButton("Phantom")

        # Rows that come and go are kept by name so _update_visibility can find them.
        self.rows:dict[str,QWidget] = {}

        self._section("Acquisition")
        self._add("sampling", "sampling", self.sampling)
        self._add("mask", "mask", self.mask)
        self._add("R", "acceleration R", self._slider_row(self.R, self.R_lbl))
        self._add("spokes", "spokes", self._slider_row(self.spokes, self.spokes_lbl))
        self._add("order", "spoke order", self.order)
        self._add("arms", "spiral arms", self._slider_row(self.arms, self.arms_lbl))
        self._add("density", None, self.density)

        self._section("Signal")
        self._add("window", "window", self.window)
        self._add("sigma", "window sigma", self._slider_row(self.sigma, self.sigma_lbl))
        self._add("noise", "noise", self._slider_row(self.noise, self.noise_lbl))

        self._section("Patient motion")
        self._add("motion", "motion", self.motion)
        self._add("motion_amp", "amplitude (px)", self._slider_row(self.motion_amp, self.motion_amp_lbl))
        self._add("motion_cyc", "cycles", self._slider_row(self.motion_cyc, self.motion_cyc_lbl))

        self._section("Reconstruction")
        self._add("recon", "method", self.recon)

        self._section("Field of view")
        self._add("fov", "reduce FOV by", self.fov)
        self._add("crop", "crop k-space by", self.crop)

        self.form.addRow(self._row(self.load_btn, None, self.phantom_btn))

        for combo in (self.sampling, self.mask, self.order, self.window,
                      self.motion, self.recon, self.fov, self.crop):
            combo.currentTextChanged.connect(self._emit)
        for slider in (self.R, self.sigma, self.noise, self.spokes, self.arms,
                       self.motion_amp, self.motion_cyc):
            slider.valueChanged.connect(self._emit)
        self.density.toggled.connect(self._emit)
        self.load_btn.clicked.connect(self.loadImageClicked)

        self._update_visibility()

    # ------
    # Layout helpers
    # ------

    def _section(self, text:str)->None:
        """A small section heading with a rule under it."""
        lbl = QLabel(text); lbl.setStyleSheet("font-weight: bold; margin-top: 6px;")
        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)
        self.form.addRow(self._row(lbl, line))

    def _add(self, name:str, label:str|None, field:QWidget)->None:
        """Adds a row and remembers its field widget so it can be hidden later."""
        if label is None:
            self.form.addRow(field)
        else:
            self.form.addRow(label, field)
        self.rows[name] = field

    def _slider(self, lo, hi, val):
        s = QSlider(Qt.Horizontal); s.setRange(lo, hi); s.setValue(val)
        lbl = QLabel(str(val)); lbl.setMinimumWidth(34)
        return s, lbl

    def _step_btn(self, slider, delta):
        b = QToolButton(); b.setText("-" if delta < 0 else "+")
        b.setAutoRepeat(True)  # hold to keep stepping
        b.clicked.connect(lambda: slider.setValue(slider.value() + delta))
        return b

    def _slider_row(self, slider, lbl):
        return self._row(self._step_btn(slider, -1), slider,
                         self._step_btn(slider, +1), lbl)

    def _row(self, *widgets):
        box = QHBoxLayout(); w = QWidget()
        for x in widgets:
            if x is not None: box.addWidget(x, stretch=1 if isinstance(x, QFrame) else 0)
        box.setContentsMargins(0, 0, 0, 0); w.setLayout(box)
        return w

    # ------
    # Which rows apply right now
    # ------

    def _update_visibility(self)->None:
        """
        Hides every row the current mode would ignore.

        Cartesian sampling reads the mask and R; radial reads spokes and the
        spoke order; spiral reads the arm count. Density compensation only means
        anything where samples are accumulated, so never for Cartesian. Window
        sigma is read only by the gaussian window, and the motion shape decides
        whether an amplitude or a cycle count is meaningful.
        """
        samp = self.sampling.currentText()
        motion = self.motion.currentText()
        visible = {
            "mask":       samp == "cartesian",
            "R":          samp == "cartesian",
            "spokes":     samp == "radial",
            "order":      samp == "radial",
            "arms":       samp == "spiral",
            "density":    samp in ("radial","spiral"),
            "sigma":      self.window.currentText() == "gaussian",
            "motion_amp": motion != "none",
            "motion_cyc": motion == "periodic",
        }
        for name, shown in visible.items():
            self.form.setRowVisible(self.rows[name], shown)

    # ------
    # State
    # ------

    def settings(self)->dict:
        """Current state of every knob, in pipeline.reconstruct's vocabulary."""
        return {
            "sampling": self.sampling.currentText(),
            "mask": self.mask.currentText(),
            "R": self.R.value(),
            "window": self.window.currentText(),
            "sigma": self.sigma.value()/100.0,
            "noise": self.noise.value()/100.0,
            "seed": 0,  # fixed seed so random masks and noise stay comparable
            "n_spokes": self.spokes.value(),
            "n_interleaves": self.arms.value(),
            "order": self.order.currentText(),
            "density_comp": self.density.isChecked(),
            "motion": self.motion.currentText(),
            "motion_amp": float(self.motion_amp.value()),
            "motion_cycles": float(self.motion_cyc.value()),
            "recon": RECON_LABELS[self.recon.currentText()],
            "fov": int(self.fov.currentText()),
            "crop": int(self.crop.currentText()),
        }

    def _emit(self, *_):
        self.R_lbl.setText(str(self.R.value()))
        self.sigma_lbl.setText(f"{self.sigma.value()/100:.2f}")
        self.noise_lbl.setText(f"{self.noise.value()/100:.2f}")
        self.spokes_lbl.setText(str(self.spokes.value()))
        self.arms_lbl.setText(str(self.arms.value()))
        self.motion_amp_lbl.setText(str(self.motion_amp.value()))
        self.motion_cyc_lbl.setText(str(self.motion_cyc.value()))
        self._update_visibility()
        self.settingsChanged.emit(self.settings())
