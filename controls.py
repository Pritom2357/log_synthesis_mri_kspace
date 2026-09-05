"""
The control panel. Every knob the simulator has, collected into one widget
that emits a single signal: settingsChanged(dict).

The dict it emits is exactly what pipeline.reconstruct expects, so app.py
never translates anything.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QFormLayout, QComboBox, QSlider,
                               QLabel, QCheckBox, QPushButton, QHBoxLayout,
                               QToolButton)


class Controls(QWidget):
    settingsChanged = Signal(dict)
    loadImageClicked = Signal()

    def __init__(self):
        super().__init__()
        form = QFormLayout(self)

        self.sampling = QComboBox(); self.sampling.addItems(["cartesian","radial","spiral"])
        self.mask = QComboBox(); self.mask.addItems(["uniform","variable_density","random","corner_cut"])
        self.window = QComboBox(); self.window.addItems(["none","hamming","gaussian"])

        self.R, self.R_lbl = self._slider(1, 8, 1)
        self.sigma, self.sigma_lbl = self._slider(5, 100, 35)          # /100 -> 0.05..1.0
        self.noise, self.noise_lbl = self._slider(0, 30, 0)            # /100 -> 0.0..0.3
        self.spokes, self.spokes_lbl = self._slider(16, 256, 64)
        self.arms, self.arms_lbl = self._slider(4, 32, 16)

        self.density = QCheckBox("density compensation"); self.density.setChecked(True)
        self.load_btn = QPushButton("Load image..."); self.phantom_btn = QPushButton("Phantom")

        form.addRow("sampling", self.sampling)
        form.addRow("mask", self.mask)
        form.addRow("acceleration R", self._slider_row(self.R, self.R_lbl))
        form.addRow("window", self.window)
        form.addRow("window sigma", self._slider_row(self.sigma, self.sigma_lbl))
        form.addRow("noise", self._slider_row(self.noise, self.noise_lbl))
        form.addRow("spokes", self._slider_row(self.spokes, self.spokes_lbl))
        form.addRow("spiral arms", self._slider_row(self.arms, self.arms_lbl))
        form.addRow(self.density)
        form.addRow(self._row(self.load_btn, None, self.phantom_btn))

        for combo in (self.sampling, self.mask, self.window):
            combo.currentTextChanged.connect(self._emit)
        for slider in (self.R, self.sigma, self.noise, self.spokes, self.arms):
            slider.valueChanged.connect(self._emit)
        self.density.toggled.connect(self._emit)
        self.load_btn.clicked.connect(self.loadImageClicked)

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
            if x is not None: box.addWidget(x)
        box.setContentsMargins(0, 0, 0, 0); w.setLayout(box)
        return w

    def settings(self)->dict:
        """Current state of every knob, in pipeline.reconstruct's vocabulary."""
        return {
            "sampling": self.sampling.currentText(),
            "mask": self.mask.currentText(),
            "R": self.R.value(),
            "window": self.window.currentText(),
            "sigma": self.sigma.value()/100.0,
            "noise": self.noise.value()/100.0,
            "seed": 0,  # fixed seed so dragging the noise slider is repeatable
            "n_spokes": self.spokes.value(),
            "n_interleaves": self.arms.value(),
            "density_comp": self.density.isChecked(),
        }

    def _emit(self, *_):
        self.R_lbl.setText(str(self.R.value()))
        self.sigma_lbl.setText(f"{self.sigma.value()/100:.2f}")
        self.noise_lbl.setText(f"{self.noise.value()/100:.2f}")
        self.spokes_lbl.setText(str(self.spokes.value()))
        self.arms_lbl.setText(str(self.arms.value()))
        self.settingsChanged.emit(self.settings())
