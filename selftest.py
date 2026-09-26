"""Headless wiring check for app.py; run with `python app.py --selftest`."""
import sys
import time
import numpy as np
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QGroupBox

from controls import LIMITS, RangeDialog
from mri.pipeline import reconstruct
from mri.raw_data import find_repetitions


def run(app, win):
    main = sys.modules["__main__"]  # app.py, run as a script
    GREEN, RED, Prolog = main.GREEN, main.RED, main.Prolog
    c, kv, bv, rv = win.controls, win.kview, win.bview, win.rview
    vis = lambda n: c._forms[n].isRowVisible(c.rows[n])

    def wait_done():
        for _ in range(600):
            app.processEvents()
            if win.pending is None and win.state and not win.state.startswith("reconstructing"):
                return
            if win.worker is not None:
                win.worker.wait(50)
        raise AssertionError(win.state)

    wait_done()
    assert "256 of 256 rows" in win.details_text and "256 of 256 columns" in win.details_text, win.details_text
    assert win.state.startswith("ready") and "0.0 %" in win.info_text
    assert all(n in win.info_text for n in ("Aliasing loss", "SNR", "SSIM", "PSNR"))

    before = rv.get_array().copy()
    c.rate.setValue(50); wait_done()
    assert not (before == rv.get_array()).all(), "rate changed nothing"
    now, delta = win.score_cells["Aliasing loss"]
    assert float(now.text().split()[0]) > 20 and delta.get_color() == RED
    assert win.score_cells["SSIM"][1].text().startswith("-") and win.score_cells["SSIM"][1].get_color() == RED
    c.rate.setValue(100); wait_done()
    assert win.score_cells["Aliasing loss"][1].text() == "±0"
    c.rate.setValue(50); wait_done()

    untouched, base = bv.get_array().copy(), rv.get_array().copy()
    for name, w, v in (("dc", c.dc, 40), ("spike", c.spike, 20), ("erase", c.erase, 40),
                       ("partial fourier", c.pf, 60), ("sharpen", c.sharpen, 50)):
        w.setValue(v); wait_done()
        assert not (base == rv.get_array()).all(), f"{name} changed nothing"
        assert (untouched == bv.get_array()).all(), f"{name} moved the base"
        w.setValue(100 if w in (c.dc, c.pf) else 0); wait_done()
    c.keep.setCurrentText("phase only"); wait_done()
    assert not (base == rv.get_array()).all()
    c.keep.setCurrentText("magnitude and phase"); wait_done()

    kbefore = kv.get_array().copy()
    c.erase.setValue(40); wait_done()
    assert np.abs(kv.get_array()[:40, :40] - kbefore[:40, :40]).max() < 1e-12, "erase moved the k-space scale"
    c.erase.setValue(0); wait_done()
    c.rate.setValue(100); wait_done()

    for order in ("linear", "centric"):
        c.acq_order.setCurrentText(order)
        c.acq.setValue(250); wait_done()
        shown = kv.get_array() > 0
        if order == "linear":
            assert shown[:60].any() and not shown[70:].any()
        else:
            assert shown[108:148].any() and not shown[:90].any()
    c.acq_order.setCurrentText("linear")
    c.acq.setValue(0); wait_done()
    c.play_btn.setChecked(True)
    assert c.play_btn.text() == "Pause"
    seen, t0 = set(), time.perf_counter()
    while c.play_btn.isChecked() and time.perf_counter() - t0 < 30:
        app.processEvents(); seen.add(round(c.settings()["acquired"], 1)); time.sleep(0.005)
    wait_done()
    assert not c.play_btn.isChecked() and c.acq.value() == c.acq.maximum() and len(seen) >= 8, sorted(seen)
    assert c.play_btn.text() == "Play"
    c.rewind_btn.click(); wait_done()
    assert c.acq.value() == 0
    c.acq.setValue(c.acq.maximum()); wait_done()

    assert not bv.isVisible()
    c.rate.setValue(50); wait_done(); app.processEvents()
    alone = rv.grab().toImage()
    c.compare.setChecked(True); app.processEvents()
    assert rv._compare is bv and rv.get_title().startswith("base")
    split, t = rv.grab().toImage(), rv._fit()
    lp, rp = QPoint(t.x() + t.width()//4, t.center().y()), QPoint(t.x() + 3*t.width()//4, t.center().y())
    assert split.pixel(rp) == alone.pixel(rp) and split.pixel(lp) != alone.pixel(lp)
    app.sendEvent(rv, QMouseEvent(QMouseEvent.MouseButtonPress, QPointF(t.x() + t.width()*0.2, t.center().y()),
                                  Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    assert abs(rv.get_split() - 0.2) < 0.02, rv.get_split()
    c.compare.setChecked(False); app.processEvents()
    assert rv._compare is None and not rv.get_title().startswith("base")
    c.rate.setValue(100); wait_done()

    full_k = kv.get_array().copy()
    for frac in (100, 400, 800):
        c.acq.setValue(frac); wait_done()
        part = kv.get_array()
        assert np.abs(part[part > 0] - full_k[part > 0]).max() < 1e-12, "a sample changed colour"
    c.acq.setValue(c.acq.maximum()); wait_done()

    assert not vis("interp")
    c.upscale.setCurrentText("2")
    assert vis("interp")
    wait_done()
    assert rv.get_array().shape == (512, 512) and "image size 512 × 512" in win.details_text
    assert bv.get_array().shape == (256, 256)
    c.upscale.setCurrentText("1"); wait_done()
    assert "image size" not in win.details_text

    groups = c.findChildren(QGroupBox)
    assert len(groups) == 6 and all(g.isVisible() for g in groups)
    assert c.compare.parent() is not c and rv.isAncestorOf(c.compare)
    sizes = win.centralWidget().sizes()
    assert abs(sizes[0]/sum(sizes) - 0.62) < 0.05, sizes
    assert kv.geometry().top() == rv.geometry().top() and kv.height() > win.score_box.height()

    assert c._forms["dc"] is c._forms["keep"], "DC sits with the k-space edits"
    assert vis("rate") and not vis("average") and not vis("sigma")
    c.window.setCurrentText("gaussian")
    assert vis("sigma")
    c.window.setCurrentText("none"); wait_done()

    dlg = RangeDialog(c, win)
    assert set(dlg.spins) == set(LIMITS) and c.sigma.maximum() == 100
    lo_spin, hi_spin = dlg.spins["sigma"]
    hi_spin.setValue(10_000)
    assert hi_spin.value() == LIMITS["sigma"][2], "spin box caps at the limit"
    hi_spin.setValue(300); dlg.apply()
    assert c.sigma.maximum() == 300
    c.sigma.setValue(280); wait_done()
    assert c.settings()["sigma"] == 2.8
    lo_spin.setValue(300); hi_spin.setValue(300); dlg.apply()
    assert c.sigma.minimum() < c.sigma.maximum(), "an empty range is refused"
    dlg.restore(); wait_done()
    assert (c.sigma.minimum(), c.sigma.maximum()) == c.default_ranges["sigma"]

    pixels = rv.get_array().copy()
    c.wl_centre.setValue(30); c.wl_width.setValue(40); app.processEvents()
    lo, hi = rv.get_clim()
    assert abs(lo - 0.10) < 1e-9 and abs(hi - 0.50) < 1e-9 and (rv.get_array() == pixels).all()
    assert "display window" in rv.get_title() and bv.get_clim() == (0.0, 1.0)
    c.wl_reset.click(); app.processEvents()
    assert rv.get_clim() == (0.0, 1.0) and rv.get_title() == "augmented reconstruction"
    c.rate.setValue(50); wait_done()
    c.wl_auto.click(); app.processEvents()
    lo, hi = rv.get_clim()
    r = rv.get_array(); tissue = r[r > r[:32, :32].max()]
    assert 0.0 <= lo < hi < 1.0 and (tissue > hi + 0.01).mean() < 0.02
    c.wl_reset.click(); app.processEvents()
    c.rate.setValue(100); wait_done()

    key = win._base_key
    c.sharpen.setValue(10); wait_done()
    assert win._base_key == key, "a slider must not rebuild the base"
    c.sharpen.setValue(0); wait_done()

    for w in (c.dc, c.window):
        get = w.value if hasattr(w, "value") else w.currentIndex
        v = get()
        app.sendEvent(w, QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120),
                                     Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
        assert get() == v, f"the wheel changed {w}"

    bar = win.colorbar
    assert kv.geometry().right() < bar.geometry().left() < rv.geometry().left()
    app.processEvents()
    img, t = bar.grab().toImage(), kv._fit()
    assert img.pixelColor(30, t.top() + 20).lightness() > img.pixelColor(30, t.bottom() - 20).lightness()
    assert abs(kv._fit().top() - rv._fit().top()) <= 1 and abs(kv._fit().height() - rv._fit().height()) <= 1
    assert not c.prolog_btn.isEnabled(), "the phantom has no coils"

    multi = [f for f in win.datasets if len(find_repetitions(str(f))) > 1]
    if not multi:
        print("  no repeated scans on disk, real-data checks skipped")
    else:
        c.source.setCurrentText(multi[0].name); wait_done(); app.processEvents()
        top_h, k_h = win.score_box.height(), kv.height()
        assert vis("average")
        single = rv.get_array().copy()
        c.average.setCurrentIndex(c.average.count() - 1); wait_done()
        msg = win.details_text
        assert f"averaged {c.average.count()} scans" in msg and "phase offsets" in msg, msg
        assert "0.3 T" in msg and "reference" in msg, msg
        c.upscale.setCurrentText("2"); wait_done(); app.processEvents()
        assert "image size" in win.details_text and win.score_box.height() == top_h and kv.height() == k_h
        c.upscale.setCurrentText("1"); wait_done()
        assert win.score_cells["SNR"][1].get_color() == GREEN
        z = c.slice_idx.value()
        k1, ref1, _ = win._scan(z)
        direct = reconstruct(None, c.settings(), kspace=k1, reference=ref1, repeats=win._other_repetitions(z))["recon"]
        assert np.abs(direct - rv.get_array()).max() < 1e-9, "cached average = pipeline average"
        assert not (single == rv.get_array()).all() and (win.base == bv.get_array()).all()

        assert c.prolog_btn.isEnabled()
        win._open_prolog()
        p = win._prolog
        assert p.windowFlags() & Qt.WindowMaximizeButtonHint and any(a.shortcut().toString() == "F11" for a in p.actions())
        assert len(p.coil_k_views) == k1.shape[0] == len(p.coil_img_views) and not p.next_btn.isEnabled()
        app.processEvents(); early = p.coil_k_views[0].get_array().copy()
        t0 = time.perf_counter()
        while not p.revealed and time.perf_counter() - t0 < 20:
            app.processEvents(); time.sleep(0.005)
        assert p.revealed and p.next_btn.isEnabled()
        assert (early > 0).sum() < (p.coil_k_views[0].get_array() > 0).sum(), "reveals outward"
        per = np.abs(np.fft.ifft2(np.fft.ifftshift(k1, axes=(-2, -1)), axes=(-2, -1)))
        assert np.allclose(p.coil_img_views[1].get_array(), per[1]/per[1].max())
        p.next_btn.click(); app.processEvents()
        assert not p.coil_k_views[0].isVisible() and np.allclose(p.final_img_view.get_array(), win.base)
        assert "root sum of squares" in p.caption.text().lower() and p.close_btn.isVisible()
        p.close_btn.click(); app.processEvents()
        assert not p.isVisible()
        eight = Prolog(np.concatenate([k1, k1]), win)
        grid = eight.stage.layout()
        assert len(eight.coil_k_views) == 8 and grid.rowCount() == 4 and grid.columnCount() == 4
        eight._timer.stop(); eight.deleteLater()

        real_load, reads, times = main.load_raw_kspace, [], []
        main.load_raw_kspace = lambda *a, **k: (reads.append(a), real_load(*a, **k))[1]
        for v in (0, 30, 50, 30, 0):
            t0 = time.perf_counter(); c.nfloor.setValue(v); wait_done()
            times.append((time.perf_counter() - t0)*1000)
        main.load_raw_kspace = real_load
        assert reads == [], "a slider tick must not re-read the scan"
        print(f"  slider tick on real data, 3 scans averaged: {np.median(times):.0f} ms (median)")
        c.average.setCurrentIndex(0); wait_done()
        import tempfile, os
        from skimage.io import imsave
        png = os.path.join(tempfile.mkdtemp(), "grad.png")
        imsave(png, (np.outer(np.arange(64), np.ones(64))*4).astype(np.uint8))
        real_dialog = main.QFileDialog.getOpenFileName
        main.QFileDialog.getOpenFileName = lambda *a, **k: (png, "")
        win._load_image(); wait_done()  # loading a picture while on a raw scan must keep the picture
        main.QFileDialog.getOpenFileName = real_dialog
        assert win.raw_path is None and win.image[-1].mean() > win.image[0].mean() and win.state.startswith("ready")

    win._load_phantom(); wait_done()
    assert win.raw_path is None and c.source.currentIndex() == 0 and win.state.startswith("ready")
    print("SELFTEST PASSED")
