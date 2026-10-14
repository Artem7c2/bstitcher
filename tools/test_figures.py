"""Visual test report: for every test scenario plot input, expected and actual output.

Re-runs the scenarios of the test suite (same generators, seeds and parameters)
and draws one figure per scenario, panels left to right:
    Input | Expected | Actual | Difference (actual - expected)
Each figure gets a PASS/FAIL verdict computed with the same criterion as the test.

    python tools/test_figures.py --out tmp/test_figures [--fixture <bruker job dir>]

Writes fig_XX_*.png and figures.json (title, test, caption, verdict) to --out.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from synthetic import make_mosaic, relative, textured_image  # noqa: E402

from bstitcher.georef import physical_geotransform, pixel_geotransform  # noqa: E402
from bstitcher.layout import nominal_positions  # noqa: E402
from bstitcher.registration import _ncc, box_smooth, register_pair  # noqa: E402
from bstitcher.stitch import (auto_max_shift, compose, owner_map, solve_positions,  # noqa: E402
                              stitch_layout, write_mosaic_streamed)
from bstitcher.tiff_io import read_info, read_tiff, write_tiff  # noqa: E402
from bstitcher.tile import Tile, export_tile, integrate_windows  # noqa: E402

FIGS: list[dict] = []
OUT: Path
TMP: Path


# ------------------------------------------------------------------ helpers
def save(fig, name, title, test, caption, passed, step):
    fig.suptitle(f"{title}   [{'PASS' if passed else 'FAIL'}]", fontsize=12,
                 color="darkgreen" if passed else "firebrick", fontweight="bold", y=1.02)
    fig.savefig(OUT / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    FIGS.append({"file": f"{name}.png", "step": step, "title": title, "test": test,
                 "caption": caption, "passed": bool(passed)})
    print(f"{'PASS' if passed else 'FAIL'}  {name}")


def show(ax, img, title, cmap="gray", **kw):
    im = ax.imshow(img, cmap=cmap, interpolation="nearest", **kw)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def show_diff(ax, d, title="actual − expected"):
    d = np.asarray(d, dtype=np.float64)
    lim = max(float(np.nanmax(np.abs(d))), 1e-9)
    im = show(ax, d, f"{title}\nmax |Δ| = {np.nanmax(np.abs(d)):g}", cmap="RdBu_r", vmin=-lim, vmax=lim)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def boxes(ax, positions, shapes, color, ls="-", label=True):
    for idx, (r, c) in positions.items():
        h, w = shapes[idx]
        ax.add_patch(Rectangle((c - .5, r - .5), w, h, fill=False, ec=color, lw=1.2, ls=ls))
        if label:
            ax.text(c + 3, r + 9, f"{idx[0]},{idx[1]}", color=color, fontsize=7)


def nominal_layout(layout):
    """Same layout, but every tile at its rounded stage-nominal position (what Bruker does)."""
    nom = {k: (round(v[0]), round(v[1])) for k, v in layout.nominal.items() if k in layout.positions}
    r0, c0 = min(p[0] for p in nom.values()), min(p[1] for p in nom.values())
    pos = {k: (p[0] - r0, p[1] - c0) for k, p in nom.items()}
    H = max(pos[i][0] + layout.shapes[i][0] for i in pos)
    W = max(pos[i][1] + layout.shapes[i][1] for i in pos)
    return dataclasses.replace(layout, positions=pos, canvas_shape=(H, W))


def truth_crop(truth, true_pos, shape):
    r0 = min(p[0] for p in true_pos.values())
    c0 = min(p[1] for p in true_pos.values())
    return truth[r0:r0 + shape[0], c0:c0 + shape[1]]


def ncc_surface(a, b, offset, max_shift, smooth=1, min_frac=0.5):
    """NCC for every candidate shift, exactly as register_pair scores it (NaN = not evaluated)."""
    a = box_smooth(np.asarray(a, float), smooth)
    b = box_smooth(np.asarray(b, float), smooth)
    (ha, wa), (hb, wb) = a.shape, b.shape
    nom = max(min(ha, offset[0] + hb) - max(0, offset[0]), 0) * max(min(wa, offset[1] + wb) - max(0, offset[1]), 0)
    s = np.full((2 * max_shift + 1,) * 2, np.nan)
    for i, dr in enumerate(range(-max_shift, max_shift + 1)):
        for j, dc in enumerate(range(-max_shift, max_shift + 1)):
            ro, co = offset[0] + dr, offset[1] + dc
            r0, r1, c0, c1 = max(0, ro), min(ha, ro + hb), max(0, co), min(wa, co + wb)
            n = max(r1 - r0, 0) * max(c1 - c0, 0)
            if n >= 4 and n >= min_frac * nom:
                s[i, j] = _ncc(a[r0:r1, c0:c1], b[r0 - ro:r1 - ro, c0 - co:c1 - co])
    return s


# ------------------------------------------------------- step 1: TIFF I/O
def fig_tiff_roundtrip():
    img = textured_image((120, 160), seed=21)
    cube = np.stack([(img >> s) % 50 for s in range(0, 12)], axis=-1).astype(np.uint8)
    write_tiff(TMP / "rt.tif", img)
    write_tiff(TMP / "rt_cube.tif", cube, band_names=[f"ch{i}" for i in range(cube.shape[2])])
    back, _ = read_tiff(TMP / "rt.tif")
    cback, _ = read_tiff(TMP / "rt_cube.tif")
    ok = np.array_equal(back, img) and back.dtype == img.dtype and np.array_equal(cback, cube)
    size = (TMP / "rt.tif").stat().st_size
    fig, ax = plt.subplots(1, 5, figsize=(19, 3.6))
    show(ax[0], img, f"input array\n{img.dtype}, {img.shape}")
    show(ax[1], img, "expected read-back\n(identical to input)")
    show(ax[2], back, f"actual read-back\n{back.dtype}, file {size / img.nbytes:.0%} of raw (ZSTD)")
    show_diff(ax[3], back.astype(int) - img.astype(int))
    ax[4].plot(cube[60, 80], "o-", label="input spectrum @ (60,80)")
    ax[4].plot(cback[60, 80], "x--", label="read-back")
    ax[4].set_title("3-D cube (H, W, C) round trip", fontsize=9)
    ax[4].set_xlabel("channel")
    ax[4].legend(fontsize=7)
    save(fig, "fig_01_tiff_roundtrip", "Lossless TIFF round trip", "test_tiff_io::test_2d_roundtrip_is_bit_exact, "
         "test_spectral_cube_roundtrip_hwc, test_compression_is_applied_and_lossless_codec",
         "An image and a spectral cube are written with ZSTD + predictor and read back. Pass = bit-identical "
         "values and dtype.", ok, 1)


def fig_georef():
    import rasterio
    img = textured_image((40, 60), seed=22)
    stage, px = (-2993.75, 3665.25), (4.0, 4.0)
    # tile centre at -stage (world = -stage), top-left = centre - half size
    origin = (-stage[0] - 60 / 2 * px[0], -stage[1] + 40 / 2 * px[1])
    gt = physical_geotransform(origin_um=origin, pixel_size_um=px)
    write_tiff(TMP / "loc.tif", img, geotransform=gt, crs="local")
    write_tiff(TMP / "pix.tif", img, geotransform=pixel_geotransform(40), crs="pixel")
    with rasterio.open(TMP / "loc.tif") as ds:
        b_loc = ds.bounds
    with rasterio.open(TMP / "pix.tif") as ds:
        b_pix, epsg = ds.bounds, ds.crs.to_epsg()
    exp_loc = (origin[0] * 1e-6, (origin[1] - 40 * px[1]) * 1e-6, (origin[0] + 60 * px[0]) * 1e-6, origin[1] * 1e-6)
    exp_pix = (0, 0, 60, 40)
    ok = np.allclose(tuple(b_loc), exp_loc) and np.allclose(tuple(b_pix), exp_pix) and epsg == 4326
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    show(ax[0], img, f"input tile 60×40 px\nstage = {stage} µm, pixel {px[0]} µm")
    ax[0].plot(29.5, 19.5, "r+", ms=14)
    for a, b_act, b_exp, unit, f, t in (
            (ax[1], b_loc, exp_loc, "µm", 1e6, "--georef local (LOCAL_CS, metres)"),
            (ax[2], b_pix, exp_pix, "px", 1, "--georef pixel (legacy EPSG:4326)")):
        L, B, R, T = (v * f for v in b_act)
        a.imshow(img, cmap="gray", extent=(L, R, B, T))
        el, eb, er, et = (v * f for v in b_exp)
        a.add_patch(Rectangle((el, eb), er - el, et - eb, fill=False, ec="lime", lw=2, ls="--"))
        if unit == "µm":
            a.plot(-stage[0], -stage[1], "r+", ms=14)
        a.set_title(f"{t}\nimage = actual GDAL bounds, green = expected", fontsize=9)
        a.set_xlabel(f"x [{unit}]")
        a.set_ylabel(f"y [{unit}]")
    save(fig, "fig_02_georef", "Georeferencing as read by GDAL", "test_tiff_io::test_gdal_reads_physical_georef_in_metres, "
         "test_gdal_reads_pixel_georef; test_tile_export::test_tile_geotransform_pixel_and_local",
         "The tile is written with the local and the pixel georeference and opened with rasterio (GDAL). "
         "Pass = GDAL bounds equal the expected extent. In the local frame the stage position (red +) "
         "is the tile centre, and world = −stage.", ok, 1)


# -------------------------------------------- step 2: tile model / export
def _synthetic_cube():
    rng = np.random.default_rng(3)
    h, w, c = 60, 80, 400
    energy = {"offset_keV": -0.2, "scale_keV": 0.025, "size": c}
    e = energy["offset_keV"] + energy["scale_keV"] * np.arange(c)
    lines = {"Mg": 1.25, "Si": 1.74, "Fe": 6.40}
    amps = {el: textured_image((h, w), seed=30 + i) / 4000 for i, el in enumerate(lines)}
    lam = 0.3 + sum(8 * amps[el][..., None] * np.exp(-0.5 * ((e - x) / 0.06) ** 2) for el, x in lines.items())
    return rng.poisson(lam).astype(np.uint8), energy, e, lines


def fig_element_windows():
    cube, energy, e, lines = _synthetic_cube()
    windows = {el: (x - 0.12, x + 0.12) for el, x in lines.items()}
    maps = integrate_windows(cube, energy, windows)
    fig, ax = plt.subplots(len(lines), 4, figsize=(15, 3.2 * len(lines)),
                           gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
    ok = True
    for k, (el, (lo, hi)) in enumerate(windows.items()):
        # expected: channels whose centre lies in [lo, hi], summed by hand
        sel = (e >= lo - 1e-9) & (e <= hi + 1e-9)
        exp = cube[..., sel].sum(-1)
        ok &= np.array_equal(maps[el], exp)
        a = ax[k, 0]
        a.plot(e, cube.mean((0, 1)), lw=.8, color="k")
        a.axvspan(lo, hi, color="orange", alpha=.4)
        a.set_title(f"input: mean spectrum, {el} window {lo:.2f}–{hi:.2f} keV", fontsize=9)
        a.set_xlabel("keV")
        show(ax[k, 1], exp, f"expected {el}: Σ channels in window\n({sel.sum()} channels)", cmap="viridis")
        show(ax[k, 2], maps[el], f"actual integrate_windows → {el}", cmap="viridis")
        show_diff(ax[k, 3], maps[el].astype(int) - exp.astype(int))
    save(fig, "fig_03_element_windows", "Element maps from EDS windows", "test_tile_export::"
         "test_integrate_windows_sums_channel_ranges", "A synthetic EDS cube with Mg, Si and Fe peaks of known "
         "spatial pattern is integrated over keV windows. Pass = equal to a hand-summed channel range.", ok, 2)


def fig_tile_export():
    cube, energy, e, lines = _synthetic_cube()
    bse = textured_image(cube.shape[:2], seed=40)
    fe = integrate_windows(cube, energy, {"Fe": (6.28, 6.52)})["Fe"].astype(np.uint32)
    t = Tile(index=(1, 1), images={"BSE": bse}, element_maps={"Fe": fe}, eds=cube, energy=energy,
             pixel_size_um=(2.0, 2.0), stage_um=(-100.0, 50.0), windows_keV={"Fe": [6.28, 6.52]},
             metadata={"beam_energy_keV": 15.0, "bruker": {"Stage": {"X": -100.0, "Y": 50.0}}},
             source="Mapping_(1,1).bcf")
    files = export_tile(t, TMP / "export", eds=True, georef="local")
    got = {k: read_tiff(v) for k, v in files.items()}
    ok = (np.array_equal(got["BSE"][0], bse) and np.array_equal(got["elem_Fe"][0], fe)
          and np.array_equal(got["EDS"][0], cube) and got["elem_Fe"][0].dtype == np.uint8)
    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    for k, (name, src) in enumerate((("BSE", bse), ("elem_Fe", fe))):
        arr = got[name][0]
        show(ax[k, 0], src, f"input Tile.{'images' if k == 0 else 'element_maps'}[{name.split('_')[-1]!r}]\n{src.dtype}")
        show(ax[k, 1], src, f"expected Mapping_(1,1)_{name}.tif\n(same values)")
        show(ax[k, 2], arr, f"actual file, stored as {arr.dtype}\n{Path(files[name]).stat().st_size / 1024:.1f} kB")
        show_diff(ax[k, 3], arr.astype(int) - src.astype(int))
    md = got["elem_Fe"][1]["metadata"]
    txt = json.dumps({k: md[k] for k in ("modality", "element", "tile_index", "stage_um", "pixel_size_um",
                                          "integration_window_keV", "beam_energy_keV", "bruker")}, indent=1)
    fig.text(0.5, -0.02, "metadata read back from elem_Fe.tif:\n" + txt, ha="center", va="top",
             family="monospace", fontsize=8)
    save(fig, "fig_04_tile_export", "Per-tile export (BSE, element map, EDS)", "test_tile_export::"
         "test_export_tile_writes_all_modalities_losslessly", "A Tile is exported to Mapping_(c,r)_*.tif. Pass = "
         "all files bit-identical, and the uint32 element map (max ≤ 255) is shrunk to the smallest lossless type, "
         "uint8. The metadata is read back as JSON.", ok, 2)


# ----------------------------------------------------- step 3: BCF reader
def fig_bcf_reader(fixture):
    import hyperspy.api as hs
    from bstitcher_bcf.reader import read_bcf
    t = read_bcf(fixture / "Mapping_(2,1).bcf", eds=True)
    s = hs.load(str(fixture / "Mapping_(2,1).bcf"))[-1]
    ref = {m.metadata.Sample.elements[0]: m.data for m in s.get_lines_intensity()}
    els = ["Fe", "Mg", "Si", "Ca"]
    e = t.energy["offset_keV"] + t.energy["scale_keV"] * np.arange(t.energy["size"])
    ok = all(np.array_equal(t.element_maps[el], ref[el].astype(np.uint64)) for el in ref)
    fig = plt.figure(figsize=(16, 3.1 * (len(els) + 1)))
    gs = fig.add_gridspec(len(els) + 1, 4)
    a = fig.add_subplot(gs[0, :3])
    a.semilogy(e, t.eds.sum((0, 1)) + 1, lw=.7, color="k")
    for el, (lo, hi) in t.windows_keV.items():
        a.axvspan(lo, hi, color="orange", alpha=.35)
        a.text((lo + hi) / 2, a.get_ylim()[1] * 0.3, el, ha="center", fontsize=8)
    a.set_xlim(0, 10)
    a.set_title("input Mapping_(2,1).bcf: sum spectrum with the 9 job elements' windows (2 × FWHM)", fontsize=9)
    a.set_xlabel("keV")
    show(fig.add_subplot(gs[0, 3]), t.images["BSE"], f"input BSE {t.images['BSE'].shape}")
    for k, el in enumerate(els, start=1):
        show(fig.add_subplot(gs[k, 0]), t.eds[..., slice(*np.searchsorted(e, t.windows_keV[el]))].sum(-1),
             f"input: cube summed over {el} window", cmap="viridis")
        show(fig.add_subplot(gs[k, 1]), ref[el], f"expected eXSpy get_lines_intensity {el}", cmap="viridis")
        show(fig.add_subplot(gs[k, 2]), t.element_maps[el], f"actual bstitcher-bcf {el}", cmap="viridis")
        show_diff(fig.add_subplot(gs[k, 3]), t.element_maps[el].astype(float) - ref[el])
    fig.tight_layout()
    save(fig, "fig_05_bcf_reader", "BCF reader vs. eXSpy (real tile)", "test_bcf_reader::"
         "test_element_maps_match_exspy_lines_intensity, test_tile_geometry_and_types, test_energy_axis",
         "Real Bruker tile Mapping_(2,1).bcf (88×118 px, 4096 channels). Pass = the element maps are "
         "bit-identical to what eXSpy (the legacy pipeline) produces, for all 9 elements.", ok, 3)


# ---------------------------------------------------- step 4: registration
def fig_registration():
    img = textured_image((260, 300), seed=3)
    rng = np.random.default_rng(0)
    cases = []
    for ts in [(0, 0), (2, -3), (-4, 1), (3, 3)]:
        a = img[20:80, 20:100]
        b = img[20 + ts[0]:80 + ts[0], 20 + 66 + ts[1]:100 + 66 + ts[1]]
        cases.append((f"true shift {ts}", a, b, (0, 66), 5, ts, "match"))
    a = (1000 + rng.normal(0, 3, (60, 80))).astype(np.uint16)
    b = (1000 + rng.normal(0, 3, (60, 80))).astype(np.uint16)
    cases.append(("featureless overlap", a, b, (0, 66), 5, None, "lowscore"))
    img2 = textured_image((260, 300), seed=13)
    a, b = img2[20:120, 20:140], img2[27:127, 122:242]
    cases.append(("true (7,2) inside max_shift=10", a, b, (0, 100), 10, (7, 2), "match"))
    cases.append(("true (7,2) outside max_shift=4", a, b, (0, 100), 4, (7, 2), "boundary"))
    fig, ax = plt.subplots(len(cases), 4, figsize=(15, 3.1 * len(cases)),
                           gridspec_kw={"width_ratios": [1.3, 1.3, 1, 1.3]})
    ok = True
    for k, (name, a, b, off, ms, ts, kind) in enumerate(cases):
        res = register_pair(a, b, off, max_shift=ms)
        surf = ncc_surface(a, b, off, ms)
        if kind == "match":
            good = res.shift == ts and res.score > 0.9 and not res.at_boundary
        elif kind == "lowscore":
            good = res.score < 0.5
        else:
            good = res.at_boundary
        ok &= good
        show(ax[k, 0], a, f"input tile A — {name}")
        show(ax[k, 1], b, f"input tile B, nominal offset {off}")
        im = ax[k, 2].imshow(surf, cmap="magma", extent=(-ms - .5, ms + .5, ms + .5, -ms - .5), vmin=-.2, vmax=1)
        if ts is not None:  # keep the expected marker visible even outside the searched window
            lim = max(ms, abs(ts[0]), abs(ts[1])) + .5
            ax[k, 2].set_xlim(-lim, lim)
            ax[k, 2].set_ylim(lim, -lim)
        plt.colorbar(im, ax=ax[k, 2], fraction=0.046)
        if ts is not None:
            ax[k, 2].plot(ts[1], ts[0], "x", color="cyan", ms=12, mew=2, label="expected")
        ax[k, 2].plot(res.shift[1], res.shift[0], "o", mfc="none", mec="lime", ms=12, mew=2, label="actual")
        ax[k, 2].set_title(f"NCC over ±{ms} px  {'✓' if good else '✗'}", fontsize=9)
        ax[k, 2].set_xlabel("Δcol")
        ax[k, 2].set_ylabel("Δrow")
        ax[k, 2].legend(fontsize=7, loc="upper left")
        ax[k, 3].axis("off")
        exp_txt = {"match": f"shift = {ts}, score > 0.9", "lowscore": "score < 0.5 (unreliable)",
                   "boundary": "at_boundary = True\n(peak outside search → pair not used)"}[kind]
        ax[k, 3].text(0, .5, f"expected: {exp_txt}\n\nactual: shift = {res.shift}\n        score = {res.score:.3f}\n"
                             f"        at_boundary = {res.at_boundary}", family="monospace", fontsize=9, va="center")
    fig.tight_layout()
    save(fig, "fig_06_registration", "Pairwise registration (NCC search)", "test_stitching::"
         "test_register_pair_recovers_shift, test_register_pair_featureless_has_low_score, "
         "test_peak_on_search_boundary_is_flagged", "Two tiles cut from a known image with a known extra shift. "
         "The NCC surface over all candidate shifts is shown: cyan × = expected, green ○ = found. A featureless "
         "overlap must give a low score. A peak outside the search window must be flagged, not trusted.", ok, 4)


def _placement_row(ax, label, truth, recs, images, true_pos, **kw):
    layout = stitch_layout(recs, lambda i: images[i], **kw)
    nl = nominal_layout(layout)
    nom_canvas, _ = compose(nl, lambda i: images[i])
    canvas, own = compose(layout, lambda i: images[i])
    exp = truth_crop(truth, true_pos, canvas.shape)
    cov = own > 0  # pixels covered by a tile (a missing tile leaves a hole)
    good = relative(layout.positions) == relative(true_pos) and np.array_equal(canvas[cov], exp[cov])
    show(ax[0], nom_canvas, f"input: tiles at stage-nominal positions\n{label}")
    boxes(ax[0], nl.positions, layout.shapes, "orange")
    show(ax[1], exp, "expected: ground truth + true tile boxes")
    boxes(ax[1], relative(true_pos), layout.shapes, "cyan")
    show(ax[2], canvas, f"actual: stitched  {'✓' if good else '✗'}")
    boxes(ax[2], layout.positions, layout.shapes, "lime")
    show_diff(ax[3], np.where(cov, canvas.astype(int) - exp.astype(int), np.nan), "actual − expected (covered px)")
    return good, layout


def fig_placement():
    rows = [("4×3, jitter ±3 px", dict(nx=4, ny=3, jitter=3, seed=1), dict(max_shift=6)),
            ("3×3, jitter ±2 px", dict(nx=3, ny=3, jitter=2, seed=2), dict(max_shift=4)),
            ("5×1 strip", dict(nx=5, ny=1, jitter=2, seed=6), dict(max_shift=4)),
            ("tile (2,2) missing", dict(nx=3, ny=3, jitter=2, seed=3, missing=[(2, 2)]), dict(max_shift=4)),
            ("3×2, jitter ±6, auto max_shift", dict(nx=3, ny=2, jitter=6, seed=15, tile=(200, 260),
                                                     step=(170, 220)), {})]
    fig, ax = plt.subplots(len(rows), 4, figsize=(17, 3.6 * len(rows)))
    ok = True
    for k, (label, mk, kw) in enumerate(rows):
        truth, recs, images, true_pos = make_mosaic(**mk)
        good, layout = _placement_row(ax[k], label, truth, recs, images, true_pos, **kw)
        ok &= good
    fig.tight_layout()
    save(fig, "fig_07_placement", "Global placement and composition (synthetic)", "test_stitching::"
         "test_recovers_true_positions_exactly, test_composed_canvas_equals_ground_truth, test_single_row_strip, "
         "test_missing_tile_is_tolerated, test_auto_max_shift_scales_with_tile_size", "Tiles are cut from a known "
         "image at randomly jittered positions, while the stage coordinates give only the regular grid. Left: what "
         "stage positions alone give (seams visible). Pass = every tile at its true position and the mosaic "
         "bit-identical to the ground truth.", ok, 4)


def fig_fallback_and_outlier():
    fig, ax = plt.subplots(2, 4, figsize=(17, 7.5))
    truth, recs, images, true_pos = make_mosaic(nx=3, ny=1, jitter=0, seed=4, blank=[(3, 1)])
    layout = stitch_layout(recs, lambda i: images[i], max_shift=4)
    canvas, _ = compose(layout, lambda i: images[i])
    nom = relative({k: (round(v[0]), round(v[1])) for k, v in nominal_positions(recs).items()})
    ok1 = relative(layout.positions)[(3, 1)] == nom[(3, 1)] and layout.n_fallback_pairs >= 1
    show(ax[0, 0], images[(3, 1)], "input: tile (3,1) is featureless")
    show(ax[0, 1], truth_crop(truth, true_pos, canvas.shape), "expected: (3,1) stays at nominal (orange)")
    boxes(ax[0, 1], nom, layout.shapes, "orange")
    show(ax[0, 2], canvas, f"actual: placed (lime)  {'✓' if ok1 else '✗'}")
    boxes(ax[0, 2], layout.positions, layout.shapes, "lime")
    ax[0, 3].axis("off")
    ax[0, 3].text(0, .5, "pair report:\n" + "\n".join(
        f"{tuple(p['a'])}-{tuple(p['b'])}: score {p['score']:.2f} used={p['used']}" for p in layout.pairs),
                  family="monospace", fontsize=9, va="center")
    # outlier rejection on a 2x2 loop
    nominal = {(1, 1): (0.0, 0.0), (2, 1): (0.0, 50.0), (1, 2): (40.0, 0.0), (2, 2): (40.0, 50.0)}
    meas = {((1, 1), (2, 1)): ((0, 50), 0.9), ((1, 1), (1, 2)): ((40, 0), 0.9),
            ((2, 1), (2, 2)): ((40, 0), 0.9), ((1, 2), (2, 2)): ((0, 58), 0.8)}
    pos, rejected = solve_positions(nominal, meas, max_residual=2.0)
    expected = {(1, 1): (0, 0), (2, 1): (0, 50), (1, 2): (40, 0), (2, 2): (40, 50)}
    ok2 = relative(pos) == expected and ((1, 2), (2, 2)) in rejected
    for a_, panel, title, P in ((ax[1, 0], 0, "input: 4 pair measurements\n(1,2)→(2,2) is 8 px off", None),
                                (ax[1, 1], 1, "expected positions", expected),
                                (ax[1, 2], 2, f"actual solved positions  {'✓' if ok2 else '✗'}", relative(pos))):
        a_.set_xlim(-15, 75)
        a_.set_ylim(60, -15)
        a_.set_aspect("equal")
        a_.set_title(title, fontsize=9)
        base = P or nominal
        for (u, v), (m, _) in meas.items():
            bad = (u, v) in rejected and panel == 2
            col = "red" if (panel == 0 and m == (0, 58)) or bad else "gray"
            a_.annotate("", xy=(base[u][1] + m[1], base[u][0] + m[0]), xytext=(base[u][1], base[u][0]),
                        arrowprops=dict(arrowstyle="->", color=col, ls="--" if bad else "-"))
            a_.text(base[u][1] + m[1] / 2, base[u][0] + m[0] / 2, str(m), fontsize=7, color=col)
        for idx, (r, c) in base.items():
            a_.plot(c, r, "s", color="lime" if panel == 2 else "cyan", ms=10)
            a_.text(c + 2, r - 3, f"{idx}", fontsize=8)
    ax[1, 3].axis("off")
    ax[1, 3].text(0, .5, f"rejected as outlier:\n  {sorted(rejected)}", family="monospace", fontsize=9, va="center")
    fig.tight_layout()
    save(fig, "fig_08_fallback_outlier", "Featureless tile fallback and outlier rejection", "test_stitching::"
         "test_featureless_tile_falls_back_to_nominal, test_solve_positions_rejects_inconsistent_measurement",
         "Top: a tile with no texture cannot be registered, so it keeps its stage position instead of jumping. "
         "Bottom: one of four pair measurements contradicts the loop by 8 px. Pass = it is rejected (red dashed) "
         "and the positions follow the consistent three.", ok1 and ok2, 4)


def fig_seams():
    truth, recs, images, _ = make_mosaic(nx=2, ny=1, jitter=0, tile=(40, 60), step=(0, 40))
    layout = stitch_layout(recs, lambda i: images[i])
    ids = layout.tile_ids
    oc = owner_map(layout, "center")
    ow = owner_map(layout, "overwrite")
    H, W = layout.canvas_shape
    cols = np.arange(W)
    exp_c = np.where(cols < 50, ids[(1, 1)], ids[(2, 1)])[None].repeat(H, 0)  # midpoint between centres 30, 70
    exp_o = np.where(cols < 40, ids[(1, 1)], ids[(2, 1)])[None].repeat(H, 0)
    ok = np.array_equal(oc, exp_c) and np.array_equal(ow, exp_o)
    fig, ax = plt.subplots(2, 4, figsize=(16, 5))
    canvas, _ = compose(layout, lambda i: images[i])
    for k, (name, exp, act) in enumerate((("center", exp_c, oc), ("overwrite", exp_o, ow))):
        show(ax[k, 0], canvas, "input: 2 tiles, overlap = cols 40–59")
        boxes(ax[k, 0], layout.positions, layout.shapes, "orange")
        show(ax[k, 1], exp, f"expected owner, --seam {name}", cmap="tab10", vmin=0, vmax=9)
        show(ax[k, 2], act, "actual owner_map (tile_map.tif)", cmap="tab10", vmin=0, vmax=9)
        show_diff(ax[k, 3], act.astype(int) - exp.astype(int))
    save(fig, "fig_09_seams", "Seam rule: which tile supplies overlap pixels", "test_stitching::"
         "test_seam_center_owner_is_nearest_tile_centre, test_seam_overwrite_last_tile_wins", "center: each "
         "overlap pixel comes from the tile whose centre is nearest, so the seam lies at the middle of the overlap "
         "(col 50). overwrite: the later tile wins. Pixels are never blended.", ok, 4)


def fig_streamed():
    truth, recs, images, _ = make_mosaic(nx=3, ny=3, jitter=2, seed=9, tile=(70, 90), step=(55, 75))
    cubes = {k: np.stack([v >> 4, v >> 6, v % 251], axis=-1).astype(np.uint16) for k, v in images.items()}
    layout = stitch_layout(recs, lambda i: images[i], max_shift=4)
    expected, owner = compose(layout, lambda i: cubes[i])
    calls = []

    def load(i):
        calls.append(i)
        return cubes[i]

    write_mosaic_streamed(TMP / "eds_stream.tif", layout, load, owner=owner, n_bands=3, dtype=np.uint16,
                          tile=(32, 48))
    got, _ = read_tiff(TMP / "eds_stream.tif")
    counts = {i: calls.count(i) for i in layout.positions}
    ok = np.array_equal(got, expected) and set(counts.values()) == {1}
    fig, ax = plt.subplots(1, 5, figsize=(20, 3.8))
    show(ax[0], owner, "input: 9 tiles, 3 bands each\n(tile_map)", cmap="tab10")
    show(ax[1], expected[..., 1], "expected: in-memory compose (band 1)")
    show(ax[2], got[..., 1], "actual: streamed in 32×48 blocks")
    show_diff(ax[3], got.astype(int) - expected.astype(int), "actual − expected (all bands)")
    ax[4].bar([f"{i[0]},{i[1]}" for i in counts], list(counts.values()), color="lime")
    ax[4].axhline(1, color="cyan", ls="--", label="expected: 1")
    ax[4].set_title("tile decodes during streaming", fontsize=9)
    ax[4].legend(fontsize=7)
    save(fig, "fig_10_streamed", "Streamed EDS mosaic", "test_stitching::"
         "test_streamed_mosaic_equals_in_memory_and_loads_each_tile_once", "The EDS cube is written block by block "
         "without holding the mosaic in memory. Pass = identical to the in-memory result, and every tile is "
         "decoded exactly once.", ok, 4)


# --------------------------------------------------- step 5: pipeline
def _npz_session(d, nx, ny, seed, sizes=None, scales=None):
    import test_pipeline_synthetic  # noqa: F401  (registers the throw-away .npz reader)
    truth, recs, images, true_pos = make_mosaic(nx=nx, ny=ny, jitter=2 if sizes is None else 0, seed=seed)
    d.mkdir(parents=True, exist_ok=True)
    for r in recs:
        extra = {} if sizes is None else {"c": sizes[r.index], "scale": scales[r.index]}
        np.savez(d / f"Mapping_({r.index[0]},{r.index[1]}).npz", index=r.index, bse=images[r.index],
                 px=r.pixel_size_um, stage=r.stage_um, **extra)
    return truth, images, true_pos


def fig_pipeline_synthetic():
    from bstitcher.pipeline import run_session
    d = TMP / "syn_session"
    truth, images, true_pos = _npz_session(d, 3, 2, 11)
    out = TMP / "syn_out"
    run_session(d, out, eds=True, log=lambda *a: None)
    bse, info = read_tiff(out / "stitched" / "BSE.tif")
    fe, _ = read_tiff(out / "stitched" / "elem_Fe.tif")
    owner, _ = read_tiff(out / "stitched" / "tile_map.tif")
    exp = truth_crop(truth, true_pos, bse.shape)
    md = info["metadata"]
    pos = {tuple(t["index"]): tuple(t["position_px"]) for t in md["tiles"]}
    exp_fe = np.zeros_like(fe)
    for t in md["tiles"]:
        tile, _ = read_tiff(out / "tiles" / f"Mapping_({t['index'][0]},{t['index'][1]})_elem_Fe.tif")
        y, x = t["position_px"]
        h, w = tile.shape
        m = owner[y:y + h, x:x + w] == t["tile_id"]
        exp_fe[y:y + h, x:x + w][m] = tile[m]
    cov = owner > 0
    ok = (relative(pos) == relative(true_pos) and np.array_equal(bse[cov], exp[cov])
          and np.array_equal(fe, exp_fe))
    fig, ax = plt.subplots(2, 4, figsize=(17, 6.5))
    shapes = {k: v.shape for k, v in images.items()}
    gap = np.zeros((2 * 60 + 8, 3 * 80 + 16), np.uint16)
    for (c, r), im in images.items():
        gap[(r - 1) * 68:(r - 1) * 68 + 60, (c - 1) * 88:(c - 1) * 88 + 80] = im
    show(ax[0, 0], gap, "input: 6 raw tiles Mapping_(c,r)")
    show(ax[0, 1], exp, "expected BSE: ground truth")
    boxes(ax[0, 1], relative(true_pos), shapes, "cyan")
    show(ax[0, 2], bse, "actual stitched/BSE.tif")
    boxes(ax[0, 2], pos, shapes, "lime")
    show_diff(ax[0, 3], np.where(cov, bse.astype(int) - exp.astype(int), np.nan), "actual − expected (covered px)")
    show(ax[1, 0], owner, "actual stitched/tile_map.tif", cmap="tab10")
    show(ax[1, 1], exp_fe, "expected elem_Fe: tile files pasted by tile_map", cmap="viridis")
    show(ax[1, 2], fe, "actual stitched/elem_Fe.tif", cmap="viridis")
    show_diff(ax[1, 3], fe.astype(int) - exp_fe.astype(int))
    fig.tight_layout()
    save(fig, "fig_11_pipeline_synthetic", "Full pipeline, synthetic session", "test_pipeline_synthetic::"
         "test_stitched_positions_and_lossless_pixels, test_run_session_outputs", "run_session on 6 synthetic "
         "tiles (BSE, 2 element maps, EDS). Pass = true positions recovered and every stitched modality "
         "pixel-identical to the tile files, with tile_map telling which tile each pixel came from.", ok, 5)


def fig_eds_padding():
    from bstitcher.pipeline import run_session
    sizes = {(1, 1): 30, (2, 1): 32, (3, 1): 27}
    scales = {(1, 1): 0.01, (2, 1): 0.01, (3, 1): 0.0101}
    d = TMP / "pad_session"
    _npz_session(d, 3, 1, 12, sizes, scales)
    out = TMP / "pad_out"
    run_session(d, out, eds=True, log=lambda *a: None)
    eds, info = read_tiff(out / "stitched" / "EDS.tif")
    owner, _ = read_tiff(out / "stitched" / "tile_map.tif")
    md = info["metadata"]
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8), gridspec_kw={"width_ratios": [1.2, 1, 1, 1]})
    show(ax[0], owner, "input: 3 tiles with 30, 32, 27 channels\n(3rd tile has a different keV/channel)",
         cmap="tab10")
    ok = eds.shape[2] == 32 and md["energy_axis_uniform"] is False
    for k, t in enumerate(sorted(md["tiles"], key=lambda t: t["index"])):
        tile, _ = read_tiff(out / "tiles" / f"Mapping_({t['index'][0]},{t['index'][1]})_EDS.tif")
        y, x = t["position_px"]
        h, w, c = tile.shape
        m = owner[y:y + h, x:x + w] == t["tile_id"]
        got = eds[y:y + h, x:x + w][m].sum(0)
        exp = np.zeros(eds.shape[2], np.int64)
        exp[:c] = tile[m].sum(0)
        good = np.array_equal(got, exp)
        ok &= good
        a = ax[k + 1]
        a.bar(np.arange(len(exp)), exp, color="cyan", alpha=.6, label="expected: tile channels + 0-pad")
        a.step(np.arange(len(got)), got, where="mid", color="lime", label="actual from stitched EDS")
        a.axvspan(c - .5, eds.shape[2] - .5, color="gray", alpha=.3)
        a.set_title(f"tile {tuple(t['index'])}: {c} ch, {t['energy']['scale_keV'] * 1000:.1f} eV/ch  "
                    f"{'✓' if good else '✗'}", fontsize=9)
        a.set_xlabel("channel (gray = zero padding)")
        a.legend(fontsize=7)
    save(fig, "fig_12_eds_padding", "EDS cube with differing channel counts and calibrations", "test_pipeline_synthetic::"
         "test_eds_with_varying_channel_count_and_calibration", "Bruker cuts spectra and calibrates per tile. The "
         "stitched cube keeps every tile's raw channels, zero-padded to the longest (never resampled). Each tile's "
         "energy axis is stored in the metadata. Pass = summed spectra identical, padding zero.", ok, 5)


def fig_real_pipeline(fixture):
    import tifffile

    from bstitcher.pipeline import load_tile_records, run_session, stitch_tiles
    out = TMP / "bcf_out"
    res = run_session(fixture, out, eds=True, log=lambda *a: None)
    stitch_tiles(out / "tiles", TMP / "bcf_nominal", modalities=["BSE"], max_shift=0, log=lambda *a: None)
    nom, _ = read_tiff(TMP / "bcf_nominal" / "BSE.tif")
    reg, info = read_tiff(out / "stitched" / "BSE.tif")
    owner, _ = read_tiff(out / "stitched" / "tile_map.tif")
    ref = tifffile.imread(fixture / "Mosaic sample_Ch 1.tif").astype(float)
    ref = ref.mean(axis=2) if ref.ndim == 3 else ref
    md = info["metadata"]
    recs = {r.index: r for r in load_tile_records(out / "tiles")}
    tiles = {i: read_tiff(r.files["BSE"])[0] for i, r in recs.items()}

    def z(a):
        a = a.astype(float)
        return (a - a.mean()) / a.std()

    h, w = min(nom.shape[0], ref.shape[0]), min(nom.shape[1], ref.shape[1])
    zn, zr = z(nom[:h, :w]), z(ref[:h, :w])
    corr = np.array([[np.mean(z(nom[3:h - 3, 3:w - 3]) * z(ref[3 + dy:h - 3 + dy, 3 + dx:w - 3 + dx]))
                      for dx in range(-2, 3)] for dy in range(-2, 3)])
    ok1 = np.unravel_index(corr.argmax(), corr.shape) == (2, 2) and corr[2, 2] > 0.85

    # figure 13: nominal geometry == Bruker
    fig, ax = plt.subplots(1, 5, figsize=(21, 3.8))
    gap = np.zeros((2 * 88 + 6, 2 * 118 + 6), np.uint16)
    for (c, r), im in tiles.items():
        gap[(r - 1) * 94:(r - 1) * 94 + 88, (c - 1) * 124:(c - 1) * 124 + 118] = im
    show(ax[0], gap, "input: 4 BSE tiles from Mapping_(c,r).bcf")
    show(ax[1], ref, "expected: Bruker 'Mosaic sample_Ch 1.tif'\n(8-bit display, stage-nominal)")
    show(ax[2], nom, "actual: bstitcher --max-shift 0\n(stage positions only)")
    show_diff(ax[3], zn - zr, "actual − expected (z-scored)")
    im = ax[4].imshow(corr, cmap="viridis", extent=(-2.5, 2.5, 2.5, -2.5))
    plt.colorbar(im, ax=ax[4], fraction=0.046)
    ax[4].plot(0, 0, "x", color="cyan", ms=12, mew=2)
    ax[4].set_title(f"correlation vs offset: peak at 0,0 = {corr[2, 2]:.3f}", fontsize=9)
    ax[4].set_xlabel("Δcol")
    ax[4].set_ylabel("Δrow")
    save(fig, "fig_13_real_vs_bruker", "Real job: stage geometry reproduces Bruker's mosaic", "test_pipeline_bcf::"
         "test_nominal_layout_reproduces_bruker_mosaic", "With registration off, our mosaic must have the same "
         "geometry as Bruker's own. Pass = correlation peaks at zero offset (> 0.85; not 1.0 because Bruker exports "
         "an 8-bit display rendering). The remaining difference lies on the seam lines (Bruker lets another tile win in the "
         "overlap) and inside tile (2,2), which Bruker renders with a different grey mapping. Neither is a position "
         "error.", ok1, 5)

    # figure 14: registration improves the overlaps
    solved = {tuple(t["index"]): tuple(t["position_px"]) for t in md["tiles"]}
    nominal = {k: (round(v[0]), round(v[1])) for k, v in nominal_positions(list(recs.values())).items()}
    r0, c0 = min(p[0] for p in nominal.values()), min(p[1] for p in nominal.values())
    nominal = {k: (p[0] - r0, p[1] - c0) for k, p in nominal.items()}
    pairs = [p for p in res["report"]["pairs"] if p["a"][0] == p["b"][0] or p["a"][1] == p["b"][1]]
    fig, ax = plt.subplots(len(pairs), 5, figsize=(21, 3.0 * len(pairs)),
                           gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.3]})
    gain = []
    ok2 = all(p["used"] for p in pairs)
    for k, p in enumerate(pairs):
        a, b = tuple(p["a"]), tuple(p["b"])
        vals = []
        for j, (P, lab) in enumerate(((nominal, "stage-nominal"), (solved, "registered"))):
            (ra, ca), (rb, cb) = P[a], P[b]
            hh, ww = tiles[a].shape
            rr0, rr1, cc0, cc1 = max(ra, rb), min(ra, rb) + hh, max(ca, cb), min(ca, cb) + ww
            A = tiles[a][rr0 - ra:rr1 - ra, cc0 - ca:cc1 - ca].astype(float)
            B = tiles[b][rr0 - rb:rr1 - rb, cc0 - cb:cc1 - cb].astype(float)
            v = _ncc(A, B)
            vals.append(v)
            both = np.concatenate([A, np.full((A.shape[0], 2), np.nan), B], axis=1) if A.shape[0] > A.shape[1] \
                else np.concatenate([A, np.full((2, A.shape[1]), np.nan), B], axis=0)
            show(ax[k, 2 * j], both, f"{a}|{b} overlap, {lab}\nA beside B, NCC = {v:.2f}", aspect="auto")
            d = z(A) - z(B)
            im = show(ax[k, 2 * j + 1], d, f"z(A) − z(B), {lab}\nmean |Δ| = {np.abs(d).mean():.2f}",
                      cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
            plt.colorbar(im, ax=ax[k, 2 * j + 1], fraction=0.046)
        gain.append(vals)
        ax[k, 4].axis("off")
        ax[k, 4].text(0, .5, f"pair {a} → {b}\nexpected: overlap NCC rises,\n          pair used\n"
                             f"actual:   NCC {vals[0]:.2f} → {vals[1]:.2f}\n          shift {tuple(p['shift'])} px"
                             f", used={p['used']}", family="monospace", fontsize=9, va="center")
    g = np.mean(gain, axis=0)
    ok2 = ok2 and g[1] > g[0] + 0.2
    fig.text(0.5, -0.01, f"mean overlap NCC: stage-nominal {g[0]:.2f} → registered {g[1]:.2f} "
                         "(test requires a gain > 0.2)", ha="center", fontsize=11)
    fig.tight_layout()
    save(fig, "fig_14_real_registration", "Real job: registration aligns the overlaps", "test_pipeline_bcf::"
         "test_registration_improves_overlap_agreement, test_loop_closure", "For each neighbouring pair the "
         "shared strip of tile A and tile B is shown at stage-nominal and at registered positions. Stage positions "
         "are a few px off (structures doubled, high |Δ|). After registration A and B show the same structures. "
         "Pass = all edge pairs used, and the pipeline test requires mean NCC +0.2.", ok2, 5)

    # figure 15: stitched EDS lossless + element map + georef
    eds, einfo = read_tiff(out / "stitched" / "EDS.tif")
    fe, finfo = read_tiff(out / "stitched" / "elem_Fe.tif")
    exp_sum = np.zeros(eds.shape[2], np.int64)
    exp_fe = np.zeros_like(fe)
    for t in md["tiles"]:
        i = tuple(t["index"])
        cube, _ = read_tiff(recs[i].files["EDS"])
        fet, _ = read_tiff(recs[i].files["elem_Fe"])
        y, x = t["position_px"]
        m = owner[y:y + 88, x:x + 118] == t["tile_id"]
        exp_sum += cube[m].sum(0, dtype=np.int64)
        exp_fe[y:y + 88, x:x + 118][m] = fet[m]
    act_sum = eds.reshape(-1, eds.shape[2]).sum(0, dtype=np.int64)
    same_grid = all(read_info(p)["geotransform"] == info["geotransform"] for p in (out / "stitched").glob("*.tif"))
    ok3 = np.array_equal(act_sum, exp_sum) and np.array_equal(fe, exp_fe) and same_grid
    e = einfo["metadata"]["energy"]
    keV = e["offset_keV"] + e["scale_keV"] * np.arange(e["size"])
    fig, ax = plt.subplots(1, 5, figsize=(21, 3.8), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.6]})
    show(ax[0], owner, "input: tile_map (registered)", cmap="tab10")
    show(ax[1], exp_fe, "expected elem_Fe: tile maps pasted", cmap="viridis")
    show(ax[2], fe, "actual stitched/elem_Fe.tif", cmap="viridis")
    show_diff(ax[3], fe.astype(int) - exp_fe.astype(int))
    ax[4].semilogy(keV, exp_sum + 1, color="cyan", lw=3, alpha=.6, label="expected: Σ tile cubes (owned px)")
    ax[4].semilogy(keV, act_sum + 1, color="k", lw=.7, label="actual: Σ stitched EDS.tif")
    ax[4].set_xlim(0, 10)
    ax[4].set_xlabel("keV")
    ax[4].legend(fontsize=7)
    sz = (out / "stitched" / "EDS.tif").stat().st_size
    ax[4].set_title(f"EDS {eds.shape}, file {sz / eds.nbytes:.0%} of raw; max |Δ| = "
                    f"{np.abs(act_sum - exp_sum).max()}", fontsize=9)
    save(fig, "fig_15_real_eds", "Real job: stitched element map and EDS cube are lossless",
         "test_pipeline_bcf::test_eds_mosaic_is_lossless, test_stitched_modalities_share_canvas_and_georef, "
         "test_eds_compressed_smaller_than_raw", "Pass = the stitched Fe map equals the tile maps pasted according "
         "to tile_map, the stitched cube's total spectrum equals that of the source pixels, and all stitched "
         f"files share one canvas and georeference (same grid: {same_grid}).", ok3, 5)


def main():
    global OUT, TMP
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "tmp" / "test_figures"))
    ap.add_argument("--fixture", default=None, help="real Bruker mapping job (enables figures 5, 13–15)")
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        TMP = Path(tmp)
        fig_tiff_roundtrip()
        fig_georef()
        fig_element_windows()
        fig_tile_export()
        if args.fixture:
            fig_bcf_reader(Path(args.fixture))
        fig_registration()
        fig_placement()
        fig_fallback_and_outlier()
        fig_seams()
        fig_streamed()
        fig_pipeline_synthetic()
        fig_eds_padding()
        if args.fixture:
            fig_real_pipeline(Path(args.fixture))
    FIGS.sort(key=lambda f: f["file"])
    (OUT / "figures.json").write_text(json.dumps(FIGS, indent=1, ensure_ascii=False))
    n_fail = sum(not f["passed"] for f in FIGS)
    print(f"{len(FIGS)} figures, {n_fail} failed -> {OUT}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
