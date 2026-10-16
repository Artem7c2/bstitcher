# SPDX-License-Identifier: Apache-2.0
"""PNG previews of stitched mosaics: grey scale, colour bar and scale bar.

The previews are for looking at, not for analysis: large mosaics are
block-averaged to at most ``max_px`` pixels on the long side and the grey
scale is stretched to the 0.5-99.5 % percentiles (the colour bar shows the
original values).  Pixels outside every tile (``tile_map == 0``) are drawn
in a pale blue so they are not mistaken for low signal.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from .tiff_io import read_info, read_tiff

VIZ_DIR = "visualization_stitched"
MAX_PX = 4000
NO_DATA_COLOUR = "#c6d7ea"
PERCENTILES = (0.5, 99.5)
_SKIP = {"EDS", "tile_map"}  # spectrum cube / tile ids: no grey-scale preview


def nice_scalebar_um(width_um: float, fraction: float = 0.2) -> float:
    """Largest 1-2-5 x 10^n length not exceeding ``fraction`` of the image width."""
    target = width_um * fraction
    exp = math.floor(math.log10(target))
    for m in (5, 2, 1):
        if m * 10 ** exp <= target * (1 + 1e-9):
            return m * 10 ** exp
    return 10 ** exp


def _format_um(um: float) -> str:
    if um >= 1000:
        return f"{um / 1000:g} mm"
    if um < 1:
        return f"{um * 1000:g} nm"
    return f"{um:g} µm"


def block_reduce(a, factor: int, mask=None):
    """Mean over ``factor`` x ``factor`` blocks of valid pixels; returns ``(image, valid)``.

    Works in strips, so a mosaic far larger than memory as float is fine.
    """
    a = np.asarray(a)
    if mask is None:
        mask = np.ones(a.shape, bool)
    if factor == 1:
        return a.astype(np.float32), mask.copy()
    H, W = a.shape
    h, w = -(-H // factor), -(-W // factor)
    out = np.zeros((h, w), np.float32)
    valid = np.zeros((h, w), bool)
    pad_w = w * factor - W
    for i in range(h):
        rows = a[i * factor:(i + 1) * factor].astype(np.float64)
        m = mask[i * factor:(i + 1) * factor]
        if pad_w:
            rows = np.pad(rows, ((0, 0), (0, pad_w)))
            m = np.pad(m, ((0, 0), (0, pad_w)))
        s = (rows * m).reshape(len(rows), w, factor).sum(axis=(0, 2))
        n = m.reshape(len(m), w, factor).sum(axis=(0, 2))
        valid[i] = n > 0
        out[i] = np.where(n > 0, s / np.maximum(n, 1), 0)
    return out, valid


def render_png(a, path, *, pixel_size_um: float, title: str, label: str = "value",
               mask=None, max_px: int = MAX_PX) -> dict:
    """Write a grey-scale PNG with colour bar and scale bar; returns what was drawn."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    factor = max(1, math.ceil(max(a.shape) / max_px))
    img, valid = block_reduce(a, factor, mask)
    vals = img[valid]
    if vals.size:
        vmin, vmax = (float(v) for v in np.percentile(vals, PERCENTILES))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    h, w = img.shape
    px_um = pixel_size_um * factor
    bar_um = nice_scalebar_um(w * px_um)
    fs = max(9.0, math.sqrt(h * w) / 150)  # font size in pt == px at dpi 72
    dpi = 72
    left, right_extra, top, bottom = 2 * fs, 9 * fs, 2.6 * fs, 1.0 * fs
    W, H = w + left + right_extra, h + top + bottom
    fig = Figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([left / W, bottom / H, w / W, h / H])
    cmap = matplotlib.colormaps["gray"].copy()
    cmap.set_bad(NO_DATA_COLOUR)
    im = ax.imshow(np.ma.masked_array(img, ~valid), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(title, fontsize=1.3 * fs, loc="left")
    cax = fig.add_axes([(left + w + 1.2 * fs) / W, bottom / H, 1.2 * fs / W, h / H])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.ax.tick_params(labelsize=fs)
    cb.set_label(label, fontsize=fs)
    ax.add_artist(AnchoredSizeBar(ax.transData, bar_um / px_um, _format_um(bar_um), "lower right",
                                  pad=0.4, borderpad=0.6, sep=0.3 * fs, color="black", frameon=True,
                                  size_vertical=max(2.0, 0.35 * fs),
                                  fontproperties=FontProperties(size=fs)))
    os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
    fig.savefig(str(path), dpi=dpi, facecolor="white", metadata={"Software": None})
    return {"path": str(path), "downsample": factor, "scalebar_um": bar_um, "vmin": vmin, "vmax": vmax}


def _title_and_label(mod: str, md: dict):
    if mod.startswith("elem_"):
        el = md.get("element") or mod[5:]
        win = md.get("integration_window_keV")
        title = f"{el}" + (f"  ({win[0]:.2f}-{win[1]:.2f} keV)" if win else "")
        return title, "X-ray counts per pixel"
    return mod, f"{mod} intensity (detector units)"


def visualize_stitched(stitched_dir, viz_dir, *, modalities=None, max_px: int = MAX_PX,
                       log=print) -> dict:
    """PNG preview of every 2-D mosaic in ``stitched_dir``; returns ``{modality: png}``."""
    stitched_dir = Path(stitched_dir)
    tile_map = stitched_dir / "tile_map.tif"
    mask = read_tiff(tile_map)[0] > 0 if tile_map.exists() else None
    out = {}
    for p in sorted(stitched_dir.glob("*.tif"), key=lambda p: (p.stem != "BSE", p.stem)):
        mod = p.stem
        if mod in _SKIP or (modalities is not None and mod not in modalities):
            continue
        info = read_info(p)
        if len(info["shape"]) != 2:
            continue
        a, _ = read_tiff(p)
        md = info["metadata"]
        px = md.get("pixel_size_um")
        px = float(px[0]) if isinstance(px, (list, tuple)) else float(px or 1.0)
        title, label = _title_and_label(mod, md)
        m = mask if mask is not None and mask.shape == a.shape else None
        png = os.path.join(str(viz_dir), f"{mod}.png")
        render_png(a, png, pixel_size_um=px, title=title, label=label, mask=m, max_px=max_px)
        out[mod] = png
        log(f"preview {mod}: {png}")
    return out
