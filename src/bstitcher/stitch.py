# SPDX-License-Identifier: Apache-2.0
"""Global tile placement and lossless mosaic composition.

1. Nominal tile positions come from the stage coordinates.
2. Every overlapping pair is registered (NCC) -> measured relative offsets.
3. A least-squares solve over all reliable measurements (plus a weak prior to
   the nominal positions, which anchors the solution and places tiles that
   have no reliable overlap) yields integer positions; measurements that
   disagree with the consensus by more than ``max_residual`` px are rejected
   iteratively.
4. Tiles are pasted at integer positions without resampling or blending, so
   every mosaic pixel is an original measured value.  In overlaps each pixel is
   taken from the tile whose centre is nearest (``seam="center"``) or from the
   last tile in raster order (``seam="overwrite"``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .layout import check_pixel_size, nominal_positions, overlapping_pairs, world_origins
from .registration import register_pair


@dataclass
class Layout:
    positions: dict                     # index -> (row, col) int, min 0
    shapes: dict                        # index -> (H, W)
    nominal: dict                       # index -> (row, col) float
    tile_ids: dict                      # index -> 1-based id used in owner maps
    canvas_shape: tuple
    origin_um: tuple = (0.0, 0.0)       # world (µm) top-left of the canvas
    pixel_size_um: tuple = (1.0, 1.0)
    pairs: list = field(default_factory=list)
    max_shift: int = 0

    @property
    def n_fallback_pairs(self) -> int:
        return sum(1 for p in self.pairs if not p["used"])

    def order(self) -> list:
        """Raster order (row index, then column index)."""
        return sorted(self.positions, key=lambda i: (i[1], i[0]))


def solve_positions(nominal: dict, measurements: dict, *, max_residual: float = 2.0,
                    prior_weight: float = 1e-3):
    """Least-squares integer positions.

    measurements: {(a, b): ((drow, dcol) measured offset b - a, score)}
    Returns (positions, rejected_pairs).
    """
    idx = sorted(nominal)
    col = {k: n for n, k in enumerate(idx)}
    meas = dict(measurements)
    rejected = []
    while True:
        rows, rhs_r, rhs_c, w = [], [], [], []
        for (a, b), ((mr, mc), score) in meas.items():
            v = np.zeros(len(idx))
            v[col[b]], v[col[a]] = 1.0, -1.0
            rows.append(v); rhs_r.append(mr); rhs_c.append(mc); w.append(1.0)
        for k in idx:
            v = np.zeros(len(idx))
            v[col[k]] = 1.0
            rows.append(v); rhs_r.append(nominal[k][0]); rhs_c.append(nominal[k][1])
            w.append(prior_weight)
        A = np.array(rows) * np.array(w)[:, None]
        sol_r = np.linalg.lstsq(A, np.array(rhs_r) * w, rcond=None)[0]
        sol_c = np.linalg.lstsq(A, np.array(rhs_c) * w, rcond=None)[0]
        worst, worst_res = None, max_residual
        for (a, b), ((mr, mc), _) in meas.items():
            res = max(abs(sol_r[col[b]] - sol_r[col[a]] - mr), abs(sol_c[col[b]] - sol_c[col[a]] - mc))
            if res > worst_res:
                worst, worst_res = (a, b), res
        if worst is None:
            break
        rejected.append(worst)
        del meas[worst]
    pos = {k: (int(round(sol_r[col[k]])), int(round(sol_c[col[k]]))) for k in idx}
    r0 = min(p[0] for p in pos.values())
    c0 = min(p[1] for p in pos.values())
    return {k: (p[0] - r0, p[1] - c0) for k, p in pos.items()}, rejected


def auto_max_shift(shapes) -> int:
    """Default search radius: 5 % of the largest tile side, at least 8 px.

    Stage-to-image scale and rotation errors of a few percent are common
    (e.g. ~13 px per 322 px step on 1 µm Bruker jobs), so the radius has to
    grow with the tile size.
    """
    return max(8, int(np.ceil(0.05 * max(max(s) for s in shapes))))


def stitch_layout(records, load_reference, *, max_shift: int | None = None, min_score: float = 0.5,
                  max_residual: float = 2.0, min_overlap: int = 4,
                  min_overlap_frac: float = 0.5, smooth: int = 1) -> Layout:
    """Compute tile placements.

    load_reference(index) -> 2D (or 3D, summed) image used for registration
    (normally BSE).
    """
    if not records:
        raise ValueError("no tiles")
    px = check_pixel_size(records)
    shapes = {r.index: tuple(r.shape) for r in records}
    if max_shift is None:
        max_shift = auto_max_shift(shapes.values())
    nominal = nominal_positions(records)
    pairs = overlapping_pairs(nominal, shapes, min_overlap=min_overlap)
    cache = {}

    def img(i):
        if i not in cache:
            cache[i] = np.asarray(load_reference(i))
        return cache[i]

    measurements, report = {}, []
    for a, b in pairs:
        nom_off = (round(nominal[b][0]) - round(nominal[a][0]),
                   round(nominal[b][1]) - round(nominal[a][1]))
        res = register_pair(img(a), img(b), nom_off, max_shift=max_shift,
                            min_overlap_frac=min_overlap_frac, smooth=smooth)
        meas = (nom_off[0] + res.shift[0], nom_off[1] + res.shift[1])
        ok = res.score >= min_score and (not res.at_boundary or max_shift == 0)
        if ok:
            measurements[(a, b)] = (meas, res.score)
        report.append({"a": list(a), "b": list(b), "nominal_offset": list(nom_off),
                       "shift": list(res.shift), "score": round(res.score, 4),
                       "n_pixels": res.n_pixels, "at_boundary": res.at_boundary, "used": ok})
    positions, rejected = solve_positions(nominal, measurements, max_residual=max_residual)
    for p in report:
        if (tuple(p["a"]), tuple(p["b"])) in rejected:
            p["used"] = False
            p["rejected_as_outlier"] = True

    H = max(positions[i][0] + shapes[i][0] for i in positions)
    W = max(positions[i][1] + shapes[i][1] for i in positions)
    # World origin of the canvas: consistent with every tile's stage position on average.
    org = world_origins(records)
    ox = float(np.mean([org[i][0] - positions[i][1] * px[0] for i in positions]))
    oy = float(np.mean([org[i][1] + positions[i][0] * px[1] for i in positions]))
    order = sorted(positions, key=lambda i: (i[1], i[0]))
    return Layout(positions=positions, shapes=shapes, nominal=nominal,
                  tile_ids={k: n + 1 for n, k in enumerate(order)}, canvas_shape=(H, W),
                  origin_um=(ox, oy), pixel_size_um=px, pairs=report, max_shift=max_shift)


def owner_map(layout: Layout, seam: str = "center") -> np.ndarray:
    """uint16/uint32 map of which tile id supplies each canvas pixel (0 = no data)."""
    H, W = layout.canvas_shape
    dt = np.uint16 if len(layout.tile_ids) < 2**16 else np.uint32
    owner = np.zeros((H, W), dtype=dt)
    if seam == "overwrite":
        for i in layout.order():
            (r, c), (h, w) = layout.positions[i], layout.shapes[i]
            owner[r:r + h, c:c + w] = layout.tile_ids[i]
        return owner
    if seam != "center":
        raise ValueError(f"unknown seam {seam!r}")
    best = np.full((H, W), np.inf, dtype=np.float32)
    for i in layout.order():
        (r, c), (h, w) = layout.positions[i], layout.shapes[i]
        rr = (np.arange(h, dtype=np.float32) + 0.5 - h / 2.0) / h
        cc = (np.arange(w, dtype=np.float32) + 0.5 - w / 2.0) / w
        d = rr[:, None] ** 2 + cc[None, :] ** 2
        sub = best[r:r + h, c:c + w]
        m = d < sub
        sub[m] = d[m]
        owner[r:r + h, c:c + w][m] = layout.tile_ids[i]
    return owner


def compose(layout: Layout, load, *, seam: str = "center", owner=None, out=None, dtype=None):
    """Paste tiles into a canvas.  Returns ``(canvas, owner)``.

    load(index) -> (H, W) or (H, W, C) array (missing tiles may return None).
    ``out`` may be a pre-allocated array / memmap (e.g. for large EDS cubes).
    """
    if owner is None:
        owner = owner_map(layout, seam)
    canvas = out
    for i in layout.order():
        tile = load(i)
        if tile is None:
            continue
        tile = np.asarray(tile)
        if canvas is None:
            dt = dtype or tile.dtype
            canvas = np.zeros(layout.canvas_shape + tile.shape[2:], dtype=dt)
        elif out is None and np.result_type(canvas.dtype, tile.dtype) != canvas.dtype:
            canvas = canvas.astype(np.result_type(canvas.dtype, tile.dtype))
        (r, c), (h, w) = layout.positions[i], layout.shapes[i]
        m = owner[r:r + h, c:c + w] == layout.tile_ids[i]
        if tile.shape[:2] != (h, w):
            raise ValueError(f"tile {i} has shape {tile.shape[:2]}, layout expects {(h, w)}")
        dst = canvas[r:r + h, c:c + w]
        dst[m] = tile[m]
    return canvas, owner


def iter_mosaic_tiles(layout: Layout, load, *, owner, tile=(128, 128), n_bands=None, dtype=None):
    """Yield the mosaic as ``(th, tw[, C])`` output tiles in row-major order.

    Source tiles are loaded lazily, each exactly once, and released as soon as
    the output has passed below them; peak memory is about one row of source
    tiles, independent of the mosaic size.
    """
    th, tw = tile
    H, W = layout.canvas_shape
    extra = () if not n_bands else (n_bands,)
    cache = {}
    for y0 in range(0, H, th):
        y1 = min(H, y0 + th)
        for x0 in range(0, W, tw):
            x1 = min(W, x0 + tw)
            out = np.zeros((th, tw) + extra, dtype=dtype)
            for i in layout.order():
                (r, c), (h, w) = layout.positions[i], layout.shapes[i]
                if r >= y1 or r + h <= y0 or c >= x1 or c + w <= x0:
                    continue
                if i not in cache:
                    cache[i] = np.asarray(load(i))
                src = cache[i]
                ry0, ry1 = max(r, y0), min(r + h, y1)
                rx0, rx1 = max(c, x0), min(c + w, x1)
                m = owner[ry0:ry1, rx0:rx1] == layout.tile_ids[i]
                dst = out[ry0 - y0:ry1 - y0, rx0 - x0:rx1 - x0]
                dst[m] = src[ry0 - r:ry1 - r, rx0 - c:rx1 - c][m]
            yield out
        for i in [k for k in cache if layout.positions[k][0] + layout.shapes[k][0] <= y1]:
            del cache[i]


def write_mosaic_streamed(path, layout: Layout, load, *, owner, dtype, n_bands=None,
                          tile=(128, 128), **write_kwargs):
    """Write a (possibly huge) mosaic straight to a tiled GeoTIFF without a canvas in memory."""
    from .tiff_io import write_tiff
    shape = layout.canvas_shape + ((n_bands,) if n_bands else ())
    return write_tiff(path, iter_mosaic_tiles(layout, load, owner=owner, tile=tile, n_bands=n_bands,
                                              dtype=dtype),
                      shape=shape, dtype=dtype, tile=tile, **write_kwargs)
