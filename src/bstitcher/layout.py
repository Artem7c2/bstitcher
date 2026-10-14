# SPDX-License-Identifier: Apache-2.0
"""Tile grid geometry from stage coordinates."""
from __future__ import annotations

from dataclasses import dataclass, field

from .tile import tile_origin_um


@dataclass
class TileRecord:
    """Geometry of one tile plus (optionally) its exported files per modality."""
    index: tuple
    shape: tuple            # (H, W)
    stage_um: tuple         # stage (x, y) of the tile centre
    pixel_size_um: tuple    # (dx, dy)
    files: dict = field(default_factory=dict)


def check_pixel_size(records, rel_tol: float = 1e-3) -> tuple:
    dx0, dy0 = records[0].pixel_size_um
    for r in records:
        dx, dy = r.pixel_size_um
        if abs(dx - dx0) > rel_tol * dx0 or abs(dy - dy0) > rel_tol * dy0:
            raise ValueError(f"tile {r.index} pixel size {r.pixel_size_um} differs from "
                             f"{(dx0, dy0)}; mixed magnifications cannot be stitched")
    return dx0, dy0


def world_origins(records) -> dict:
    """World (µm) top-left corner of every tile."""
    return {r.index: tile_origin_um(r.stage_um, r.pixel_size_um, r.shape) for r in records}


def nominal_positions(records) -> dict:
    """Float (row, col) pixel position of each tile's top-left corner, min at 0."""
    dx, dy = check_pixel_size(records)
    org = world_origins(records)
    x_min = min(o[0] for o in org.values())
    y_max = max(o[1] for o in org.values())
    return {i: ((y_max - o[1]) / dy, (o[0] - x_min) / dx) for i, o in org.items()}


def overlap_box(pa, sa, pb, sb):
    """Overlap height/width of two rectangles at integer positions."""
    oh = min(pa[0] + sa[0], pb[0] + sb[0]) - max(pa[0], pb[0])
    ow = min(pa[1] + sa[1], pb[1] + sb[1]) - max(pa[1], pb[1])
    return oh, ow


def overlapping_pairs(positions: dict, shapes: dict, min_overlap: int = 4) -> list:
    """Pairs ``(a, b)`` (a < b) whose nominal rectangles overlap by >= min_overlap px."""
    idx = sorted(positions)
    rp = {i: (round(positions[i][0]), round(positions[i][1])) for i in idx}
    pairs = []
    for n, a in enumerate(idx):
        for b in idx[n + 1:]:
            oh, ow = overlap_box(rp[a], shapes[a], rp[b], shapes[b])
            if oh >= min_overlap and ow >= min_overlap:
                pairs.append((a, b))
    return pairs
