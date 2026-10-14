# SPDX-License-Identifier: Apache-2.0
"""Format-independent tile model and per-tile GeoTIFF export.

Readers (e.g. the ``bstitcher-bcf`` plugin) produce :class:`Tile` objects;
everything downstream only depends on this model.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from .georef import CRS_LOCAL, CRS_PIXEL, physical_geotransform, pixel_geotransform
from .metadata import tile_stem
from .tiff_io import write_tiff

EDS_TILE_LEVEL = 9  # EDS cubes are big; ZSTD-9 is ~10x faster than 22 at ~same size


@dataclass
class Tile:
    index: tuple                       # (col, row) from the acquisition job
    images: dict                       # name -> (H, W) array; "BSE" is the primary image
    element_maps: dict                 # element -> (H, W) integer counts
    pixel_size_um: tuple               # (dx, dy)
    stage_um: tuple                    # (x, y) stage position of the tile centre
    eds: np.ndarray | None = None      # (H, W, C) counts
    energy: dict | None = None         # {"offset_keV", "scale_keV", "size"}
    windows_keV: dict = field(default_factory=dict)  # element -> [lo, hi] channel-centre energies
    metadata: dict = field(default_factory=dict)     # acquisition metadata (JSON-able)
    source: str = ""

    @property
    def shape(self) -> tuple:
        return self.images["BSE"].shape[:2]


def channel_range(energy: dict, lo_keV: float, hi_keV: float) -> tuple[int, int]:
    """Half-open channel index range of channels whose centre lies in [lo, hi]."""
    off, sc, n = energy["offset_keV"], energy["scale_keV"], energy["size"]
    i0 = math.ceil((lo_keV - off) / sc - 1e-6)
    i1 = math.floor((hi_keV - off) / sc + 1e-6) + 1
    return max(0, i0), min(n, i1)


def integrate_windows(cube: np.ndarray, energy: dict, windows_keV: dict) -> dict:
    """Element maps by summing EDS counts over energy windows."""
    out = {}
    for name, (lo, hi) in windows_keV.items():
        i0, i1 = channel_range(energy, lo, hi)
        out[name] = cube[..., i0:i1].sum(axis=-1, dtype=np.uint64)
    return out


def minimal_uint(a: np.ndarray) -> np.ndarray:
    """Cast integer-valued data to the smallest unsigned dtype, losslessly."""
    a = np.asarray(a)
    if a.dtype.kind == "f":
        if not np.all(np.isfinite(a)) or not np.all(np.mod(a, 1) == 0):
            raise ValueError("data is not integer-valued; cannot cast losslessly")
    if a.size and a.min() < 0:
        raise ValueError("negative values; cannot cast to unsigned")
    vmax = int(a.max()) if a.size else 0
    for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
        if vmax <= np.iinfo(dt).max:
            return a.astype(dt, copy=False)
    raise ValueError("values exceed uint64")


def tile_origin_um(stage_um, pixel_size_um, shape) -> tuple:
    """World (µm) top-left corner of a tile whose centre sits at ``stage_um``.

    World coordinates are the negated stage coordinates (see :mod:`georef`).
    """
    h, w = shape
    dx, dy = pixel_size_um
    return (-stage_um[0] - w * dx / 2.0, -stage_um[1] + h * dy / 2.0)


def tile_geotransform(tile: Tile, georef: str) -> tuple:
    if georef == CRS_PIXEL:
        return pixel_geotransform(tile.shape[0])
    if georef == CRS_LOCAL:
        return physical_geotransform(tile_origin_um(tile.stage_um, tile.pixel_size_um, tile.shape),
                                     tile.pixel_size_um)
    raise ValueError(f"unknown georef {georef!r}")


def _base_metadata(tile: Tile) -> dict:
    md = {"tile_index": list(tile.index), "stage_um": list(tile.stage_um),
          "pixel_size_um": list(tile.pixel_size_um), "source_file": tile.source}
    md.update(tile.metadata)
    return md


def export_tile(tile: Tile, out_dir, *, eds: bool = False, georef: str = CRS_LOCAL,
                level: int | None = None) -> dict:
    """Write one tile as ``Mapping_(c,r)_<modality>.tif`` files; returns ``{modality: path}``."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(str(out_dir), tile_stem(tile.index))
    gt = tile_geotransform(tile, georef)
    base = _base_metadata(tile)
    lvl = {} if level is None else {"level": level}
    files = {}
    for name, img in tile.images.items():
        p = f"{stem}_{name}.tif"
        write_tiff(p, img, metadata={"modality": name, **base}, geotransform=gt, crs=georef, **lvl)
        files[name] = p
    for el, m in tile.element_maps.items():
        p = f"{stem}_elem_{el}.tif"
        md = {"modality": f"elem_{el}", "element": el,
              "integration_window_keV": tile.windows_keV.get(el), **base}
        write_tiff(p, minimal_uint(m), metadata=md, geotransform=gt, crs=georef, **lvl)
        files[f"elem_{el}"] = p
    if eds and tile.eds is not None:
        p = f"{stem}_EDS.tif"
        md = {"modality": "EDS", "energy": tile.energy, **base}
        write_tiff(p, tile.eds, metadata=md, geotransform=gt, crs=georef,
                   level=EDS_TILE_LEVEL if level is None else level)
        files["EDS"] = p
    return files
