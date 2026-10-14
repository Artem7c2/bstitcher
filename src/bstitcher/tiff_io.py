# SPDX-License-Identifier: Apache-2.0
"""Lossless GeoTIFF writing / reading with embedded metadata.

* Compression: ZSTD (lossless) with a differencing predictor.
* Full metadata: JSON in the TIFF ImageDescription tag
  (``{"bstitcher": version, "metadata": {...}, "band_names": [...]}``).
* Key items: top-level scalar metadata entries and band names are mirrored
  into the GDAL_METADATA tag so that GDAL / QGIS show them.
* Images are ``(H, W)``; multi-band data (element stacks, EDS cubes) are
  ``(H, W, C)`` pixel-interleaved.
"""
from __future__ import annotations

import json
from xml.sax.saxutils import escape

import numpy as np
import tifffile

from . import __version__
from .georef import CRS_PIXEL, geotiff_extratags, read_georef
from .metadata import jsonable

GDAL_METADATA_TAG = 42112
DEFAULT_LEVEL = 22


def _gdal_metadata_xml(metadata: dict | None, band_names) -> str | None:
    items = []
    for k, v in (metadata or {}).items():
        if isinstance(v, (str, int, float, bool)):
            items.append(f'<Item name="{escape(str(k))}">{escape(str(v))}</Item>')
    for i, name in enumerate(band_names or []):
        items.append(f'<Item name="DESCRIPTION" sample="{i}" role="description">'
                     f'{escape(str(name))}</Item>')
    if not items:
        return None
    return "<GDALMetadata>" + "".join(items) + "</GDALMetadata>"


def write_tiff(path, data, *, metadata: dict | None = None, geotransform=None,
               crs: str = CRS_PIXEL, band_names=None, tile=None,
               level: int = DEFAULT_LEVEL, shape=None, dtype=None) -> str:
    """Write ``data`` (``(H, W)`` or ``(H, W, C)``) as a lossless GeoTIFF.

    ``data`` may also be an iterator over ``(th, tw[, C])`` tiles in row-major
    tile order; ``shape``, ``dtype`` and ``tile`` are then required.
    """
    if isinstance(data, np.ndarray):
        shape, dtype = data.shape, data.dtype
    elif shape is None or dtype is None or tile is None:
        raise ValueError("shape, dtype and tile are required when writing from an iterator")
    if len(shape) not in (2, 3):
        raise ValueError(f"expected (H, W) or (H, W, C) data, got shape {shape}")
    if band_names is not None:
        n = 1 if len(shape) == 2 else shape[2]
        if len(band_names) != n:
            raise ValueError(f"{len(band_names)} band names for {n} bands")
    md = jsonable(metadata) if metadata is not None else {}
    description = json.dumps({"bstitcher": __version__, "metadata": md,
                              "band_names": list(band_names) if band_names else None})
    extratags = []
    if geotransform is not None:
        extratags += geotiff_extratags(geotransform, crs)
    xml = _gdal_metadata_xml(md, band_names)
    if xml is not None:
        extratags.append((GDAL_METADATA_TAG, 2, 0, xml, True))

    kwargs = dict(photometric="minisblack", compression="zstd",
                  compressionargs={"level": level}, predictor=True,
                  description=description, metadata=None, extratags=extratags)
    if len(shape) == 3:
        kwargs["planarconfig"] = "contig"
    if tile is not None:
        kwargs["tile"] = tuple(tile)
    if int(np.prod(shape)) * np.dtype(dtype).itemsize > 2**32 - 2**25:
        kwargs["bigtiff"] = True
    if not isinstance(data, np.ndarray):
        kwargs.update(shape=tuple(shape), dtype=np.dtype(dtype))
    tifffile.imwrite(str(path), data, **kwargs)
    return str(path)


def read_info(path) -> dict:
    """Metadata, band names and georeferencing without decoding pixels."""
    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        try:
            desc = json.loads(page.description)
        except (ValueError, TypeError):
            desc = {}
        gt, crs = read_georef(page)
        shape = page.shape
        dtype = page.dtype
    return {"metadata": desc.get("metadata", {}), "band_names": desc.get("band_names"),
            "geotransform": gt, "crs": crs, "shape": tuple(shape), "dtype": np.dtype(dtype)}


def read_tiff(path):
    """Return ``(data, info)``; see :func:`read_info` for ``info`` keys."""
    data = tifffile.imread(str(path))
    return data, read_info(path)
