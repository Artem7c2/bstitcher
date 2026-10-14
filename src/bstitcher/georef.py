# SPDX-License-Identifier: Apache-2.0
"""GeoTIFF georeferencing without GDAL.

Two frames are supported:

``"pixel"``
    Legacy frame of the LS-Miner pipeline: EPSG:4326, 1 unit per pixel, origin
    at (0, height) so that y grows upward.  Web viewers (Leaflet) and the
    existing SAM pipeline rely on it.

``"local"``
    Physical frame: a local engineering CRS in metres derived from the SEM stage
    position and pixel size.  World coordinates are the *negated* stage
    coordinates (Bruker stage X decreases to the right of the image and stage Y
    increases downward), so tiles and mosaics of different sessions on the same
    sample mount line up in absolute coordinates.

A geotransform uses GDAL order ``(x0, px_w, 0, y0, 0, -px_h)``.
"""
from __future__ import annotations

UM = 1e-6  # metres per micrometre

CRS_PIXEL = "pixel"
CRS_LOCAL = "local"
LOCAL_CS_NAME = "SEM stage (negated stage coordinates)"

# EPSG:4326 key set, identical to what the gdal-based legacy pipeline wrote.
_GEOKEYS_PIXEL = (
    1, 1, 0, 7,
    1024, 0, 1, 2,             # GTModelTypeGeoKey = Geographic
    1025, 0, 1, 1,             # GTRasterTypeGeoKey = PixelIsArea
    2048, 0, 1, 4326,          # GeographicTypeGeoKey = WGS 84
    2049, 34737, 7, 0,         # GeogCitationGeoKey -> "WGS 84|"
    2054, 0, 1, 9102,          # GeogAngularUnitsGeoKey = degree
    2057, 34736, 1, 1,         # GeogSemiMajorAxisGeoKey
    2059, 34736, 1, 0,         # GeogInvFlatteningGeoKey
)
_DOUBLES_PIXEL = (298.257223563, 6378137.0)
_ASCII_PIXEL = "WGS 84|"


def _local_geokeys(citation: str):
    return (
        1, 1, 0, 3,
        1025, 0, 1, 1,                         # PixelIsArea
        1026, 34737, len(citation) + 1, 0,     # GTCitationGeoKey -> LOCAL_CS name
        3076, 0, 1, 9001,                      # ProjLinearUnitsGeoKey = metre
    )


def pixel_geotransform(height: int) -> tuple:
    """Legacy pixel frame: 1 unit per pixel, y up, origin at (0, height)."""
    return (0.0, 1.0, 0.0, float(height), 0.0, -1.0)


def physical_geotransform(origin_um, pixel_size_um) -> tuple:
    """Geotransform in metres.

    origin_um : (x, y) world coordinate (µm) of the top-left corner of the
        top-left pixel.
    pixel_size_um : (dx, dy) pixel size in µm.
    """
    x0, y0 = origin_um
    dx, dy = pixel_size_um
    return (x0 * UM, dx * UM, 0.0, y0 * UM, 0.0, -dy * UM)


def geotiff_extratags(geotransform, crs: str = CRS_PIXEL) -> list:
    """tifffile ``extratags`` entries for a north-up geotransform."""
    x0, sx, rx, y0, ry, sy = (float(v) for v in geotransform)
    if rx != 0.0 or ry != 0.0:
        raise ValueError("rotated geotransforms are not supported")
    tags = [
        (33550, 12, 3, (sx, -sy, 0.0), True),                 # ModelPixelScale
        (33922, 12, 6, (0.0, 0.0, 0.0, x0, y0, 0.0), True),   # ModelTiepoint
    ]
    if crs == CRS_PIXEL:
        tags += [
            (34735, 3, len(_GEOKEYS_PIXEL), _GEOKEYS_PIXEL, True),
            (34736, 12, len(_DOUBLES_PIXEL), _DOUBLES_PIXEL, True),
            (34737, 2, 0, _ASCII_PIXEL, True),
        ]
    elif crs == CRS_LOCAL:
        keys = _local_geokeys(LOCAL_CS_NAME)
        tags += [
            (34735, 3, len(keys), keys, True),
            (34737, 2, 0, LOCAL_CS_NAME + "|", True),
        ]
    else:
        raise ValueError(f"unknown crs {crs!r}; use 'pixel' or 'local'")
    return tags


def read_georef(page):
    """Return ``(geotransform, crs)`` from a tifffile page, or ``(None, None)``."""
    scale = page.tags.get(33550)
    tie = page.tags.get(33922)
    if scale is None or tie is None:
        return None, None
    sx, sy = scale.value[0], scale.value[1]
    i, j, _, x, y, _ = tie.value[:6]
    gt = (x - i * sx, sx, 0.0, y + j * sy, 0.0, -sy)
    keys = page.tags.get(34735)
    crs = None
    if keys is not None:
        k = keys.value
        ids = {k[n] for n in range(4, len(k), 4)}
        if 2048 in ids:
            crs = CRS_PIXEL
        elif 3076 in ids:
            crs = CRS_LOCAL
    return tuple(float(v) for v in gt), crs
