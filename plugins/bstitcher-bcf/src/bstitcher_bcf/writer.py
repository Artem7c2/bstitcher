# SPDX-License-Identifier: GPL-3.0-or-later
"""Write a stitched mosaic as one Bruker ``.bcf`` (spectrum map + BSE + tile header).

Source: the stitched ``EDS.tif`` / ``BSE.tif`` of a bstitcher run, plus one raw
tile ``.bcf`` of the same job as template for the instrument / detector /
element metadata.  The EDS cube is streamed band by band (its GeoTIFF tile
rows) as sparse nonzero counts, so mosaics far larger than memory work.

Counts are lossless; the energy axis is the reference calibration of the
stitched cube (``energy`` in its metadata).  Every pixel of the mosaic gets a
record, pixels outside all tiles have zero counts.
"""
from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from bstitcher.tiff_io import read_info, read_tiff

from .header import iter_header
from .hypermap import HypermapWriter
from .sfs import SFSWriter

STITCHED_BCF = "stitched.bcf"


def _template_internals(template):
    from rsciio.bruker._api import BCF_reader
    b = BCF_reader(str(template))
    xml = b.get_file("EDSDatabase/HeaderData").get_as_BytesIO_string().getvalue()
    sd = b.get_file("EDSDatabase/SpectrumData0")
    first = next(sd.get_iter_and_properties()[0])
    return xml, first[:0x1A0]


def _iter_bands(eds_path):
    """Yield ``(y0, rows, entries)`` per tile row of the stitched EDS GeoTIFF.

    ``entries[r]`` = list of ``(x, c, v)`` arrays for line ``y0 + r``, one per tile.
    """
    with tifffile.TiffFile(str(eds_path)) as tf:
        page = tf.pages[0]
        H, W, C = page.shape
        th = page.tilelength if page.is_tiled else H
        band, band_y = None, None
        for seg, idx, _shape in page.segments():
            y0, x0 = idx[2], idx[3]
            if band_y != y0:
                if band is not None:
                    yield band_y, band
                band_y, band = y0, [[] for _ in range(min(th, H - y0))]
            seg = seg.reshape(seg.shape[-3:])[: len(band), : max(0, min(seg.shape[-2], W - x0))]
            yy, xx, cc = np.nonzero(seg)
            if yy.size:
                vv = seg[yy, xx, cc]
                cuts = np.searchsorted(yy, np.arange(len(band) + 1))
                for r in range(len(band)):
                    a, b = cuts[r], cuts[r + 1]
                    if b > a:
                        band[r].append((xx[a:b] + x0, cc[a:b], vv[a:b]))
        if band is not None:
            yield band_y, band


def _merge(parts):
    if not parts:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)
    x = np.concatenate([p[0] for p in parts])
    c = np.concatenate([p[1] for p in parts])
    v = np.concatenate([p[2] for p in parts])
    order = np.lexsort((c, x))  # tiles of one band are disjoint in x, but keep it general
    return x[order], c[order], v[order]


def write_stitched_bcf(stitched_dir, template, out=None, *, log=print) -> str:
    """Write ``<stitched_dir>/stitched.bcf`` (or ``out``); returns its path."""
    stitched_dir = Path(stitched_dir)
    eds_path, bse_path = stitched_dir / "EDS.tif", stitched_dir / "BSE.tif"
    if not eds_path.exists():
        raise FileNotFoundError(f"{eds_path} missing: the stitched BCF needs the EDS cube (run with --eds)")
    out = Path(out) if out else stitched_dir / STITCHED_BCF
    info = read_info(eds_path)
    H, W, C = info["shape"]
    md = info["metadata"]
    bse, _ = read_tiff(bse_path)
    if bse.shape != (H, W):
        raise ValueError(f"BSE {bse.shape} and EDS {(H, W)} mosaics differ in size")
    xml, sd_header = _template_internals(template)
    px = md["pixel_size_um"]
    ox, oy = md["origin_um"]
    # world = negated stage; the stage position of a field is its centre
    stage = (-(ox + W * px[0] / 2.0), -(oy - H * px[1] / 2.0))

    total = np.zeros(C, np.int64)
    tmp_pos = tempfile.NamedTemporaryFile(prefix="bstitcher_pos_", suffix=".tmp", dir=str(out.parent), delete=False)
    tmp_pos.close()
    tmp_out = out.with_suffix(".bcf.part")
    try:
        with SFSWriter(tmp_out) as sfs:
            with sfs.open_file("EDSDatabase/SpectrumData0") as f:
                n_ch = struct.unpack("<i", sd_header[8:12])[0]  # the template's channel count
                if C > n_ch:
                    raise ValueError(f"EDS cube has {C} channels, the template BCF only {n_ch}")
                hw = HypermapWriter(f, H, W, positions=tmp_pos.name, header=sd_header, channels=n_ch)
                for y0, band in _iter_bands(eds_path):
                    for parts in band:
                        x, c, v = _merge(parts)
                        if v.size:
                            np.add.at(total, c, v)
                        hw.write_line_sparse(x, c, v)
                    log(f"stitched.bcf: lines {y0 + len(band)}/{H}")
                hw.close()
            with sfs.open_file("EDSDatabase/HeaderData", position=0) as f:
                for piece in iter_header(xml, bse, pixel_size_um=px, stage_um=stage,
                                         energy=md["energy"], sum_spectrum=total):
                    f.write(piece)
            with sfs.open_file("EDSDatabase/SpectrumPositions0") as f, open(tmp_pos.name, "rb") as pf:
                while chunk := pf.read(1 << 24):
                    f.write(chunk)
        os.replace(tmp_out, out)
    finally:
        os.remove(tmp_pos.name)
        if tmp_out.exists():
            os.remove(tmp_out)
    log(f"stitched.bcf: {out} ({out.stat().st_size / 2**20:.0f} MB)")
    return str(out)
