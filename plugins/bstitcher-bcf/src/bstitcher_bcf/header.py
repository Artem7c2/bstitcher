# SPDX-License-Identifier: GPL-3.0-or-later
"""``EDSDatabase/HeaderData`` for a stitched BCF, adapted from a tile's header.

The XML of the first tile is kept (instrument, detector, elements, colours ...)
and only what describes the map is changed: image size, pixel size, stage
position (mosaic centre), energy calibration, line counters and the sum
spectrum.  The BSE image is the stitched one; planes marked invalid
(``Valid = 0``) and the tile's overview image are dropped, they would only
inflate the header.  The BSE pixels are base64 text inside the XML, so the
header is returned in two parts around that text and the image is streamed
in between (see :func:`iter_header`).
"""
from __future__ import annotations

import base64
import xml.etree.ElementTree as ET

import numpy as np

_PLACEHOLDER = "@@BSTITCHER_BSE_DATA@@"
_B64_CHUNK = 3 * 1 << 20  # multiple of 3: chunked base64 == base64 of the whole


def _fmt(v: float) -> str:
    return repr(float(v))


def _set(parent, path, value):
    e = parent.find(path)
    if e is None:
        raise KeyError(f"{path} not in BCF header")
    e.text = value if isinstance(value, str) else _fmt(value) if isinstance(value, float) else str(value)


def _image_instances(root, name=None):
    for ci in root.iter("ClassInstance"):
        if ci.get("Type") == "TRTImageData" and ci.get("Name") == name:
            yield ci


def _planes(img):
    return [e for e in img if e.tag.startswith("Plane") and e.tag[5:].isdigit()]


def _line_counter(height: int, value: str) -> str:
    return ",".join([value] * height) + ","


def adapt_header(template_xml: bytes, *, height: int, width: int, pixel_size_um, stage_um,
                 energy: dict, sum_spectrum, bse_dtype=np.uint16) -> tuple[bytes, bytes, bytes]:
    """Return ``(declaration_and_head, tail, encoding)``; the BSE base64 text goes between."""
    decl_end = template_xml.index(b"?>") + 2
    declaration = template_xml[:decl_end]
    encoding = "cp1252" if b"WINDOWS-1252" in declaration.upper() else "utf-8"
    root = ET.fromstring(template_xml)
    db = root.find("ClassInstance[@Type='TRTSpectrumDatabase']")
    h, w = str(height), str(width)

    _set(db, "ClassInstance[@Type='TRTSEMData']/DX", float(pixel_size_um[0]))
    _set(db, "ClassInstance[@Type='TRTSEMData']/DY", float(pixel_size_um[1]))
    _set(db, "ClassInstance[@Type='TRTSEMStageData']/X", float(stage_um[0]))
    _set(db, "ClassInstance[@Type='TRTSEMStageData']/Y", float(stage_um[1]))
    _set(db, "ClassInstance[@Type='TRTDSPConfiguration']/ImageWidth", w)
    _set(db, "ClassInstance[@Type='TRTDSPConfiguration']/ImageHeight", h)

    # the map image: BSE in plane 0, invalid planes dropped
    img = next(_image_instances(db))
    _set(img, "Width", w)
    _set(img, "Height", h)
    _set(img, "ItemSize", str(np.dtype(bse_dtype).itemsize))
    planes = sorted(_planes(img), key=lambda e: int(e.tag[5:]))
    for p in planes[1:]:
        if p.findtext("Valid") == "0":
            img.remove(p)
    kept = _planes(img)
    _set(img, "PlaneCount", str(len(kept)))
    if len(kept) != 1:
        raise ValueError("stitched BCF supports one valid image plane (BSE)")
    p0 = kept[0]
    lc_value = (p0.findtext("LineCounter") or "1,").split(",")[0] or "1"
    _set(p0, "LineCounter", _line_counter(height, lc_value))
    _set(p0, "Data", _PLACEHOLDER)
    _set(p0, "Size", str(height * width * np.dtype(bse_dtype).itemsize))
    for name in ("Counter", "PixelTimes"):
        for ci in _image_instances(db, name):
            _set(ci, "Width", w)
            _set(ci, "Height", h)
            for p in _planes(ci):
                ci.remove(p)
            _set(ci, "PlaneCount", "0")
    for ov in db.iter("ClassInstance"):
        if ov.get("Type") == "TRTContainerClass" and ov.get("Name") == "OverviewImages":
            children = ov.find("ChildClassInstances")
            if children is not None:
                for c in list(children):
                    children.remove(c)

    # spectrum map: line counter, calibration, sum spectrum
    lc = db.find("LineCounter")
    _set(db, "LineCounter", _line_counter(height, (lc.text or "1,").split(",")[0] or "1"))
    spec = db.find("SpectrumData0/ClassInstance[@Type='TRTSpectrum']")
    sh = spec.find("ClassInstance[@Type='TRTSpectrumHeader']")
    n_ch = int(sh.findtext("ChannelCount"))
    _set(sh, "CalibAbs", float(energy["offset_keV"]))
    _set(sh, "CalibLin", float(energy["scale_keV"]))
    s = np.zeros(n_ch, np.int64)
    s[:min(n_ch, len(sum_spectrum))] = np.asarray(sum_spectrum)[:n_ch]
    _set(spec, "Channels", ",".join(map(str, s.tolist())))

    text = ET.tostring(root, encoding="unicode")
    head, tail = text.split(_PLACEHOLDER)
    return declaration + head.encode(encoding), tail.encode(encoding), encoding.encode()


def iter_header(template_xml: bytes, bse, **kwargs):
    """Yield the complete header XML in pieces (the BSE image streamed as base64)."""
    bse = np.ascontiguousarray(bse)
    head, tail, _ = adapt_header(template_xml, height=bse.shape[0], width=bse.shape[1],
                                 bse_dtype=bse.dtype, **kwargs)
    yield head
    raw = memoryview(bse.astype(bse.dtype.newbyteorder("<"), copy=False)).cast("B")
    for i in range(0, len(raw), _B64_CHUNK):
        yield base64.b64encode(raw[i:i + _B64_CHUNK])
    yield tail
