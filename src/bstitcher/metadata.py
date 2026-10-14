# SPDX-License-Identifier: Apache-2.0
"""Metadata helpers: JSON-safe conversion and tile-name parsing."""
from __future__ import annotations

import datetime as _dt
import math
import os
import re

import numpy as np

_TILE_RE = re.compile(r"Mapping_\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def jsonable(obj):
    """Recursively convert ``obj`` into plain JSON-serialisable Python types."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    if isinstance(obj, (_dt.date, _dt.datetime, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def parse_tile_index(name) -> tuple[int, int]:
    """``Mapping_(a,b)...`` -> ``(a, b)``.

    In Bruker Esprit jobs ``a`` is the column (stage X) and ``b`` the row
    (stage Y); stitching itself relies on stage coordinates, not on this.
    """
    m = _TILE_RE.search(os.path.basename(str(name)))
    if m is None:
        raise ValueError(f"not a tile name: {name!r}")
    return int(m.group(1)), int(m.group(2))


def tile_stem(index) -> str:
    return f"Mapping_({index[0]},{index[1]})"
