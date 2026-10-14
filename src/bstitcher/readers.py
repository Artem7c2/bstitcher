# SPDX-License-Identifier: Apache-2.0
"""Reader registry.

A reader is ``callable(path, *, eds=False, elements=None, all_images=False) -> Tile``.
Readers are registered by file extension, either explicitly with
:func:`register_reader` or by installed packages through the
``bstitcher.readers`` entry-point group (e.g. ``bcf = bstitcher_bcf.reader:read_bcf``).
Keeping format readers in separate distributions lets the core stay
permissively licensed while e.g. the BCF reader builds on GPL libraries.
"""
from __future__ import annotations

import os
from importlib.metadata import entry_points

_READERS: dict = {}
_LOADED = False


def register_reader(ext: str, func) -> None:
    _READERS[ext.lower().lstrip(".")] = func


def _load_entry_points() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for ep in entry_points(group="bstitcher.readers"):
        _READERS.setdefault(ep.name.lower(), ep.load())


def get_reader(path):
    _load_entry_points()
    ext = os.path.splitext(str(path))[1].lower().lstrip(".")
    try:
        return _READERS[ext]
    except KeyError:
        raise LookupError(
            f"no reader for '.{ext}' files; installed: {sorted(_READERS) or 'none'} "
            "(the Bruker .bcf reader is provided by the bstitcher-bcf package)") from None


def available_formats() -> list:
    _load_entry_points()
    return sorted(_READERS)
