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
_LOAD_ERRORS: dict = {}  # extension -> "ExcType: message" of a plugin that failed to import
_LOADED = False


def register_reader(ext: str, func) -> None:
    _READERS[ext.lower().lstrip(".")] = func


def _load_entry_points() -> None:
    global _LOADED
    if _LOADED:
        return
    eps = entry_points(group="bstitcher.readers")  # if this raises, the next call retries
    for ep in eps:
        name = ep.name.lower()
        if name in _READERS:
            continue
        try:
            _READERS[name] = ep.load()
        except Exception as e:  # keep the reason: "no reader" alone is undiagnosable
            _LOAD_ERRORS[name] = f"{type(e).__name__}: {e}"
    _LOADED = True


def reader_errors() -> dict:
    """``{extension: error}`` for installed reader plugins that failed to load."""
    _load_entry_points()
    return dict(_LOAD_ERRORS)


def describe_errors() -> str:
    errs = reader_errors()
    return "".join(f"; the .{k} reader failed to load: {v}" for k, v in sorted(errs.items()))


def get_reader(path):
    _load_entry_points()
    ext = os.path.splitext(str(path))[1].lower().lstrip(".")
    try:
        return _READERS[ext]
    except KeyError:
        raise LookupError(
            f"no reader for '.{ext}' files; installed: {sorted(_READERS) or 'none'} "
            "(the Bruker .bcf reader is provided by the bstitcher-bcf package)" + describe_errors()) from None


def available_formats() -> list:
    _load_entry_points()
    return sorted(_READERS)
