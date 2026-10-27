# SPDX-License-Identifier: GPL-3.0-or-later
"""Read one Bruker Esprit mapping tile (.bcf) into a :class:`bstitcher.tile.Tile`.

Uses HyperSpy/RosettaSciIO to parse the file and eXSpy to determine X-ray line
integration windows (same defaults as ``EDSSpectrum.get_lines_intensity``).
"""
from __future__ import annotations

import logging
import os

import numpy as np

from bstitcher.metadata import jsonable, parse_tile_index
from bstitcher.tile import Tile, integrate_windows

WINDOW_WIDTH = 2.0  # integration window = WINDOW_WIDTH * FWHM of the line (exspy default)


def _not_numba_notice(record) -> bool:
    # HyperSpy warns on import when the optional Numba accelerator is missing.  Reading
    # .bcf files does not use Numba, so the notice only alarms users; others pass.
    return not record.getMessage().startswith("Numba is not installed")


logging.getLogger("hyperspy.decorators").addFilter(_not_numba_notice)


def _split_signals(signals):
    images, eds = [], None
    for s in signals:
        if s.metadata.Signal.get_item("signal_type", "") .startswith("EDS"):
            eds = s
        else:
            images.append(s)
    if eds is None or not images:
        raise ValueError("BCF does not contain both an image and an EDS hypermap")
    return images, eds


def _clean_hyperspy_md(md: dict) -> dict:
    md = dict(md)
    md.pop("_HyperSpy", None)
    gen = dict(md.get("General", {}))
    gen.pop("FileIO", None)  # load timestamps: not acquisition metadata
    md["General"] = gen
    return md


def _line_windows(eds, elements):
    """{label: (i0, i1)} channel ranges and {label: [lo, hi]} channel-centre keV."""
    if elements is not None:
        eds.metadata.Sample.elements = []
        if eds.metadata.Sample.has_item("xray_lines"):
            del eds.metadata.Sample.xray_lines
        eds.add_elements(list(elements))
    if not eds.metadata.Sample.has_item("xray_lines"):
        eds.add_lines()
    lines = list(eds.metadata.Sample.xray_lines)
    windows = eds.estimate_integration_windows(windows_width=WINDOW_WIDTH, xray_lines=lines)
    ax = eds.axes_manager.signal_axes[0]
    per_element = {}
    for line in lines:
        per_element.setdefault(line.split("_")[0], []).append(line)
    out_idx, out_keV = {}, {}
    for line, (lo, hi) in zip(lines, windows):
        el = line.split("_")[0]
        label = el if len(per_element[el]) == 1 else line
        i0, i1 = ax.value2index(lo), ax.value2index(hi)  # as in eds.isig[lo:hi]
        out_idx[label] = (i0, i1)
        out_keV[label] = [float(ax.index2value(i0)), float(ax.index2value(i1 - 1))]
    return out_idx, out_keV


def read_bcf(path, *, eds: bool = False, elements=None, all_images: bool = False) -> Tile:
    import hyperspy.api as hs

    path = str(path)
    signals = hs.load(path)
    if not isinstance(signals, list):
        signals = [signals]
    images, s_eds = _split_signals(signals)

    img_dict = {"BSE": np.asarray(images[0].data)}
    if all_images:
        for k, s in enumerate(images[1:], start=1):
            img_dict[f"image{k}"] = np.asarray(s.data)

    ax = s_eds.axes_manager.signal_axes[0]
    energy = {"offset_keV": float(ax.offset), "scale_keV": float(ax.scale), "size": int(ax.size)}
    cube = np.asarray(s_eds.data)
    _, windows_keV = _line_windows(s_eds, elements)
    maps = integrate_windows(cube, energy, windows_keV)

    om = images[0].original_metadata.as_dictionary()
    mic, stage = om["Microscope"], om["Stage"]
    hmd = s_eds.metadata.as_dictionary()
    sem = hmd.get("Acquisition_instrument", {}).get("SEM", {})
    gen = hmd.get("General", {})
    metadata = {
        "beam_energy_keV": sem.get("beam_energy", mic.get("HV")),
        "magnification": sem.get("magnification", mic.get("Mag")),
        "working_distance_mm": mic.get("WD"),
        "acquisition_date": gen.get("date"),
        "acquisition_time": gen.get("time"),
        "eds_detector": sem.get("Detector", {}).get("EDS", {}).get("detector_type"),
        "eds_real_time_s": sem.get("Detector", {}).get("EDS", {}).get("real_time"),
        "bruker": jsonable(om),
        "hyperspy": jsonable(_clean_hyperspy_md(hmd)),
        "reader": "bstitcher-bcf (rosettasciio/exspy)",
    }
    return Tile(
        index=parse_tile_index(os.path.basename(path)),
        images=img_dict,
        element_maps=maps,
        eds=cube if eds else None,
        energy=energy,
        windows_keV=windows_keV,
        pixel_size_um=(float(mic["DX"]), float(mic["DY"])),
        stage_um=(float(stage["X"]), float(stage["Y"])),
        metadata=metadata,
        source=os.path.basename(path),
    )
