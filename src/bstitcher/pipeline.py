# SPDX-License-Identifier: Apache-2.0
"""End-to-end pipeline: raw tiles -> per-tile GeoTIFFs -> stitched GeoTIFFs.

Output layout::

    <out>/tiles/Mapping_(c,r)_BSE.tif, _elem_<El>.tif, [_EDS.tif]
    <out>/stitched/BSE.tif, elem_<El>.tif, [EDS.tif], tile_map.tif,
                   placements.csv, stitch_report.json
    <out>/visualization_stitched/BSE.png, elem_<El>.png   previews (colour + scale bar)
    <out>/tiles/failed_tiles.json   only if some tiles could not be read
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from . import __version__
from .georef import CRS_LOCAL, CRS_PIXEL, physical_geotransform, pixel_geotransform
from .layout import TileRecord
from .metadata import parse_tile_index
from .readers import available_formats, describe_errors, get_mosaic_writer, get_reader
from .stitch import compose, owner_map, stitch_layout, write_mosaic_streamed
from .tiff_io import read_info, read_tiff, write_tiff
from .tile import EDS_TILE_LEVEL, export_tile
from .visualize import VIZ_DIR, visualize_stitched

EDS_STITCHED_TILE = (128, 128)
_TILE_KEYS = ("tile_index", "stage_um", "pixel_size_um", "source_file", "modality", "element",
              "integration_window_keV", "energy", "bruker", "hyperspy")


def find_tile_files(session_dir) -> list:
    """Raw tile files ``Mapping_(c,r).<ext>`` with an installed reader."""
    exts = set(available_formats())
    out = []
    for p in sorted(Path(session_dir).iterdir()):
        if p.suffix.lower().lstrip(".") not in exts:
            continue
        try:
            parse_tile_index(p.name)
        except ValueError:
            continue
        if p.stem.startswith("Mapping_(") and p.stem.endswith(")"):
            out.append(p)
    return out


FAILED_TILES_FILE = "failed_tiles.json"


def convert_tiles(session_dir, tiles_dir, *, eds: bool = False, elements=None,
                  all_images: bool = False, georef: str = CRS_LOCAL, strict: bool = False,
                  log=print) -> list:
    """Read every raw tile and export it as GeoTIFFs; returns TileRecords.

    Unreadable tiles (e.g. truncated by an aborted acquisition) are skipped and
    listed in ``<tiles_dir>/failed_tiles.json`` unless ``strict`` is set.  The
    file exists only when tiles failed; a stale one from an earlier run is removed.
    """
    if not available_formats():
        raise LookupError("no tile reader is available (install the bstitcher-bcf plugin for .bcf)"
                          + describe_errors())
    files = find_tile_files(session_dir)
    if not files:
        raise FileNotFoundError(f"no Mapping_(c,r).<{'|'.join(available_formats())}> tiles in "
                                f"{session_dir}")
    os.makedirs(tiles_dir, exist_ok=True)
    records, failed = [], []
    for n, f in enumerate(files, 1):
        try:
            tile = get_reader(f)(str(f), eds=eds, elements=elements, all_images=all_images)
        except Exception as e:
            if strict:
                raise
            failed.append({"file": f.name, "error": f"{type(e).__name__}: {e}"})
            log(f"[{n}/{len(files)}] {f.name} SKIPPED (unreadable: {e})")
            continue
        written = export_tile(tile, tiles_dir, eds=eds, georef=georef)
        records.append(TileRecord(tile.index, tile.shape, tile.stage_um, tile.pixel_size_um, written))
        log(f"[{n}/{len(files)}] {f.name} -> {len(written)} files")
    failed_path = os.path.join(str(tiles_dir), FAILED_TILES_FILE)
    if failed:
        with open(failed_path, "w") as fh:
            json.dump(failed, fh, indent=1)
    elif os.path.exists(failed_path):
        os.remove(failed_path)
    if not records:
        raise RuntimeError(f"none of the {len(files)} tiles could be read; first error "
                           f"({failed[0]['file']}): {failed[0]['error']}")
    return records


def read_failed_tiles(tiles_dir) -> list:
    p = os.path.join(str(tiles_dir), FAILED_TILES_FILE)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return json.load(fh)


def load_tile_records(tiles_dir) -> list:
    """Rebuild TileRecords from exported tile GeoTIFFs (metadata in each BSE tile)."""
    tiles_dir = Path(tiles_dir)
    records = {}
    for p in sorted(tiles_dir.glob("Mapping_(*)_BSE.tif")):
        md = read_info(p)
        m = md["metadata"]
        idx = tuple(m["tile_index"])
        records[idx] = TileRecord(idx, md["shape"][:2], tuple(m["stage_um"]),
                                  tuple(m["pixel_size_um"]), {})
    for p in sorted(tiles_dir.glob("Mapping_(*)_*.tif")):
        idx = parse_tile_index(p.name)
        if idx in records:
            mod = p.stem.split(")_", 1)[1]
            records[idx].files[mod] = str(p)
    if not records:
        raise FileNotFoundError(f"no Mapping_(c,r)_BSE.tif tiles in {tiles_dir}")
    return [records[k] for k in sorted(records)]


def _session_metadata(records, layout, tile_infos: dict) -> dict:
    """Common acquisition scalars + per-tile entries (incl. full Bruker metadata)."""
    first = tile_infos[records[0].index]["metadata"]
    common = {k: v for k, v in first.items()
              if k not in _TILE_KEYS and isinstance(v, (str, int, float, bool))}
    tiles = []
    for r in records:
        if r.index not in layout.positions:
            continue
        m = tile_infos[r.index]["metadata"]
        tiles.append({"index": list(r.index), "tile_id": layout.tile_ids[r.index],
                      "position_px": list(layout.positions[r.index]),
                      "stage_um": list(r.stage_um), "source_file": m.get("source_file"),
                      "bruker": m.get("bruker")})
    common.update({"pixel_size_um": list(layout.pixel_size_um), "origin_um": list(layout.origin_um),
                   "n_tiles": len(tiles), "hyperspy": first.get("hyperspy"), "tiles": tiles,
                   "stitcher": f"bstitcher {__version__}"})
    return common


def energy_axis_summary(axes: dict) -> dict:
    """Compare per-tile EDS energy axes against the most common calibration.

    ``max_deviation_eV``: largest energy difference of the same channel index
    between any tile and the reference, over that tile's channels.
    """
    keys = [(a["offset_keV"], a["scale_keV"]) for a in axes.values()]
    ref_key = max(set(keys), key=keys.count)
    ref = {"offset_keV": ref_key[0], "scale_keV": ref_key[1],
           "size": max(a["size"] for a in axes.values())}
    dev = 0.0
    for a in axes.values():
        k = a["size"] - 1
        for kk in (0, k):
            e = a["offset_keV"] + kk * a["scale_keV"]
            e_ref = ref["offset_keV"] + kk * ref["scale_keV"]
            dev = max(dev, abs(e - e_ref) * 1000.0)
    return {"reference": ref, "uniform": len(set(keys)) == 1,
            "n_calibrations": len(set(keys)), "max_deviation_eV": round(dev, 3),
            "sizes": sorted({a["size"] for a in axes.values()})}


def _geotransform(layout, georef):
    if georef == CRS_PIXEL:
        return pixel_geotransform(layout.canvas_shape[0])
    if georef == CRS_LOCAL:
        return physical_geotransform(layout.origin_um, layout.pixel_size_um)
    raise ValueError(f"unknown georef {georef!r}")


def stitch_tiles(tiles_dir, stitched_dir, *, viz_dir=None, log=print, **kwargs) -> dict:
    """Stitch exported tiles; returns ``{modality: path}``.  See :func:`_stitch`.

    With ``viz_dir`` PNG previews of the 2-D mosaics are written there as well.
    """
    outputs = _stitch(tiles_dir, stitched_dir, log=log, **kwargs)[0]
    if viz_dir is not None:
        visualize_stitched(stitched_dir, viz_dir, modalities=list(outputs), log=log)
    return outputs


def _stitch(tiles_dir, stitched_dir, *, modalities=None, georef: str = CRS_LOCAL,
            seam: str = "center", max_shift: int | None = None, min_score: float = 0.5,
            max_residual: float = 2.0, smooth: int = 1, log=print):
    """Stitch exported tiles; returns ``({modality: path}, report)``.

    Placement is computed once on BSE and applied to every modality, so all
    mosaics share one canvas.  The EDS cube is streamed tile by tile into a
    tiled GeoTIFF, so it never has to fit in memory or on scratch disk.
    """
    records = load_tile_records(tiles_dir)
    os.makedirs(stitched_dir, exist_ok=True)
    rec = {r.index: r for r in records}
    layout = stitch_layout(records, lambda i: read_tiff(rec[i].files["BSE"])[0],
                           max_shift=max_shift, min_score=min_score, max_residual=max_residual,
                           smooth=smooth)
    owner = owner_map(layout, seam)
    gt = _geotransform(layout, georef)
    tile_infos = {r.index: read_info(r.files["BSE"]) for r in records}
    session_md = _session_metadata(records, layout, tile_infos)

    all_mods = sorted({m for r in records for m in r.files}, key=lambda m: (m != "BSE", m == "EDS", m))
    mods = all_mods if modalities is None else [m for m in all_mods if m in modalities]
    outputs, eds_axis = {}, None
    for mod in mods:
        infos = {i: read_info(r.files[mod]) for i, r in rec.items() if mod in r.files}
        if not infos:
            continue
        dtype = np.result_type(*[v["dtype"] for v in infos.values()])
        first_md = next(iter(infos.values()))["metadata"]
        md = {"modality": mod, **{k: first_md[k] for k in ("element", "integration_window_keV",
                                                           "energy") if k in first_md},
              **session_md}
        path = os.path.join(str(stitched_dir), f"{mod}.tif")

        def load(i, mod=mod):
            r = rec[i]
            return read_tiff(r.files[mod])[0] if mod in r.files else None

        if mod == "EDS":
            axes = {i: v["metadata"]["energy"] for i, v in infos.items()}
            eds_axis = energy_axis_summary(axes)
            md.update(energy=eds_axis["reference"], energy_axis_uniform=eds_axis["uniform"],
                      tiles=[{**t, "energy": axes.get(tuple(t["index"]))} for t in md["tiles"]])
            C = max(v["shape"][2] for v in infos.values())

            def load(i, _load=load, C=C):  # zero-pad shorter spectra; never resample
                a = _load(i)
                if a is not None and a.shape[2] < C:
                    a = np.concatenate([a, np.zeros(a.shape[:2] + (C - a.shape[2],), a.dtype)], 2)
                return a

            write_mosaic_streamed(path, layout, load, owner=owner, dtype=dtype, n_bands=C,
                                  tile=EDS_STITCHED_TILE, metadata=md, geotransform=gt, crs=georef,
                                  level=EDS_TILE_LEVEL)
        else:
            canvas, _ = compose(layout, load, owner=owner, dtype=dtype)
            write_tiff(path, canvas, metadata=md, geotransform=gt, crs=georef)
        outputs[mod] = path
        log(f"stitched {mod}: {layout.canvas_shape} -> {path}")

    write_tiff(os.path.join(str(stitched_dir), "tile_map.tif"), owner,
               metadata={"modality": "tile_map", "seam": seam, **session_md},
               geotransform=gt, crs=georef)
    with open(os.path.join(str(stitched_dir), "placements.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["col_index", "row_index", "tile_id", "row", "col", "nominal_row", "nominal_col",
                    "stage_x_um", "stage_y_um"])
        for i in layout.order():
            r = rec[i]
            w.writerow([i[0], i[1], layout.tile_ids[i], *layout.positions[i],
                        *(round(v, 3) for v in layout.nominal[i]), *r.stage_um])
    report = {"n_tiles": len(layout.positions), "canvas_shape": list(layout.canvas_shape),
              "pixel_size_um": list(layout.pixel_size_um), "origin_um": list(layout.origin_um),
              "georef": georef, "seam": seam,
              "params": {"max_shift": layout.max_shift, "min_score": min_score, "max_residual": max_residual,
                         "smooth": smooth},
              "n_fallback_pairs": layout.n_fallback_pairs,
              "failed_tiles": read_failed_tiles(tiles_dir),
              **({"eds_energy_axis": eds_axis} if eds_axis else {}), "pairs": layout.pairs,
              "stitcher": f"bstitcher {__version__}"}
    with open(os.path.join(str(stitched_dir), "stitch_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return outputs, report


def run_session(session_dir, out_dir, *, eds: bool = False, elements=None, all_images: bool = False,
                georef: str = CRS_LOCAL, seam: str = "center", max_shift: int | None = None,
                min_score: float = 0.5, smooth: int = 1, strict: bool = False, visualize: bool = True,
                bcf: bool = False, log=print) -> dict:
    """Convert all raw tiles of a session, stitch them and (``visualize``) write PNG previews.

    ``bcf``: also write ``stitched/stitched.bcf`` (needs ``eds`` and the bstitcher-bcf plugin).
    """
    if bcf and not eds:
        raise ValueError("the stitched BCF needs the EDS cube: add --eds")
    bcf_writer = get_mosaic_writer("bcf") if bcf else None
    out_dir = Path(out_dir)
    tiles_dir, stitched_dir = out_dir / "tiles", out_dir / "stitched"
    records = convert_tiles(session_dir, tiles_dir, eds=eds, elements=elements,
                            all_images=all_images, georef=georef, strict=strict, log=log)
    stitched, report = _stitch(tiles_dir, stitched_dir, georef=georef, seam=seam,
                               max_shift=max_shift, min_score=min_score, smooth=smooth, log=log)
    viz = visualize_stitched(stitched_dir, out_dir / VIZ_DIR, modalities=list(stitched), log=log) \
        if visualize else {}
    res = {"tiles": [r.files for r in records], "stitched": stitched, "report": report,
           "visualization": viz}
    if bcf_writer is not None:
        failed = {f["file"] for f in report["failed_tiles"]}
        template = next(f for f in find_tile_files(session_dir) if f.name not in failed)
        res["bcf"] = bcf_writer(stitched_dir, template, log=log)
    return res
