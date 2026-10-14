# SPDX-License-Identifier: Apache-2.0
"""Command line interface: ``bstitcher run|convert|stitch|info``."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .pipeline import convert_tiles, run_session, stitch_tiles
from .tiff_io import read_info


def _stitch_args(p):
    p.add_argument("--seam", choices=["center", "overwrite"], default="center",
                   help="which tile supplies overlap pixels (default: nearest tile centre)")
    p.add_argument("--max-shift", type=int, default=None,
                   help="registration search radius around the stage position, px "
                        "(default: 5%% of tile size, >= 8; 0 = stage positions only)")
    p.add_argument("--min-score", type=float, default=0.5,
                   help="minimum NCC for a pair to be trusted")
    p.add_argument("--smooth", type=int, default=1, help="box-filter radius used for matching only")


def _common(p):
    p.add_argument("--georef", choices=["local", "pixel"], default="local",
                   help="local: stage coordinates in metres; pixel: legacy EPSG:4326 pixel frame")


def build_parser():
    ap = argparse.ArgumentParser(prog="bstitcher", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="convert raw tiles and stitch them")
    run.add_argument("session")
    run.add_argument("--out", required=True)
    run.add_argument("--eds", action="store_true", help="also export / stitch the full EDS cube")
    run.add_argument("--elements", nargs="+", default=None, help="override the job's element list")
    run.add_argument("--all-images", action="store_true", help="export every image channel")
    run.add_argument("--strict", action="store_true", help="abort on unreadable tiles instead of skipping")
    _common(run)
    _stitch_args(run)

    conv = sub.add_parser("convert", help="convert raw tiles to GeoTIFF tiles only")
    conv.add_argument("session")
    conv.add_argument("--out", required=True)
    conv.add_argument("--eds", action="store_true")
    conv.add_argument("--elements", nargs="+", default=None)
    conv.add_argument("--all-images", action="store_true")
    _common(conv)

    st = sub.add_parser("stitch", help="stitch previously converted tiles")
    st.add_argument("tiles")
    st.add_argument("--out", required=True)
    st.add_argument("--modalities", nargs="+", default=None, help="e.g. BSE elem_Fe EDS (default: all)")
    _common(st)
    _stitch_args(st)

    info = sub.add_parser("info", help="print metadata / georeference of a bstitcher TIFF as JSON")
    info.add_argument("tiff")

    sub.add_parser("gui", help="open the desktop window")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "gui":
        from .gui import main as gui_main
        return gui_main([])
    logging.getLogger("hyperspy").setLevel(logging.ERROR)
    try:
        if args.cmd == "run":
            run_session(args.session, args.out, eds=args.eds, elements=args.elements,
                        all_images=args.all_images, georef=args.georef, seam=args.seam,
                        max_shift=args.max_shift, min_score=args.min_score, smooth=args.smooth,
                        strict=args.strict)
        elif args.cmd == "convert":
            convert_tiles(args.session, args.out, eds=args.eds, elements=args.elements,
                          all_images=args.all_images, georef=args.georef)
        elif args.cmd == "stitch":
            stitch_tiles(args.tiles, args.out, modalities=args.modalities, georef=args.georef,
                         seam=args.seam, max_shift=args.max_shift, min_score=args.min_score,
                         smooth=args.smooth)
        elif args.cmd == "info":
            info = read_info(args.tiff)
            info["dtype"] = str(info["dtype"])
            print(json.dumps(info, indent=1))
    except (FileNotFoundError, LookupError, ValueError, RuntimeError) as e:
        print(f"bstitcher: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
