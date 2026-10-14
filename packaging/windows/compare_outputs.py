"""Compare two bstitcher output folders (e.g. Linux run vs. Windows exe run).

Pixel data, dtype, georeference and metadata must be identical; only values that
legitimately differ between runs (absolute paths) are ignored.

    python compare_outputs.py <out_a> <out_b>
"""
import json
import sys
from pathlib import Path

import numpy as np

from bstitcher.tiff_io import read_tiff

IGNORE_KEYS = {"source_dir", "tiles_dir", "stitched_dir", "session_dir", "out_dir", "path"}


def scrub(obj):
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if k not in IGNORE_KEYS}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def main(a, b) -> int:
    a, b = Path(a), Path(b)
    fa = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    problems = []
    if fa != fb:
        problems.append(f"file lists differ: only in A {sorted(set(fa) - set(fb))}, only in B {sorted(set(fb) - set(fa))}")
    n_tif = 0
    for rel in sorted(set(fa) & set(fb)):
        pa, pb = a / rel, b / rel
        if rel.suffix == ".tif":
            n_tif += 1
            try:
                da, ia = read_tiff(pa)
                db, ib = read_tiff(pb)
            except Exception as e:
                problems.append(f"{rel}: unreadable ({type(e).__name__}: {e})")
                continue
            if da.dtype != db.dtype or da.shape != db.shape or not np.array_equal(da, db):
                problems.append(f"{rel}: pixel data differ")
            if ia["geotransform"] != ib["geotransform"] or ia["crs"] != ib["crs"]:
                problems.append(f"{rel}: georeference differs")
            if scrub(ia["metadata"]) != scrub(ib["metadata"]):
                problems.append(f"{rel}: metadata differ")
        elif rel.suffix == ".json":
            if scrub(json.loads(pa.read_text())) != scrub(json.loads(pb.read_text())):
                problems.append(f"{rel}: JSON content differs")
        elif pa.read_bytes().replace(b"\r\n", b"\n") != pb.read_bytes().replace(b"\r\n", b"\n"):
            problems.append(f"{rel}: content differs")
    for p in problems:
        print("DIFF", p)
    print(f"compared {len(set(fa) & set(fb))} files ({n_tif} TIFFs): "
          + ("IDENTICAL" if not problems else f"{len(problems)} difference(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
