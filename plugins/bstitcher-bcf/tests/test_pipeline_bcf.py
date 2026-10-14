"""Step 5b — full pipeline on the real 2x2 Bruker job.

Bruker's own ``Mosaic sample_Ch 1.tif`` places tiles at their stage-nominal
positions only.  It therefore validates our stage -> pixel convention (run with
registration disabled), while the registered result is validated by overlap
agreement and loop closure (on this job the nominal offsets are ~3 px off).
"""
import numpy as np
import pytest
import tifffile

from bstitcher.pipeline import load_tile_records, run_session, stitch_tiles
from bstitcher.registration import _ncc
from bstitcher.tiff_io import read_info, read_tiff

pytestmark = pytest.mark.integration
ELEMENTS = ["Al", "Ca", "Cr", "Fe", "K", "Mg", "Na", "Si", "Ti"]


@pytest.fixture(scope="module")
def result(fixture_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("bcf_out")
    res = run_session(fixture_dir, out, eds=True, log=lambda *a: None)
    return out, res


def _nz(a):
    a = a.astype(np.float64)
    return (a - a.mean()) / (a.std() + 1e-12)


def _overlap_ncc(out, positions):
    """Mean NCC of BSE between tiles in their overlaps at the given positions."""
    recs = {r.index: r for r in load_tile_records(out / "tiles")}
    img = {i: read_tiff(r.files["BSE"])[0].astype(float) for i, r in recs.items()}
    vals = []
    for a in img:
        for b in img:
            if a >= b or a[0] != b[0] and a[1] != b[1]:
                continue
            (ra, ca), (rb, cb) = positions[a], positions[b]
            (h, w) = img[a].shape
            r0, r1 = max(ra, rb), min(ra, rb) + h
            c0, c1 = max(ca, cb), min(ca, cb) + w
            vals.append(_ncc(img[a][r0 - ra:r1 - ra, c0 - ca:c1 - ca], img[b][r0 - rb:r1 - rb, c0 - cb:c1 - cb]))
    return float(np.mean(vals))


def test_all_outputs_written(result):
    out, _ = result
    assert len(list((out / "tiles").glob("*_BSE.tif"))) == 4
    assert len(list((out / "tiles").glob("*_elem_*.tif"))) == 4 * 9
    assert len(list((out / "tiles").glob("*_EDS.tif"))) == 4
    for mod in ["BSE", "EDS"] + [f"elem_{e}" for e in ELEMENTS]:
        assert (out / "stitched" / f"{mod}.tif").exists(), mod


def test_nominal_layout_reproduces_bruker_mosaic(result, fixture_dir, tmp_path):
    out, _ = result
    stitch_tiles(out / "tiles", tmp_path / "nominal", modalities=["BSE"], max_shift=0,
                 log=lambda *a: None)
    bse, _ = read_tiff(tmp_path / "nominal" / "BSE.tif")
    ref = tifffile.imread(fixture_dir / "Mosaic sample_Ch 1.tif").astype(float)
    ref = ref.mean(axis=2) if ref.ndim == 3 else ref
    assert abs(bse.shape[0] - ref.shape[0]) <= 1 and abs(bse.shape[1] - ref.shape[1]) <= 1
    # Geometry is pixel-identical: the correlation peaks at zero offset.  The
    # peak stays < 1 because Bruker's export is an 8-bit display rendering.
    h, w = min(bse.shape[0], ref.shape[0]) - 6, min(bse.shape[1], ref.shape[1]) - 6
    corr = np.array([[np.mean(_nz(bse[3:3 + h, 3:3 + w]) * _nz(ref[3 + dy:3 + dy + h, 3 + dx:3 + dx + w]))
                      for dx in range(-2, 3)] for dy in range(-2, 3)])
    assert np.unravel_index(corr.argmax(), corr.shape) == (2, 2)
    assert corr[2, 2] > 0.85


def test_registration_improves_overlap_agreement(result):
    out, res = result
    rep = res["report"]
    md = read_info(out / "stitched" / "BSE.tif")["metadata"]
    solved = {tuple(t["index"]): tuple(t["position_px"]) for t in md["tiles"]}
    recs = load_tile_records(out / "tiles")
    from bstitcher.layout import nominal_positions
    nominal = {k: (round(v[0]), round(v[1])) for k, v in nominal_positions(recs).items()}
    assert _overlap_ncc(out, solved) > _overlap_ncc(out, nominal) + 0.2
    edge = [p for p in rep["pairs"] if p["a"][0] == p["b"][0] or p["a"][1] == p["b"][1]]
    assert all(p["used"] for p in edge)


def test_loop_closure(result):
    """All accepted pair measurements are reproduced by the solved positions to <= 1 px."""
    out, res = result
    md = read_info(out / "stitched" / "BSE.tif")["metadata"]
    pos = {tuple(t["index"]): t["position_px"] for t in md["tiles"]}
    for p in res["report"]["pairs"]:
        if not p["used"]:
            continue
        a, b = tuple(p["a"]), tuple(p["b"])
        meas = np.add(p["nominal_offset"], p["shift"])
        solved = np.subtract(pos[b], pos[a])
        assert np.abs(solved - meas).max() <= 1, p


def test_stitched_modalities_share_canvas_and_georef(result):
    out, _ = result
    ref = read_info(out / "stitched" / "BSE.tif")
    for p in (out / "stitched").glob("*.tif"):
        info = read_info(p)
        assert info["shape"][:2] == ref["shape"][:2], p.name
        assert info["geotransform"] == pytest.approx(ref["geotransform"]), p.name
        assert info["crs"] == "local"


def test_eds_mosaic_is_lossless(result):
    out, _ = result
    eds, info = read_tiff(out / "stitched" / "EDS.tif")
    assert eds.shape[2] == 4096 and eds.dtype == np.uint8
    assert info["metadata"]["energy"]["size"] == 4096
    owner, _ = read_tiff(out / "stitched" / "tile_map.tif")
    md = read_info(out / "stitched" / "BSE.tif")["metadata"]
    for t in md["tiles"]:
        tile, _ = read_tiff(out / "tiles" / f"Mapping_({t['index'][0]},{t['index'][1]})_EDS.tif")
        y, x = t["position_px"]
        m = owner[y:y + 88, x:x + 118] == t["tile_id"]
        np.testing.assert_array_equal(eds[y:y + 88, x:x + 118][m], tile[m])
    assert (owner > 0).sum() == int(sum((owner == t["tile_id"]).sum() for t in md["tiles"]))


def test_bruker_metadata_in_stitched_tiff(result):
    out, _ = result
    md = read_info(out / "stitched" / "elem_Fe.tif")["metadata"]
    assert md["element"] == "Fe" and md["beam_energy_keV"] == 15.0
    assert {tuple(t["index"]) for t in md["tiles"]} == {(1, 1), (1, 2), (2, 1), (2, 2)}
    assert all("Stage" in t["bruker"] for t in md["tiles"])


def test_eds_compressed_smaller_than_raw(result):
    out, _ = result
    p = out / "stitched" / "EDS.tif"
    info = read_info(p)
    raw = np.prod(info["shape"]) * info["dtype"].itemsize
    assert p.stat().st_size < raw / 3
