"""Step 5a — full pipeline on synthetic tiles via a throw-away reader (no GPL plugin, no real data)."""
import csv
import json

import numpy as np
import pytest

from bstitcher.pipeline import load_tile_records, run_session, stitch_tiles
from bstitcher.readers import register_reader
from bstitcher.tiff_io import read_info, read_tiff
from bstitcher.tile import Tile
from synthetic import make_mosaic, relative

C = 32


def read_npz(path, *, eds=False, elements=None, all_images=False):
    z = np.load(path)
    idx = tuple(int(v) for v in z["index"])
    bse = z["bse"]
    c = int(z["c"]) if "c" in z else C
    scale = float(z["scale"]) if "scale" in z else 0.01
    cube = np.stack([(bse >> 6).astype(np.uint8) % 7 + k % 3 for k in range(c)], axis=-1).astype(np.uint8)
    return Tile(index=idx, images={"BSE": bse},
                element_maps={"Fe": (bse // 16).astype(np.uint16), "Mg": (bse % 200).astype(np.uint8)},
                eds=cube if eds else None,
                energy={"offset_keV": 0.0, "scale_keV": scale, "size": c},
                windows_keV={"Fe": [6.3, 6.5], "Mg": [1.2, 1.3]},
                pixel_size_um=tuple(z["px"]), stage_um=tuple(z["stage"]),
                metadata={"beam_energy_keV": 15.0, "bruker": {"Stage": {"X": float(z["stage"][0])}}},
                source=path.split("/")[-1])


register_reader("npz", read_npz)


@pytest.fixture()
def session(tmp_path):
    truth, recs, images, true_pos = make_mosaic(nx=3, ny=2, jitter=2, seed=11)
    d = tmp_path / "session"
    d.mkdir()
    for r in recs:
        np.savez(d / f"Mapping_({r.index[0]},{r.index[1]}).npz", index=r.index, bse=images[r.index],
                 px=r.pixel_size_um, stage=r.stage_um)
    (d / "Mosaic map_0.npz").write_bytes(b"")  # non-tile files are ignored
    return d, truth, images, true_pos


def test_run_session_outputs(session, tmp_path):
    d, truth, images, true_pos = session
    out = tmp_path / "out"
    res = run_session(d, out, eds=True)
    tiles = sorted(p.name for p in (out / "tiles").glob("*.tif"))
    assert len(tiles) == 6 * 4  # BSE, elem_Fe, elem_Mg, EDS per tile
    st = {p.name for p in (out / "stitched").iterdir()}
    assert {"BSE.tif", "elem_Fe.tif", "elem_Mg.tif", "EDS.tif", "tile_map.tif",
            "placements.csv", "stitch_report.json"} <= st
    assert set(res["stitched"]) >= {"BSE", "elem_Fe", "elem_Mg", "EDS"}
    # PNG previews of the 2-D mosaics (not of the EDS cube or the tile map)
    assert sorted(p.name for p in (out / "visualization_stitched").iterdir()) == [
        "BSE.png", "elem_Fe.png", "elem_Mg.png"]
    assert sorted(res["visualization"]) == ["BSE", "elem_Fe", "elem_Mg"]


def test_no_visualization_when_disabled(session, tmp_path):
    d, *_ = session
    res = run_session(d, tmp_path / "out", visualize=False, log=lambda *a: None)
    assert not (tmp_path / "out" / "visualization_stitched").exists()
    assert res["visualization"] == {}


def test_no_error_file_without_errors(session, tmp_path):
    d, *_ = session
    res = run_session(d, tmp_path / "out", log=lambda *a: None)
    assert not (tmp_path / "out" / "tiles" / "failed_tiles.json").exists()
    assert res["report"]["failed_tiles"] == []


def test_stale_error_file_removed_on_clean_rerun(session, tmp_path):
    d, *_ = session
    bad = d / "Mapping_(9,9).npz"
    bad.write_bytes(b"truncated acquisition")
    run_session(d, tmp_path / "out", log=lambda *a: None)
    assert (tmp_path / "out" / "tiles" / "failed_tiles.json").exists()
    bad.rename(tmp_path / "moved_away.npz")
    res = run_session(d, tmp_path / "out", log=lambda *a: None)
    assert not (tmp_path / "out" / "tiles" / "failed_tiles.json").exists()
    assert res["report"]["failed_tiles"] == []


def test_stitched_positions_and_lossless_pixels(session, tmp_path):
    d, truth, images, true_pos = session
    out = tmp_path / "out"
    run_session(d, out, eds=True)
    with open(out / "stitched" / "placements.csv") as f:
        rows = list(csv.DictReader(f))
    pos = {(int(r["col_index"]), int(r["row_index"])): (int(r["row"]), int(r["col"])) for r in rows}
    assert relative(pos) == relative(true_pos)
    owner, _ = read_tiff(out / "stitched" / "tile_map.tif")
    for mod in ("BSE", "elem_Fe", "elem_Mg", "EDS"):
        mosaic, _ = read_tiff(out / "stitched" / f"{mod}.tif")
        assert mosaic.shape[:2] == owner.shape
        for r in rows:
            idx = (int(r["col_index"]), int(r["row_index"]))
            tile, _ = read_tiff(out / "tiles" / f"Mapping_({idx[0]},{idx[1]})_{mod}.tif")
            h, w = tile.shape[:2]
            y, x = pos[idx]
            m = owner[y:y + h, x:x + w] == int(r["tile_id"])
            assert m.any()
            np.testing.assert_array_equal(mosaic[y:y + h, x:x + w][m], tile[m])


def test_stitched_metadata_and_georef(session, tmp_path):
    d, *_ = session
    out = tmp_path / "out"
    run_session(d, out, georef="local")
    info = read_info(out / "stitched" / "BSE.tif")
    md = info["metadata"]
    assert md["modality"] == "BSE" and md["n_tiles"] == 6
    assert md["beam_energy_keV"] == 15.0
    assert len(md["tiles"]) == 6
    assert {"index", "position_px", "stage_um", "source_file", "bruker"} <= set(md["tiles"][0])
    assert info["crs"] == "local"
    # canvas origin agrees with each tile's own georeference (to 1 px)
    px = md["pixel_size_um"][0] * 1e-6
    for t in md["tiles"]:
        ti = read_info(out / "tiles" / f"Mapping_({t['index'][0]},{t['index'][1]})_BSE.tif")
        x_from_canvas = info["geotransform"][0] + t["position_px"][1] * px
        y_from_canvas = info["geotransform"][3] - t["position_px"][0] * px
        assert abs(x_from_canvas - ti["geotransform"][0]) <= 2.5 * px  # jitter ±2 px
        assert abs(y_from_canvas - ti["geotransform"][3]) <= 2.5 * px
    eds = read_info(out / "stitched" / "elem_Fe.tif")
    assert eds["metadata"]["integration_window_keV"] == [6.3, 6.5]
    report = json.loads((out / "stitched" / "stitch_report.json").read_text())
    assert report["n_tiles"] == 6 and len(report["pairs"]) >= 7


def test_pixel_georef_option(session, tmp_path):
    d, *_ = session
    out = tmp_path / "out"
    run_session(d, out, georef="pixel")
    info = read_info(out / "stitched" / "BSE.tif")
    H = info["shape"][0]
    assert info["crs"] == "pixel"
    assert info["geotransform"] == pytest.approx((0, 1, 0, H, 0, -1))


def test_restitch_from_existing_tiles(session, tmp_path):
    d, *_ = session
    out = tmp_path / "out"
    run_session(d, out)
    recs = load_tile_records(out / "tiles")
    assert len(recs) == 6 and "elem_Fe" in recs[0].files
    res = stitch_tiles(out / "tiles", tmp_path / "restitched", seam="overwrite", modalities=["BSE"])
    assert set(res) == {"BSE"}
    a, _ = read_tiff(out / "stitched" / "BSE.tif")
    b, _ = read_tiff(tmp_path / "restitched" / "BSE.tif")
    assert a.shape == b.shape


def test_no_tiles_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_session(tmp_path, tmp_path / "out")


def test_unreadable_tile_is_skipped_and_reported(session, tmp_path):
    d, *_ = session
    (d / "Mapping_(9,9).npz").write_bytes(b"truncated acquisition")
    res = run_session(d, tmp_path / "out", log=lambda *a: None)
    assert res["report"]["n_tiles"] == 6
    assert [f["file"] for f in res["report"]["failed_tiles"]] == ["Mapping_(9,9).npz"]
    assert res["report"]["failed_tiles"][0]["error"]
    rep = json.loads((tmp_path / "out" / "stitched" / "stitch_report.json").read_text())
    assert rep["failed_tiles"][0]["file"] == "Mapping_(9,9).npz"
    assert (tmp_path / "out" / "tiles" / "failed_tiles.json").exists()


def test_unreadable_tile_strict_raises(session, tmp_path):
    d, *_ = session
    (d / "Mapping_(9,9).npz").write_bytes(b"truncated acquisition")
    with pytest.raises(Exception):
        run_session(d, tmp_path / "out", strict=True, log=lambda *a: None)


def test_all_tiles_unreadable_names_the_reason(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    for i in (1, 2):
        (d / f"Mapping_({i},1).npz").write_bytes(b"not a tile")
    with pytest.raises(RuntimeError, match=r"none of the 2 tiles could be read.*Mapping_\(1,1\)\.npz"):
        run_session(d, tmp_path / "out", log=lambda *a: None)


def test_eds_with_varying_channel_count_and_calibration(tmp_path):
    """Bruker cuts spectra per tile and calibrates each acquisition: pad, never resample."""
    truth, recs, images, true_pos = make_mosaic(nx=3, ny=1, jitter=0, seed=12)
    d = tmp_path / "s"
    d.mkdir()
    sizes = {(1, 1): 30, (2, 1): 32, (3, 1): 27}
    scales = {(1, 1): 0.01, (2, 1): 0.01, (3, 1): 0.0101}
    for r in recs:
        np.savez(d / f"Mapping_({r.index[0]},{r.index[1]}).npz", index=r.index, bse=images[r.index],
                 px=r.pixel_size_um, stage=r.stage_um, c=sizes[r.index], scale=scales[r.index])
    out = tmp_path / "out"
    res = run_session(d, out, eds=True, log=lambda *a: None)
    eds, info = read_tiff(out / "stitched" / "EDS.tif")
    assert eds.shape[2] == 32
    md = info["metadata"]
    assert md["energy_axis_uniform"] is False
    assert {t["energy"]["size"] for t in md["tiles"]} == {27, 30, 32}
    owner, _ = read_tiff(out / "stitched" / "tile_map.tif")
    for t in md["tiles"]:
        tile, _ = read_tiff(out / "tiles" / f"Mapping_({t['index'][0]},{t['index'][1]})_EDS.tif")
        y, x = t["position_px"]
        h, w, c = tile.shape
        m = owner[y:y + h, x:x + w] == t["tile_id"]
        np.testing.assert_array_equal(eds[y:y + h, x:x + w, :c][m], tile[m])
        assert not eds[y:y + h, x:x + w, c:][m].any()  # padding is zero
    ea = res["report"]["eds_energy_axis"]
    assert ea["uniform"] is False and ea["max_deviation_eV"] == pytest.approx(0.0001 * 26 * 1000, rel=0.05)
