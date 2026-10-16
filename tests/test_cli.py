"""Step 6 — command line interface (in-process, synthetic reader)."""
import json

import pytest

from bstitcher.cli import main
from bstitcher.tiff_io import read_info
import test_pipeline_synthetic as tps  # registers the .npz test reader

session = tps.session


def test_run_command(session, tmp_path, capsys):
    d, *_ = session
    out = tmp_path / "out"
    assert main(["run", str(d), "--out", str(out), "--eds", "--georef", "pixel"]) == 0
    assert (out / "stitched" / "EDS.tif").exists()
    assert read_info(out / "stitched" / "BSE.tif")["crs"] == "pixel"
    assert (out / "visualization_stitched" / "BSE.png").exists()


def test_run_no_viz(session, tmp_path):
    d, *_ = session
    assert main(["run", str(d), "--out", str(tmp_path / "o"), "--no-viz"]) == 0
    assert not (tmp_path / "o" / "visualization_stitched").exists()


def test_convert_then_stitch(session, tmp_path):
    d, *_ = session
    tiles = tmp_path / "tiles"
    assert main(["convert", str(d), "--out", str(tiles)]) == 0
    assert len(list(tiles.glob("*_BSE.tif"))) == 6
    assert main(["stitch", str(tiles), "--out", str(tmp_path / "st"), "--seam", "overwrite",
                 "--modalities", "BSE", "elem_Fe"]) == 0
    assert sorted(p.name for p in (tmp_path / "st").glob("*.tif")) == ["BSE.tif", "elem_Fe.tif", "tile_map.tif"]
    rep = json.loads((tmp_path / "st" / "stitch_report.json").read_text())
    assert rep["seam"] == "overwrite"
    assert sorted(p.name for p in (tmp_path / "st" / "visualization_stitched").iterdir()) == [
        "BSE.png", "elem_Fe.png"]


def test_visualize_command(session, tmp_path):
    d, *_ = session
    main(["run", str(d), "--out", str(tmp_path / "o"), "--no-viz"])
    assert main(["visualize", str(tmp_path / "o" / "stitched"), "--out", str(tmp_path / "v")]) == 0
    assert (tmp_path / "v" / "BSE.png").exists()


def test_info_command(session, tmp_path, capsys):
    d, *_ = session
    main(["run", str(d), "--out", str(tmp_path / "o")])
    capsys.readouterr()
    assert main(["info", str(tmp_path / "o" / "stitched" / "BSE.tif")]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["metadata"]["modality"] == "BSE" and info["crs"] == "local"


def test_bad_session_returns_error(tmp_path, capsys):
    assert main(["run", str(tmp_path), "--out", str(tmp_path / "o")]) == 1
    assert "no Mapping_" in capsys.readouterr().err
