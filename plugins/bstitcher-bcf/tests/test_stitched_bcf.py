"""Step 8c — the stitched mosaic as one Bruker .bcf, checked by reading it back with HyperSpy."""
import logging
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from bstitcher.tiff_io import read_tiff

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def stitched(fixture_dir, tmp_path_factory):
    from bstitcher.pipeline import run_session
    from bstitcher_bcf.writer import write_stitched_bcf
    out = tmp_path_factory.mktemp("stitched_bcf")
    run_session(fixture_dir, out, eds=True, visualize=False, log=lambda *a: None)
    path = write_stitched_bcf(out / "stitched", fixture_dir / "Mapping_(1,1).bcf", log=lambda *a: None)
    logging.disable(logging.WARNING)
    import hyperspy.api as hs
    signals = hs.load(path)
    logging.disable(logging.NOTSET)
    return out / "stitched", path, signals


def _eds(signals):
    return next(s for s in signals if s.data.ndim == 3)


def _bse(signals):
    return next(s for s in signals if s.data.ndim == 2)


def test_written_next_to_the_stitched_tiffs(stitched):
    st, path, signals = stitched
    assert path == str(st / "stitched.bcf")
    assert not list(st.glob("*.part")) and not list(st.glob("bstitcher_pos_*"))
    assert len(signals) == 2  # BSE + spectrum map (the per-tile overview image is not carried over)


def test_spectrum_map_is_the_stitched_cube(stitched):
    st, _, signals = stitched
    cube, _ = read_tiff(st / "EDS.tif")
    data = _eds(signals).data
    assert data.shape[:2] == cube.shape[:2]
    c = min(data.shape[2], cube.shape[2])
    np.testing.assert_array_equal(data[..., :c], cube[..., :c])
    assert not cube[..., c:].any() and not data[..., c:].any()


def test_bse_image_is_the_stitched_bse(stitched):
    st, _, signals = stitched
    bse, _ = read_tiff(st / "BSE.tif")
    np.testing.assert_array_equal(_bse(signals).data, bse)


def test_calibration_and_position(stitched):
    st, _, signals = stitched
    _, info = read_tiff(st / "EDS.tif")
    md = info["metadata"]
    eds = _eds(signals)
    e = eds.axes_manager["Energy"]
    assert e.scale == pytest.approx(md["energy"]["scale_keV"]) and e.offset == pytest.approx(md["energy"]["offset_keV"])
    assert eds.axes_manager["height"].scale == pytest.approx(md["pixel_size_um"][1])
    H, W = eds.data.shape[:2]
    stage = eds.metadata.Acquisition_instrument.SEM.Stage
    ox, oy = md["origin_um"]
    assert stage.x == pytest.approx(-(ox + W * md["pixel_size_um"][0] / 2))
    assert stage.y == pytest.approx(-(oy - H * md["pixel_size_um"][1] / 2))
    assert stage.x != pytest.approx(md["tiles"][0]["stage_um"][0])  # mosaic centre, not tile (1,1)


def test_header_sum_spectrum_and_sizes(stitched):
    from rsciio.bruker._api import BCF_reader
    st, path, signals = stitched
    xml = BCF_reader(path).get_file("EDSDatabase/HeaderData").get_as_BytesIO_string().getvalue()
    root = ET.fromstring(xml)
    H, W = _eds(signals).data.shape[:2]
    assert root.findtext(".//ClassInstance[@Type='TRTDSPConfiguration']/ImageWidth") == str(W)
    assert root.findtext(".//ClassInstance[@Type='TRTDSPConfiguration']/ImageHeight") == str(H)
    assert len(root.find("ClassInstance/LineCounter").text.rstrip(",").split(",")) == H
    chans = np.array(root.find(".//SpectrumData0/ClassInstance/Channels").text.split(","), int)
    cube, _ = read_tiff(st / "EDS.tif")
    s = cube.reshape(-1, cube.shape[2]).sum(0, dtype=np.int64)
    np.testing.assert_array_equal(chans[:len(s)], s)
    assert xml.startswith(b'<?xml version="1.0" encoding="WINDOWS-1252"')
