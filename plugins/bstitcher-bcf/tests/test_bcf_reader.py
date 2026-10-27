"""Step 3b — Bruker .bcf reader plugin on a real 2x2 mapping job."""
import json

import numpy as np
import pytest

from bstitcher.readers import get_reader
from bstitcher.tile import Tile, integrate_windows

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tile(fixture_dir):
    from bstitcher_bcf.reader import read_bcf
    return read_bcf(fixture_dir / "Mapping_(2,1).bcf", eds=True)


def test_reader_is_registered_via_entry_point():
    from bstitcher_bcf.reader import read_bcf
    assert get_reader("x/Mapping_(1,1).bcf") is read_bcf
    assert get_reader("x/Mapping_(1,1).BCF") is read_bcf


def test_stitched_bcf_writer_is_registered_via_entry_point():
    from bstitcher.readers import get_mosaic_writer
    from bstitcher_bcf.writer import write_stitched_bcf
    assert get_mosaic_writer("bcf") is write_stitched_bcf


def test_tile_geometry_and_types(tile):
    assert isinstance(tile, Tile)
    assert tile.index == (2, 1)
    assert tile.images["BSE"].shape == (88, 118)
    assert tile.images["BSE"].dtype == np.uint16
    assert tile.eds.shape == (88, 118, 4096) and tile.eds.dtype == np.uint8
    assert tile.pixel_size_um == pytest.approx((4.060401343, 4.077776586))
    assert tile.stage_um == pytest.approx((-3424.75008964539, 3665.2500629425))


def test_energy_axis(tile):
    assert tile.energy["size"] == 4096
    assert tile.energy["scale_keV"] == pytest.approx(0.0049995, rel=1e-4)
    assert tile.energy["offset_keV"] == pytest.approx(-0.47735, abs=1e-4)


def test_elements_from_job(tile):
    assert sorted(tile.element_maps) == ["Al", "Ca", "Cr", "Fe", "K", "Mg", "Na", "Si", "Ti"]
    assert set(tile.windows_keV) == set(tile.element_maps)


def test_element_maps_reproducible_from_cube_and_windows(tile):
    again = integrate_windows(tile.eds, tile.energy, tile.windows_keV)
    for el, m in tile.element_maps.items():
        np.testing.assert_array_equal(m, again[el])


def test_element_maps_match_exspy_lines_intensity(fixture_dir, tile):
    import hyperspy.api as hs
    s = hs.load(str(fixture_dir / "Mapping_(2,1).bcf"))[-1]
    for m in s.get_lines_intensity():
        el = m.metadata.Sample.elements[0]
        np.testing.assert_array_equal(tile.element_maps[el], m.data.astype(np.uint64))


def test_metadata_is_json_and_complete(tile):
    md = json.loads(json.dumps(tile.metadata))
    assert md["beam_energy_keV"] == 15.0
    assert md["magnification"] == 250.0
    assert md["acquisition_date"] == "2024-10-15"
    assert md["bruker"]["Stage"]["X"] == pytest.approx(-3424.75008964539)
    assert md["bruker"]["DSP Configuration"]["ImageWidth"] == 118
    assert md["hyperspy"]["Acquisition_instrument"]["SEM"]["Detector"]["EDS"]["detector_type"]


def test_eds_optional(fixture_dir):
    from bstitcher_bcf.reader import read_bcf
    t = read_bcf(fixture_dir / "Mapping_(1,1).bcf", eds=False)
    assert t.eds is None and len(t.element_maps) == 9


def test_custom_element_list(fixture_dir):
    from bstitcher_bcf.reader import read_bcf
    t = read_bcf(fixture_dir / "Mapping_(1,1).bcf", elements=["Fe", "Mg", "O"])
    assert sorted(t.element_maps) == ["Fe", "Mg", "O"]


def test_all_images_option(fixture_dir):
    from bstitcher_bcf.reader import read_bcf
    t = read_bcf(fixture_dir / "Mapping_(1,1).bcf", all_images=True)
    assert list(t.images)[:1] == ["BSE"] and len(t.images) == 2
