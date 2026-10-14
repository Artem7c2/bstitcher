"""Step 3a — Tile model, element-window integration, per-tile GeoTIFF export (no BCF needed)."""
import numpy as np
import pytest

from bstitcher.tile import Tile, export_tile, integrate_windows, minimal_uint, tile_geotransform
from bstitcher.tiff_io import read_tiff

RNG = np.random.default_rng(1)


def make_tile(index=(1, 1), h=20, w=30, c=64, stage=(-100.0, 50.0), px=(2.0, 2.0)):
    eds = RNG.integers(0, 5, size=(h, w, c), dtype=np.uint8)
    return Tile(
        index=index,
        images={"BSE": RNG.integers(0, 4096, size=(h, w), dtype=np.uint16)},
        element_maps={"Fe": eds[..., 10:20].sum(-1, dtype=np.uint32)},
        eds=eds,
        energy={"offset_keV": -0.5, "scale_keV": 0.01, "size": c},
        pixel_size_um=px,
        stage_um=stage,
        metadata={"beam_energy_keV": 15.0, "bruker": {"Stage": {"X": stage[0]}}},
        windows_keV={"Fe": [6.2, 6.3]},
        source="Mapping_(1,1).bcf",
    )


def test_integrate_windows_sums_channel_ranges():
    cube = RNG.integers(0, 9, size=(5, 6, 100), dtype=np.uint8)
    energy = {"offset_keV": 0.0, "scale_keV": 0.1, "size": 100}
    maps = integrate_windows(cube, energy, {"A": (1.0, 2.0), "B": (5.05, 5.25)})
    np.testing.assert_array_equal(maps["A"], cube[..., 10:21].sum(-1))
    np.testing.assert_array_equal(maps["B"], cube[..., 51:53].sum(-1))


@pytest.mark.parametrize("vmax,expected", [(200, np.uint8), (255, np.uint8), (256, np.uint16),
                                           (70000, np.uint32)])
def test_minimal_uint_is_lossless(vmax, expected):
    a = np.array([[0, vmax]], dtype=np.float64)
    out = minimal_uint(a)
    assert out.dtype == expected
    np.testing.assert_array_equal(out, a)


def test_minimal_uint_rejects_non_integer():
    with pytest.raises(ValueError):
        minimal_uint(np.array([0.5]))


def test_tile_geotransform_pixel_and_local():
    t = make_tile(h=20, w=30, stage=(-100.0, 50.0), px=(2.0, 2.0))
    assert tile_geotransform(t, "pixel") == (0.0, 1.0, 0.0, 20.0, 0.0, -1.0)
    # world = -stage; stage is the tile centre -> top-left = (100 - 30, -50 + 20) µm
    gt = tile_geotransform(t, "local")
    assert gt == pytest.approx((70e-6, 2e-6, 0.0, -30e-6, 0.0, -2e-6))


def test_export_tile_writes_all_modalities_losslessly(tmp_path):
    t = make_tile()
    files = export_tile(t, tmp_path, eds=True, georef="local")
    assert set(files) == {"BSE", "elem_Fe", "EDS"}
    for name, path in files.items():
        assert path.endswith(f"Mapping_(1,1)_{name}.tif")
    bse, info = read_tiff(files["BSE"])
    np.testing.assert_array_equal(bse, t.images["BSE"])
    assert info["crs"] == "local"
    md = info["metadata"]
    assert md["modality"] == "BSE"
    assert md["tile_index"] == [1, 1]
    assert md["stage_um"] == [-100.0, 50.0] and md["pixel_size_um"] == [2.0, 2.0]
    assert md["beam_energy_keV"] == 15.0
    assert md["bruker"]["Stage"]["X"] == -100.0
    fe, info = read_tiff(files["elem_Fe"])
    np.testing.assert_array_equal(fe, t.element_maps["Fe"])
    assert fe.dtype == np.uint8  # stored in the smallest lossless unsigned dtype
    assert info["metadata"]["element"] == "Fe"
    assert info["metadata"]["integration_window_keV"] == [6.2, 6.3]
    eds, info = read_tiff(files["EDS"])
    np.testing.assert_array_equal(eds, t.eds)
    assert info["metadata"]["energy"] == {"offset_keV": -0.5, "scale_keV": 0.01, "size": 64}


def test_export_tile_without_eds(tmp_path):
    files = export_tile(make_tile(), tmp_path, eds=False)
    assert "EDS" not in files
