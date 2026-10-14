"""Step 1 — lossless TIFF writing with embedded metadata and georeferencing."""
import json

import numpy as np
import pytest
import tifffile

from bstitcher.georef import physical_geotransform, pixel_geotransform
from bstitcher.tiff_io import read_tiff, write_tiff

RNG = np.random.default_rng(0)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.int16, np.float32])
def test_2d_roundtrip_is_bit_exact(tmp_path, dtype):
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    if info is not None:
        data = RNG.integers(info.min, info.max, size=(61, 97), dtype=dtype, endpoint=True)
    else:
        data = RNG.normal(size=(61, 97)).astype(dtype)
    p = tmp_path / "a.tif"
    write_tiff(p, data)
    back, _ = read_tiff(p)
    assert back.dtype == data.dtype
    np.testing.assert_array_equal(back, data)


def test_spectral_cube_roundtrip_hwc(tmp_path):
    cube = RNG.integers(0, 40, size=(23, 31, 300), dtype=np.uint8)
    p = tmp_path / "eds.tif"
    write_tiff(p, cube, band_names=[f"ch{i}" for i in range(300)])
    back, info = read_tiff(p)
    assert back.shape == (23, 31, 300)
    np.testing.assert_array_equal(back, cube)
    assert info["band_names"][:2] == ["ch0", "ch1"]


def test_tiled_write_of_memmap_is_lossless(tmp_path):
    mm = np.lib.format.open_memmap(tmp_path / "buf.npy", mode="w+", dtype=np.uint16, shape=(700, 530))
    mm[:] = RNG.integers(0, 4096, size=mm.shape, dtype=np.uint16)
    p = tmp_path / "big.tif"
    write_tiff(p, mm, tile=(256, 256))
    back, _ = read_tiff(p)
    np.testing.assert_array_equal(back, np.asarray(mm))
    with tifffile.TiffFile(p) as tf:
        assert tf.pages[0].is_tiled


def test_compression_is_applied_and_lossless_codec(tmp_path):
    data = np.tile(np.arange(256, dtype=np.uint16), (256, 1))
    p = tmp_path / "c.tif"
    write_tiff(p, data)
    with tifffile.TiffFile(p) as tf:
        page = tf.pages[0]
        assert page.compression.name in ("ZSTD", "ADOBE_DEFLATE", "LZMA")
        assert int(page.predictor) == 2  # horizontal differencing
    assert p.stat().st_size < data.nbytes / 4


def test_metadata_roundtrip_json(tmp_path):
    md = {"Stage": {"X": -2993.75, "Y": 3665.25}, "HV": 15.0, "elements": ["Fe", "Mg"],
          "nested": {"a": [1, 2, {"b": None}]}}
    p = tmp_path / "m.tif"
    write_tiff(p, np.zeros((4, 5), np.uint16), metadata=md)
    _, info = read_tiff(p)
    assert info["metadata"] == md
    # Full JSON also readable by any TIFF tool via ImageDescription.
    with tifffile.TiffFile(p) as tf:
        assert json.loads(tf.pages[0].description)["metadata"] == md


def test_metadata_with_numpy_values_is_serialised(tmp_path):
    md = {"x": np.float64(1.5), "arr": np.arange(3), "i": np.int32(7)}
    p = tmp_path / "n.tif"
    write_tiff(p, np.zeros((4, 5), np.uint8), metadata=md)
    _, info = read_tiff(p)
    assert info["metadata"] == {"x": 1.5, "arr": [0, 1, 2], "i": 7}


def test_pixel_georef_matches_legacy_convention(tmp_path):
    h, w = 40, 70
    p = tmp_path / "g.tif"
    write_tiff(p, np.zeros((h, w), np.uint16), geotransform=pixel_geotransform(h), crs="pixel")
    _, info = read_tiff(p)
    assert info["geotransform"] == pytest.approx((0.0, 1.0, 0.0, float(h), 0.0, -1.0))
    assert info["crs"] == "pixel"


def test_physical_georef_roundtrip(tmp_path):
    gt = physical_geotransform(origin_um=(2993.75, -3665.25), pixel_size_um=(4.06, 4.08))
    p = tmp_path / "phys.tif"
    write_tiff(p, np.zeros((10, 12), np.uint16), geotransform=gt, crs="local")
    _, info = read_tiff(p)
    assert info["geotransform"] == pytest.approx(gt)
    assert info["crs"] == "local"


rasterio = pytest.importorskip("rasterio")


def test_gdal_reads_pixel_georef(tmp_path):
    h, w = 40, 70
    p = tmp_path / "g.tif"
    write_tiff(p, np.zeros((h, w), np.uint16), geotransform=pixel_geotransform(h), crs="pixel",
               metadata={"HV": 15.0})
    with rasterio.open(p) as ds:
        assert ds.crs.to_epsg() == 4326
        assert tuple(ds.transform)[:6] == pytest.approx((1.0, 0.0, 0.0, 0.0, -1.0, float(h)))
        assert ds.tags()["HV"] == "15.0"  # key items surface as GDAL metadata


def test_gdal_reads_physical_georef_in_metres(tmp_path):
    gt = physical_geotransform(origin_um=(2993.75, -3665.25), pixel_size_um=(4.0, 4.0))
    p = tmp_path / "phys.tif"
    write_tiff(p, np.zeros((10, 12), np.uint16), geotransform=gt, crs="local")
    with rasterio.open(p) as ds:
        a = ds.transform
        assert a.c == pytest.approx(2993.75e-6)
        assert a.a == pytest.approx(4.0e-6)
        assert a.e == pytest.approx(-4.0e-6)
        assert ds.crs is not None and not ds.crs.is_geographic


def test_gdal_band_descriptions(tmp_path):
    cube = np.zeros((5, 6, 3), np.uint8)
    p = tmp_path / "b.tif"
    write_tiff(p, cube, band_names=["Fe", "Mg", "Si"])
    with rasterio.open(p) as ds:
        assert ds.count == 3
        assert list(ds.descriptions) == ["Fe", "Mg", "Si"]
