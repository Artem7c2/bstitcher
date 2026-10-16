"""Step 7 — PNG previews of the stitched mosaics (grey scale, colour bar, scale bar)."""
import numpy as np
import pytest
from matplotlib.image import imread

from bstitcher.tiff_io import write_tiff
from bstitcher.visualize import block_reduce, nice_scalebar_um, render_png, visualize_stitched


@pytest.mark.parametrize("width_um,expected", [(1234, 200), (100, 20), (60, 10), (5000, 1000), (0.9, 0.1)])
def test_nice_scalebar_is_1_2_5_and_about_a_fifth(width_um, expected):
    assert nice_scalebar_um(width_um) == pytest.approx(expected)


def test_block_reduce_averages_valid_pixels_only():
    a = np.array([[1, 3, 10, 10], [5, 7, 10, 10], [0, 0, 4, 4]], np.uint16)
    m = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 1, 1]], bool)
    r, rm = block_reduce(a, 2, m)
    # partial last row / column blocks are kept
    assert r.shape == (2, 2)
    np.testing.assert_allclose(r[0], [4.0, 10.0])
    assert r[1, 1] == 4.0
    np.testing.assert_array_equal(rm, [[True, True], [False, True]])


def test_block_reduce_factor_one_is_identity():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    r, rm = block_reduce(a, 1)
    np.testing.assert_array_equal(r, a)
    assert rm.all()


def test_render_png(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.integers(1000, 3000, (300, 500)).astype(np.uint16)
    mask = np.ones(a.shape, bool)
    mask[:50, :80] = False  # outside every tile
    info = render_png(a, tmp_path / "x.png", pixel_size_um=2.0, title="BSE", mask=mask, max_px=250)
    assert info["downsample"] == 2
    assert info["scalebar_um"] == 200  # 1000 µm wide image -> 200 µm bar
    lo, hi = info["vmin"], info["vmax"]
    assert 1000 <= lo < hi <= 3000
    img = imread(tmp_path / "x.png")
    h, w = img.shape[:2]
    assert w > 250 and h > 150  # image plus colour bar / labels
    rgb = img[..., :3]
    grey = np.isclose(rgb[..., 0], rgb[..., 1]) & np.isclose(rgb[..., 1], rgb[..., 2])
    assert grey.mean() > 0.9  # grey-scale rendering; only the no-data colour is tinted
    assert not grey.all()  # ...and the masked region is visibly marked


def test_render_constant_image(tmp_path):
    info = render_png(np.full((20, 30), 7, np.uint8), tmp_path / "c.png", pixel_size_um=1.0, title="c")
    assert info["vmax"] > info["vmin"]
    assert (tmp_path / "c.png").exists()


def test_visualize_stitched_skips_cube_and_tile_map(tmp_path):
    st = tmp_path / "stitched"
    st.mkdir()
    md = {"pixel_size_um": [1.5, 1.5]}
    write_tiff(st / "BSE.tif", np.arange(600, dtype=np.uint16).reshape(20, 30), metadata={**md, "modality": "BSE"})
    write_tiff(st / "elem_Fe.tif", np.ones((20, 30), np.uint16),
               metadata={**md, "modality": "elem_Fe", "element": "Fe", "integration_window_keV": [6.3, 6.5]})
    write_tiff(st / "EDS.tif", np.ones((20, 30, 4), np.uint8), metadata={**md, "modality": "EDS"})
    write_tiff(st / "tile_map.tif", np.ones((20, 30), np.uint16), metadata={**md, "modality": "tile_map"})
    out = visualize_stitched(st, tmp_path / "viz", log=lambda *a: None)
    assert sorted(out) == ["BSE", "elem_Fe"]
    assert sorted(p.name for p in (tmp_path / "viz").iterdir()) == ["BSE.png", "elem_Fe.png"]
