"""Step 4 — layout, pairwise registration, global placement, composition (synthetic data)."""
import numpy as np
import pytest

from bstitcher.layout import TileRecord, nominal_positions, overlapping_pairs
from bstitcher.registration import register_pair
from bstitcher.stitch import compose, solve_positions, stitch_layout
from synthetic import make_mosaic, relative, textured_image


# ---------------------------------------------------------------- layout
def test_nominal_positions_follow_bruker_stage_convention():
    px = (4.0, 4.0)
    recs = [
        TileRecord((1, 1), (88, 118), (-2993.75, 3665.25), px),
        TileRecord((2, 1), (88, 118), (-2993.75 - 106 * 4.0, 3665.25), px),  # X decreases -> right
        TileRecord((1, 2), (88, 118), (-2993.75, 3665.25 + 79 * 4.0), px),   # Y increases -> down
    ]
    pos = nominal_positions(recs)
    assert pos[(1, 1)] == pytest.approx((0.0, 0.0))
    assert pos[(2, 1)] == pytest.approx((0.0, 106.0))
    assert pos[(1, 2)] == pytest.approx((79.0, 0.0))


def test_nominal_positions_reject_mixed_pixel_sizes():
    recs = [TileRecord((1, 1), (10, 10), (0, 0), (1.0, 1.0)),
            TileRecord((2, 1), (10, 10), (-8, 0), (2.0, 2.0))]
    with pytest.raises(ValueError, match="pixel size"):
        nominal_positions(recs)


def test_overlapping_pairs_grid():
    _, recs, _, _ = make_mosaic(nx=3, ny=3, jitter=0)
    pos = nominal_positions(recs)
    pairs = overlapping_pairs(pos, {r.index: r.shape for r in recs})
    edges = {(a, b) for a, b in pairs if a[0] == b[0] or a[1] == b[1]}
    assert len(edges) == 12  # 6 horizontal + 6 vertical neighbours
    assert ((1, 1), (2, 1)) in pairs and ((1, 1), (1, 2)) in pairs
    assert all(a < b for a, b in pairs)


# ---------------------------------------------------------- registration
@pytest.mark.parametrize("true_shift", [(0, 0), (2, -3), (-4, 1), (3, 3)])
def test_register_pair_recovers_shift(true_shift):
    img = textured_image((200, 260), seed=3)
    a = img[20:80, 20:100]
    nominal = (0, 66)  # b nominally 66 px right of a (14 px overlap)
    tr = (nominal[0] + true_shift[0], nominal[1] + true_shift[1])
    b = img[20 + tr[0]:80 + tr[0], 20 + tr[1]:100 + tr[1]]
    res = register_pair(a, b, nominal, max_shift=5)
    assert res.shift == true_shift
    assert res.score > 0.9


def test_register_pair_featureless_has_low_score():
    rng = np.random.default_rng(0)
    a = (1000 + rng.normal(0, 3, size=(60, 80))).astype(np.uint16)
    b = (1000 + rng.normal(0, 3, size=(60, 80))).astype(np.uint16)
    res = register_pair(a, b, (0, 66), max_shift=5)
    assert res.score < 0.5


def test_register_pair_robust_to_noise():
    rng = np.random.default_rng(5)
    img = textured_image((200, 260), seed=4).astype(float)
    a = img[20:80, 20:100] + rng.normal(0, 150, (60, 80))
    b = img[20 + 48:80 + 48, 20 + 2:100 + 2] + rng.normal(0, 150, (60, 80))
    res = register_pair(a, b, (50, 0), max_shift=5)
    assert res.shift == (-2, 2)


# ----------------------------------------------------- global placement
def _solve(recs, images, **kw):
    return stitch_layout(recs, lambda idx: images[idx], **kw)


def test_recovers_true_positions_exactly():
    truth, recs, images, true_pos = make_mosaic(nx=4, ny=3, jitter=3, seed=1)
    layout = _solve(recs, images, max_shift=6)
    assert relative(layout.positions) == relative(true_pos)
    edge = [p for p in layout.pairs if p["a"][0] == p["b"][0] or p["a"][1] == p["b"][1]]
    assert all(p["used"] for p in edge)  # tiny diagonal corners may be left out conservatively


def test_composed_canvas_equals_ground_truth():
    truth, recs, images, true_pos = make_mosaic(nx=3, ny=3, jitter=2, seed=2)
    layout = _solve(recs, images, max_shift=4)
    canvas, owner = compose(layout, lambda idx: images[idx])
    r0 = min(p[0] for p in true_pos.values())
    c0 = min(p[1] for p in true_pos.values())
    for idx, (r, c) in relative(true_pos).items():
        h, w = images[idx].shape
        np.testing.assert_array_equal(canvas[r:r + h, c:c + w], truth[r0 + r:r0 + r + h, c0 + c:c0 + c + w])
    assert canvas.dtype == np.uint16
    assert owner.max() == len(recs) and owner.min() >= 0


def test_seam_center_owner_is_nearest_tile_centre():
    truth, recs, images, _ = make_mosaic(nx=2, ny=1, jitter=0, tile=(40, 60), step=(0, 40))
    layout = _solve(recs, images)
    _, owner = compose(layout, lambda idx: images[idx], seam="center")
    ids = layout.tile_ids
    # overlap spans cols 40..59 of the canvas; left half -> left tile, right half -> right tile
    assert owner[20, 45] == ids[(1, 1)]
    assert owner[20, 55] == ids[(2, 1)]


def test_seam_overwrite_last_tile_wins():
    truth, recs, images, _ = make_mosaic(nx=2, ny=1, jitter=0, tile=(40, 60), step=(0, 40))
    layout = _solve(recs, images)
    _, owner = compose(layout, lambda idx: images[idx], seam="overwrite")
    assert owner[20, 41] == layout.tile_ids[(2, 1)]


def test_missing_tile_is_tolerated():
    truth, recs, images, true_pos = make_mosaic(nx=3, ny=3, jitter=2, seed=3, missing=[(2, 2)])
    layout = _solve(recs, images, max_shift=4)
    assert (2, 2) not in layout.positions
    assert relative(layout.positions) == relative(true_pos)


def test_featureless_tile_falls_back_to_nominal():
    truth, recs, images, true_pos = make_mosaic(nx=3, ny=1, jitter=0, seed=4, blank=[(3, 1)])
    layout = _solve(recs, images, max_shift=4)
    nom = nominal_positions(recs)
    rel_nom = relative({k: (round(v[0]), round(v[1])) for k, v in nom.items()})
    assert relative(layout.positions)[(3, 1)] == rel_nom[(3, 1)]
    assert layout.n_fallback_pairs >= 1


def test_single_row_strip():
    truth, recs, images, true_pos = make_mosaic(nx=5, ny=1, jitter=2, seed=6)
    layout = _solve(recs, images, max_shift=4)
    assert relative(layout.positions) == relative(true_pos)


def test_single_tile():
    truth, recs, images, true_pos = make_mosaic(nx=1, ny=1, jitter=0)
    layout = _solve(recs, images)
    assert layout.positions == {(1, 1): (0, 0)}
    assert layout.canvas_shape == images[(1, 1)].shape


def test_solve_positions_rejects_inconsistent_measurement():
    nominal = {(1, 1): (0.0, 0.0), (2, 1): (0.0, 50.0), (1, 2): (40.0, 0.0), (2, 2): (40.0, 50.0)}
    good = {((1, 1), (2, 1)): ((0, 50), 0.9), ((1, 1), (1, 2)): ((40, 0), 0.9),
            ((2, 1), (2, 2)): ((40, 0), 0.9), ((1, 2), (2, 2)): ((0, 50), 0.9)}
    bad = dict(good)
    bad[((1, 2), (2, 2))] = ((0, 58), 0.8)  # outlier: 8 px off, contradicts the loop
    pos, rejected = solve_positions(nominal, bad, max_residual=2.0)
    assert relative(pos) == {(1, 1): (0, 0), (2, 1): (0, 50), (1, 2): (40, 0), (2, 2): (40, 50)}
    assert ((1, 2), (2, 2)) in rejected


def test_smoothing_raises_score_on_noisy_tiles_without_changing_shift():
    rng = np.random.default_rng(7)
    img = textured_image((200, 260), seed=8).astype(float)
    a = img[20:80, 20:100] + rng.normal(0, 600, (60, 80))
    b = img[20 + 3:80 + 3, 20 + 68:100 + 68] + rng.normal(0, 600, (60, 80))
    raw = register_pair(a, b, (0, 66), max_shift=5, smooth=0)
    sm = register_pair(a, b, (0, 66), max_shift=5, smooth=1)
    assert sm.shift == (3, 2)
    assert sm.score > raw.score


def test_box_smooth_preserves_mean_and_shape():
    from bstitcher.registration import box_smooth
    a = np.arange(30.0).reshape(5, 6)
    s = box_smooth(a, 1)
    assert s.shape == a.shape
    assert s[2, 2] == pytest.approx(a[1:4, 1:4].mean())


def test_streamed_mosaic_equals_in_memory_and_loads_each_tile_once(tmp_path):
    from bstitcher.stitch import write_mosaic_streamed
    from bstitcher.tiff_io import read_tiff
    truth, recs, images, _ = make_mosaic(nx=3, ny=3, jitter=2, seed=9, tile=(70, 90), step=(55, 75))
    cubes = {k: np.stack([v >> 4, v >> 6, v % 251], axis=-1).astype(np.uint16) for k, v in images.items()}
    layout = _solve(recs, images, max_shift=4)
    expected, owner = compose(layout, lambda i: cubes[i])
    calls = []

    def load(i):
        calls.append(i)
        return cubes[i]

    p = tmp_path / "eds.tif"
    write_mosaic_streamed(p, layout, load, owner=owner, n_bands=3, dtype=np.uint16, tile=(32, 48),
                          metadata={"modality": "EDS"})
    got, info = read_tiff(p)
    np.testing.assert_array_equal(got, expected)
    assert sorted(calls) == sorted(layout.positions)  # every tile decoded exactly once
    assert info["metadata"]["modality"] == "EDS"


def test_peak_on_search_boundary_is_flagged():
    img = textured_image((260, 300), seed=13)
    a = img[20:120, 20:140]
    true = (7, 100 + 2)  # 7 rows beyond nominal (0, 100)
    b = img[20 + true[0]:120 + true[0], 20 + true[1]:140 + true[1]]
    inside = register_pair(a, b, (0, 100), max_shift=10)
    assert inside.shift == (7, 2) and not inside.at_boundary
    clamped = register_pair(a, b, (0, 100), max_shift=4)
    assert clamped.at_boundary


def test_boundary_pairs_are_not_used():
    truth, recs, images, true_pos = make_mosaic(nx=3, ny=2, jitter=5, seed=14, tile=(80, 100), step=(60, 80))
    layout = _solve(recs, images, max_shift=3)
    for p in layout.pairs:
        if p["at_boundary"]:
            assert not p["used"]
    assert any(p["at_boundary"] for p in layout.pairs)


def test_auto_max_shift_scales_with_tile_size():
    from bstitcher.stitch import auto_max_shift
    assert auto_max_shift([(88, 118)]) == 8
    assert auto_max_shift([(357, 478)]) == 24
    truth, recs, images, true_pos = make_mosaic(nx=3, ny=2, jitter=6, seed=15, tile=(200, 260), step=(170, 220))
    layout = _solve(recs, images)  # max_shift=None -> 13 px
    assert layout.max_shift == 13
    assert relative(layout.positions) == relative(true_pos)
