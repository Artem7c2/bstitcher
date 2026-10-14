"""Synthetic mosaics with known ground truth for stitching tests."""
import numpy as np
from scipy import ndimage

from bstitcher.layout import TileRecord


def textured_image(shape, seed=0):
    rng = np.random.default_rng(seed)
    a = ndimage.gaussian_filter(rng.normal(size=shape), 2.0)
    a += 0.3 * ndimage.gaussian_filter(rng.normal(size=shape), 8.0)
    a = (a - a.min()) / (a.max() - a.min())
    return (a * 4000 + 50).astype(np.uint16)


def make_mosaic(nx=4, ny=3, tile=(60, 80), step=(50, 66), jitter=3, px=(2.0, 2.0),
                seed=0, missing=(), blank=()):
    """Cut jittered tiles out of a textured ground truth.

    Tile (c, r) nominally sits at row r*step_y, col c*step_x (1-based indices),
    stage coordinates are derived from the *nominal* grid (Bruker convention:
    stage X decreases to the right, stage Y increases downward), and the true
    position adds a random jitter that registration has to recover.
    Returns (truth_image, records, images, true_positions).
    """
    rng = np.random.default_rng(seed)
    h, w = tile
    sy, sx = step
    margin = jitter + 2
    H = margin * 2 + (ny - 1) * sy + h
    W = margin * 2 + (nx - 1) * sx + w
    truth = textured_image((H, W), seed)
    records, images, true_pos = [], {}, {}
    for r in range(1, ny + 1):
        for c in range(1, nx + 1):
            if (c, r) in missing:
                continue
            nom = (margin + (r - 1) * sy, margin + (c - 1) * sx)
            dj = rng.integers(-jitter, jitter + 1, size=2) if jitter else np.zeros(2, int)
            if (c, r) == (1, 1):
                dj[:] = 0
            pos = (int(nom[0] + dj[0]), int(nom[1] + dj[1]))
            img = truth[pos[0]:pos[0] + h, pos[1]:pos[1] + w].copy()
            if (c, r) in blank:
                img[:] = 1000
                truth[pos[0]:pos[0] + h, pos[1]:pos[1] + w] = 1000
            stage = (1000.0 - (nom[1] + w / 2) * px[0], 500.0 + (nom[0] + h / 2) * px[1])
            records.append(TileRecord(index=(c, r), shape=(h, w), stage_um=stage, pixel_size_um=px))
            images[(c, r)] = img
            true_pos[(c, r)] = pos
    # re-cut after blanking so neighbours see the blank area too
    for idx, pos in true_pos.items():
        images[idx] = truth[pos[0]:pos[0] + h, pos[1]:pos[1] + w].copy()
    return truth, records, images, true_pos


def relative(pos: dict) -> dict:
    """Positions relative to the top-left-most tile, for translation-free comparison."""
    r0 = min(p[0] for p in pos.values())
    c0 = min(p[1] for p in pos.values())
    return {k: (int(p[0] - r0), int(p[1] - c0)) for k, p in pos.items()}
