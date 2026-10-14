# SPDX-License-Identifier: Apache-2.0
"""Pairwise tile registration by exhaustive normalised cross-correlation.

Bruker mapping jobs often overlap by only ~10 % (a 9-12 px strip on small
tiles), where FFT phase correlation is unreliable.  Instead every integer shift
within ``±max_shift`` around the stage-nominal offset is scored by the
normalised cross-correlation (NCC) of the actual overlap region.  Shifts that
shrink the overlap below ``min_overlap_frac`` of the nominal overlap are not
considered, which avoids spurious maxima on tiny overlaps.  An optimum on the
edge of the searched region is only trusted if it is a local maximum, checked
by scoring its outside neighbours; otherwise it is flagged ``at_boundary``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout import overlap_box


@dataclass
class PairResult:
    shift: tuple        # (drow, dcol) correction to the nominal offset
    score: float        # NCC at the optimum (-1..1); low = unreliable
    n_pixels: int       # overlap size used at the optimum
    at_boundary: bool = False  # optimum touches the edge of the searched region -> unreliable


def _ncc(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else 0.0


def box_smooth(a: np.ndarray, radius: int) -> np.ndarray:
    """Mean filter of size (2r+1)^2 with edge padding (noise suppression for matching only)."""
    if radius <= 0:
        return a
    k = 2 * radius + 1
    c = np.pad(a, radius, mode="edge").cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def register_pair(img_a, img_b, offset, *, max_shift: int = 8, min_overlap_frac: float = 0.5,
                  smooth: int = 1) -> PairResult:
    """Best integer correction of ``offset`` = position(b) - position(a).

    ``smooth`` is the radius of a box filter applied to both images before
    matching; it raises the NCC of noisy (short dwell time) tiles without
    affecting the output, which is always composed from the raw tiles.
    """
    a = np.asarray(img_a, dtype=np.float64)
    b = np.asarray(img_b, dtype=np.float64)
    if a.ndim == 3:
        a = a.sum(axis=2)
    if b.ndim == 3:
        b = b.sum(axis=2)
    a, b = box_smooth(a, smooth), box_smooth(b, smooth)
    (ha, wa), (hb, wb) = a.shape, b.shape
    o_r, o_c = int(round(offset[0])), int(round(offset[1]))
    oh, ow = overlap_box((0, 0), (ha, wa), (o_r, o_c), (hb, wb))
    nominal_area = max(oh, 0) * max(ow, 0)
    def score_at(dr, dc, min_frac):
        r_off, c_off = o_r + dr, o_c + dc
        r0, r1 = max(0, r_off), min(ha, r_off + hb)
        c0, c1 = max(0, c_off), min(wa, c_off + wb)
        n = max(r1 - r0, 0) * max(c1 - c0, 0)
        if n < 4 or n < min_frac * nominal_area:
            return None
        return _ncc(a[r0:r1, c0:c1], b[r0 - r_off:r1 - r_off, c0 - c_off:c1 - c_off]), n

    scores = {}
    for dr in range(-max_shift, max_shift + 1):
        for dc in range(-max_shift, max_shift + 1):
            v = score_at(dr, dc, min_overlap_frac)
            if v is not None:
                scores[(dr, dc)] = v
    if not scores:
        return PairResult((0, 0), 0.0, 0, True)
    # ties -> smallest correction
    (dr, dc), (s, n) = max(scores.items(), key=lambda kv: (kv[1][0], -(abs(kv[0][0]) + abs(kv[0][1]))))
    # The optimum must be a local maximum: neighbours outside the searched region
    # are evaluated once (relaxed overlap minimum); if one of them is at least
    # as good, or cannot be evaluated, the true peak may lie outside.
    at_boundary = False
    for nb in ((dr - 1, dc), (dr + 1, dc), (dr, dc - 1), (dr, dc + 1)):
        if nb in scores:
            continue
        v = score_at(*nb, min_overlap_frac / 2)
        if v is None or v[0] >= s:
            at_boundary = True
            break
    best = PairResult((dr, dc), s, n, at_boundary)
    return best
