# SPDX-License-Identifier: GPL-3.0-or-later
"""Encode spectrum counts as a Bruker hypermap (``SpectrumData0``) plus its pixel index.

``SpectrumData0``: a 0x1A0-byte header (``int32`` height, width, channel count,
1, then reserved bytes), then per line ``int32`` pixels-in-line followed by one
record per pixel::

    uint32 x, uint16 last channel + 1, uint16 0, uint32 786432,
    uint16 flag = 0 (pulse list), uint16 0, uint16 n pulses, uint32 2 * n pulses,
    n x uint16 channel number of each X-ray count

``SpectrumPositions0``: ``int64`` height | width << 32, then for every pixel
(row-major) the offset of its record in ``SpectrumData0`` + 4.
"""
from __future__ import annotations

import struct

import numpy as np

HEADER_SIZE = 0x1A0
_D1 = 786432  # constant in every Bruker pixel record seen
_REC = np.dtype([("x", "<u4"), ("c1", "<u2"), ("c2", "<u2"), ("d1", "<u4"), ("flag", "<u2"),
                 ("ds1", "<u2"), ("n", "<u2"), ("ds2", "<u4")])
MAX_PULSES = 65535  # the pulse count field is 16 bit


class HypermapWriter:
    """Stream lines of a ``(height, width, channels)`` count cube into an open SFS file."""

    def __init__(self, fh, height: int, width: int, *, positions, channels: int = 4096,
                 header: bytes | None = None):
        self.fh, self.h, self.w = fh, int(height), int(width)
        head = bytearray(header[:HEADER_SIZE] if header else bytes(HEADER_SIZE))
        head[:16] = struct.pack("<iiii", self.h, self.w, channels, 1)
        fh.write(bytes(head))
        self.offset = HEADER_SIZE
        self.pos = open(positions, "wb")
        self.pos.write(struct.pack("<q", self.h | (self.w << 32)))
        self.lines = 0

    def write_line(self, counts) -> None:
        """One line as a dense ``(width, channels)`` count array."""
        counts = np.asarray(counts)
        if counts.shape[0] != self.w:
            raise ValueError(f"line has {counts.shape[0]} pixels, expected {self.w}")
        x, c = np.nonzero(counts)
        self.write_line_sparse(x, c, counts[x, c])

    def write_line_sparse(self, x, c, v) -> None:
        """One line from its nonzero counts: pixel ``x``, channel ``c``, count ``v``,
        sorted by ``x`` then ``c`` (the order of :func:`numpy.nonzero`)."""
        w = self.w
        x, c = np.asarray(x, np.int64), np.asarray(c, np.int64)
        v = np.asarray(v, np.int64)
        n = np.bincount(x, weights=v, minlength=w).astype(np.int64) if x.size else np.zeros(w, np.int64)
        if n.max(initial=0) > MAX_PULSES:
            raise ValueError(f"a pixel has {int(n.max())} counts; a Bruker pixel record holds at most {MAX_PULSES}")
        last = np.zeros(w, np.int64)
        if x.size:
            np.maximum.at(last, x, c + 1)
        rec = np.zeros(w, _REC)
        rec["x"] = np.arange(w)
        rec["c1"], rec["d1"], rec["n"], rec["ds2"] = last, _D1, n, 2 * n
        sizes = 22 + 2 * n
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])  # record starts within this line
        buf = np.zeros(int(sizes.sum()), np.uint8)
        buf[starts[:, None] + np.arange(22)] = rec.view(np.uint8).reshape(w, 22)
        if n.sum():
            ch = np.repeat(c.astype("<u2"), v)  # pixel by pixel, channels ascending
            first = np.concatenate([[0], np.cumsum(n)[:-1]])
            rank = np.arange(ch.size) - np.repeat(first, n)
            dest = np.repeat(starts + 22, n) + 2 * rank
            b = ch.view(np.uint8).reshape(-1, 2)
            buf[dest], buf[dest + 1] = b[:, 0], b[:, 1]
        self.fh.write(struct.pack("<i", w))
        line_start = self.offset + 4
        self.pos.write((line_start + starts + 4).astype("<i8").tobytes())
        self.fh.write(buf.tobytes())
        self.offset = line_start + buf.size
        self.lines += 1

    def close(self) -> None:
        if self.lines != self.h:
            raise ValueError(f"wrote {self.lines} lines, expected {self.h}")
        self.pos.close()
