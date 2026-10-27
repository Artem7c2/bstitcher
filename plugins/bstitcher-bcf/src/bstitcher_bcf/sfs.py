# SPDX-License-Identifier: GPL-3.0-or-later
"""Writer for AidAim SFS containers, the single-file file system inside Bruker ``.bcf``.

Layout, as written by Bruker Esprit and parsed by RosettaSciIO's ``SFS_reader``:

* bytes ``0 .. 0x118``: signature ``AAMVHFSS``; then chunk ``k`` occupies
  ``0x118 + k * chunksize`` .. ``+ chunksize``.
* every chunk starts with a 32-byte header: ``int32 next chunk`` (-1 = none),
  ``uint32 XOR of all 32-bit words of the chunk's data``, ``uint8 type``
  (1 file tree, 2/3/4 allocation maps, 5 pointer table, 6 data) + 3 flag bytes.
* chunk 0: container header (version, chunk size, tree chunk, item and chunk count).
* chunk 1: free-chunk bitmap (bit ``i`` <-> chunk ``i + 1``, 1 = free);
  chunk 2: one bit per byte of chunk 1 (1 = that byte is all free); chunk 3: reserved.
* chunk 4: the file tree, 512-byte items (pointer-table chunk, size, Windows
  FILETIMEs, attributes, parent index, is-dir flag, name).
* a file = chained pointer-table chunks (uint32 data-chunk numbers) + data chunks.

Files are written uncompressed and streamed: data chunks first, the pointer
table after them, the tree/allocation chunks and header at the end.
"""
from __future__ import annotations

import os
import struct
import time

import numpy as np

CHUNK_HEADER = 32
PREFIX = 0x118  # bytes before chunk 0
ITEM_SIZE = 0x200
_TYPE = {"tree": b"\x01\x00\x00\x01", "free2": b"\x02\x00\x00\x18", "free3": b"\x03\x00\x00\x18",
         "free1": b"\x04\x00\x00\x18", "pointers": b"\x05\x00\x00\x01", "data": b"\x06\x00\x00\x00"}
_FIRST_DATA_CHUNK = 5  # 0 header, 1-3 allocation maps, 4 file tree
_VERSION = 2.8
_ATTR_DIR = 0x10


def _filetime(t=None) -> int:
    return int(((time.time() if t is None else t) + 11644473600) * 10_000_000)


def _xor32(data: bytes) -> int:
    return int(np.bitwise_xor.reduce(np.frombuffer(data, "<u4"))) if data else 0


class _File:
    """Write handle returned by :meth:`SFSWriter.open_file`."""

    def __init__(self, sfs, name):
        self.sfs, self.name = sfs, name
        self.size, self.chunks, self._buf = 0, [], bytearray()

    def write(self, data) -> int:
        data = memoryview(data).cast("B")
        self._buf += data
        self.size += len(data)
        u = self.sfs.usable
        n_full = len(self._buf) // u
        for i in range(n_full):
            self.chunks.append(self.sfs._append_chunk("data", bytes(self._buf[i * u:(i + 1) * u])))
        del self._buf[:n_full * u]
        return len(data)

    def close(self):
        if self._buf:
            self.chunks.append(self.sfs._append_chunk("data", bytes(self._buf)))
            self._buf.clear()
        self.sfs._finish_file(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()


class SFSWriter:
    """Create an SFS container; ``add_file``/``open_file`` with paths like ``"Dir/name"``."""

    def __init__(self, path, *, chunksize: int = 65536, mtime=None):
        if chunksize % 4 or chunksize < 4096:
            raise ValueError("chunksize must be a multiple of 4 and >= 4096")
        self.path, self.chunksize = str(path), chunksize
        self.usable = chunksize - CHUNK_HEADER
        self.time = _filetime(mtime)
        self._fh = open(self.path, "wb")
        self._n_chunks = _FIRST_DATA_CHUNK
        self._items = []  # dicts: name, parent, is_dir, ptr, size
        self._dirs = {}
        self._open = None

    # -- chunk level
    def _write_chunk(self, k: int, kind: str, data: bytes, next_chunk: int = -1, word0=None):
        if len(data) > self.usable:
            raise ValueError("chunk data too long")
        data = data + bytes(self.usable - len(data))
        head = struct.pack("<iI", next_chunk if word0 is None else word0, _xor32(data)) + _TYPE[kind]
        self._fh.seek(PREFIX + k * self.chunksize)
        self._fh.write(head + bytes(CHUNK_HEADER - len(head)) + data)

    def _append_chunk(self, kind: str, data: bytes, next_chunk: int = -1) -> int:
        k = self._n_chunks
        self._n_chunks += 1
        self._write_chunk(k, kind, data, next_chunk)
        return k

    # -- files
    def _parent(self, path: str) -> int:
        parent = -1
        parts = path.split("/")[:-1]
        for i in range(len(parts)):
            key = "/".join(parts[:i + 1])
            if key not in self._dirs:
                self._items.append({"name": parts[i], "parent": parent, "is_dir": True, "ptr": -1, "size": 0})
                self._dirs[key] = len(self._items) - 1
            parent = self._dirs[key]
        return parent

    def open_file(self, path: str, *, position: int | None = None) -> _File:
        """Stream a file.  ``position``: its place in the file tree (default: next),
        so a header computed last can still be listed first, as in Bruker files."""
        if self._open is not None:
            raise RuntimeError(f"close {self._open.name!r} first")
        self._open = _File(self, path)
        self._open.position = position
        return self._open

    def add_file(self, path: str, data: bytes, *, position: int | None = None) -> None:
        with self.open_file(path, position=position) as f:
            f.write(data)

    def _finish_file(self, f: _File):
        per = self.usable // 4
        tables = [f.chunks[i:i + per] for i in range(0, max(len(f.chunks), 1), per)] or [[]]
        first = self._n_chunks
        for i, t in enumerate(tables):
            nxt = first + i + 1 if i + 1 < len(tables) else -1
            self._append_chunk("pointers", np.asarray(t, "<u4").tobytes(), nxt)
        parent = self._parent(f.name)
        self._items.append({"name": f.name.split("/")[-1], "parent": parent, "is_dir": False,
                            "ptr": first, "size": f.size, "position": f.position})
        self._open = None

    def _ordered_items(self) -> list:
        """Directories first (in creation order), then files by requested position."""
        dirs = [it for it in self._items if it["is_dir"]]
        files = [it for it in self._items if not it["is_dir"]]
        placed = sorted((it for it in files if it.get("position") is not None), key=lambda it: it["position"])
        rest = [it for it in files if it.get("position") is None]
        for it in placed:
            rest.insert(min(it["position"], len(rest)), it)
        order = dirs + rest
        new_index = {id(it): i for i, it in enumerate(order)}
        old = {i: it for i, it in enumerate(self._items)}
        return [{**it, "parent": -1 if it["parent"] < 0 else new_index[id(old[it["parent"]])]} for it in order]

    # -- finishing
    def close(self):
        if self._fh.closed:
            return
        if self._open is not None:
            self._open.close()
        n = self._n_chunks
        if len(self._items) * ITEM_SIZE > self.usable:
            raise ValueError("too many files for a single-chunk file tree")
        if n - 1 > self.usable * 8:
            raise ValueError("container too large for one allocation bitmap; use a larger chunksize")
        tree = b"".join(
            struct.pack("<iQQQQIi176s?3s256s32s", it["ptr"], it["size"], self.time, self.time, self.time,
                        _ATTR_DIR if it["is_dir"] else 0, it["parent"], b"", it["is_dir"], b"",
                        it["name"].encode("utf-8"), b"")
            for it in self._ordered_items())
        self._write_chunk(4, "tree", tree)
        free1 = np.ones(self.usable * 8, np.uint8)
        free1[:n - 1] = 0  # chunks 1 .. n-1 are in use
        b1 = np.packbits(free1, bitorder="little")
        free2 = (b1 == 0xFF).astype(np.uint8)
        b2 = np.packbits(np.concatenate([free2, np.ones(self.usable * 8 - len(free2), np.uint8)]),
                         bitorder="little")[:self.usable]
        self._write_chunk(1, "free1", b1.tobytes(), word0=int(np.argmax(b1 != 0)))
        self._write_chunk(2, "free2", b2.tobytes(), word0=int(np.argmax(b2 != 0)))
        self._write_chunk(3, "free3", b"", word0=0)
        head = (b"AASFSSGN" + struct.pack("<ffII", 2.78, _VERSION, self.chunksize, 8)
                + struct.pack("<4I", 1, 1, 1, 1) + struct.pack("<III", 4, len(self._items), n))
        self._fh.seek(0)
        self._fh.write(b"AAMVHFSS" + b"\xfe\xff\xff\xff\xff\xff\xff\x7f" + bytes(PREFIX - 16) + head
                       + bytes(self.chunksize - len(head)))
        self._fh.seek(0, os.SEEK_END)
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()
        else:
            self._fh.close()
