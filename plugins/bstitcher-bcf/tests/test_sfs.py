"""Step 8a — SFS container writer (the file system inside a Bruker .bcf)."""
import struct

import numpy as np
import pytest
from rsciio.bruker._api import SFS_reader

from bstitcher_bcf.sfs import CHUNK_HEADER, SFSWriter


def _read(path, name):
    return SFS_reader(str(path)).get_file(name).get_as_BytesIO_string().getvalue()


def _chunks(path):
    raw = open(path, "rb").read()
    cs = struct.unpack("<I", raw[0x128:0x12C])[0]
    n = (len(raw) - 0x118) // cs
    return raw, cs, n


@pytest.mark.parametrize("chunksize", [4096, 65536])
def test_files_round_trip_through_rosettasciio(tmp_path, chunksize):
    rng = np.random.default_rng(1)
    small = b"<xml>tiny</xml>"
    big = rng.integers(0, 256, 5_000_000, dtype=np.uint8).tobytes()  # >1 pointer-table chunk at 4096
    p = tmp_path / "t.bcf"
    with SFSWriter(p, chunksize=chunksize) as w:
        w.add_file("EDSDatabase/HeaderData", small)
        with w.open_file("EDSDatabase/SpectrumData0") as f:  # streamed in pieces
            for i in range(0, len(big), 77777):
                f.write(big[i:i + 77777])
    assert _read(p, "EDSDatabase/HeaderData") == small
    assert _read(p, "EDSDatabase/SpectrumData0") == big
    sfs = SFS_reader(str(p))
    assert sfs.compression == "None" and sfs.chunksize == chunksize
    assert sfs.vfs["EDSDatabase"]["SpectrumData0"].size == len(big)


def test_rejects_unsupported_chunksize(tmp_path):
    with pytest.raises(ValueError):
        SFSWriter(tmp_path / "x.bcf", chunksize=512)


def test_chunk_checksums_and_free_bitmap(tmp_path):
    p = tmp_path / "t.bcf"
    with SFSWriter(p, chunksize=4096) as w:
        w.add_file("EDSDatabase/A", bytes(range(256)) * 9)
    raw, cs, n = _chunks(p)
    assert raw[:8] == b"AAMVHFSS" and raw[0x118:0x120] == b"AASFSSGN"
    assert struct.unpack("<III", raw[0x140:0x14C])[2] == n  # chunk count in the header
    for k in range(1, n):  # Bruker: header word 1 = XOR of the chunk's 32-bit data words
        o = 0x118 + k * cs
        words = np.frombuffer(raw[o + CHUNK_HEADER:o + cs], "<u4")
        assert struct.unpack("<I", raw[o + 4:o + 8])[0] == int(np.bitwise_xor.reduce(words)), k
    # chunk 1 = free bitmap: bit i <-> chunk i+1, 1 = free; every chunk of the file is in use
    o = 0x118 + cs + CHUNK_HEADER
    bits = np.unpackbits(np.frombuffer(raw[o:o + cs - CHUNK_HEADER], np.uint8), bitorder="little")
    assert not bits[:n - 1].any() and bits[n - 1:].all()
    assert [raw[0x118 + k * cs + 8] for k in (1, 2, 3, 4)] == [4, 2, 3, 1]  # chunk types


@pytest.mark.integration
def test_rewrites_a_real_bcf_identically(fixture_dir, tmp_path):
    """Repacking a real tile's internal files gives a file HyperSpy reads identically."""
    import hyperspy.api as hs
    from rsciio.bruker._api import BCF_reader
    src = fixture_dir / "Mapping_(1,1).bcf"
    b = BCF_reader(str(src))
    names = ["HeaderData", "SpectrumData0", "SpectrumPositions0"]
    p = tmp_path / "repacked.bcf"
    with SFSWriter(p) as w:
        for n in names:
            w.add_file(f"EDSDatabase/{n}", b.get_file(f"EDSDatabase/{n}").get_as_BytesIO_string().getvalue())
    for n in names:
        assert _read(p, f"EDSDatabase/{n}") == _read(src, f"EDSDatabase/{n}")
    a, c = hs.load(str(src)), hs.load(str(p))
    for sa, sc in zip(a, c):
        np.testing.assert_array_equal(sa.data, sc.data)


def test_tree_position_lists_a_late_file_first(tmp_path):
    p = tmp_path / "t.bcf"
    with SFSWriter(p, chunksize=4096) as w:
        w.add_file("EDSDatabase/SpectrumData0", b"data" * 3000)
        w.add_file("EDSDatabase/HeaderData", b"<xml/>" * 10, position=0)
    sfs = SFS_reader(str(p))
    assert list(sfs.vfs["EDSDatabase"]) == ["HeaderData", "SpectrumData0"]
    assert _read(p, "EDSDatabase/HeaderData") == b"<xml/>" * 10
    assert _read(p, "EDSDatabase/SpectrumData0") == b"data" * 3000
