"""Step 8b — encoding a spectrum cube as a Bruker hypermap (SpectrumData0 + SpectrumPositions0)."""
import struct

import numpy as np
import pytest
from rsciio.bruker._api import SFS_reader, py_parse_hypermap
from rsciio.bruker import unbcf_fast

from bstitcher_bcf.hypermap import HypermapWriter
from bstitcher_bcf.sfs import SFSWriter


def _cube(h=5, w=7, c=60, seed=3):
    rng = np.random.default_rng(seed)
    cube = rng.poisson(0.08, (h, w, c)).astype(np.uint16)
    cube[1, 2] = 0          # empty pixel (outside every tile in a mosaic)
    cube[3, :, :] = 0       # empty line
    cube[0, 0, c - 1] = 3   # highest channel used
    return cube


def _write(tmp_path, cube):
    h, w, c = cube.shape
    p = tmp_path / "h.bcf"
    pos_tmp = tmp_path / "pos.tmp"
    with SFSWriter(p, chunksize=4096) as sfs:
        with sfs.open_file("EDSDatabase/SpectrumData0") as f:
            hw = HypermapWriter(f, h, w, positions=pos_tmp)
            for row in cube:
                hw.write_line(row)
            hw.close()
        sfs.add_file("EDSDatabase/SpectrumPositions0", pos_tmp.read_bytes())
    return SFS_reader(str(p))


@pytest.mark.parametrize("parser", ["python", "cython"])
@pytest.mark.parametrize("empty_pixels", [False, True])
def test_cube_round_trips_through_rosettasciio_parsers(tmp_path, parser, empty_pixels):
    if parser == "python" and empty_pixels:
        pytest.xfail("RosettaSciIO's pure-Python fallback cannot parse a pixel record with 0 counts "
                     "(bincount minlength -1); the compiled parser, used by default, can")
    cube = _cube()
    if not empty_pixels:
        cube[cube.sum(2) == 0, 0] = 1
    vf = _write(tmp_path, cube).get_file("EDSDatabase/SpectrumData0")
    shape = cube.shape
    if parser == "python":
        out = py_parse_hypermap(vf, shape, np.uint16)
    else:
        out = unbcf_fast.parse_to_numpy(vf, shape, np.uint16)
    np.testing.assert_array_equal(out, cube)


def test_pixel_records_and_positions(tmp_path):
    cube = _cube()
    h, w, c = cube.shape
    sfs = _write(tmp_path, cube)
    data = sfs.get_file("EDSDatabase/SpectrumData0").get_as_BytesIO_string().getvalue()
    pos = np.frombuffer(sfs.get_file("EDSDatabase/SpectrumPositions0").get_as_BytesIO_string().getvalue(), "<i8")
    assert struct.unpack("<ii", data[:8]) == (h, w)
    assert pos[0] == h | (w << 32) and len(pos) == h * w + 1
    for y in range(h):
        for x in range(w):
            o = int(pos[1 + y * w + x]) - 4  # Bruker points 4 bytes into the record
            px, c1, c2, d1, flag, ds1, npul, ds2 = struct.unpack("<IHHIHHHI", data[o:o + 22])
            spec = cube[y, x]
            assert px == x and flag == 0 and c2 == 0 and d1 == 786432
            assert npul == spec.sum() and ds2 == 2 * npul
            assert c1 == (np.flatnonzero(spec).max() + 1 if spec.any() else 0)


def test_too_many_counts_in_one_pixel_is_an_error(tmp_path):
    with SFSWriter(tmp_path / "x.bcf", chunksize=4096) as sfs, sfs.open_file("EDSDatabase/S") as f:
        hw = HypermapWriter(f, 1, 1, positions=tmp_path / "p.tmp")
        with pytest.raises(ValueError, match="65535"):
            hw.write_line(np.array([[70000]], np.uint32))


def test_sparse_and_dense_lines_give_identical_bytes(tmp_path):
    cube = _cube()
    h, w, c = cube.shape
    a = _write(tmp_path, cube).get_file("EDSDatabase/SpectrumData0").get_as_BytesIO_string().getvalue()
    p = tmp_path / "s.bcf"
    with SFSWriter(p, chunksize=4096) as sfs:
        with sfs.open_file("EDSDatabase/SpectrumData0") as f:
            hw = HypermapWriter(f, h, w, positions=tmp_path / "p2.tmp")
            for row in cube:
                x, ch = np.nonzero(row)
                hw.write_line_sparse(x, ch, row[x, ch])
            hw.close()
    assert SFS_reader(str(p)).get_file("EDSDatabase/SpectrumData0").get_as_BytesIO_string().getvalue() == a
