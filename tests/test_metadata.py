"""Step 2 — metadata helpers and tile naming."""
import datetime as dt
import json

import numpy as np
import pytest

from bstitcher.metadata import jsonable, parse_tile_index


def test_jsonable_converts_numpy_and_dates():
    obj = {"a": np.float32(1.25), "b": np.arange(2, dtype=np.int64), "c": dt.date(2024, 10, 15),
           "d": (1, 2), "e": b"raw", "f": {np.str_("k"): np.bool_(True)}}
    out = jsonable(obj)
    json.dumps(out)  # must not raise
    assert out == {"a": 1.25, "b": [0, 1], "c": "2024-10-15", "d": [1, 2], "e": "raw",
                   "f": {"k": True}}


def test_jsonable_handles_nan():
    assert jsonable({"x": float("nan")}) == {"x": None}


@pytest.mark.parametrize("name,expected", [
    ("Mapping_(1,1).bcf", (1, 1)),
    ("Mapping_(12,3)_BSE.tif", (12, 3)),
    ("/a/b/Mapping_(2, 19).bcf", (2, 19)),
])
def test_parse_tile_index(name, expected):
    assert parse_tile_index(name) == expected


def test_parse_tile_index_rejects_other_names():
    with pytest.raises(ValueError):
        parse_tile_index("Mosaic map_0.tif")
