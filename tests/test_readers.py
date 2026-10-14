"""Reader registry (core, no plugin required)."""
import pytest

from bstitcher.readers import get_reader, register_reader


def test_unknown_extension_raises():
    with pytest.raises(LookupError, match="no reader"):
        get_reader("tile.xyz")


def test_manual_registration():
    def fake(path, **kw):
        return path
    register_reader("fake", fake)
    assert get_reader("a/b/Mapping_(1,1).fake") is fake
