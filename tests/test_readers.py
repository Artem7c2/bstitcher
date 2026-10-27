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


class _EP:
    def __init__(self, name, err=None, func=None):
        self.name, self.err, self.func = name, err, func

    def load(self):
        if self.err:
            raise self.err
        return self.func


def _only_readers(eps):
    return lambda group: eps if group == "bstitcher.readers" else []


@pytest.fixture()
def fresh_registry(monkeypatch):
    from bstitcher import readers
    monkeypatch.setattr(readers, "_READERS", {})
    monkeypatch.setattr(readers, "_LOAD_ERRORS", {})
    monkeypatch.setattr(readers, "_MOSAIC_WRITERS", {})
    monkeypatch.setattr(readers, "_LOADED", False)
    return readers


def test_failing_plugin_is_reported_not_hidden(fresh_registry, monkeypatch):
    """A plugin that fails to import must name its error -- every time, not only the first."""
    r = fresh_registry
    monkeypatch.setattr(r, "entry_points", _only_readers([_EP("bcf", ImportError("DLL load failed: numpy"))]))
    for _ in range(2):
        with pytest.raises(LookupError, match=r"bcf.*ImportError: DLL load failed: numpy"):
            r.get_reader("Mapping_(1,1).bcf")
    assert r.available_formats() == []
    assert r.reader_errors() == {"bcf": "ImportError: DLL load failed: numpy"}


def test_listing_entry_points_failure_is_retried(fresh_registry, monkeypatch):
    r = fresh_registry
    calls = []

    def eps(group):
        if group != "bstitcher.readers":
            return []
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("metadata not ready")
        return [_EP("bcf", func=print)]
    monkeypatch.setattr(r, "entry_points", eps)
    with pytest.raises(RuntimeError):
        r.available_formats()
    assert r.available_formats() == ["bcf"]  # not stuck at "no readers"


def test_no_reader_session_error_names_the_cause(fresh_registry, monkeypatch, tmp_path):
    from bstitcher.pipeline import convert_tiles
    r = fresh_registry
    monkeypatch.setattr(r, "entry_points", _only_readers([_EP("bcf", ImportError("no module hyperspy"))]))
    (tmp_path / "Mapping_(1,1).bcf").write_bytes(b"x")
    with pytest.raises(LookupError, match=r"no tile reader.*bcf.*no module hyperspy"):
        convert_tiles(tmp_path, tmp_path / "out")
