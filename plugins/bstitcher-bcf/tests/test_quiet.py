"""HyperSpy's harmless "Numba is not installed" warning is not shown to users."""
import logging

import pytest


def _record(name, msg):
    return logging.LogRecord(name, logging.WARNING, __file__, 1, msg, None, None)


def test_numba_warning_is_dropped_other_warnings_pass():
    import bstitcher_bcf.reader  # noqa: F401  (installs the filter)
    lg = logging.getLogger("hyperspy.decorators")
    assert not lg.filter(_record(lg.name, "Numba is not installed, falling back to non-accelerated implementation."))
    assert lg.filter(_record(lg.name, "something that matters"))


@pytest.mark.integration
def test_reading_a_tile_logs_no_numba_warning(fixture_dir, caplog):
    from bstitcher_bcf.reader import read_bcf
    with caplog.at_level(logging.WARNING):
        read_bcf(fixture_dir / "Mapping_(1,1).bcf")
    assert not [r for r in caplog.records if "Numba" in r.getMessage()]
