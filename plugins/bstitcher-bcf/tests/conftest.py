import os
from pathlib import Path

import pytest

FIXTURE_ENV = "BSTITCHER_FIXTURE"


@pytest.fixture(scope="session")
def fixture_dir():
    """Directory with a real Bruker mapping job (Mapping_(c,r).bcf files).

    Set BSTITCHER_FIXTURE; tests using it are skipped otherwise.
    """
    p = os.environ.get(FIXTURE_ENV)
    if not p or not Path(p).is_dir():
        pytest.skip(f"{FIXTURE_ENV} not set / not a directory")
    return Path(p)
