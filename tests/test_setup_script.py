"""setup.sh — install routine (checked via --dry-run; real installs are exercised in CI / docker)."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.sh"

pytestmark = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None,
                                reason="setup.sh is the Unix installer (needs bash)")


def run(*args):
    return subprocess.run(["bash", str(SETUP), *args], capture_output=True, text=True, cwd=ROOT)


def test_syntax():
    assert subprocess.run(["bash", "-n", str(SETUP)]).returncode == 0


def test_help():
    r = run("--help")
    assert r.returncode == 0
    for opt in ("--venv", "--no-bcf", "--dev", "--docker", "--dry-run"):
        assert opt in r.stdout


def test_unknown_option_fails():
    r = run("--bogus")
    assert r.returncode != 0 and "--bogus" in r.stderr


def test_default_installs_core_and_plugin_into_venv(tmp_path):
    r = run("--dry-run", "--venv", str(tmp_path / "env"))
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert f"-m venv {tmp_path / 'env'}" in out
    pip = [l for l in out.splitlines() if "pip install" in l and "plugins/bstitcher-bcf" in l]
    assert pip, out
    assert " -e " not in pip[0]  # regular install unless --dev
    assert "GPL" in out  # licence notice when the plugin is installed
    assert "bstitcher --help" in out  # post-install check


def test_no_bcf_is_core_only(tmp_path):
    out = run("--dry-run", "--no-bcf", "--venv", str(tmp_path / "env")).stdout
    assert "plugins/bstitcher-bcf" not in out
    assert "GPL" not in out
    assert any("pip install" in l and str(ROOT) in l for l in out.splitlines())


def test_dev_is_editable_with_tests(tmp_path):
    out = run("--dry-run", "--dev", "--venv", str(tmp_path / "env")).stdout
    pip = [l for l in out.splitlines() if "pip install" in l and "bstitcher-bcf" in l][0]
    assert pip.count(" -e ") == 2 and "[test]" in pip
    assert "pytest" in out


def test_docker_builds_image():
    out = run("--dry-run", "--docker").stdout
    assert "docker build" in out and "docker/Dockerfile" in out and "bstitcher-dev" in out
    assert "-m venv" not in out
