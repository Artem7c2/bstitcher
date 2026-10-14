# PyInstaller spec for the Windows build (run by packaging/windows/build.sh or the GitHub workflow).
#
# Produces, in --distpath:
#   bstitcher/bstitcher.exe        console CLI      } one folder, shared libraries
#   bstitcher/bstitcher-gui.exe    windowed GUI     }
#   bstitcher-gui-onefile.exe      self-contained single-file GUI (unpacks to %TEMP% on start)
#
# The bundle contains the GPL-3.0 bcf plugin, so the executables are distributed under GPL-3.0-or-later.
import os

from PyInstaller.utils.hooks import collect_all, copy_metadata

here = SPECPATH  # noqa: F821  (defined by PyInstaller)
root = os.path.abspath(os.path.join(here, "..", ".."))

datas, binaries, hidden = [], [], ["bstitcher_bcf.reader"]
# Packages that import parts of themselves only at runtime, which PyInstaller cannot see:
# HyperSpy / RosettaSciIO / eXSpy (readers, signal classes, YAML registries), dask, and
# imagecodecs (one compiled module per codec, e.g. ZSTD and the delta predictor).
for pkg in ("hyperspy", "exspy", "rsciio", "dask", "imagecodecs", "bstitcher", "bstitcher_bcf"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hidden += h
# importlib.metadata must work when frozen: entry points (readers, HyperSpy extensions) and
# version checks of dependencies (dask checks numpy's version) -> metadata of the whole tree.
datas += copy_metadata("bstitcher-bcf", recursive=True)
datas += [(os.path.join(root, "LICENSE"), "licenses/bstitcher-core"),
          (os.path.join(root, "plugins", "bstitcher-bcf", "LICENSE"), "licenses/bstitcher-bcf")]

excludes = ["IPython", "jupyter", "notebook", "PyQt5", "PySide2", "PyQt6", "PySide6", "pytest"]


def analysis(script):
    return Analysis(  # noqa: F821
        [os.path.join(here, script)], pathex=[], binaries=binaries, datas=datas, hiddenimports=hidden,
        hookspath=[], runtime_hooks=[], excludes=excludes, noarchive=False)


a_cli = analysis("entry_cli.py")
a_gui = analysis("entry_gui.py")
pyz_cli = PYZ(a_cli.pure)  # noqa: F821
pyz_gui = PYZ(a_gui.pure)  # noqa: F821

exe_cli = EXE(pyz_cli, a_cli.scripts, [], exclude_binaries=True, name="bstitcher", console=True,  # noqa: F821
              upx=False)
exe_gui = EXE(pyz_gui, a_gui.scripts, [], exclude_binaries=True, name="bstitcher-gui",  # noqa: F821
              console=False, upx=False)
COLLECT(exe_cli, exe_gui, a_cli.binaries, a_cli.datas, a_gui.binaries, a_gui.datas,  # noqa: F821
        name="bstitcher", upx=False)

EXE(pyz_gui, a_gui.scripts, a_gui.binaries, a_gui.datas, [], name="bstitcher-gui-onefile",  # noqa: F821
    console=False, upx=False, runtime_tmpdir=None)
