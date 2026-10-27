"""Step 7 — small desktop GUI: form handling and background job (headless), window (needs a display)."""
import os
import time
from pathlib import Path

import pytest

from bstitcher import gui
import test_pipeline_synthetic as tps  # registers the .npz test reader

session = tps.session


def form(**kw):
    v = {"session": "/data/job1", "out": "", "eds": False, "strict": False, "georef": "local",
         "seam": "center", "max_shift": ""}
    v.update(kw)
    return v


def test_form_defaults():
    s, out, kw = gui.form_to_job(form())
    assert s == Path("/data/job1")
    assert out == Path("/data/job1_bstitcher")  # default output next to the session
    assert kw == {"eds": False, "strict": False, "georef": "local", "seam": "center", "max_shift": None,
                  "visualize": True}


def test_form_options():
    s, out, kw = gui.form_to_job(form(out="/tmp/o", eds=True, strict=True, georef="pixel", seam="overwrite",
                                      max_shift=" 12 ", visualize=False))
    assert out == Path("/tmp/o")
    assert kw == {"eds": True, "strict": True, "georef": "pixel", "seam": "overwrite", "max_shift": 12,
                  "visualize": False}


@pytest.mark.parametrize("bad,msg", [({"session": ""}, "session folder"),
                                     ({"max_shift": "abc"}, "max shift"),
                                     ({"max_shift": "-1"}, "max shift")])
def test_form_errors(bad, msg):
    with pytest.raises(ValueError, match=msg):
        gui.form_to_job(form(**bad))


def test_parse_progress():
    assert gui.parse_progress("[3/77] Mapping_(1,3).bcf -> 11 files") == (3, 77)
    assert gui.parse_progress("[2/4] Mapping_(2,1).bcf SKIPPED (unreadable: x)") == (2, 4)
    assert gui.parse_progress("stitched BSE: (100, 200) -> x") is None


def test_job_runs_session_and_collects_log(session, tmp_path):
    d, *_ = session
    job = gui.Job(d, tmp_path / "out", {"eds": True})
    job.run()  # synchronous
    assert job.error is None
    assert job.result["report"]["n_tiles"] == 6
    lines = list(job.drain())
    assert any(l.startswith("[6/6]") for l in lines)
    assert any("stitched EDS" in l for l in lines)
    assert (tmp_path / "out" / "stitched" / "EDS.tif").exists()


def test_job_reports_errors(tmp_path):
    job = gui.Job(tmp_path, tmp_path / "out", {})
    job.start()
    job.join(10)
    assert not job.is_alive()
    assert "no Mapping_" in job.error and job.result is None


# ------------------------------------------------------------ real window
def _tk_ok():
    if not os.environ.get("DISPLAY") and os.name != "nt":
        return False
    try:
        import tkinter
        tkinter.Tk().destroy()
        return True
    except Exception:
        return False


needs_display = pytest.mark.skipif(not _tk_ok(), reason="needs tkinter and a display (e.g. xvfb-run)")


@needs_display
def test_smoke_test_flag():
    assert gui.main(["--smoke-test"]) == 0


@needs_display
def test_window_runs_a_job(session, tmp_path):
    d, *_ = session
    app = gui.App()
    try:
        app.vars["session"].set(str(d))
        app.vars["out"].set(str(tmp_path / "o"))
        app.vars["eds"].set(True)
        app.start()
        t0 = time.time()
        while app.job is not None and app.job.is_alive() and time.time() - t0 < 60:
            app.update()
            time.sleep(0.02)
        for _ in range(20):
            app.update()
            time.sleep(0.02)
        assert "Done" in app.status.get()
        assert "[6/6]" in app.log.get("1.0", "end")
        assert (tmp_path / "o" / "stitched" / "EDS.tif").exists()
        assert str(app.run_btn["state"]) == "normal"
    finally:
        app.destroy()


@needs_display
def test_window_shows_form_error():
    app = gui.App()
    try:
        app.start()  # empty session field
        assert "session folder" in app.status.get()
        assert app.job is None
    finally:
        app.destroy()


def test_check_readers_in_worker_thread(tmp_path):
    """``--check-readers FILE`` loads the readers in a background thread, like a job does."""
    import json
    from bstitcher.readers import available_formats
    out = tmp_path / "check.json"
    rc = gui.main(["--check-readers", str(out)])
    res = json.loads(out.read_text())
    assert res["formats"] == available_formats()
    assert rc == (0 if "bcf" in res["formats"] else 1)
    assert "errors" in res


def test_job_reports_traceback_for_unexpected_errors(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise KeyError("deep inside")
    monkeypatch.setattr(gui, "run_session", boom)
    job = gui.Job(tmp_path, tmp_path / "o", {})
    job.start()
    job.join()
    assert job.error == "KeyError: 'deep inside'"
    lines = "\n".join(job.drain())
    assert "Traceback" in lines and "boom" in lines


def test_cli_has_gui_command():
    from bstitcher.cli import build_parser
    assert build_parser().parse_args(["gui"]).cmd == "gui"
