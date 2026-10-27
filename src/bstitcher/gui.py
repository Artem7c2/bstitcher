# SPDX-License-Identifier: Apache-2.0
"""Small desktop window for ``run_session``: pick a mapping-job folder, choose options, run.

Start with ``bstitcher-gui`` or ``bstitcher gui``.  The work runs in a
background thread; its log lines are passed to the window through a queue.
Tkinter is imported only when the window is opened, so the form logic and
:class:`Job` work (and are tested) without a display.
"""
from __future__ import annotations

import logging
import os
import queue
import re
import subprocess
import sys
import threading
import traceback
from pathlib import Path

from . import __version__
from .pipeline import run_session

GEOREFS = ("local", "pixel")
SEAMS = ("center", "overwrite")
_PROGRESS = re.compile(r"^\[(\d+)/(\d+)\]")


def default_out(session) -> Path:
    s = Path(session)
    return s.parent / f"{s.name}_bstitcher"


def form_to_job(values: dict):
    """Validate the form values; returns ``(session, out, run_session kwargs)``."""
    session = str(values.get("session", "")).strip()
    if not session:
        raise ValueError("Choose the session folder with the Mapping_(c,r) files.")
    out = str(values.get("out", "")).strip()
    ms = str(values.get("max_shift", "")).strip()
    if ms:
        try:
            max_shift = int(ms)
        except ValueError:
            max_shift = -1
        if max_shift < 0:
            raise ValueError(f"The max shift must be a whole number of pixels ≥ 0 (or empty for auto), not {ms!r}.")
    else:
        max_shift = None
    georef = values.get("georef", "local")
    seam = values.get("seam", "center")
    if georef not in GEOREFS or seam not in SEAMS:
        raise ValueError(f"Unknown georef {georef!r} or seam {seam!r}.")
    if values.get("bcf") and not values.get("eds"):
        raise ValueError("The stitched BCF needs the EDS spectrum cube: tick 'Export the full EDS spectrum cube'.")
    kwargs = {"eds": bool(values.get("eds")), "strict": bool(values.get("strict")),
              "georef": georef, "seam": seam, "max_shift": max_shift,
              "visualize": bool(values.get("visualize", True)), "bcf": bool(values.get("bcf"))}
    return Path(session), Path(out) if out else default_out(session), kwargs


def parse_progress(line: str):
    """``(done, total)`` from a ``[n/N] ...`` tile-conversion log line, else None."""
    m = _PROGRESS.match(line)
    return (int(m.group(1)), int(m.group(2))) if m else None


class _QueueHandler(logging.Handler):
    def __init__(self, q):
        super().__init__(logging.WARNING)
        self.q = q

    def emit(self, record):
        self.q.put(f"{record.levelname.lower()}: {record.getMessage()}")


class Job(threading.Thread):
    """Runs one session in the background; log lines are read with :meth:`drain`."""

    def __init__(self, session, out, kwargs):
        super().__init__(daemon=True)
        self.session, self.out, self.kwargs = Path(session), Path(out), dict(kwargs)
        self.q: queue.Queue = queue.Queue()
        self.result = None
        self.error: str | None = None

    def run(self):
        handler = _QueueHandler(self.q)
        logging.getLogger().addHandler(handler)
        try:
            self.result = run_session(self.session, self.out, log=self.q.put, **self.kwargs)
        except Exception as e:  # shown to the user, never raised in the thread
            if type(e) in (FileNotFoundError, ValueError, LookupError, RuntimeError):  # our own messages
                self.error = str(e)
            else:  # unexpected: the traceback goes to the log so it can be reported
                self.error = f"{type(e).__name__}: {e}"
                for line in traceback.format_exc().rstrip().splitlines():
                    self.q.put(line)
        finally:
            logging.getLogger().removeHandler(handler)

    def drain(self):
        while True:
            try:
                yield self.q.get_nowait()
            except queue.Empty:
                return


def open_folder(path) -> None:
    path = str(path)
    if os.name == "nt":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


class App:
    """The window.  ``vars`` holds the form fields, ``start()`` launches a :class:`Job`."""

    def __init__(self):
        import tkinter as tk
        from tkinter import filedialog, ttk
        from tkinter.scrolledtext import ScrolledText

        self.tk, self.filedialog = tk, filedialog
        self.root = root = tk.Tk()
        root.title(f"bstitcher {__version__}")
        root.minsize(620, 460)
        self.job: Job | None = None
        self.vars = {
            "session": tk.StringVar(), "out": tk.StringVar(), "eds": tk.BooleanVar(value=False),
            "strict": tk.BooleanVar(value=False), "georef": tk.StringVar(value="local"),
            "seam": tk.StringVar(value="center"), "max_shift": tk.StringVar(value=""),
            "visualize": tk.BooleanVar(value=True), "bcf": tk.BooleanVar(value=False),
        }
        self.status = tk.StringVar(value="Choose a mapping-job folder.")

        f = ttk.Frame(root, padding=12)
        f.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

        def folder_row(r, label, key, cmd):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(f, textvariable=self.vars[key]).grid(row=r, column=1, sticky="ew", padx=6)
            ttk.Button(f, text="Browse…", command=cmd).grid(row=r, column=2)

        folder_row(0, "Session folder", "session", self._pick_session)
        folder_row(1, "Output folder", "out", self._pick_out)
        ttk.Label(f, text="(empty = next to the session, <name>_bstitcher)", foreground="gray").grid(
            row=2, column=1, sticky="w", padx=6)

        opt = ttk.LabelFrame(f, text="Options", padding=8)
        opt.grid(row=3, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Checkbutton(opt, text="Export the full EDS spectrum cube (large)", variable=self.vars["eds"]).grid(
            row=0, column=0, columnspan=4, sticky="w")
        ttk.Checkbutton(opt, text="Stop on unreadable tiles (default: skip and report them)",
                        variable=self.vars["strict"]).grid(row=1, column=0, columnspan=4, sticky="w")
        ttk.Label(opt, text="Georeference").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(opt, textvariable=self.vars["georef"], values=GEOREFS, state="readonly", width=8).grid(
            row=2, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(opt, text="Seam").grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Combobox(opt, textvariable=self.vars["seam"], values=SEAMS, state="readonly", width=10).grid(
            row=2, column=3, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(opt, text="Max shift (px)").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(opt, textvariable=self.vars["max_shift"], width=8).grid(row=3, column=1, sticky="w", padx=6,
                                                                         pady=(6, 0))
        ttk.Label(opt, text="empty = auto (5 % of the tile size); 0 = stage positions only",
                  foreground="gray").grid(row=3, column=2, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(opt, text="PNG previews with colour bar and scale bar (visualization_stitched)",
                        variable=self.vars["visualize"]).grid(row=4, column=0, columnspan=4, sticky="w",
                                                              pady=(6, 0))
        ttk.Checkbutton(opt, text="Stitched Bruker .bcf of the whole mosaic (needs the EDS cube; large)",
                        variable=self.vars["bcf"]).grid(row=5, column=0, columnspan=4, sticky="w")

        bar = ttk.Frame(f)
        bar.grid(row=4, column=0, columnspan=3, sticky="ew")
        bar.columnconfigure(2, weight=1)
        self.run_btn = ttk.Button(bar, text="Run", command=self.start)
        self.run_btn.grid(row=0, column=0)
        self.open_btn = ttk.Button(bar, text="Open output folder", command=self._open_out, state="disabled")
        self.open_btn.grid(row=0, column=1, padx=6)
        self.progress = ttk.Progressbar(bar, mode="determinate")
        self.progress.grid(row=0, column=2, sticky="ew", padx=6)

        ttk.Label(f, textvariable=self.status).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 2))
        self.log = ScrolledText(f, height=12, state="disabled", font=("TkFixedFont", 9))
        self.log.grid(row=6, column=0, columnspan=3, sticky="nsew")
        f.rowconfigure(6, weight=1)

    # -- delegation so tests / callers can drive the window
    def update(self):
        self.root.update()

    def destroy(self):
        self.root.destroy()

    def mainloop(self):
        self.root.mainloop()

    # -- actions
    def _pick_session(self):
        d = self.filedialog.askdirectory(title="Mapping-job folder (Mapping_(c,r).bcf)")
        if d:
            self.vars["session"].set(d)

    def _pick_out(self):
        d = self.filedialog.askdirectory(title="Output folder")
        if d:
            self.vars["out"].set(d)

    def _open_out(self):
        if self.job is not None:
            open_folder(self.job.out)

    def _append(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        if self.job is not None and self.job.is_alive():
            return
        try:
            session, out, kwargs = form_to_job({k: v.get() for k, v in self.vars.items()})
        except ValueError as e:
            self.status.set(str(e))
            return
        self.job = Job(session, out, kwargs)
        self.run_btn.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        self.progress.configure(value=0, maximum=1)
        self.status.set(f"Running… output: {out}")
        self._append(f"== {session} -> {out}  {kwargs}")
        self.job.start()
        self.root.after(100, self._poll)

    def _poll(self):
        job = self.job
        for line in job.drain():
            self._append(line)
            p = parse_progress(line)
            if p:
                self.progress.configure(value=p[0], maximum=p[1])
                self.status.set(f"Converting tile {p[0]} of {p[1]}…")
            elif line.startswith("stitched "):
                self.status.set(f"Stitching… {line.split(':')[0][9:]}")
            elif line.startswith("preview "):
                self.status.set(f"Writing previews… {line.split(':')[0][8:]}")
        if job.is_alive():
            self.root.after(100, self._poll)
            return
        self.run_btn.configure(state="normal")
        if job.error:
            self.status.set(f"Failed: {job.error}")
            self._append(f"!! {job.error}")
        else:
            rep = job.result["report"]
            failed = len(rep.get("failed_tiles", []))
            self.status.set(f"Done: {rep['n_tiles']} tiles stitched"
                            + (f", {failed} unreadable tile(s) skipped" if failed else "") + f". Output: {job.out}")
            self.open_btn.configure(state="normal")


def check_readers(path) -> int:
    """Load the readers in a worker thread (as a job does); write ``{formats, errors}`` JSON to path.

    Diagnostic for packaged builds (``bstitcher-gui --check-readers out.json``); exit 0 if
    the .bcf reader is usable.
    """
    import json
    from . import readers
    res = {}

    def work():
        try:
            res.update(formats=readers.available_formats(), errors=readers.reader_errors())
        except Exception:
            res.update(formats=[], errors={"*": traceback.format_exc()})
    t = threading.Thread(target=work)
    t.start()
    t.join()
    Path(path).write_text(json.dumps(res, indent=1))
    return 0 if "bcf" in res["formats"] else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # A windowed Windows executable has no console: stdout/stderr are None.
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w"))
    logging.getLogger("hyperspy").setLevel(logging.ERROR)
    if "--check-readers" in argv:
        return check_readers(argv[argv.index("--check-readers") + 1])
    from .readers import available_formats, describe_errors
    try:  # import the readers (HyperSpy & co.) here in the main thread, not first inside a job
        formats, problem = available_formats(), describe_errors()
    except Exception as e:
        formats, problem = [], f"; {type(e).__name__}: {e}"
    app = App()
    if "bcf" not in formats:
        msg = "The .bcf reader is not available" + problem
        app.status.set(msg[:200])
        app._append("!! " + msg)
    if "--smoke-test" in argv:  # build the window, then close it (used by packaging tests)
        app.root.after(300, app.destroy)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
