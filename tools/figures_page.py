"""Build a review page (index.html) from the output of tools/test_figures.py.

    python tools/figures_page.py tmp/test_figures
"""
import html
import json
import sys
from pathlib import Path

STEPS = {
    1: ("TIFF writing, georeferencing, metadata", "tests/test_tiff_io.py, tests/test_metadata.py"),
    2: ("Tile model and per-tile export", "tests/test_tile_export.py, tests/test_readers.py"),
    3: ("Bruker .bcf reader (real data)", "plugins/bstitcher-bcf/tests/test_bcf_reader.py"),
    4: ("Registration, placement, composition", "tests/test_stitching.py"),
    5: ("Full pipeline", "tests/test_pipeline_synthetic.py, plugins/bstitcher-bcf/tests/test_pipeline_bcf.py"),
}

CSS = """
/* Layout: one reading column of step sections; each figure is a captioned plate that can expand to full width. */
:root {
  --paper: #f3f5f4; --surface: #ffffff; --ink: #1b2326; --muted: #59666b; --rule: #d5dcdc;
  --accent: #0d6f73; --pass: #2e7a3a; --pass-bg: #e3f1e4; --fail: #b02a1f; --fail-bg: #f7e2df;
  --plate: #ffffff;
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --f-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --paper: #12181a; --surface: #1a2225; --ink: #e3e9ea; --muted: #9aa8ac; --rule: #2c373a;
    --accent: #5cc3c4; --pass: #7fd08b; --pass-bg: #1d3322; --fail: #f08a7e; --fail-bg: #3a1f1c;
    --plate: #f4f6f5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --paper: #12181a; --surface: #1a2225; --ink: #e3e9ea; --muted: #9aa8ac; --rule: #2c373a;
  --accent: #5cc3c4; --pass: #7fd08b; --pass-bg: #1d3322; --fail: #f08a7e; --fail-bg: #3a1f1c;
  --plate: #f4f6f5;
}
* { box-sizing: border-box; }
body {
  background: var(--paper); color: var(--ink); font: 15px/1.55 var(--f-body);
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 32px 64px;
}
.wrap { max-width: 1180px; margin: 0 auto; display: grid; gap: 40px; }
header { display: grid; gap: 14px; }
.eyebrow { font: 600 12px/1 var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
h1 { font: 600 34px/1.15 var(--f-display); margin: 0; text-wrap: balance; }
.lede { max-width: 68ch; margin: 0; color: var(--muted); }
.summary { display: flex; flex-wrap: wrap; gap: 10px 22px; font: 13px/1.4 var(--f-mono); color: var(--muted);
  font-variant-numeric: tabular-nums; }
.summary b { color: var(--ink); font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 13px; color: var(--muted); }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px;
  margin-right: 6px; vertical-align: -1px; background: var(--c); }
nav { display: flex; flex-wrap: wrap; gap: 6px; }
nav a { font: 500 13px/1 var(--f-body); color: var(--ink); text-decoration: none; padding: 8px 12px;
  border: 1px solid var(--rule); border-radius: 999px; background: var(--surface); }
nav a:hover, nav a:focus-visible { border-color: var(--accent); color: var(--accent); outline: none; }
section.step { display: grid; gap: 22px; scroll-margin-top: 16px; }
.step-head { display: grid; gap: 4px; border-top: 2px solid var(--ink); padding-top: 12px; }
.step-head h2 { font: 600 22px/1.2 var(--f-display); margin: 0; text-wrap: balance; }
.step-head .files { font: 12px/1.4 var(--f-mono); color: var(--muted); overflow-wrap: anywhere; }
.step-head .num { font: 600 12px/1 var(--f-mono); letter-spacing: .08em; color: var(--accent); }
figure { margin: 0; display: grid; gap: 10px; background: var(--surface); border: 1px solid var(--rule);
  border-radius: 6px; padding: 16px; }
.fig-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 12px; }
.fig-head h3 { font: 600 17px/1.3 var(--f-body); margin: 0; flex: 1 1 280px; text-wrap: balance; }
.chip { font: 600 11px/1 var(--f-mono); letter-spacing: .06em; padding: 5px 8px; border-radius: 4px; }
.chip.pass { color: var(--pass); background: var(--pass-bg); }
.chip.fail { color: var(--fail); background: var(--fail-bg); }
.tests { font: 12px/1.5 var(--f-mono); color: var(--muted); overflow-wrap: anywhere; margin: 0; }
figcaption { max-width: 75ch; margin: 0; }
.plate { background: var(--plate); border-radius: 4px; overflow: auto; padding: 6px; }
.plate img { display: block; width: 100%; height: auto; max-width: 100%; cursor: zoom-in; }
.plate.zoom img { width: auto; max-width: none; cursor: zoom-out; }
.hint { font-size: 12px; color: var(--muted); margin: 0; }
footer { font-size: 13px; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 14px; }
code { font: 13px var(--f-mono); }
@media (max-width: 520px) { h1 { font-size: 27px; } figure { padding: 12px; } }
"""

JS = """
document.querySelectorAll('.plate').forEach(p => {
  const t = () => p.classList.toggle('zoom');
  p.querySelector('img').addEventListener('click', t);
  p.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); t(); } });
});
"""


def build(out: Path) -> Path:
    figs = json.loads((out / "figures.json").read_text())
    n, n_pass = len(figs), sum(f["passed"] for f in figs)
    e = html.escape
    parts = [
        '<title>bstitcher Test Atlas</title>',
        '<meta name="claude:viewport-fit-sides" content="cover">',
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600'
        '&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
        f"<style>{CSS}</style>",
        '<div class="wrap"><header>',
        '<div class="eyebrow">bstitcher · test review</div>',
        "<h1>What each test feeds in, expects and gets</h1>",
        '<p class="lede">Every figure re-runs a test scenario with the same data, seeds and parameters as the test '
        "suite. Panels read left to right: <b>input</b>, <b>expected</b> output, <b>actual</b> output, and the "
        "<b>difference</b> (actual − expected; white = 0). The verdict uses the same criterion as the test.</p>",
        f'<div class="summary"><span><b>{n_pass}/{n}</b> figures pass</span><span>5 pipeline steps</span>'
        "<span>synthetic data + real Bruker job KL-1-1 (2×2 tiles, 88×118 px, 4096 ch)</span></div>",
        '<div class="legend"><span style="--c:#00e5ff">cyan = expected</span>'
        '<span style="--c:#32cd32">green = actual</span><span style="--c:#ffa500">orange = stage-nominal</span></div>',
        "<nav>" + "".join(f'<a href="#step{k}">{k} · {e(v[0])}</a>' for k, v in STEPS.items()) + "</nav>",
        "</header>",
    ]
    for k, (name, files) in STEPS.items():
        sf = [f for f in figs if f["step"] == k]
        if not sf:
            continue
        parts.append(f'<section class="step" id="step{k}"><div class="step-head"><span class="num">STEP {k}</span>'
                     f'<h2>{e(name)}</h2><span class="files">{e(files)}</span></div>')
        for f in sf:
            status = "pass" if f["passed"] else "fail"
            parts.append(
                f'<figure><div class="fig-head"><h3>{e(f["title"])}</h3>'
                f'<span class="chip {status}">{status.upper()}</span></div>'
                f'<p class="tests">{e(f["test"])}</p>'
                f'<figcaption>{e(f["caption"])}</figcaption>'
                f'<div class="plate" tabindex="0" aria-label="Toggle full size">'
                f'<img src="{e(f["file"])}" alt="{e(f["title"])}" loading="lazy"></div></figure>')
        parts.append("</section>")
    parts += [
        '<p class="hint">Click a figure to see it at full resolution; click again to fit.</p>',
        "<footer>Generated by <code>tools/test_figures.py</code> (figures) and <code>tools/figures_page.py</code> "
        "(this page) in the <code>bstitcher-dev</code> Docker image. Setup-script tests (tests/test_setup_script.py) "
        "and CLI tests (tests/test_cli.py) check text output and have no figure.</footer>",
        f"</div><script>{JS}</script>",
    ]
    p = out / "index.html"
    p.write_text("\n".join(parts), encoding="utf-8")
    return p


if __name__ == "__main__":
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "tmp/test_figures")))
