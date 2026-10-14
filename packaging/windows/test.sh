#!/usr/bin/env bash
# Test the Windows build under Wine (after build.sh):
#   1. the library test suite on Windows Python      2. bstitcher.exe --help / run on a real job
#   3. GUI executables open and close (Xvfb)          4. screenshot of the GUI
#   5. the exe's output is identical to the Linux pipeline's output
#   packaging/windows/test.sh <bruker mapping job dir>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB="$(cd "${1:?usage: test.sh <bruker mapping job dir>}" && pwd)"
OUT="$ROOT/tmp/wintest"
mkdir -p "$OUT"
[[ -e "$OUT/win" || -e "$OUT/linux" ]] && mkdir -p "$ROOT/trash" && mv "$OUT" "$ROOT/trash/wintest-$(date +%s)" && mkdir -p "$OUT"

echo "== Linux reference run"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$JOB":/job:ro -v "$ROOT":/src bstitcher-dev \
  bstitcher run /job --out /src/tmp/wintest/linux --eds >/dev/null

docker run --rm --init -v "$JOB":/job:ro -v "$ROOT":/src -e BSTITCHER_FIXTURE='Z:\job' bstitcher-winbuild bash -c "
  set -e
  Xvfb :99 -nolisten tcp >/dev/null 2>&1 & export DISPLAY=:99; sleep 1
  mkdir -p /build && cd /src && cp -r src plugins tests pyproject.toml README.md LICENSE /build/
  wine python -m pip install --no-deps --no-cache-dir /build /build/plugins/bstitcher-bcf >/dev/null
  cd /build
  echo '== 1. test suite on Windows Python (Wine)'
  # --capture=sys: pytest's default fd capture breaks Tcl start-up under Wine (not on Windows)
  wine python -m pytest -q -p no:cacheprovider -W ignore --capture=sys tests plugins/bstitcher-bcf/tests 2>&1 | tail -3
  EXE='Z:\\src\\dist\\windows\\bstitcher'
  echo '== 2. bstitcher.exe'
  wine \"\$EXE\\bstitcher.exe\" --help | head -3
  t0=\$SECONDS; wine \"\$EXE\\bstitcher.exe\" run 'Z:\\job' --out 'Z:\\src\\tmp\\wintest\\win' --eds | tail -2
  echo \"run took \$((SECONDS - t0)) s (incl. Wine start-up)\"
  echo '== 3. GUI executables (--smoke-test)'
  wine \"\$EXE\\bstitcher-gui.exe\" --smoke-test && echo 'bstitcher-gui.exe ok'
  wine 'Z:\\src\\dist\\windows\\bstitcher-gui-onefile.exe' --smoke-test && echo 'bstitcher-gui-onefile.exe ok'
  echo '== 4. screenshot'
  wine \"\$EXE\\bstitcher-gui.exe\" & sleep 15; import -window root -display :99 /src/tmp/wintest/gui.png; kill %2 || true
  chown -R $(id -u):$(id -g) /src/tmp/wintest
"
echo "== 5. Windows exe output vs Linux output"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$ROOT":/src bstitcher-dev \
  python packaging/windows/compare_outputs.py tmp/wintest/linux tmp/wintest/win
