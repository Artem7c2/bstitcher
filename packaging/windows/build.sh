#!/usr/bin/env bash
# Build the Windows executables on Linux: Windows Python + PyInstaller under Wine (Docker).
#   packaging/windows/build.sh            -> dist/windows/bstitcher/{bstitcher,bstitcher-gui}.exe
#                                            dist/windows/bstitcher-gui-onefile.exe
#                                            dist/bstitcher-windows-<version>.zip
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE=bstitcher-winbuild
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)

docker build -f "$ROOT/packaging/windows/Dockerfile" -t "$IMAGE" "$ROOT"
docker run --rm -v "$ROOT":/src "$IMAGE" bash -c "
  set -e
  mkdir -p /build && cd /src && cp -r src plugins pyproject.toml README.md LICENSE /build/
  wine python -m pip install --no-deps --no-cache-dir /build /build/plugins/bstitcher-bcf >/dev/null
  wine pyinstaller --noconfirm --log-level WARN --distpath 'Z:\\src\\dist\\windows' \
       --workpath 'Z:\\build\\work' 'Z:\\src\\packaging\\windows\\bstitcher.spec'
  chown -R $(id -u):$(id -g) /src/dist
"
docker run --rm --user "$(id -u):$(id -g)" -v "$ROOT/dist":/dist -w /dist/windows bstitcher-dev \
  python -m zipfile -c "/dist/bstitcher-windows-$VERSION.zip" bstitcher
ls -la "$ROOT/dist" "$ROOT/dist/windows"
du -sh "$ROOT/dist/windows/bstitcher"
