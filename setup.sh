#!/usr/bin/env bash
# Install routine for bstitcher (+ optional GPL bcf reader plugin).
#
#   ./setup.sh                    venv in ./.venv, core + bcf plugin
#   ./setup.sh --no-bcf           core only (Apache-2.0, no GPL code)
#   ./setup.sh --dev              editable install + test deps, runs the test suite
#   ./setup.sh --docker           build the bstitcher-dev image and run the tests in it
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN="$ROOT/plugins/bstitcher-bcf"
VENV="$ROOT/.venv"
PYTHON="${PYTHON:-python3}"
IMAGE="bstitcher-dev"
BCF=1 DEV=0 DOCKER=0 DRY=0

usage() {
    cat <<EOF
Usage: ./setup.sh [options]

  --venv DIR     virtualenv to create / reuse (default: $ROOT/.venv)
  --python EXE   interpreter used to create the venv, >= 3.10 (default: \$PYTHON or python3)
  --no-bcf       install only the Apache-2.0 core, without the GPL Bruker .bcf plugin
  --dev          editable install with test dependencies, then run the test suite
  --docker       build the Docker image '$IMAGE' instead of a venv and run the tests in it
  --dry-run      print the commands without executing them
  -h, --help     show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)    VENV="${2:?--venv needs a directory}"; shift ;;
        --python)  PYTHON="${2:?--python needs an executable}"; shift ;;
        --no-bcf)  BCF=0 ;;
        --dev)     DEV=1 ;;
        --docker)  DOCKER=1 ;;
        --dry-run) DRY=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "setup.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

run() {
    echo "+ $*"
    [[ $DRY -eq 1 ]] || "$@"
}

if [[ $DOCKER -eq 1 ]]; then
    run docker build -f "$ROOT/docker/Dockerfile" -t "$IMAGE" "$ROOT"
    run docker run --rm "$IMAGE" pytest -q
    echo "Image '$IMAGE' ready, e.g.: docker run --rm -v /data:/data $IMAGE bstitcher run <session> --out <out>"
    exit 0
fi

if [[ $DRY -eq 0 ]]; then
    "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
        || { echo "setup.sh: $PYTHON is older than Python 3.10" >&2; exit 1; }
fi

[[ -x "$VENV/bin/python" ]] || run "$PYTHON" -m venv "$VENV"
PY="$VENV/bin/python"
run "$PY" -m pip install --upgrade pip

core="$ROOT"
[[ $DEV -eq 1 ]] && core="$ROOT[test]"
pkgs=()
if [[ $DEV -eq 1 ]]; then
    pkgs+=(-e "$core")
    [[ $BCF -eq 1 ]] && pkgs+=(-e "$PLUGIN")
else
    pkgs+=("$core")
    [[ $BCF -eq 1 ]] && pkgs+=("$PLUGIN")
fi
run "$PY" -m pip install "${pkgs[@]}"

run "$VENV/bin/bstitcher" --help
run "$PY" -c 'from bstitcher.readers import available_formats; print("readers:", available_formats())'
[[ $DEV -eq 1 ]] && (cd "$ROOT" && run "$PY" -m pytest -q)

if [[ $BCF -eq 1 ]]; then
    echo "Note: bstitcher-bcf is GPL-3.0-or-later (HyperSpy/RosettaSciIO/eXSpy); distributing it together"
    echo "      with bstitcher makes the combination GPL. Use --no-bcf for the Apache-2.0 core only."
fi
echo "Done. Activate with: source $VENV/bin/activate"
