# Windows build image: Windows CPython 3.11 under Wine (tobix/pywine) + bstitcher dependencies.
# Build from the repo root:  docker build -f packaging/windows/Dockerfile -t bstitcher-winbuild .
FROM tobix/pywine:3.11
RUN apt-get update && apt-get install -y --no-install-recommends x11-apps && rm -rf /var/lib/apt/lists/*
COPY packaging/windows/requirements-win.txt /tmp/requirements-win.txt
RUN wine python -m pip install --no-cache-dir -r /tmp/requirements-win.txt
# screenshots of the GUI under Xvfb (packaging/windows/test.sh)
RUN apt-get update && apt-get install -y --no-install-recommends imagemagick && rm -rf /var/lib/apt/lists/*
