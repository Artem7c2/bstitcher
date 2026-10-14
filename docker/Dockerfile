# Dev / test image for bstitcher (+ GPL bcf plugin).
# Build from the repo root:  docker build -f docker/Dockerfile -t bstitcher-dev .
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 libtk8.6 xvfb xauth && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        "numpy==1.26.4" "scipy==1.13.1" "tifffile==2024.9.20" "imagecodecs==2024.9.22" \
        "hyperspy==2.2.0" "rosettasciio==0.13.0" "exspy==0.2.1" \
        "pytest==8.3.3" "rasterio==1.4.3"

WORKDIR /src
COPY . /src
RUN pip install --no-cache-dir --no-deps -e /src -e /src/plugins/bstitcher-bcf

ENV MPLCONFIGDIR=/tmp/mpl
CMD ["pytest", "-q"]
