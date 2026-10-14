# bstitcher

Lossless, georeferenced GeoTIFF export and stitching of SEM mapping jobs
(BSE images, element maps and optionally full EDS spectrum cubes), e.g. Bruker
Esprit `Mapping_(c,r).bcf` tiles.

* **Lossless**: ZSTD + differencing predictor; values are never resampled or
  blended — every mosaic pixel is an original measured value.
* **Georeferenced**: physical stage coordinates (local CRS, metres) by default,
  or the legacy EPSG:4326 1-unit-per-pixel frame (`--georef pixel`).
* **Self-describing**: the acquisition metadata of every tile (full Bruker
  header + HyperSpy metadata) is embedded as JSON in the TIFF
  `ImageDescription`; key items also appear as GDAL metadata (QGIS / `gdalinfo`).
* **Scales**: the stitched EDS cube is streamed tile by tile, so a 60 GB cube
  needs neither 60 GB RAM nor scratch disk.

## Install

```bash
pip install bstitcher                 # core (Apache-2.0)
pip install bstitcher-bcf             # Bruker .bcf reader plugin (GPL-3.0-or-later)
```

From this repository, use the setup routine (Python ≥ 3.10):

```bash
git clone https://github.com/Artem7c2/bstitcher.git && cd bstitcher
./setup.sh                  # .venv with core + bcf plugin, then checks the install
./setup.sh --no-bcf         # Apache-2.0 core only (no GPL code)
./setup.sh --dev            # editable install + test deps, runs the test suite
./setup.sh --docker         # build the bstitcher-dev image and run the tests in it
source .venv/bin/activate
```

Other options: `--venv DIR`, `--python EXE`, `--dry-run` (print the commands only).
Manual equivalent: `pip install -e ".[test]" -e plugins/bstitcher-bcf`.

## Usage

```bash
bstitcher run  <session_dir> --out <out_dir> [--eds] [--georef local|pixel] [--elements Fe Mg ...]
bstitcher convert <session_dir> --out <tiles_dir>          # tiles only
bstitcher stitch  <tiles_dir>   --out <stitched_dir> [--modalities BSE elem_Fe EDS] [--seam center|overwrite]
bstitcher info    <file.tif>                                 # metadata + georef as JSON
```

```python
from bstitcher.pipeline import run_session
res = run_session("Mapping_job/", "out/", eds=True)
res["report"]["pairs"]          # per-overlap registration shift and score
```

### Output

```
out/tiles/Mapping_(c,r)_BSE.tif          (H, W)       uint16
out/tiles/Mapping_(c,r)_elem_Fe.tif      (H, W)       smallest lossless uint
out/tiles/Mapping_(c,r)_EDS.tif          (H, W, C)    counts, C energy channels  [--eds]
out/stitched/BSE.tif, elem_<El>.tif, EDS.tif          same canvas + georeference
out/stitched/tile_map.tif                which tile supplied each pixel (provenance)
out/stitched/placements.csv              nominal vs. registered tile positions
out/stitched/stitch_report.json          per-pair shift / NCC score / used flag
```

Element maps are the EDS counts summed over `2 × FWHM` windows around each
line (eXSpy `get_lines_intensity` defaults, bit-identical).  The window
actually used (`integration_window_keV`) and the energy axis (`energy`) are
stored in the metadata, so maps can be re-derived from the EDS cube.

## Stitching method

1. Nominal tile positions from the stage coordinates (Bruker: stage X
   decreases to the right, stage Y increases downward; the stage position is
   the tile centre).  With registration disabled (`--max-shift 0`) this
   reproduces Bruker's own mosaic pixel for pixel.
2. Every overlapping pair is registered by exhaustive normalised
   cross-correlation within `±max_shift` px (box-smoothed for matching only),
   which is robust on the narrow (≈10 %) overlaps of mapping jobs.
3. Integer positions from a least-squares solve over all pairs with
   `NCC ≥ min_score`, with a weak prior to the nominal positions (anchors
   tiles without a reliable overlap); inconsistent pairs (`> max_residual` px)
   are rejected iteratively.
4. Composition without resampling; in overlaps each pixel comes from the tile
   whose centre is nearest (`--seam center`) or from the last tile
   (`--seam overwrite`).

## Georeferencing

| `--georef` | CRS | geotransform |
|---|---|---|
| `local` (default) | LOCAL_CS, metres | world = −stage; origin = top-left of the canvas; pixel size from the microscope calibration |
| `pixel` | EPSG:4326 | `(0, 1, 0, H, 0, −1)` — legacy LS-Miner / Leaflet frame |

## Tests

```bash
docker run --rm -v "$PWD":/src bstitcher-dev pytest                       # synthetic + unit
docker run --rm -v "$PWD":/src -v /path/to/job:/fixture:ro \
           -e BSTITCHER_FIXTURE=/fixture bstitcher-dev pytest             # + real Bruker job
```

Synthetic tests cut jittered tiles from a known image and require exact
recovery of positions and pixel values; integration tests (marker
`integration`) run the full pipeline on a real 2×2 Bruker job and compare
with Bruker's own mosaic and with eXSpy.

## Licensing

| Package | License | Depends on |
|---|---|---|
| `bstitcher` (this directory, `src/`) | **Apache-2.0** | numpy, tifffile, imagecodecs (BSD) |
| `bstitcher-bcf` (`plugins/bstitcher-bcf/`) | **GPL-3.0-or-later** | HyperSpy, RosettaSciIO, eXSpy (GPL-3.0) |

The core never imports GPL code; readers are discovered through the
`bstitcher.readers` entry point.  The core can therefore be used, modified and
embedded in closed-source / commercial software.  Installing the BCF plugin
makes the *combination* subject to the GPL when it is distributed; to stay
permissive, provide an own reader (`bstitcher.readers.register_reader`).
Running the tools internally or as a service does not trigger GPL obligations.
