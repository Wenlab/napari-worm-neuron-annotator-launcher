# Repository guidance

## Purpose

This repository prepares volumetric worm-neuron datasets and opens them in
napari with `napari-worm-neuron-annotator`. Keep the launcher small and keep
dataset conversion in the standalone preprocessing script.

## Commands

- Install the environment with `pixi install`.
- Convert a dataset with `pixi run preprocess`.
- Open a prepared or raw dataset with `pixi run start`.
- Check Python syntax with
  `pixi run python -m py_compile script/launch_datasets.py preprocess/export_dataset_to_npy.py preprocess/raw_dataset_loader.py preprocess/tiff_source.py preprocess/roi_source.py preprocess/geometry.py preprocess/export_pipeline.py`.

## Repository layout

- `script/launch_datasets.py` loads prepared NPY files and opens napari.
- `preprocess/export_dataset_to_npy.py` holds editable preprocessing
  settings and starts the export.
- `preprocess/tiff_source.py`, `preprocess/roi_source.py`, and
  `preprocess/geometry.py` handle source reading and shared transforms;
  `preprocess/raw_dataset_loader.py` assembles eager or virtual raw inputs and
  `preprocess/export_pipeline.py` writes prepared launcher inputs.
- `pixi.toml` defines the environment and the `start` and `preprocess` tasks.
- `README.md` documents dataset preparation and interactive use.

## Dataset contract

NPY mode requires this file in `DATA_DIR`:

- `volumes.npy`: a `(T, Z, Y, X)` Image array.

`neuron_point_tuple.npy` is an optional `(T, N, K)` ROI array with `K >= 6`.
`ROI_PATH = None` disables ROI loading. `neuron_mask.npy` is optional;
`LABELS_PATH = None` disables Labels loading, while a `Path` loads an overlay.

The preprocessing script stores ROI Z coordinates in scaled units. Keep
`Z_SCALE_RATIO` in the preprocessing script equal to `Z_DIVISOR` in the
launcher. The repository does not use a metadata JSON file or a preprocessing
marker file to synchronize these values.

Do not reintroduce uint8 export, dataset metadata, or preprocessing marker
files unless the user requests them.

## Configuration

Keep frequently edited preprocessing settings at the top of the preprocessing
script. This group includes `TIFF_PATH`, `DYNAMICS_PATH`, `OUTPUT_DIR`,
`SELECTED_VOLUMES`, and `SAVE_MASK`.

Keep launcher settings near the top of the launcher. Users edit `DATA_DIR`,
raw source paths, `SOURCE_MODE`, `LABELS_PATH`, `Z_DIVISOR`, layer scale, and
contrast limits there.
Do not add command-line configuration unless the user asks for it.

## Development notes

- Load NPY inputs with `allow_pickle=False` and read-only memory mapping.
- Preserve the `(T, Z, Y, X)` Image layout and `(T, N, K)` ROI layout.
- Apply the same XY alignment and flip operations to images and ROI points.
- Keep virtual TIFF chunks in `(1, 1, Y, X)` units and open TIFF handles inside
  individual Dask tasks rather than capturing live handles in the graph.
- Keep the raw-mode ROI temporary NPY alive for the complete napari session;
  the plugin memory-maps it and Windows cannot delete an open mapping.
- Keep optional Labels separate from the authoritative ROI data.
- Do not commit generated datasets or machine-specific caches.
- Preserve existing user changes in a dirty worktree.

This repository does not maintain a full automated test suite. Do not recreate
the `tests/` directory unless the user requests tests. Use syntax checks and a
small manual dataset run when verification is needed.
