"""Open exported NPY/TIFF or raw datasets with Worm Neuron Annotator."""

from __future__ import annotations

import gc
import sys
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Literal

import dask.array as da
import napari
import numpy as np
import tifffile
from napari_worm_neuron_annotator import NeuronAnnotatorWidget

APP_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = APP_DIR.parent
PREPROCESS_DIR = REPOSITORY_DIR / "preprocess"

SourceMode = Literal["npy", "tiff", "raw-eager", "raw-virtual"]

# Choose "npy", "tiff", "raw-eager", or "raw-virtual".
SOURCE_MODE: SourceMode = "tiff"

# ---------------------------------------------------------------------------
# Prepared NPY or compressed TIFF source
# ---------------------------------------------------------------------------

DATA_DIR = Path(
    r"/path/to/data"
)
IMAGE_PATH = DATA_DIR / "volumes.npy"
TIFF_STACK_PATH = DATA_DIR / "volumes.tif"
# Number of decompressed (Z,Y,X) volumes retained in RAM in TIFF mode.
TIFF_VOLUME_CACHE_SIZE = 3
# Worker threads used to decode the compressed pages of one volume.
TIFF_DECODE_WORKERS = 4
# Used by both prepared modes. Set to None to open without ROI data.
ROI_PATH: Path | None = DATA_DIR / "neuron_point_tuple.npy"
# ROI_PATH: Path | None = Path(r"data\20260304_w3\neuron_pt_tuple_corrected_wjh.npy")

# ---------------------------------------------------------------------------
# Raw TIFF + ROI source
# ---------------------------------------------------------------------------

TIFF_PATH = Path(
    r"\\192.168.1.192\Ikrma-2\20260304\W3IMMOB_2026-03-05_01-27-19"
    r"\0_Camera-Red_VSC-10629"
)
# Choose "dynamics" for one dynamics.h5 file or "realtime-results" for a
# directory containing volume_XXXXXXXX.h5 files.
RAW_ROI_SOURCE_MODE = "dynamics"
RAW_ROI_SOURCE_PATH = Path(
    r"Z:\data5\CBMI_inferred_results\Ikrma\proxy"
    r"\20260304_w3_immobile\dynamics.h5"
)
SELECTED_VOLUMES = [346, 361, 396]
FRAMES_PER_VOLUME = 20
Z_START_FRAME = 0
Z_END_FRAME = 17
REVERSE_Z_BY_VOLUME_PARITY = (False, False)
DYNAMICS_FIRST_VOLUME = 0
ALIGN_XY = True
GOAL_ANGLE_DEGREES = -90.0
FLIP_X = False
FLIP_Y = False
IMAGE_INTERPOLATION_ORDER = 1
COORDINATE_ORDER = "xyz"
XY_PIXEL_SIZE = 0.3
Z_STEP_SIZE = 1.5
Z_SCALE_RATIO = Z_STEP_SIZE / XY_PIXEL_SIZE

# ---------------------------------------------------------------------------
# Viewer configuration
# ---------------------------------------------------------------------------

# Set this to a compatible NPY Labels path for an optional overlay.
LABELS_PATH: Path | None = None
Z_DIVISOR = 5.0
LAYER_SCALE_TZYX = (1.0, 5.0, 1.0, 1.0)
IMAGE_CONTRAST_LIMITS = (102, 400)


_TIFF_VOLUME_CACHE: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
_TIFF_VOLUME_CACHE_LOCK = Lock()


def _check_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def _load_npy_inputs() -> tuple[np.ndarray, Path | None]:
    _check_file(IMAGE_PATH, "Image NPY")
    volumes = np.load(IMAGE_PATH, mmap_mode="r", allow_pickle=False)
    return volumes, _load_prepared_roi(volumes)


def _load_prepared_roi(volumes) -> Path | None:
    if ROI_PATH is None:
        _validate_image_and_roi(volumes, None)
        return None

    _check_file(ROI_PATH, "ROI NPY")
    roi = np.load(ROI_PATH, mmap_mode="r", allow_pickle=False)
    _validate_image_and_roi(volumes, roi)
    return ROI_PATH


def _clear_tiff_volume_cache() -> None:
    with _TIFF_VOLUME_CACHE_LOCK:
        _TIFF_VOLUME_CACHE.clear()


def _read_tiff_stack_volume(
    path: Path,
    time_index: int,
    z_dim: int,
    expected_shape: tuple[int, int],
    expected_dtype: np.dtype,
    file_signature: tuple[int, int],
    ifd_offsets: tuple[int, ...],
) -> np.ndarray:
    """Read and cache one compressed ``(Z,Y,X)`` TIFF volume."""

    cache_key = (
        str(path),
        time_index,
        z_dim,
        expected_shape,
        expected_dtype.str,
        file_signature,
    )
    with _TIFF_VOLUME_CACHE_LOCK:
        cached = _TIFF_VOLUME_CACHE.pop(cache_key, None)
        if cached is not None:
            _TIFF_VOLUME_CACHE[cache_key] = cached
            return cached

        first_page = time_index * z_dim
        last_page = first_page + z_dim
        if len(ifd_offsets) != z_dim:
            raise ValueError(
                f"TIFF volume {time_index} has {len(ifd_offsets)} IFD offsets; "
                f"expected {z_dim}"
            )
        expected_volume_shape = (z_dim, *expected_shape)
        volume = np.empty(expected_volume_shape, dtype=expected_dtype)
        with tifffile.TiffFile(path) as tiff:
            keyframe = tiff.pages[0]
            for z_index, (page_index, ifd_offset) in enumerate(
                zip(
                    range(first_page, last_page),
                    ifd_offsets,
                    strict=True,
                )
            ):
                frame = (
                    keyframe
                    if page_index == 0
                    else tifffile.TiffFrame(
                        tiff,
                        page_index,
                        offset=ifd_offset,
                        keyframe=keyframe,
                    )
                )
                frame.asarray(
                    out=volume[z_index],
                    maxworkers=TIFF_DECODE_WORKERS,
                )

        volume.setflags(write=False)
        _TIFF_VOLUME_CACHE[cache_key] = volume
        while len(_TIFF_VOLUME_CACHE) > TIFF_VOLUME_CACHE_SIZE:
            _TIFF_VOLUME_CACHE.popitem(last=False)
        return volume


class _TiffStackBackend:
    """Array-like TIFF backend used by Dask's efficient slice graph."""

    def __init__(
        self,
        path: Path,
        shape: tuple[int, int, int, int],
        dtype: np.dtype,
        file_signature: tuple[int, int],
        page_offsets: tuple[int, ...],
    ) -> None:
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.file_signature = file_signature
        self.page_offsets = page_offsets
        self.ndim = len(shape)
        self.size = int(np.prod(shape, dtype=np.int64))

    def __dask_tokenize__(self):
        return (
            type(self).__name__,
            str(self.path),
            self.shape,
            self.dtype.str,
            self.file_signature,
        )

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != self.ndim:
            raise IndexError("TIFF backend requires one index per TZYX axis")
        time_key, *volume_key = key
        if isinstance(time_key, int):
            time_index = time_key
            if time_index < 0:
                time_index += self.shape[0]
            if not 0 <= time_index < self.shape[0]:
                raise IndexError("TIFF time index is out of range")
            selected = self._volume(time_index)[tuple(volume_key)]
            return np.array(selected, copy=True, order="C")
        if not isinstance(time_key, slice):
            raise TypeError("TIFF backend supports integer and slice indexing")

        time_indices = range(*time_key.indices(self.shape[0]))
        selected = [
            np.array(
                self._volume(time_index)[tuple(volume_key)],
                copy=True,
                order="C",
            )
            for time_index in time_indices
        ]
        if len(selected) == 1:
            return selected[0][np.newaxis]
        if selected:
            return np.stack(selected, axis=0)

        empty = np.empty((0, *self.shape[1:]), dtype=self.dtype)
        return empty[(slice(None), *volume_key)]

    def _volume(self, time_index: int) -> np.ndarray:
        z_dim = self.shape[1]
        start = time_index * z_dim
        return _read_tiff_stack_volume(
            self.path,
            time_index,
            z_dim,
            self.shape[2:],
            self.dtype,
            self.file_signature,
            self.page_offsets[start : start + z_dim],
        )


def _load_tiff_inputs() -> tuple[da.Array, Path | None]:
    """Open a compressed TIFF with lazy per-volume decompression and caching."""

    _check_file(TIFF_STACK_PATH, "Image TIFF stack")
    if TIFF_VOLUME_CACHE_SIZE < 1:
        raise ValueError("TIFF_VOLUME_CACHE_SIZE must be at least 1")
    if TIFF_DECODE_WORKERS < 1:
        raise ValueError("TIFF_DECODE_WORKERS must be at least 1")
    _clear_tiff_volume_cache()
    file_stat = TIFF_STACK_PATH.stat()
    file_signature = (file_stat.st_size, file_stat.st_mtime_ns)
    with tifffile.TiffFile(TIFF_STACK_PATH) as tiff:
        series = tiff.series[0]
        image_shape = tuple(int(size) for size in series.shape)
        axes = series.axes
        image_dtype = np.dtype(series.dtype)
        page_count = len(series.pages)
        page_offsets = tuple(int(page.offset) for page in series.pages)

    if len(image_shape) != 4 or axes != "TZYX":
        raise ValueError(
            "Expected a TIFF series with axes TZYX and shape (T,Z,Y,X), "
            f"got axes={axes!r}, shape={image_shape}"
        )
    t_dim, z_dim, y_dim, x_dim = image_shape
    if page_count != t_dim * z_dim:
        raise ValueError(
            f"Expected {t_dim * z_dim} two-dimensional TIFF pages for "
            f"shape {image_shape}, found {page_count}"
        )

    backend = _TiffStackBackend(
        TIFF_STACK_PATH,
        image_shape,
        image_dtype,
        file_signature,
        page_offsets,
    )
    volumes = da.from_array(
        backend,
        chunks=(1, 1, y_dim, x_dim),
        asarray=False,
        fancy=False,
        lock=False,
    )
    volume_mib = z_dim * y_dim * x_dim * image_dtype.itemsize / 1024**2
    print(
        f"TIFF cache:      {TIFF_VOLUME_CACHE_SIZE} volumes  "
        f"(~{TIFF_VOLUME_CACHE_SIZE * volume_mib:.0f} MiB)"
    )
    print(f"TIFF decoding:   {TIFF_DECODE_WORKERS} workers per volume")
    _validate_image_and_roi(volumes, None)
    return volumes, _load_prepared_roi(volumes)


def _enable_raw_loader_imports() -> None:
    """Make the sibling preprocessing modules importable in raw modes."""

    if str(PREPROCESS_DIR) not in sys.path:
        sys.path.insert(0, str(PREPROCESS_DIR))


def _raw_config():
    _enable_raw_loader_imports()
    from raw_dataset_loader import RawDatasetConfig

    if not np.isclose(Z_DIVISOR, Z_SCALE_RATIO):
        raise ValueError(
            "Raw loading requires Z_DIVISOR to equal Z_SCALE_RATIO"
        )
    return RawDatasetConfig(
        tiff_path=TIFF_PATH,
        roi_source_mode=RAW_ROI_SOURCE_MODE,
        roi_source_path=RAW_ROI_SOURCE_PATH,
        selected_volumes=tuple(SELECTED_VOLUMES),
        frames_per_volume=FRAMES_PER_VOLUME,
        z_start_frame=Z_START_FRAME,
        z_end_frame=Z_END_FRAME,
        reverse_z_by_volume_parity=REVERSE_Z_BY_VOLUME_PARITY,
        dynamics_first_volume=DYNAMICS_FIRST_VOLUME,
        align_xy=ALIGN_XY,
        goal_angle_degrees=GOAL_ANGLE_DEGREES,
        flip_x=FLIP_X,
        flip_y=FLIP_Y,
        image_interpolation_order=IMAGE_INTERPOLATION_ORDER,
        coordinate_order=COORDINATE_ORDER,
        z_scale_ratio=Z_SCALE_RATIO,
    )


def _load_raw_inputs(mode: SourceMode):
    _enable_raw_loader_imports()
    from raw_dataset_loader import load_raw_dataset

    load_mode = "eager" if mode == "raw-eager" else "virtual"
    dataset = load_raw_dataset(_raw_config(), mode=load_mode)
    _validate_image_and_roi(dataset.volumes, dataset.roi)
    return dataset


def _validate_image_and_roi(volumes, roi: np.ndarray | None) -> None:
    if volumes.ndim != 4:
        raise ValueError("Expected a (T,Z,Y,X) Image array")
    if roi is None:
        return
    if roi.ndim != 3 or roi.shape[2] < 6:
        raise ValueError("Expected a (T,N,K>=6) ROI array")
    if roi.shape[0] != volumes.shape[0]:
        raise ValueError("ROI time dimension must match the Image time dimension")


def _write_session_roi(temporary_dir: TemporaryDirectory, roi: np.ndarray) -> Path:
    roi_path = Path(temporary_dir.name) / "neuron_point_tuple.npy"
    with roi_path.open("wb") as output_file:
        np.save(output_file, roi)
    return roi_path


def _add_optional_labels(viewer: napari.Viewer, volumes):
    if LABELS_PATH is None:
        return None
    _check_file(LABELS_PATH, "Labels NPY")
    labels = np.load(LABELS_PATH, mmap_mode="r", allow_pickle=False)
    if labels.shape != volumes.shape:
        raise ValueError("Expected Labels to match the Image shape")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Expected an integer Labels array")
    return viewer.add_labels(
        labels,
        name="neuron mask",
        scale=LAYER_SCALE_TZYX,
        axis_labels=("t", "z", "y", "x"),
    )


def main() -> None:
    if SOURCE_MODE not in ("npy", "tiff", "raw-eager", "raw-virtual"):
        raise ValueError(
            "SOURCE_MODE must be 'npy', 'tiff', 'raw-eager', or 'raw-virtual'"
        )

    temporary_dir: TemporaryDirectory | None = None
    widget = None
    viewer = None
    try:
        if SOURCE_MODE == "npy":
            volumes, roi_path = _load_npy_inputs()
        elif SOURCE_MODE == "tiff":
            volumes, roi_path = _load_tiff_inputs()
        else:
            dataset = _load_raw_inputs(SOURCE_MODE)
            volumes = dataset.volumes
            temporary_dir = TemporaryDirectory(prefix="worm-neuron-roi-")
            roi_path = _write_session_roi(temporary_dir, dataset.roi)

        viewer = napari.Viewer()
        image_layer = viewer.add_image(
            volumes,
            name="volumes",
            # The bounded TIFF volume cache is sufficient. Dask otherwise
            # retains a second, much larger cache of displayed slices.
            cache=SOURCE_MODE != "tiff",
            scale=LAYER_SCALE_TZYX,
            axis_labels=("t", "z", "y", "x"),
            contrast_limits=IMAGE_CONTRAST_LIMITS,
            colormap="gray",
            blending="additive",
        )
        labels_layer = _add_optional_labels(viewer, volumes)
        viewer.dims.current_step = (0, volumes.shape[1] // 2, 0, 0)

        widget = NeuronAnnotatorWidget(viewer)
        viewer.window.add_dock_widget(
            widget,
            name="Worm Neuron Annotator",
            area="right",
        )
        image_index = widget.image_combo.findData(image_layer)
        if image_index >= 0:
            widget.image_combo.setCurrentIndex(image_index)
        if labels_layer is not None:
            labels_index = widget.labels_combo.findData(labels_layer)
            if labels_index >= 0:
                widget.labels_combo.setCurrentIndex(labels_index)
        widget.z_divisor_spin.setValue(Z_DIVISOR)
        if roi_path is not None:
        widget.load_roi_path(roi_path)

        if widget.active_id is not None:
            widget.activate_id(widget.active_id, locate=True)

        viewer.window.show()
        napari.run()
    finally:
        if SOURCE_MODE == "tiff":
            _clear_tiff_volume_cache()
        if temporary_dir is not None:
            if widget is not None and widget.roi_dataset is not None:
                try:
                    widget.unload_roi()
                except RuntimeError:
                    pass
            widget = None
            viewer = None
            gc.collect()
            temporary_dir.cleanup()


if __name__ == "__main__":
    main()
