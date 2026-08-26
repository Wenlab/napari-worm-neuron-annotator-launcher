"""Open exported or raw datasets with Worm Neuron Annotator."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import napari
import numpy as np
from napari_worm_neuron_annotator import NeuronAnnotatorWidget

APP_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = APP_DIR.parent
PREPROCESS_DIR = REPOSITORY_DIR / "preprocess"

SourceMode = Literal["npy", "raw-eager", "raw-virtual"]

# Choose "npy", "raw-eager", or "raw-virtual".
SOURCE_MODE: SourceMode = "npy"

# ---------------------------------------------------------------------------
# Prepared NPY source
# ---------------------------------------------------------------------------

DATA_DIR = Path(
    r"/path/to/data"
)
IMAGE_PATH = DATA_DIR / "volumes.npy"
ROI_PATH = DATA_DIR / "neuron_point_tuple.npy"

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


def _check_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def _load_npy_inputs() -> tuple[np.ndarray, Path]:
    _check_file(IMAGE_PATH, "Image NPY")
    _check_file(ROI_PATH, "ROI NPY")
    volumes = np.load(IMAGE_PATH, mmap_mode="r", allow_pickle=False)
    roi = np.load(ROI_PATH, mmap_mode="r", allow_pickle=False)
    _validate_image_and_roi(volumes, roi)
    return volumes, ROI_PATH


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


def _validate_image_and_roi(volumes, roi: np.ndarray) -> None:
    if volumes.ndim != 4:
        raise ValueError("Expected a (T,Z,Y,X) Image array")
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
    if SOURCE_MODE not in ("npy", "raw-eager", "raw-virtual"):
        raise ValueError(
            "SOURCE_MODE must be 'npy', 'raw-eager', or 'raw-virtual'"
        )

    temporary_dir: TemporaryDirectory | None = None
    widget = None
    viewer = None
    try:
        if SOURCE_MODE == "npy":
            volumes, roi_path = _load_npy_inputs()
        else:
            dataset = _load_raw_inputs(SOURCE_MODE)
            volumes = dataset.volumes
            temporary_dir = TemporaryDirectory(prefix="worm-neuron-roi-")
            roi_path = _write_session_roi(temporary_dir, dataset.roi)

        viewer = napari.Viewer()
        image_layer = viewer.add_image(
            volumes,
            name="volumes",
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
        widget.load_roi_path(roi_path)

        if widget.active_id is not None:
            widget.activate_id(widget.active_id, locate=True)

        viewer.window.show()
        napari.run()
    finally:
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
