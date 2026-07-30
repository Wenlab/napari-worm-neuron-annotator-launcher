"""Open datasets with Worm Neuron Annotator."""

from pathlib import Path

import napari
import numpy as np

from napari_worm_neuron_annotator import LabelManager

APP_DIR = Path(__file__).resolve().parent

# Edit this value when the NPY files are stored elsewhere.
DATA_DIR = Path(r"/path/to/dataset")

IMAGE_PATH = DATA_DIR / "volumes.npy"
LABELS_PATH = DATA_DIR / "neuron_mask.npy"
ROI_PATH = DATA_DIR / "neuron_point_tuple.npy"

Z_DIVISOR = 5.0
LAYER_SCALE_TZYX = (1.0, 5.0, 1.0, 1.0)
IMAGE_CONTRAST_LIMITS = (102, 400)


def _check_input_files() -> None:
    missing = [
        path for path in (IMAGE_PATH, LABELS_PATH, ROI_PATH) if not path.is_file()
    ]
    if not missing:
        return

    missing_text = "\n".join(f"- {path}" for path in missing)
    raise FileNotFoundError(
        "Dataset files were not found:\n"
        f"{missing_text}\n"
        "Update DATA_DIR near the top of this script to the directory "
        "containing the three NPY files."
    )


def main() -> None:
    _check_input_files()

    volumes = np.load(IMAGE_PATH, mmap_mode="r", allow_pickle=False)
    labels = np.load(LABELS_PATH, mmap_mode="r", allow_pickle=False)
    if volumes.shape != labels.shape or volumes.ndim != 4:
        raise ValueError(
            "Expected matching (t,z,y,x) volumes and Labels arrays"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Expected an integer Labels array")

    viewer = napari.Viewer()
    viewer.add_image(
        volumes,
        name="volumes",
        scale=LAYER_SCALE_TZYX,
        axis_labels=("t", "z", "y", "x"),
        contrast_limits=IMAGE_CONTRAST_LIMITS,
        colormap="gray",
        blending="additive",
    )
    viewer.add_labels(
        labels,
        name="neuron mask",
        scale=LAYER_SCALE_TZYX,
        axis_labels=("t", "z", "y", "x"),
    )
    viewer.dims.current_step = (0, labels.shape[1] // 2, 0, 0)

    widget = LabelManager(viewer)
    viewer.window.add_dock_widget(
        widget,
        name="Worm Neuron Annotator",
        area="right",
    )
    widget.z_divisor_spin.setValue(Z_DIVISOR)
    widget.load_roi_path(ROI_PATH)
    if widget.active_id is not None:
        widget.activate_id(widget.active_id, locate=True)

    viewer.window.show()
    napari.run()


if __name__ == "__main__":
    main()
