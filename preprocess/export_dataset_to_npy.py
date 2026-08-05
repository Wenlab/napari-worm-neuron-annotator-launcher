"""Prepare Image, ROI, and optional Labels NPY files for the launcher.

Edit the configuration constants below, then run from the repository root::

    pixi run preprocess
"""

from pathlib import Path

from export_pipeline import ExportConfig, export_dataset
from raw_dataset_loader import RawDatasetConfig

# ---------------------------------------------------------------------------
# Frequently edited configuration
# ---------------------------------------------------------------------------

# Reference/red TIFF source. A directory should contain numbered frames such
# as 00000000.tif. A TIFF file is treated as a multi-page stack.
TIFF_PATH = Path(
    r"\\192.168.1.192\Ikrma-2\20260304\W3IMMOB_2026-03-05_01-27-19"
    r"\0_Camera-Red_VSC-10629"
)
DYNAMICS_PATH = Path(
    r"Z:\data5\CBMI_inferred_results\Ikrma\proxy"
    r"\20260304_w3_immobile\dynamics.h5"
)
OUTPUT_DIR = Path(r"/path/to/output/dataset")

# Source volume numbers, in the desired output order.
SELECTED_VOLUMES = [346, 361, 396]

# Optional compatibility output. Image + ROI workflows do not require Labels.
SAVE_MASK = False

# ---------------------------------------------------------------------------
# Acquisition and geometry configuration
# ---------------------------------------------------------------------------

# TIFF acquisition layout. Z_START_FRAME and Z_END_FRAME are inclusive offsets
# inside each complete volume.
FRAMES_PER_VOLUME = 20
Z_START_FRAME = 0
Z_END_FRAME = 17
REVERSE_Z_BY_VOLUME_PARITY = (False, False)

# dynamics.h5 reading.
DYNAMICS_FIRST_VOLUME = 0

# XY alignment and flips, applied identically to images and points.
ALIGN_XY = True
GOAL_ANGLE_DEGREES = -90.0
FLIP_X = False
FLIP_Y = False
IMAGE_INTERPOLATION_ORDER = 1

# Coordinate order of the first three columns in dynamics.h5. Output ROI files
# always store canonical x, y, z in columns 0, 1, 2.
COORDINATE_ORDER = "xyz"

# Z in dynamics.h5 is stored in XY-pixel units. The launcher divides z and
# depth by this same ratio to recover image-slice coordinates.
XY_PIXEL_SIZE = 0.3
Z_STEP_SIZE = 1.5
Z_SCALE_RATIO = Z_STEP_SIZE / XY_PIXEL_SIZE


def main() -> None:
    """Run preprocessing with the editable constants at the top of the file."""

    source = RawDatasetConfig(
        tiff_path=TIFF_PATH,
        dynamics_path=DYNAMICS_PATH,
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
    export_dataset(
        ExportConfig(
            source=source,
            output_dir=OUTPUT_DIR,
            save_mask=SAVE_MASK,
        )
    )


if __name__ == "__main__":
    main()
