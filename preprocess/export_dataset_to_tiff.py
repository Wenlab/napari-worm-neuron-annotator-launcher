"""Export a compressed foreground TIFF without changing the NPY exporter.

Edit the configuration constants below, then run from the repository root::

    pixi run python preprocess/export_dataset_to_tiff.py

The output keeps the complete ``(T, Z, Y, X)`` canvas and original float32
intensities. Voxels outside the selected foreground mask are written as zero.
The original TIFF and any existing NPY files are never modified.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import tifffile

from foreground_source import (
    ForegroundConfig,
    VolumeRecord,
    iter_volume_records,
    mask_for_record,
    validate_configuration,
)

# ---------------------------------------------------------------------------
# Frequently edited configuration
# ---------------------------------------------------------------------------

# This standalone test configuration is independent of export_dataset_to_npy.py.
TIFF_PATH = Path(
    r"\\192.168.1.192\Ikrma-2\20260304\W3_2026-03-05_01-03-23"
    r"\0_Camera-Red_VSC-10629"
)
ROI_SOURCE_MODE: Literal["dynamics", "realtime-results"] | None = "dynamics"
ROI_SOURCE_PATH: Path | None = Path(
    r"Z:\data5\CBMI_inferred_results\Ikrma\proxy"
    r"\20260304_w3\dynamics.h5"
)
OUTPUT_DIR = Path(
    r"H:\Process_temporary\WJH\napari-worm-neuron-annotator-usage"
    r"\data\20260304_w3"
)

# Use "auto" to select the projected ROI mask when ROI_SOURCE_PATH is set and
# the MIP-based method otherwise. Set this explicitly to "roi" or "image" when
# comparing both methods.
FOREGROUND_MODE: Literal["auto", "roi", "image"] = "image"

SELECTED_VOLUMES = list(range(3315))
FRAMES_PER_VOLUME = 20
Z_START_FRAME = 0
Z_END_FRAME = 17
REVERSE_Z_BY_VOLUME_PARITY = (False, False)

# Geometry. ROI mode applies the same transform to images and points as the
# existing raw loader. Image-only mode requires ALIGN_XY = False.
DYNAMICS_FIRST_VOLUME = 0
ALIGN_XY = False
GOAL_ANGLE_DEGREES = -90.0
FLIP_X = False
FLIP_Y = False
IMAGE_INTERPOLATION_ORDER = 1
COORDINATE_ORDER = "xyz"
XY_PIXEL_SIZE = 0.3
Z_STEP_SIZE = 1.5
Z_SCALE_RATIO = Z_STEP_SIZE / XY_PIXEL_SIZE

# Each ROI produces one expanded XY box. All valid boxes in the volume are
# projected into one 2-D union before dilation and closing.
ROI_MARGIN_XY_PIXELS = 6
ROI_MARGIN_Z_LAYERS = 1

# Image-only foreground settings. Background and connected components are
# estimated from the per-volume maximum-intensity projection (MIP).
BACKGROUND_SIGMA_XY = 20.0
THRESHOLD_SIGMA = 4.0
MIN_CONTRAST = 2.0
# Keep small but spatially coherent MIP candidates while rejecting isolated
# bright pixels.
MIN_COMPONENT_PIXELS = 4

# Safety dilation applied before the shared large-scale closing. ROI boxes use
# a larger margin; image-only MIP candidates remain more conservative.
ROI_DILATION_XY_PIXELS = 48
IMAGE_DILATION_XY_PIXELS = 48

# Apply one large 2-D closing to connect nearby neuronal regions. The result is
# broadcast along Z so every layer in one volume uses the same XY mask.
CLOSING_XY_PIXELS = 32

TIFF_COMPRESSION_LEVEL = 3


def _foreground_config() -> ForegroundConfig:
    return ForegroundConfig(
        tiff_path=TIFF_PATH,
        roi_source_mode=ROI_SOURCE_MODE,
        roi_source_path=ROI_SOURCE_PATH,
        foreground_mode=FOREGROUND_MODE,
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
        roi_margin_xy_pixels=ROI_MARGIN_XY_PIXELS,
        roi_margin_z_layers=ROI_MARGIN_Z_LAYERS,
        background_sigma_xy=BACKGROUND_SIGMA_XY,
        threshold_sigma=THRESHOLD_SIGMA,
        min_contrast=MIN_CONTRAST,
        min_component_pixels=MIN_COMPONENT_PIXELS,
        roi_dilation_xy_pixels=ROI_DILATION_XY_PIXELS,
        image_dilation_xy_pixels=IMAGE_DILATION_XY_PIXELS,
        closing_xy_pixels=CLOSING_XY_PIXELS,
    )


def _iter_masked_planes(
    records: Iterator[VolumeRecord],
    config: ForegroundConfig,
    roi_frames: list[np.ndarray] | None,
) -> Iterator[np.ndarray]:
    for record in records:
        if roi_frames is not None:
            if record.roi_points is None:
                raise ValueError("ROI data is missing from a source volume")
            roi_frames.append(record.roi_points)
        elif record.roi_points is not None:
            raise ValueError("Unexpected ROI data in an image-only export")
        mask = mask_for_record(record, config)
        voxel_count = int(np.count_nonzero(mask))
        print(
            f"    [{record.local_t}] foreground={voxel_count:,}/"
            f"{record.volume.size:,} ({voxel_count / record.volume.size:.1%})"
        )
        masked = np.where(mask, record.volume, 0.0).astype(
            np.float32,
            copy=False,
        )
        for plane in masked:
            yield np.ascontiguousarray(plane)


def _write_tiff_atomic(
    records: Iterator[VolumeRecord],
    config: ForegroundConfig,
    output_path: Path,
) -> tuple[tuple[int, int, int, int], int, tuple[int, ...] | None]:
    try:
        first_record = next(records)
    except StopIteration as error:
        raise RuntimeError("No source volumes were read") from error

    volume_shape = tuple(int(size) for size in first_record.volume.shape)
    image_shape = (len(config.selected_volumes), *volume_shape)
    if first_record.local_t != 0:
        raise ValueError("The first source volume must have local index 0")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".tiff-export-", dir=output_path.parent) as temp_dir:
        temporary_path = Path(temp_dir) / output_path.name
        temporary_roi_path = Path(temp_dir) / "neuron_point_tuple.npy"
        roi_frames: list[np.ndarray] | None = (
            [] if first_record.roi_points is not None else None
        )
        all_records = chain((first_record,), records)
        tifffile.imwrite(
            temporary_path,
            _iter_masked_planes(all_records, config, roi_frames),
            shape=image_shape,
            dtype=np.float32,
            photometric="minisblack",
            compression="deflate",
            compressionargs={"level": TIFF_COMPRESSION_LEVEL},
            metadata={"axes": "TZYX"},
            bigtiff=(
                int(np.prod(image_shape, dtype=np.int64))
                * np.dtype(np.float32).itemsize
                >= 2**32
            ),
        )
        roi_shape: tuple[int, ...] | None = None
        if roi_frames is not None:
            if len(roi_frames) != len(config.selected_volumes):
                raise RuntimeError(
                    f"Collected {len(roi_frames)} ROI frames; "
                    f"expected {len(config.selected_volumes)}"
                )
            roi = np.stack(roi_frames)
            roi_shape = tuple(int(size) for size in roi.shape)
            with temporary_roi_path.open("wb") as output_file:
                np.save(output_file, roi, allow_pickle=False)
        temporary_path.replace(output_path)
        if roi_frames is not None:
            temporary_roi_path.replace(
                output_path.parent / "neuron_point_tuple.npy"
            )
    return image_shape, output_path.stat().st_size, roi_shape


def main() -> None:
    config = _foreground_config()
    validate_configuration(config)
    if not 0 <= TIFF_COMPRESSION_LEVEL <= 9:
        raise ValueError("TIFF_COMPRESSION_LEVEL must be between 0 and 9")
    records = iter_volume_records(config)
    output_path = OUTPUT_DIR / "volumes.tif"
    print(f"Foreground mode: {config.mask_mode}")
    print(f"Output directory: {OUTPUT_DIR}")
    try:
        image_shape, file_size, roi_shape = _write_tiff_atomic(
            records,
            config,
            output_path,
        )
    finally:
        records.close()
    print(
        f"Saved TIFF:      {output_path}  shape={image_shape}  "
        f"dtype=float32  size={file_size / 1024**2:.1f} MiB"
    )
    if roi_shape is not None:
        print(
            f"Saved ROI:       {OUTPUT_DIR / 'neuron_point_tuple.npy'}  "
            f"shape={roi_shape}"
        )


if __name__ == "__main__":
    main()
