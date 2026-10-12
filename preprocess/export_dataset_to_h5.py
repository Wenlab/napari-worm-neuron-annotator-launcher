"""Stream foreground-masked TIFF volumes to images.h5.

Edit the settings below, then run ``pixi run preprocess-h5``. The ``volumes``
dataset stores float32 images in (T,Z,Y,X) order with lossless Zstd compression.
The complete XY canvas is retained; background voxels are zero. Optional ROI
points are written to neuron_point_tuple.npy for the annotator. Existing output
files are never overwritten.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import h5py
import hdf5plugin
import numpy as np

from foreground_source import (
    ForegroundConfig,
    VolumeRecord,
    iter_volume_records,
    mask_for_record,
    validate_configuration,
)
from tiff_source import TiffFrameSource

# ---------------------------------------------------------------------------
# Frequently edited configuration
# ---------------------------------------------------------------------------

# Numbered TIFF directory or multi-page TIFF stack. Set the acquisition's real
# source path and volume numbers before exporting.
TIFF_PATH = Path(r"\\192.168.1.192\Jinghao-Wang-2026\20260821_chemical_closeloop\w3_WEN1277_CuSO4-test_20260821_171723\w3_WEN1277_CuSO4-test_2026-08-21_17-17-53\0_Camera-Red_VSC-10629")
ROI_SOURCE_MODE: Literal["dynamics", "realtime-results"] | None = None
ROI_SOURCE_PATH: Path | None = None
OUTPUT_DIR = Path("data/")
# None or an empty sequence selects all complete acquisition volumes from 0.
SELECTED_VOLUMES: Sequence[int] | None = None

# "image" uses the per-volume XY maximum-intensity projection. "roi" uses
# projected ROI boxes; "auto" chooses ROI when an ROI source is configured.
FOREGROUND_MODE: Literal["auto", "roi", "image"] = "image"

# One time point is one complete volume. Keep offsets 0..17 from each group of
# 20 acquired TIFF frames, as in export_dataset_to_tiff.py.
FRAMES_PER_VOLUME = 20
Z_START_FRAME = 0
Z_END_FRAME = 17
REVERSE_Z_BY_VOLUME_PARITY = (False, False)

# Image and ROI geometry. Image-only export requires ALIGN_XY = False.
DYNAMICS_FIRST_VOLUME = 0
ALIGN_XY = False
GOAL_ANGLE_DEGREES = -90.0
FLIP_X = False
FLIP_Y = False
IMAGE_INTERPOLATION_ORDER = 1
COORDINATE_ORDER = "xyz"
XY_PIXEL_SIZE = 0.3  # micrometres per XY pixel
Z_STEP_SIZE = 1.5  # micrometres per Z layer
Z_SCALE_RATIO = Z_STEP_SIZE / XY_PIXEL_SIZE  # must match launcher Z_DIVISOR

# Same foreground defaults as the TIFF exporter. Each volume has one XY mask,
# expanded and closed before being broadcast along its Z axis.
ROI_MARGIN_XY_PIXELS = 6
ROI_MARGIN_Z_LAYERS = 1
BACKGROUND_SIGMA_XY = 20.0
THRESHOLD_SIGMA = 4.0
MIN_CONTRAST = 2.0
MIN_COMPONENT_PIXELS = 4
ROI_DILATION_XY_PIXELS = 48
IMAGE_DILATION_XY_PIXELS = 48
CLOSING_XY_PIXELS = 32

# One (1,1,Y,X) chunk per plane supports napari's on-demand Z slicing.
HDF5_COMPRESSION_LEVEL = 3


def _resolve_selected_volumes(
    tiff_path: Path,
    selected_volumes: Sequence[int] | None,
    frames_per_volume: int,
) -> tuple[int, ...]:
    if selected_volumes is not None and len(selected_volumes) > 0:
        return tuple(selected_volumes)
    if frames_per_volume <= 0:
        raise ValueError("FRAMES_PER_VOLUME must be positive")
    with TiffFrameSource(tiff_path) as source:
        frame_count = source.frame_count
    volume_count, trailing_frames = divmod(frame_count, frames_per_volume)
    if volume_count == 0:
        raise ValueError(
            f"No complete volumes: {frame_count} TIFF frames, "
            f"FRAMES_PER_VOLUME={frames_per_volume}"
        )
    if trailing_frames:
        print(f"Skipping {trailing_frames} trailing frames of an incomplete volume")
    return tuple(range(volume_count))


def _foreground_config() -> ForegroundConfig:
    return ForegroundConfig(
        tiff_path=TIFF_PATH,
        roi_source_mode=ROI_SOURCE_MODE,
        roi_source_path=ROI_SOURCE_PATH,
        foreground_mode=FOREGROUND_MODE,
        selected_volumes=_resolve_selected_volumes(
            TIFF_PATH, SELECTED_VOLUMES, FRAMES_PER_VOLUME
        ),
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


def _check_output_paths(output_path: Path, save_roi: bool) -> None:
    paths = [output_path]
    if save_roi:
        paths.append(output_path.parent / "neuron_point_tuple.npy")
    for path in paths:
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite {path}; choose a new OUTPUT_DIR"
            )


def _write_h5_atomic(
    records: Iterator[VolumeRecord],
    config: ForegroundConfig,
    output_path: Path,
) -> tuple[tuple[int, int, int, int], int, tuple[int, ...] | None]:
    """Write bounded volumes to a staged file and publish after validation."""

    save_roi = config.roi_source_mode is not None
    _check_output_paths(output_path, save_roi)
    try:
        first_record = next(records)
    except StopIteration as error:
        raise RuntimeError("No source volumes were read") from error

    volume_shape = tuple(int(size) for size in first_record.volume.shape)
    if len(volume_shape) != 3 or any(size == 0 for size in volume_shape):
        raise ValueError("Expected a non-empty (Z,Y,X) source volume")
    image_shape = (len(config.selected_volumes), *volume_shape)
    roi_frames: list[np.ndarray] | None = [] if save_roi else None
    all_records = chain((first_record,), records)
    del first_record

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".h5-export-", dir=output_path.parent) as temp_dir:
        temporary_path = Path(temp_dir) / output_path.name
        temporary_roi_path = Path(temp_dir) / "neuron_point_tuple.npy"
        with h5py.File(temporary_path, "w") as output:
            images = output.create_dataset(
                "volumes",
                shape=image_shape,
                dtype=np.float32,
                chunks=(1, 1, *volume_shape[1:]),
                shuffle=True,
                **hdf5plugin.Zstd(clevel=HDF5_COMPRESSION_LEVEL),
            )
            images.attrs["axes"] = "TZYX"
            images.attrs["frames_per_volume"] = config.frames_per_volume
            written_count = 0
            for record in all_records:
                if record.local_t != written_count or written_count >= image_shape[0]:
                    raise ValueError("Source volumes must have consecutive local indices")
                if record.source_volume != config.selected_volumes[written_count]:
                    raise ValueError("Source volume order does not match SELECTED_VOLUMES")
                if record.volume.shape != volume_shape:
                    raise ValueError(
                        f"Volume {record.source_volume} shape {record.volume.shape} "
                        f"does not match {volume_shape}"
                    )
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
                # Write one plane at a time; retain source intensities inside
                # the foreground without rescaling or uint8 conversion.
                for z_index, plane in enumerate(record.volume):
                    images[record.local_t, z_index] = np.where(
                        mask[z_index], plane, 0.0
                    ).astype(np.float32, copy=False)
                written_count += 1
            if written_count != image_shape[0]:
                raise RuntimeError(
                    f"Wrote {written_count} volumes; expected {image_shape[0]}"
                )

        roi_shape: tuple[int, ...] | None = None
        if roi_frames is not None:
            roi = np.stack(roi_frames)
            if roi.ndim != 3 or roi.shape[2] < 6:
                raise ValueError("Expected a (T,N,K>=6) ROI array")
            roi_shape = tuple(int(size) for size in roi.shape)
            with temporary_roi_path.open("wb") as output_file:
                np.save(output_file, roi, allow_pickle=False)
        _check_output_paths(output_path, save_roi)
        temporary_path.replace(output_path)
        if roi_frames is not None:
            temporary_roi_path.replace(output_path.parent / "neuron_point_tuple.npy")

    return image_shape, output_path.stat().st_size, roi_shape


def main() -> None:
    config = _foreground_config()
    validate_configuration(config)
    if not 1 <= HDF5_COMPRESSION_LEVEL <= 22:
        raise ValueError("HDF5_COMPRESSION_LEVEL must be between 1 and 22")
    output_path = OUTPUT_DIR / "images.h5"
    print(f"Foreground mode: {config.mask_mode}")
    print(f"Selected volumes: {len(config.selected_volumes)}")
    print(f"Output directory: {OUTPUT_DIR}")
    records = iter_volume_records(config)
    try:
        image_shape, file_size, roi_shape = _write_h5_atomic(
            records, config, output_path
        )
    finally:
        records.close()
    print(
        f"Saved HDF5:      {output_path}  dataset=volumes  shape={image_shape}  "
        f"dtype=float32  size={file_size / 1024**2:.1f} MiB"
    )
    if roi_shape is not None:
        print(
            f"Saved ROI:       {OUTPUT_DIR / 'neuron_point_tuple.npy'}  "
            f"shape={roi_shape}"
        )


if __name__ == "__main__":
    main()
