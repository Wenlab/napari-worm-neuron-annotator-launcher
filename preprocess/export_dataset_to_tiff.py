"""Export a compressed foreground TIFF without changing the NPY exporter.

Edit the configuration constants below, then run from the repository root::

    pixi run python preprocess/export_dataset_to_tiff.py

The output keeps the complete ``(T, Z, Y, X)`` canvas and original float32
intensities. Voxels outside the selected foreground mask are written as zero.
The original TIFF and any existing NPY files are never modified.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import tifffile
from scipy import ndimage

from raw_dataset_loader import (
    RawDatasetConfig,
    _transform_rois,
    iter_transformed_volumes,
    prepare_raw_inputs,
)
from tiff_source import TiffFrameSource, read_volume

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
IMAGE_DILATION_XY_PIXELS =48

# Apply one large 2-D closing to connect nearby neuronal regions. The result is
# broadcast along Z so every layer in one volume uses the same XY mask.
CLOSING_XY_PIXELS = 32

TIFF_COMPRESSION_LEVEL = 3


@dataclass(frozen=True)
class VolumeRecord:
    """One transformed volume and its optional transformed ROI frame."""

    local_t: int
    source_volume: int
    volume: np.ndarray
    roi_points: np.ndarray | None


def _validate_configuration() -> None:
    if not TIFF_PATH.exists():
        raise FileNotFoundError(f"TIFF source does not exist: {TIFF_PATH}")
    if (ROI_SOURCE_MODE is None) != (ROI_SOURCE_PATH is None):
        raise ValueError(
            "ROI_SOURCE_MODE and ROI_SOURCE_PATH must both be set or both be None"
        )
    if FOREGROUND_MODE not in ("auto", "roi", "image"):
        raise ValueError("FOREGROUND_MODE must be 'auto', 'roi', or 'image'")
    if FOREGROUND_MODE == "roi" and ROI_SOURCE_MODE is None:
        raise ValueError("FOREGROUND_MODE='roi' requires an ROI source")
    if ROI_SOURCE_MODE is None and ALIGN_XY:
        raise ValueError("Image-only mode requires ALIGN_XY = False")
    if not SELECTED_VOLUMES:
        raise ValueError("SELECTED_VOLUMES must not be empty")
    if len(set(SELECTED_VOLUMES)) != len(SELECTED_VOLUMES):
        raise ValueError("SELECTED_VOLUMES contains duplicates")
    if not 0 <= Z_START_FRAME <= Z_END_FRAME < FRAMES_PER_VOLUME:
        raise ValueError("Invalid Z_START_FRAME/Z_END_FRAME")
    if IMAGE_INTERPOLATION_ORDER not in (0, 1, 3):
        raise ValueError("IMAGE_INTERPOLATION_ORDER must be 0, 1, or 3")
    if Z_SCALE_RATIO <= 0 or not np.isfinite(Z_SCALE_RATIO):
        raise ValueError("Z_SCALE_RATIO must be positive")
    if ROI_MARGIN_XY_PIXELS < 0 or ROI_MARGIN_Z_LAYERS < 0:
        raise ValueError("ROI margins must be non-negative")
    if BACKGROUND_SIGMA_XY <= 0 or THRESHOLD_SIGMA <= 0:
        raise ValueError("Background sigma and threshold sigma must be positive")
    if MIN_CONTRAST < 0 or MIN_COMPONENT_PIXELS <= 0:
        raise ValueError("MIN_CONTRAST must be non-negative and component size positive")
    if ROI_DILATION_XY_PIXELS < 0 or IMAGE_DILATION_XY_PIXELS < 0:
        raise ValueError("Dilation sizes must be non-negative")
    if CLOSING_XY_PIXELS < 0:
        raise ValueError("CLOSING_XY_PIXELS must be non-negative")
    if not 0 <= TIFF_COMPRESSION_LEVEL <= 9:
        raise ValueError("TIFF_COMPRESSION_LEVEL must be between 0 and 9")


def _raw_config() -> RawDatasetConfig:
    """Build the shared raw-loader configuration when ROI data is available."""

    if ROI_SOURCE_MODE is None or ROI_SOURCE_PATH is None:
        raise ValueError("An ROI source is required for _raw_config")
    return RawDatasetConfig(
        tiff_path=TIFF_PATH,
        roi_source_mode=ROI_SOURCE_MODE,
        roi_source_path=ROI_SOURCE_PATH,
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


def _iter_roi_records() -> Iterator[VolumeRecord]:
    config = _raw_config()
    selected, roi_data, alignment_by_volume = prepare_raw_inputs(config)
    first_volume_shape: tuple[int, int, int] | None = None
    image_iterator = iter_transformed_volumes(
        config,
        selected,
        alignment_by_volume,
    )
    try:
        first_local_t, first_source_volume, first_volume = next(image_iterator)
        first_volume_shape = tuple(int(size) for size in first_volume.shape)
        roi = _transform_rois(
            config,
            selected,
            roi_data.points,
            roi_data.source_volumes,
            alignment_by_volume,
            first_volume_shape[1:],
        )
        yield VolumeRecord(
            first_local_t,
            first_source_volume,
            first_volume,
            roi[first_local_t],
        )
        for local_t, source_volume, volume in image_iterator:
            if volume.shape != first_volume_shape:
                raise ValueError(
                    f"Volume {source_volume} shape {volume.shape} does not match "
                    f"{first_volume_shape}"
                )
            yield VolumeRecord(
                local_t,
                source_volume,
                volume,
                roi[local_t],
            )
    finally:
        image_iterator.close()


def _transform_image_only(volume: np.ndarray) -> np.ndarray:
    """Apply image-only flips while keeping the source coordinate system."""

    if FLIP_X:
        volume = volume[:, :, ::-1]
    if FLIP_Y:
        volume = volume[:, ::-1, :]
    return np.ascontiguousarray(volume, dtype=np.float32)


def _iter_image_records() -> Iterator[VolumeRecord]:
    if ALIGN_XY:
        raise ValueError("Image-only mode requires ALIGN_XY = False")
    first_shape: tuple[int, int, int] | None = None
    with TiffFrameSource(TIFF_PATH) as source:
        for local_t, source_volume in enumerate(SELECTED_VOLUMES):
            volume = read_volume(
                source,
                source_volume,
                FRAMES_PER_VOLUME,
                Z_START_FRAME,
                Z_END_FRAME,
                REVERSE_Z_BY_VOLUME_PARITY,
            )
            volume = _transform_image_only(volume)
            if first_shape is None:
                first_shape = volume.shape
            elif volume.shape != first_shape:
                raise ValueError(
                    f"Volume {source_volume} shape {volume.shape} does not match "
                    f"{first_shape}"
                )
            print(
                f"  [{local_t}] source_volume={source_volume}  "
                f"shape={volume.shape}  dtype={volume.dtype}  "
                f"min={volume.min():.1f}  max={volume.max():.1f}"
            )
            yield VolumeRecord(local_t, int(source_volume), volume, None)


def _projected_roi_mask(
    points: np.ndarray,
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    """Project all valid ROI boxes into one per-volume XY union mask."""

    z_dim, y_dim, x_dim = volume_shape
    union_mask = np.zeros((y_dim, x_dim), dtype=bool)
    for row in points:
        values = np.asarray(row[:6], dtype=np.float32)
        if (
            values.size < 6
            or not np.isfinite(values).all()
            or np.any(values[3:] <= 0)
        ):
            continue
        x_value, y_value, z_scaled, width, height, depth_scaled = (
            float(value) for value in values
        )
        z_center = z_scaled / Z_SCALE_RATIO
        depth_layers = depth_scaled / Z_SCALE_RATIO
        x_min = max(
            0,
            int(np.floor(x_value - width / 2.0)) - ROI_MARGIN_XY_PIXELS,
        )
        x_max = min(
            x_dim,
            int(np.ceil(x_value + width / 2.0))
            + ROI_MARGIN_XY_PIXELS
            + 1,
        )
        y_min = max(
            0,
            int(np.floor(y_value - height / 2.0)) - ROI_MARGIN_XY_PIXELS,
        )
        y_max = min(
            y_dim,
            int(np.ceil(y_value + height / 2.0))
            + ROI_MARGIN_XY_PIXELS
            + 1,
        )
        z_min = max(
            0,
            int(np.floor(z_center - depth_layers / 2.0))
            - ROI_MARGIN_Z_LAYERS,
        )
        z_max = min(
            z_dim,
            int(np.ceil(z_center + depth_layers / 2.0))
            + ROI_MARGIN_Z_LAYERS
            + 1,
        )
        if z_min < z_max and y_min < y_max and x_min < x_max:
            union_mask[y_min:y_max, x_min:x_max] = True
    return union_mask


def _adaptive_mip_mask(volume: np.ndarray) -> np.ndarray:
    """Find locally bright connected components in one volume's XY MIP."""

    finite = np.isfinite(volume)
    valid = np.any(finite, axis=0)
    mip = np.max(
        np.where(finite, volume, -np.inf),
        axis=0,
    )
    mip[~valid] = 0.0
    valid &= mip > 0
    if not np.any(valid):
        return np.zeros(volume.shape[1:], dtype=bool)

    valid_float = valid.astype(np.float32)
    smoothed = ndimage.gaussian_filter(
        np.where(valid, mip, 0.0),
        sigma=BACKGROUND_SIGMA_XY,
        mode="nearest",
    )
    weights = ndimage.gaussian_filter(
        valid_float,
        sigma=BACKGROUND_SIGMA_XY,
        mode="nearest",
    )
    background = np.divide(
        smoothed,
        weights,
        out=np.zeros_like(smoothed),
        where=weights > np.finfo(np.float32).eps,
    )
    corrected = mip - background
    valid_corrected = corrected[valid]
    center = float(np.median(valid_corrected))
    mad = float(np.median(np.abs(valid_corrected - center)))
    noise = max(1.4826 * mad, MIN_CONTRAST / THRESHOLD_SIGMA)
    candidate = valid & (
        corrected > center + max(MIN_CONTRAST, THRESHOLD_SIGMA * noise)
    )

    connectivity = ndimage.generate_binary_structure(rank=2, connectivity=2)
    labels, component_count = ndimage.label(candidate, structure=connectivity)
    if component_count == 0:
        return candidate
    component_sizes = np.bincount(labels.ravel())
    keep = component_sizes >= MIN_COMPONENT_PIXELS
    keep[0] = False
    return keep[labels]


def _disk_structure(radius: int) -> np.ndarray:
    """Build a filled 2-D disk with the requested pixel radius."""

    if radius == 0:
        return np.ones((1, 1), dtype=bool)
    values = np.arange(-radius, radius + 1, dtype=np.int32)
    yy, xx = np.meshgrid(values, values, indexing="ij")
    return xx**2 + yy**2 <= radius**2


def _close_common_xy_mask(mask_yx: np.ndarray) -> np.ndarray:
    """Connect neuronal regions with one large-scale closing operation."""

    mask = np.asarray(mask_yx, dtype=bool)
    if CLOSING_XY_PIXELS == 0 or not np.any(mask):
        return mask
    radius = CLOSING_XY_PIXELS
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    # Two Euclidean distance transforms implement disk dilation followed by
    # disk erosion without the high cost of a large explicit footprint.
    dilated = ndimage.distance_transform_edt(~padded) <= radius
    closed = ndimage.distance_transform_edt(dilated) > radius
    return closed[radius:-radius, radius:-radius]


def _mask_for_record(record: VolumeRecord, mode: Literal["roi", "image"]) -> np.ndarray:
    if mode == "roi":
        if record.roi_points is None:
            raise ValueError("ROI mode requires ROI points")
        mask_yx = _projected_roi_mask(
            record.roi_points,
            record.volume.shape,
        )
        dilation_radius = ROI_DILATION_XY_PIXELS
    else:
        mask_yx = _adaptive_mip_mask(record.volume)
        dilation_radius = IMAGE_DILATION_XY_PIXELS
    if dilation_radius:
        mask_yx = ndimage.binary_dilation(
            mask_yx,
            structure=_disk_structure(dilation_radius),
        )
    common_mask_yx = _close_common_xy_mask(mask_yx)
    return np.broadcast_to(common_mask_yx, record.volume.shape)


def _iter_masked_planes(
    records: Iterator[VolumeRecord],
    mode: Literal["roi", "image"],
    roi_frames: list[np.ndarray] | None,
) -> Iterator[np.ndarray]:
    for record in records:
        if roi_frames is not None:
            if record.roi_points is None:
                raise ValueError("ROI data is missing from a source volume")
            roi_frames.append(record.roi_points)
        elif record.roi_points is not None:
            raise ValueError("Unexpected ROI data in an image-only export")
        mask = _mask_for_record(record, mode)
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
    mode: Literal["roi", "image"],
    output_path: Path,
) -> tuple[tuple[int, int, int, int], int, tuple[int, ...] | None]:
    try:
        first_record = next(records)
    except StopIteration as error:
        raise RuntimeError("No source volumes were read") from error

    volume_shape = tuple(int(size) for size in first_record.volume.shape)
    image_shape = (len(SELECTED_VOLUMES), *volume_shape)
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
            _iter_masked_planes(all_records, mode, roi_frames),
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
            if len(roi_frames) != len(SELECTED_VOLUMES):
                raise RuntimeError(
                    f"Collected {len(roi_frames)} ROI frames; "
                    f"expected {len(SELECTED_VOLUMES)}"
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
    _validate_configuration()
    if FOREGROUND_MODE == "auto":
        mode: Literal["roi", "image"] = (
            "roi" if ROI_SOURCE_MODE is not None else "image"
        )
    else:
        mode = FOREGROUND_MODE
    records = (
        _iter_roi_records()
        if ROI_SOURCE_MODE is not None
        else _iter_image_records()
    )
    output_path = OUTPUT_DIR / "volumes.tif"
    print(f"Foreground mode: {mode}")
    print(f"Output directory: {OUTPUT_DIR}")
    image_shape, file_size, roi_shape = _write_tiff_atomic(
        records,
        mode,
        output_path,
    )
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
