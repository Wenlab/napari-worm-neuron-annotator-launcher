"""Shared source transforms and foreground masks for TIFF and HDF5 export."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import ndimage

from raw_dataset_loader import (
    RawDatasetConfig,
    _transform_rois,
    iter_transformed_volumes,
    prepare_raw_inputs,
)
from tiff_source import TiffFrameSource, read_volume


@dataclass(frozen=True)
class ForegroundConfig:
    """Explicit acquisition, geometry, and per-volume masking settings."""

    tiff_path: Path
    roi_source_mode: Literal["dynamics", "realtime-results"] | None
    roi_source_path: Path | None
    foreground_mode: Literal["auto", "roi", "image"]
    selected_volumes: tuple[int, ...]
    frames_per_volume: int
    z_start_frame: int
    z_end_frame: int
    reverse_z_by_volume_parity: tuple[bool, bool]
    dynamics_first_volume: int
    align_xy: bool
    goal_angle_degrees: float
    flip_x: bool
    flip_y: bool
    image_interpolation_order: int
    coordinate_order: str
    z_scale_ratio: float
    roi_margin_xy_pixels: int
    roi_margin_z_layers: int
    background_sigma_xy: float
    threshold_sigma: float
    min_contrast: float
    min_component_pixels: int
    roi_dilation_xy_pixels: int
    image_dilation_xy_pixels: int
    closing_xy_pixels: int

    @property
    def mask_mode(self) -> Literal["roi", "image"]:
        if self.foreground_mode == "auto":
            return "roi" if self.roi_source_mode is not None else "image"
        return self.foreground_mode


@dataclass(frozen=True)
class VolumeRecord:
    """One transformed volume and its optional transformed ROI frame."""

    local_t: int
    source_volume: int
    volume: np.ndarray
    roi_points: np.ndarray | None


def validate_configuration(config: ForegroundConfig) -> None:
    if not config.tiff_path.exists():
        raise FileNotFoundError(f"TIFF source does not exist: {config.tiff_path}")
    if (config.roi_source_mode is None) != (config.roi_source_path is None):
        raise ValueError(
            "ROI_SOURCE_MODE and ROI_SOURCE_PATH must both be set or both be None"
        )
    if config.foreground_mode not in ("auto", "roi", "image"):
        raise ValueError("FOREGROUND_MODE must be 'auto', 'roi', or 'image'")
    if config.foreground_mode == "roi" and config.roi_source_mode is None:
        raise ValueError("FOREGROUND_MODE='roi' requires an ROI source")
    if config.roi_source_mode is None and config.align_xy:
        raise ValueError("Image-only mode requires ALIGN_XY = False")
    if not config.selected_volumes:
        raise ValueError("SELECTED_VOLUMES must not be empty")
    if len(set(config.selected_volumes)) != len(config.selected_volumes):
        raise ValueError("SELECTED_VOLUMES contains duplicates")
    if not 0 <= config.z_start_frame <= config.z_end_frame < config.frames_per_volume:
        raise ValueError("Invalid Z_START_FRAME/Z_END_FRAME")
    if config.image_interpolation_order not in (0, 1, 3):
        raise ValueError("IMAGE_INTERPOLATION_ORDER must be 0, 1, or 3")
    if config.z_scale_ratio <= 0 or not np.isfinite(config.z_scale_ratio):
        raise ValueError("Z_SCALE_RATIO must be positive")
    if config.roi_margin_xy_pixels < 0 or config.roi_margin_z_layers < 0:
        raise ValueError("ROI margins must be non-negative")
    if config.background_sigma_xy <= 0 or config.threshold_sigma <= 0:
        raise ValueError("Background sigma and threshold sigma must be positive")
    if config.min_contrast < 0 or config.min_component_pixels <= 0:
        raise ValueError("MIN_CONTRAST must be non-negative and component size positive")
    if config.roi_dilation_xy_pixels < 0 or config.image_dilation_xy_pixels < 0:
        raise ValueError("Dilation sizes must be non-negative")
    if config.closing_xy_pixels < 0:
        raise ValueError("CLOSING_XY_PIXELS must be non-negative")


def _raw_config(config: ForegroundConfig) -> RawDatasetConfig:
    """Build the shared raw-loader configuration when ROI data is available."""

    if config.roi_source_mode is None or config.roi_source_path is None:
        raise ValueError("An ROI source is required for _raw_config")
    return RawDatasetConfig(
        tiff_path=config.tiff_path,
        roi_source_mode=config.roi_source_mode,
        roi_source_path=config.roi_source_path,
        selected_volumes=tuple(config.selected_volumes),
        frames_per_volume=config.frames_per_volume,
        z_start_frame=config.z_start_frame,
        z_end_frame=config.z_end_frame,
        reverse_z_by_volume_parity=config.reverse_z_by_volume_parity,
        dynamics_first_volume=config.dynamics_first_volume,
        align_xy=config.align_xy,
        goal_angle_degrees=config.goal_angle_degrees,
        flip_x=config.flip_x,
        flip_y=config.flip_y,
        image_interpolation_order=config.image_interpolation_order,
        coordinate_order=config.coordinate_order,
        z_scale_ratio=config.z_scale_ratio,
    )


def _iter_roi_records(config: ForegroundConfig) -> Iterator[VolumeRecord]:
    raw_config = _raw_config(config)
    selected, roi_data, alignment_by_volume = prepare_raw_inputs(raw_config)
    first_volume_shape: tuple[int, int, int] | None = None
    image_iterator = iter_transformed_volumes(
        raw_config,
        selected,
        alignment_by_volume,
    )
    try:
        first_local_t, first_source_volume, first_volume = next(image_iterator)
        first_volume_shape = tuple(int(size) for size in first_volume.shape)
        roi = _transform_rois(
            raw_config,
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


def _transform_image_only(
    volume: np.ndarray, config: ForegroundConfig
) -> np.ndarray:
    """Apply image-only flips while keeping the source coordinate system."""

    if config.flip_x:
        volume = volume[:, :, ::-1]
    if config.flip_y:
        volume = volume[:, ::-1, :]
    return np.ascontiguousarray(volume, dtype=np.float32)


def _iter_image_records(config: ForegroundConfig) -> Iterator[VolumeRecord]:
    if config.align_xy:
        raise ValueError("Image-only mode requires ALIGN_XY = False")
    first_shape: tuple[int, int, int] | None = None
    with TiffFrameSource(config.tiff_path) as source:
        for local_t, source_volume in enumerate(config.selected_volumes):
            volume = read_volume(
                source,
                source_volume,
                config.frames_per_volume,
                config.z_start_frame,
                config.z_end_frame,
                config.reverse_z_by_volume_parity,
            )
            volume = _transform_image_only(volume, config)
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
    config: ForegroundConfig,
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
        z_center = z_scaled / config.z_scale_ratio
        depth_layers = depth_scaled / config.z_scale_ratio
        x_min = max(
            0,
            int(np.floor(x_value - width / 2.0)) - config.roi_margin_xy_pixels,
        )
        x_max = min(
            x_dim,
            int(np.ceil(x_value + width / 2.0))
            + config.roi_margin_xy_pixels
            + 1,
        )
        y_min = max(
            0,
            int(np.floor(y_value - height / 2.0)) - config.roi_margin_xy_pixels,
        )
        y_max = min(
            y_dim,
            int(np.ceil(y_value + height / 2.0))
            + config.roi_margin_xy_pixels
            + 1,
        )
        z_min = max(
            0,
            int(np.floor(z_center - depth_layers / 2.0))
            - config.roi_margin_z_layers,
        )
        z_max = min(
            z_dim,
            int(np.ceil(z_center + depth_layers / 2.0))
            + config.roi_margin_z_layers
            + 1,
        )
        if z_min < z_max and y_min < y_max and x_min < x_max:
            union_mask[y_min:y_max, x_min:x_max] = True
    return union_mask


def _adaptive_mip_mask(
    volume: np.ndarray, config: ForegroundConfig
) -> np.ndarray:
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
        sigma=config.background_sigma_xy,
        mode="nearest",
    )
    weights = ndimage.gaussian_filter(
        valid_float,
        sigma=config.background_sigma_xy,
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
    noise = max(1.4826 * mad, config.min_contrast / config.threshold_sigma)
    candidate = valid & (
        corrected > center + max(config.min_contrast, config.threshold_sigma * noise)
    )

    connectivity = ndimage.generate_binary_structure(rank=2, connectivity=2)
    labels, component_count = ndimage.label(candidate, structure=connectivity)
    if component_count == 0:
        return candidate
    component_sizes = np.bincount(labels.ravel())
    keep = component_sizes >= config.min_component_pixels
    keep[0] = False
    return keep[labels]


def _disk_structure(radius: int) -> np.ndarray:
    """Build a filled 2-D disk with the requested pixel radius."""

    if radius == 0:
        return np.ones((1, 1), dtype=bool)
    values = np.arange(-radius, radius + 1, dtype=np.int32)
    yy, xx = np.meshgrid(values, values, indexing="ij")
    return xx**2 + yy**2 <= radius**2


def _close_common_xy_mask(
    mask_yx: np.ndarray, config: ForegroundConfig
) -> np.ndarray:
    """Connect neuronal regions with one large-scale closing operation."""

    mask = np.asarray(mask_yx, dtype=bool)
    if config.closing_xy_pixels == 0 or not np.any(mask):
        return mask
    radius = config.closing_xy_pixels
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    # Two Euclidean distance transforms implement disk dilation followed by
    # disk erosion without the high cost of a large explicit footprint.
    dilated = ndimage.distance_transform_edt(~padded) <= radius
    closed = ndimage.distance_transform_edt(dilated) > radius
    return closed[radius:-radius, radius:-radius]


def mask_for_record(
    record: VolumeRecord, config: ForegroundConfig
) -> np.ndarray:
    mode = config.mask_mode
    if mode == "roi":
        if record.roi_points is None:
            raise ValueError("ROI mode requires ROI points")
        mask_yx = _projected_roi_mask(
            record.roi_points,
            record.volume.shape,
            config,
        )
        dilation_radius = config.roi_dilation_xy_pixels
    else:
        mask_yx = _adaptive_mip_mask(record.volume, config)
        dilation_radius = config.image_dilation_xy_pixels
    if dilation_radius:
        mask_yx = ndimage.binary_dilation(
            mask_yx,
            structure=_disk_structure(dilation_radius),
        )
    common_mask_yx = _close_common_xy_mask(mask_yx, config)
    return np.broadcast_to(common_mask_yx, record.volume.shape)


def iter_volume_records(config: ForegroundConfig) -> Iterator[VolumeRecord]:
    """Yield one complete transformed (Z,Y,X) volume at a time."""

    if config.roi_source_mode is not None:
        yield from _iter_roi_records(config)
    else:
        yield from _iter_image_records(config)
