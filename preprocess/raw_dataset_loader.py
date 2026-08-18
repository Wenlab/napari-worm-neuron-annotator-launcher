"""Prepare raw TIFF and ROI sources without writing dataset files."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dask.array as da
import numpy as np
from dask import delayed

from geometry import (
    align_volume_xy,
    build_alignment_map,
    transform_points_xy,
    transform_points_z_scaled,
)
from roi_source import RoiSourceMode, coordinates_xyz, load_roi_data
from tiff_source import TiffFrameSource, read_volume, volume_frame_numbers

LoadMode = Literal["eager", "virtual"]


@dataclass(frozen=True)
class RawDatasetConfig:
    """Configuration shared by eager and virtual raw-data loading."""

    tiff_path: Path
    roi_source_mode: RoiSourceMode
    roi_source_path: Path
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


@dataclass(frozen=True)
class RawDataset:
    """Prepared Image and ROI arrays from one raw source."""

    volumes: np.ndarray | da.Array
    roi: np.ndarray


def validate_raw_settings(config: RawDatasetConfig) -> None:
    """Validate raw acquisition and geometry settings."""

    if not config.tiff_path.exists():
        raise FileNotFoundError(f"TIFF source does not exist: {config.tiff_path}")
    if config.roi_source_mode == "dynamics":
        if not config.roi_source_path.is_file():
            raise FileNotFoundError(
                f"Dynamics source does not exist: {config.roi_source_path}"
            )
    elif config.roi_source_mode == "realtime-results":
        if not config.roi_source_path.is_dir():
            raise FileNotFoundError(
                f"Realtime results directory does not exist: "
                f"{config.roi_source_path}"
            )
    else:
        raise ValueError(
            "ROI source mode must be 'dynamics' or 'realtime-results'"
        )
    if not config.selected_volumes:
        raise ValueError("SELECTED_VOLUMES must not be empty")
    if len(set(config.selected_volumes)) != len(config.selected_volumes):
        raise ValueError("SELECTED_VOLUMES contains duplicates")
    try:
        invalid_volume = any(
            int(volume) != volume or int(volume) < 0
            for volume in config.selected_volumes
        )
    except (OverflowError, TypeError, ValueError):
        invalid_volume = True
    if invalid_volume:
        raise ValueError("Selected volume numbers must be non-negative integers")
    if config.frames_per_volume <= 0:
        raise ValueError("FRAMES_PER_VOLUME must be positive")
    if not (0 <= config.z_start_frame <= config.z_end_frame < config.frames_per_volume):
        raise ValueError(
            "Z frame range must satisfy 0 <= start <= end < FRAMES_PER_VOLUME"
        )
    if len(config.reverse_z_by_volume_parity) != 2:
        raise ValueError("REVERSE_Z_BY_VOLUME_PARITY must contain two booleans")
    order = config.coordinate_order.lower()
    if len(order) != 3 or set(order) != set("xyz"):
        raise ValueError("COORDINATE_ORDER must be a permutation of 'xyz'")
    if config.roi_source_mode == "realtime-results" and order != "xyz":
        raise ValueError("Realtime ROI coordinates have fixed COORDINATE_ORDER='xyz'")
    if config.image_interpolation_order not in (0, 1, 3):
        raise ValueError("IMAGE_INTERPOLATION_ORDER must be 0, 1, or 3")
    if not np.isfinite(config.z_scale_ratio) or config.z_scale_ratio <= 0:
        raise ValueError("Z_SCALE_RATIO must be positive")


def _transform_volume(
    volume: np.ndarray,
    center_xy: np.ndarray,
    rotation_xy: np.ndarray,
    config: RawDatasetConfig,
) -> np.ndarray:
    if config.align_xy:
        volume = align_volume_xy(
            volume,
            center_xy,
            rotation_xy,
            config.image_interpolation_order,
        )
    if config.flip_x:
        volume = volume[:, :, ::-1]
    if config.flip_y:
        volume = volume[:, ::-1, :]
    return np.ascontiguousarray(volume, dtype=np.float32)


def _read_transform_plane(
    tiff_path: Path,
    frame_number: int,
    expected_shape: tuple[int, int],
    center_xy: np.ndarray,
    rotation_xy: np.ndarray,
    config: RawDatasetConfig,
) -> np.ndarray:
    """Read and transform one YX plane inside a Dask task."""

    with TiffFrameSource(tiff_path) as source:
        plane = source.read(frame_number)
    if plane.shape != expected_shape:
        raise ValueError(
            f"TIFF frame {frame_number} shape {plane.shape} does not "
            f"match {expected_shape}"
        )
    return _transform_volume(
        plane[np.newaxis, ...], center_xy, rotation_xy, config
    )[0]


def _load_eager_volumes(
    config: RawDatasetConfig,
    selected: Sequence[int],
    alignment_by_volume: dict[int, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    volumes_f32: list[np.ndarray] = []
    reference_shape: tuple[int, int, int] | None = None

    with TiffFrameSource(config.tiff_path) as source:
        for local_t, source_volume in enumerate(selected):
            volume = read_volume(
                source,
                source_volume,
                config.frames_per_volume,
                config.z_start_frame,
                config.z_end_frame,
                config.reverse_z_by_volume_parity,
            )
            if reference_shape is None:
                reference_shape = volume.shape
            elif volume.shape != reference_shape:
                raise ValueError(
                    f"Volume {source_volume} shape {volume.shape} does not "
                    f"match {reference_shape}"
                )
            center_xy, rotation_xy = alignment_by_volume[source_volume]
            volume = _transform_volume(volume, center_xy, rotation_xy, config)
            volumes_f32.append(volume)
            print(
                f"  [{local_t}] source_volume={source_volume}  "
                f"shape={volume.shape}  dtype={volume.dtype}  "
                f"min={volume.min():.1f}  max={volume.max():.1f}"
            )

    if reference_shape is None:
        raise RuntimeError("No source volumes were read")
    return np.stack(volumes_f32, axis=0)


def _load_virtual_volumes(
    config: RawDatasetConfig,
    selected: Sequence[int],
    alignment_by_volume: dict[int, tuple[np.ndarray, np.ndarray]],
) -> da.Array:
    first_volume = selected[0]
    first_frame = volume_frame_numbers(
        first_volume,
        config.frames_per_volume,
        config.z_start_frame,
        config.z_end_frame,
        config.reverse_z_by_volume_parity,
    )[0]
    with TiffFrameSource(config.tiff_path) as source:
        expected_shape = tuple(int(size) for size in source.read(first_frame).shape)

    volumes: list[da.Array] = []
    for source_volume in selected:
        center_xy, rotation_xy = alignment_by_volume[source_volume]
        planes: list[da.Array] = []
        for frame_number in volume_frame_numbers(
            source_volume,
            config.frames_per_volume,
            config.z_start_frame,
            config.z_end_frame,
            config.reverse_z_by_volume_parity,
        ):
            plane = delayed(_read_transform_plane)(
                config.tiff_path,
                frame_number,
                expected_shape,
                center_xy,
                rotation_xy,
                config,
            )
            planes.append(
                da.from_delayed(plane, shape=expected_shape, dtype=np.float32)
            )
        volumes.append(da.stack(planes, axis=0))
    return da.stack(volumes, axis=0)


def _transform_rois(
    config: RawDatasetConfig,
    selected: Sequence[int],
    neuron_pt_tuple: np.ndarray,
    roi_volume_numbers: Sequence[int],
    alignment_by_volume: dict[int, tuple[np.ndarray, np.ndarray]],
    image_shape_yx: tuple[int, int],
) -> np.ndarray:
    volume_to_index = {
        int(volume): index for index, volume in enumerate(roi_volume_numbers)
    }
    transformed_frames: list[np.ndarray] = []

    for source_volume in selected:
        matrix_index = volume_to_index[source_volume]
        points = np.asarray(neuron_pt_tuple[matrix_index], dtype=np.float32).copy()
        xyz = coordinates_xyz(points, config.coordinate_order)
        center_xy, rotation_xy = alignment_by_volume[source_volume]
        xyz[:, :2] = transform_points_xy(
            xyz[:, :2],
            center_xy,
            rotation_xy,
            image_shape_yx,
            config.align_xy,
            config.flip_x,
            config.flip_y,
        )
        xyz[:, 2] = transform_points_z_scaled(
            xyz[:, 2],
            config.z_scale_ratio,
            config.z_start_frame,
            config.z_end_frame,
            config.reverse_z_by_volume_parity[source_volume % 2],
        )
        points[:, :3] = xyz
        transformed_frames.append(points)
    return np.stack(transformed_frames, axis=0)


def load_raw_dataset(
    config: RawDatasetConfig,
    mode: LoadMode = "eager",
) -> RawDataset:
    """Prepare raw sources eagerly or as a plane-chunked virtual Image."""

    validate_raw_settings(config)
    if mode not in ("eager", "virtual"):
        raise ValueError("Raw load mode must be 'eager' or 'virtual'")

    selected = [int(volume) for volume in config.selected_volumes]
    roi_data = load_roi_data(
        config.roi_source_path,
        config.roi_source_mode,
        selected,
        first_volume=config.dynamics_first_volume,
        align_xy=config.align_xy,
    )
    alignment_by_volume = build_alignment_map(
        roi_data.source_volumes,
        roi_data.centers_xy,
        roi_data.rotations_xy,
        config.goal_angle_degrees,
    )
    print(
        f"ROI source:       {config.roi_source_mode}  "
        f"selected={len(selected)}  "
        f"valid={len(selected) - len(roi_data.missing_volumes)}  "
        f"missing={len(roi_data.missing_volumes)}  "
        f"alignment_interpolated="
        f"{len(roi_data.interpolated_alignment_volumes)}"
    )

    if mode == "eager":
        volumes = _load_eager_volumes(config, selected, alignment_by_volume)
    else:
        volumes = _load_virtual_volumes(config, selected, alignment_by_volume)
    image_shape_yx = tuple(int(size) for size in volumes.shape[-2:])
    roi = _transform_rois(
        config,
        selected,
        roi_data.points,
        roi_data.source_volumes,
        alignment_by_volume,
        image_shape_yx,
    )
    return RawDataset(volumes=volumes, roi=roi)
