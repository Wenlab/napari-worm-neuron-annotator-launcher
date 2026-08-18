"""Read neuron ROI points and per-volume XY alignment data."""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from geometry import (
    estimate_pca_alignment_xy,
    interpolate_alignment_xy,
    pca_axis_rotation_matrix,
)

RoiSourceMode = Literal["dynamics", "realtime-results"]

POINT_DATASET_CANDIDATES = (
    "d_neuron_pt_tuple_matched_raw_vol",
    "neuron_pt_tuple",
)
CENTER_DATASET = "center"
ROTATION_DATASET = "rot"
REALTIME_ANALYSIS_GROUP = "real-time_analysis"
REALTIME_ID_DATASET = "neuron_pred_ids"
REALTIME_POINT_DATASET = "neuron_pt_tuple"
REALTIME_FILE_PATTERN = re.compile(r"volume_(\d+)\.h5")
REALTIME_GROUP_PATTERN = re.compile(r"volume_(\d+)")


@dataclass(frozen=True)
class RoiData:
    """Dense ROI frames and source-space XY transforms in requested order."""

    points: np.ndarray
    source_volumes: tuple[int, ...]
    centers_xy: np.ndarray
    rotations_xy: np.ndarray
    missing_volumes: tuple[int, ...]
    interpolated_alignment_volumes: tuple[int, ...]


@dataclass(frozen=True)
class _RealtimeFrame:
    points: np.ndarray
    neuron_ids: np.ndarray


def natural_key(value: str) -> list[object]:
    """Return a key that sorts embedded decimal numbers numerically."""

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def find_point_dataset(group: h5py.Group) -> h5py.Dataset | None:
    """Return the first supported point dataset in a dynamics group."""

    for name in POINT_DATASET_CANDIDATES:
        if name in group and isinstance(group[name], h5py.Dataset):
            return group[name]
    return None


def point_frames(h5_file: h5py.File) -> list[str]:
    """Return naturally sorted dynamics groups containing point data."""

    frames = [
        name
        for name, item in h5_file.items()
        if isinstance(item, h5py.Group) and find_point_dataset(item) is not None
    ]
    return sorted(frames, key=natural_key)


def _load_dynamics_arrays(
    dynamics_path: Path,
    first_volume: int,
) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
    """Load all point arrays and alignment transforms from one dynamics H5."""

    frames: list[np.ndarray] = []
    source_volumes: list[int] = []
    centers_xy: list[np.ndarray] = []
    rotations_xy: list[np.ndarray] = []
    with h5py.File(dynamics_path, "r") as dynamics:
        frame_names = point_frames(dynamics)
        if not frame_names:
            raise ValueError(
                f"No supported neuron point datasets found in {dynamics_path}"
            )

        expected_shape: tuple[int, ...] | None = None
        for frame_index, frame_name in enumerate(frame_names):
            dataset = find_point_dataset(dynamics[frame_name])
            if dataset is None:
                continue
            points = np.asarray(dataset[:], dtype=np.float32)
            if points.ndim != 2 or points.shape[1] < 6:
                raise ValueError(
                    f"Point dataset {dataset.name} must have shape (N,F>=6), "
                    f"got {points.shape}"
                )
            if expected_shape is None:
                expected_shape = points.shape
            elif points.shape != expected_shape:
                raise ValueError(
                    f"Point dataset {dataset.name} has shape {points.shape}; "
                    f"expected {expected_shape}"
                )

            group = dynamics[frame_name]
            if CENTER_DATASET not in group or ROTATION_DATASET not in group:
                raise KeyError(
                    f"Dynamics group {group.name} must contain "
                    f"'{CENTER_DATASET}' and '{ROTATION_DATASET}'"
                )
            center = np.asarray(group[CENTER_DATASET][:], dtype=np.float32)
            rotation = np.asarray(group[ROTATION_DATASET][:], dtype=np.float32)
            if center.ndim != 1 or center.size < 2:
                raise ValueError(
                    f"{group.name}/{CENTER_DATASET} must contain at least XY, "
                    f"got shape {center.shape}"
                )
            if rotation.shape != (2, 2):
                raise ValueError(
                    f"{group.name}/{ROTATION_DATASET} must have shape (2,2), "
                    f"got {rotation.shape}"
                )
            if not np.isfinite(center[:2]).all() or not np.isfinite(rotation).all():
                raise ValueError(f"Non-finite alignment transform in {group.name}")
            if not np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-4):
                raise ValueError(f"Rotation matrix is not orthonormal in {group.name}")
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
                raise ValueError(f"Rotation determinant is not +1 in {group.name}")

            frames.append(points)
            centers_xy.append(center[:2])
            rotations_xy.append(rotation)
            source_volumes.append(
                int(frame_name) if frame_name.isdigit() else first_volume + frame_index
            )

    if len(set(source_volumes)) != len(source_volumes):
        raise ValueError("dynamics.h5 contains duplicate source volume IDs")
    return (
        np.stack(frames, axis=0),
        source_volumes,
        np.stack(centers_xy, axis=0),
        np.stack(rotations_xy, axis=0),
    )


def _load_dynamics_roi_data(
    dynamics_path: Path,
    selected_volumes: Sequence[int],
    first_volume: int,
) -> RoiData:
    points, volumes, centers_xy, rotations_xy = _load_dynamics_arrays(
        dynamics_path, first_volume
    )
    volume_to_index = {int(volume): index for index, volume in enumerate(volumes)}
    missing = [
        int(volume) for volume in selected_volumes if int(volume) not in volume_to_index
    ]
    if missing:
        raise KeyError(f"Volumes missing from dynamics.h5: {missing}")
    indices = [volume_to_index[int(volume)] for volume in selected_volumes]
    requested = tuple(int(volume) for volume in selected_volumes)
    return RoiData(
        points=np.asarray(points[indices], dtype=np.float32),
        source_volumes=requested,
        centers_xy=np.asarray(centers_xy[indices], dtype=np.float32),
        rotations_xy=np.asarray(rotations_xy[indices], dtype=np.float32),
        missing_volumes=(),
        interpolated_alignment_volumes=(),
    )


def _index_realtime_files(realtime_dir: Path) -> dict[int, Path]:
    files_by_volume: dict[int, Path] = {}
    for path in realtime_dir.iterdir():
        if not path.is_file():
            continue
        match = REALTIME_FILE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        volume = int(match.group(1))
        if volume in files_by_volume:
            raise ValueError(
                f"Realtime directory contains duplicate files for volume {volume}: "
                f"{files_by_volume[volume]} and {path}"
            )
        files_by_volume[volume] = path
    if not files_by_volume:
        raise ValueError(f"No volume_*.h5 files found in {realtime_dir}")
    return files_by_volume


def _find_realtime_group(
    realtime_file: h5py.File,
    source_volume: int,
) -> h5py.Group | None:
    if REALTIME_ANALYSIS_GROUP not in realtime_file:
        return None
    analysis = realtime_file[REALTIME_ANALYSIS_GROUP]
    if not isinstance(analysis, h5py.Group):
        return None
    expected_name = f"volume_{source_volume:08d}"
    if expected_name in analysis and isinstance(analysis[expected_name], h5py.Group):
        return analysis[expected_name]
    for name, item in analysis.items():
        match = REALTIME_GROUP_PATTERN.fullmatch(name)
        if (
            match is not None
            and int(match.group(1)) == source_volume
            and isinstance(item, h5py.Group)
        ):
            return item
    return None


def _read_realtime_frame(
    path: Path | None,
    source_volume: int,
) -> _RealtimeFrame | None:
    if path is None:
        return None
    with h5py.File(path, "r") as realtime_file:
        group = _find_realtime_group(realtime_file, source_volume)
        if group is None:
            return None
        if (
            REALTIME_ID_DATASET not in group
            or REALTIME_POINT_DATASET not in group
        ):
            return None
        id_dataset = group[REALTIME_ID_DATASET]
        point_dataset = group[REALTIME_POINT_DATASET]
        if not isinstance(id_dataset, h5py.Dataset) or not isinstance(
            point_dataset, h5py.Dataset
        ):
            raise TypeError(
                f"Realtime group {group.name} must contain dataset values for "
                f"'{REALTIME_ID_DATASET}' and '{REALTIME_POINT_DATASET}'"
            )
        points = np.asarray(point_dataset[()], dtype=np.float32)
        neuron_ids = np.asarray(id_dataset[()])

    if points.ndim != 2 or points.shape[1] < 6:
        raise ValueError(
            f"Realtime points in {path} must have shape (N,K>=6), "
            f"got {points.shape}"
        )
    if neuron_ids.ndim != 1 or neuron_ids.shape[0] != points.shape[0]:
        raise ValueError(
            f"Realtime neuron IDs in {path} must have shape ({points.shape[0]},), "
            f"got {neuron_ids.shape}"
        )
    if not np.issubdtype(neuron_ids.dtype, np.integer):
        raise TypeError(f"Realtime neuron IDs in {path} must be integers")
    neuron_ids = neuron_ids.astype(np.int64, copy=False)
    valid_ids = neuron_ids[neuron_ids >= 0]
    if np.unique(valid_ids).size != valid_ids.size:
        raise ValueError(f"Realtime neuron IDs in {path} contain duplicates")
    return _RealtimeFrame(points=points, neuron_ids=neuron_ids)


def _load_realtime_roi_data(
    realtime_dir: Path,
    selected_volumes: Sequence[int],
    align_xy: bool,
) -> RoiData:
    files_by_volume = _index_realtime_files(realtime_dir)
    indexed_volumes = sorted(files_by_volume)
    requested = tuple(int(volume) for volume in selected_volumes)
    frame_cache: dict[int, _RealtimeFrame | None] = {}

    def read_frame(source_volume: int) -> _RealtimeFrame | None:
        if source_volume not in frame_cache:
            frame_cache[source_volume] = _read_realtime_frame(
                files_by_volume.get(source_volume), source_volume
            )
        return frame_cache[source_volume]

    baseline_frame: _RealtimeFrame | None = None
    for source_volume in indexed_volumes:
        frame = read_frame(source_volume)
        if frame is not None and np.any(frame.neuron_ids >= 0):
            baseline_frame = frame
            break
    if baseline_frame is None:
        raise ValueError(
            f"No realtime neuron point data found in {realtime_dir}"
        )

    selected_frames = [read_frame(source_volume) for source_volume in requested]
    shape_frames = [baseline_frame] + [
        frame for frame in selected_frames if frame is not None
    ]
    feature_counts = {frame.points.shape[1] for frame in shape_frames}
    if len(feature_counts) != 1:
        raise ValueError(
            f"Realtime point feature counts are inconsistent: "
            f"{sorted(feature_counts)}"
        )
    feature_count = feature_counts.pop()
    all_nonnegative_ids = [
        frame.neuron_ids[frame.neuron_ids >= 0]
        for frame in shape_frames
        if np.any(frame.neuron_ids >= 0)
    ]
    neuron_count = int(max(ids.max() for ids in all_nonnegative_ids)) + 1

    dense_points = np.full(
        (len(requested), neuron_count, feature_count),
        np.nan,
        dtype=np.float32,
    )
    missing_volumes: list[int] = []
    for output_index, (source_volume, frame) in enumerate(
        zip(requested, selected_frames, strict=True)
    ):
        if frame is None:
            missing_volumes.append(source_volume)
            continue
        valid = frame.neuron_ids >= 0
        if not np.any(valid):
            missing_volumes.append(source_volume)
            continue
        dense_points[output_index, frame.neuron_ids[valid]] = frame.points[valid]

    if not align_xy:
        return RoiData(
            points=dense_points,
            source_volumes=requested,
            centers_xy=np.zeros((len(requested), 2), dtype=np.float32),
            rotations_xy=np.repeat(
                np.eye(2, dtype=np.float32)[np.newaxis, ...],
                len(requested),
                axis=0,
            ),
            missing_volumes=tuple(missing_volumes),
            interpolated_alignment_volumes=(),
        )

    raw_alignments: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def estimate_alignment(
        source_volume: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if source_volume in raw_alignments:
            return raw_alignments[source_volume]
        frame = read_frame(source_volume)
        if frame is None:
            return None
        valid = frame.neuron_ids >= 0
        estimate = estimate_pca_alignment_xy(frame.points[valid, :2])
        if estimate is not None:
            raw_alignments[source_volume] = estimate
        return estimate

    for source_volume in requested:
        estimate_alignment(source_volume)

    def find_neighbor(source_volume: int, direction: int) -> None:
        insertion_index = bisect_left(indexed_volumes, source_volume)
        if direction < 0:
            candidates = reversed(indexed_volumes[:insertion_index])
        else:
            first_after = insertion_index
            if (
                first_after < len(indexed_volumes)
                and indexed_volumes[first_after] == source_volume
            ):
                first_after += 1
            candidates = iter(indexed_volumes[first_after:])
        for candidate in candidates:
            if estimate_alignment(candidate) is not None:
                return

    for source_volume in requested:
        if source_volume not in raw_alignments:
            find_neighbor(source_volume, -1)
            find_neighbor(source_volume, 1)

    if not raw_alignments:
        raise ValueError(
            f"No realtime point cloud can define XY alignment in {realtime_dir}"
        )

    sample_volumes = sorted(raw_alignments)
    sample_centers: list[np.ndarray] = []
    sample_rotations: list[np.ndarray] = []
    previous_axis: np.ndarray | None = None
    for source_volume in sample_volumes:
        center, raw_axis = raw_alignments[source_volume]
        axis = np.asarray(raw_axis, dtype=np.float32).copy()
        if previous_axis is None:
            if axis[1] < 0 or (np.isclose(axis[1], 0.0) and axis[0] < 0):
                axis *= -1
        elif float(np.dot(previous_axis, axis)) < 0:
            axis *= -1
        previous_axis = axis
        sample_centers.append(center)
        sample_rotations.append(pca_axis_rotation_matrix(axis))

    centers_xy, rotations_xy = interpolate_alignment_xy(
        requested,
        sample_volumes,
        np.stack(sample_centers, axis=0),
        np.stack(sample_rotations, axis=0),
    )
    interpolated = tuple(
        source_volume
        for source_volume in requested
        if source_volume not in raw_alignments
    )
    return RoiData(
        points=dense_points,
        source_volumes=requested,
        centers_xy=centers_xy,
        rotations_xy=rotations_xy,
        missing_volumes=tuple(missing_volumes),
        interpolated_alignment_volumes=interpolated,
    )


def load_roi_data(
    source_path: Path,
    source_mode: RoiSourceMode,
    selected_volumes: Sequence[int],
    first_volume: int = 0,
    align_xy: bool = True,
) -> RoiData:
    """Load dense ROI frames and source-space alignment transforms."""

    if source_mode == "dynamics":
        return _load_dynamics_roi_data(
            source_path, selected_volumes, first_volume
        )
    if source_mode == "realtime-results":
        return _load_realtime_roi_data(
            source_path, selected_volumes, align_xy
        )
    raise ValueError(
        "ROI source mode must be 'dynamics' or 'realtime-results'"
    )


def coordinates_xyz(points: np.ndarray, coordinate_order: str) -> np.ndarray:
    """Return canonical XYZ columns from a point matrix."""

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Neuron points must have shape (N,F>=3), got {points.shape}")
    order = coordinate_order.lower()
    indices = [order.index(axis) for axis in "xyz"]
    return np.asarray(points[:, indices], dtype=np.float32)
