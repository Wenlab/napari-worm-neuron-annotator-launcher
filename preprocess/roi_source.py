"""Read neuron ROI points and alignment data from dynamics HDF5 files."""

from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

POINT_DATASET_CANDIDATES = (
    "d_neuron_pt_tuple_matched_raw_vol",
    "neuron_pt_tuple",
)
CENTER_DATASET = "center"
ROTATION_DATASET = "rot"


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


def load_roi_data(
    dynamics_path: Path,
    first_volume: int = 0,
) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
    """Load point arrays and per-volume XY alignment transforms."""

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


def coordinates_xyz(points: np.ndarray, coordinate_order: str) -> np.ndarray:
    """Return canonical XYZ columns from a point matrix."""

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Neuron points must have shape (N,F>=3), got {points.shape}")
    order = coordinate_order.lower()
    indices = [order.index(axis) for axis in "xyz"]
    return np.asarray(points[:, indices], dtype=np.float32)
