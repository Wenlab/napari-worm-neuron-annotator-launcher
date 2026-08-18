"""Apply matching geometry transforms to image volumes and ROI points."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import affine_transform


def image_xy_rotation_matrix(angle_degrees: float) -> np.ndarray:
    """Return a row-vector XY rotation matrix for image coordinates."""

    angle_radians = np.deg2rad(float(angle_degrees))
    cosine = float(np.cos(angle_radians))
    sine = float(np.sin(angle_radians))
    return np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float32)


def estimate_pca_alignment_xy(
    points_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return median XY center and the first PCA axis for finite points."""

    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"XY points must have shape (N,2), got {points.shape}")
    finite_points = points[np.isfinite(points).all(axis=1)]
    if finite_points.shape[0] < 2:
        return None

    center = np.median(finite_points, axis=0)
    centered = finite_points - center
    _, singular_values, right_vectors = np.linalg.svd(
        centered, full_matrices=False
    )
    if singular_values.size == 0 or singular_values[0] <= np.finfo(np.float32).eps:
        return None
    principal_axis = right_vectors[0]
    return center.astype(np.float32), principal_axis.astype(np.float32)


def pca_axis_rotation_matrix(principal_axis: np.ndarray) -> np.ndarray:
    """Return the row-vector rotation that maps a PCA axis to positive Y."""

    axis = np.asarray(principal_axis, dtype=np.float64)
    if axis.shape != (2,) or not np.isfinite(axis).all():
        raise ValueError(f"Principal axis must be finite with shape (2,), got {axis}")
    norm = float(np.linalg.norm(axis))
    if norm <= np.finfo(np.float32).eps:
        raise ValueError("Principal axis must have non-zero length")
    x_component, y_component = axis / norm
    return np.asarray(
        [[y_component, x_component], [-x_component, y_component]],
        dtype=np.float32,
    )


def interpolate_alignment_xy(
    target_volumes: Sequence[int],
    sample_volumes: Sequence[int],
    sample_centers_xy: np.ndarray,
    sample_rotations_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate centers and continuous rotation angles at target volumes."""

    targets = np.asarray(target_volumes, dtype=np.float64)
    samples = np.asarray(sample_volumes, dtype=np.float64)
    centers = np.asarray(sample_centers_xy, dtype=np.float64)
    rotations = np.asarray(sample_rotations_xy, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("At least one alignment sample is required")
    if centers.shape != (samples.size, 2):
        raise ValueError(
            f"Alignment centers must have shape ({samples.size},2), "
            f"got {centers.shape}"
        )
    if rotations.shape != (samples.size, 2, 2):
        raise ValueError(
            f"Alignment rotations must have shape ({samples.size},2,2), "
            f"got {rotations.shape}"
        )
    if np.any(np.diff(samples) <= 0):
        raise ValueError("Alignment sample volumes must be strictly increasing")

    interpolated_centers = np.column_stack(
        [np.interp(targets, samples, centers[:, axis]) for axis in range(2)]
    )
    sample_angles = np.unwrap(
        np.arctan2(rotations[:, 0, 1], rotations[:, 0, 0])
    )
    target_angles = np.interp(targets, samples, sample_angles)
    interpolated_rotations = np.stack(
        [
            image_xy_rotation_matrix(np.rad2deg(angle))
            for angle in target_angles
        ],
        axis=0,
    )
    return (
        interpolated_centers.astype(np.float32),
        interpolated_rotations.astype(np.float32),
    )


def output_center_xy(image_shape_yx: Sequence[int]) -> np.ndarray:
    """Return the XY center used by image and point transforms."""

    height, width = (int(value) for value in image_shape_yx)
    return np.asarray([width / 2.0, height / 2.0], dtype=np.float32)


def align_points_xy(
    points_xy: np.ndarray,
    center_xy: np.ndarray,
    rotation_xy: np.ndarray,
    image_shape_yx: Sequence[int],
) -> np.ndarray:
    """Map raw row-vector XY points to the centered aligned canvas."""

    return (
        np.asarray(points_xy, dtype=np.float32) - center_xy
    ) @ rotation_xy + output_center_xy(image_shape_yx)


def flip_points_x(points_xy: np.ndarray, image_width: int) -> np.ndarray:
    """Mirror XY point coordinates like ``image[..., ::-1]``."""

    mirrored = np.asarray(points_xy, dtype=np.float32).copy()
    mirrored[:, 0] = (int(image_width) - 1) - mirrored[:, 0]
    return mirrored


def flip_points_y(points_xy: np.ndarray, image_height: int) -> np.ndarray:
    """Mirror XY point coordinates like ``image[:, ::-1, :]``."""

    mirrored = np.asarray(points_xy, dtype=np.float32).copy()
    mirrored[:, 1] = (int(image_height) - 1) - mirrored[:, 1]
    return mirrored


def transform_points_xy(
    points_xy: np.ndarray,
    center_xy: np.ndarray,
    rotation_xy: np.ndarray,
    image_shape_yx: Sequence[int],
    align_xy: bool,
    flip_x: bool,
    flip_y: bool,
) -> np.ndarray:
    """Apply the configured XY transforms to point coordinates."""

    height, width = (int(value) for value in image_shape_yx)
    transformed = np.asarray(points_xy, dtype=np.float32).copy()
    if align_xy:
        transformed = align_points_xy(
            transformed, center_xy, rotation_xy, (height, width)
        )
    if flip_x:
        transformed = flip_points_x(transformed, width)
    if flip_y:
        transformed = flip_points_y(transformed, height)
    return transformed


def transform_points_z_scaled(
    z_scaled: np.ndarray,
    z_scale_ratio: float,
    z_start_frame: int,
    z_end_frame: int,
    reverse_z: bool,
) -> np.ndarray:
    """Map source scaled-Z values into the exported stack coordinates."""

    transformed = np.asarray(z_scaled, dtype=np.float32).copy()
    transformed -= z_start_frame * z_scale_ratio
    if reverse_z:
        exported_span_scaled = (z_end_frame - z_start_frame) * z_scale_ratio
        transformed = exported_span_scaled - transformed
    return transformed


def align_volume_xy(
    volume_zyx: np.ndarray,
    center_xy: np.ndarray,
    rotation_xy: np.ndarray,
    interpolation_order: int = 1,
) -> np.ndarray:
    """Apply the point-cloud XY transform to every image slice."""

    volume = np.asarray(volume_zyx)
    if volume.ndim != 3:
        raise ValueError(f"Volume must have shape (Z,Y,X), got {volume.shape}")

    swap_xy_yx = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    rotation = np.asarray(rotation_xy, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    output_center = output_center_xy(volume.shape[1:]).astype(np.float64)
    inverse_matrix_yx = swap_xy_yx @ rotation @ swap_xy_yx
    inverse_offset_yx = swap_xy_yx @ (center - rotation @ output_center)

    matrix_zyx = np.eye(3, dtype=np.float64)
    matrix_zyx[1:, 1:] = inverse_matrix_yx
    offset_zyx = np.asarray(
        [0.0, inverse_offset_yx[0], inverse_offset_yx[1]],
        dtype=np.float64,
    )
    return affine_transform(
        np.asarray(volume, dtype=np.float32),
        matrix=matrix_zyx,
        offset=offset_zyx,
        output_shape=volume.shape,
        order=interpolation_order,
        mode="constant",
        cval=0.0,
        prefilter=interpolation_order > 1,
    )


def build_alignment_map(
    source_volume_numbers: Sequence[int],
    centers_xy: np.ndarray,
    rotations_xy: np.ndarray,
    goal_angle_degrees: float,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return ``source volume -> (center XY, effective rotation XY)``."""

    goal_rotation = image_xy_rotation_matrix(goal_angle_degrees)
    return {
        int(volume): (centers_xy[index], rotations_xy[index] @ goal_rotation)
        for index, volume in enumerate(source_volume_numbers)
    }
