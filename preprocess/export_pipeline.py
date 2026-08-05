"""Export prepared raw Image and ROI arrays as launcher-compatible NPY files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from raw_dataset_loader import RawDatasetConfig, load_raw_dataset


@dataclass(frozen=True)
class ExportConfig:
    """Raw source plus output-only settings for one dataset export."""

    source: RawDatasetConfig
    output_dir: Path
    save_mask: bool


def _rasterize_roi_mask(
    points: np.ndarray,
    volume_shape: tuple[int, int, int],
    z_scale_ratio: float,
) -> np.ndarray:
    z_dim, y_dim, x_dim = volume_shape
    mask = np.zeros(volume_shape, dtype=np.int16)
    for neuron_id, row in enumerate(points):
        x_value, y_value, z_value = (float(value) for value in row[:3])
        width = float(row[3])
        height = float(row[4])
        depth_scaled = float(row[5])
        values = np.asarray(
            [x_value, y_value, z_value, width, height, depth_scaled]
        )
        if not np.isfinite(values).all():
            continue
        if width <= 0 or height <= 0 or depth_scaled <= 0:
            continue

        z_center = round(z_value / z_scale_ratio)
        depth_layers = max(1, int(np.ceil(depth_scaled / z_scale_ratio)))
        half_depth = depth_layers / 2.0
        z_min = max(0, int(np.ceil(z_center - half_depth)))
        z_max = min(z_dim, int(np.ceil(z_center + half_depth)))
        x_min = max(0, int(x_value - width / 2.0))
        x_max = min(x_dim, int(x_value + width / 2.0))
        y_min = max(0, int(y_value - height / 2.0))
        y_max = min(y_dim, int(y_value + height / 2.0))
        if z_min < z_max and y_min < y_max and x_min < x_max:
            mask[z_min:z_max, y_min:y_max, x_min:x_max] = neuron_id + 1
    return mask


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    """Replace one NPY output only after its complete contents are written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("wb") as output_file:
            np.save(output_file, array)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_outputs(
    output_dir: Path,
    volumes: np.ndarray,
    roi: np.ndarray,
    masks: np.ndarray | None,
) -> None:
    image_path = output_dir / "volumes.npy"
    roi_path = output_dir / "neuron_point_tuple.npy"
    mask_path = output_dir / "neuron_mask.npy"

    output_dir.mkdir(parents=True, exist_ok=True)
    save_npy_atomic(image_path, volumes)
    save_npy_atomic(roi_path, roi)
    if masks is not None:
        save_npy_atomic(mask_path, masks)
    else:
        mask_path.unlink(missing_ok=True)

    print(
        f"\nSaved Image: {image_path}  shape={volumes.shape}  dtype={volumes.dtype}"
    )
    print(f"Saved ROI: {roi_path}  shape={roi.shape}  dtype={roi.dtype}")
    if masks is not None:
        print(
            f"Saved optional Labels: {mask_path}  "
            f"shape={masks.shape}  dtype={masks.dtype}"
        )
    print("Done.")


def export_dataset(config: ExportConfig) -> None:
    """Convert one configured TIFF/dynamics source into launcher inputs."""

    selected = [int(volume) for volume in config.source.selected_volumes]
    print(f"Source volumes: {selected}")
    print(f"TIFF path:      {config.source.tiff_path}")
    print(f"Dynamics path:  {config.source.dynamics_path}")
    print(f"Output folder:  {config.output_dir}")

    dataset = load_raw_dataset(config.source, mode="eager")
    if not isinstance(dataset.volumes, np.ndarray):
        raise TypeError("NPY export requires an eager NumPy Image array")

    masks = None
    if config.save_mask:
        volume_shape = tuple(int(size) for size in dataset.volumes.shape[1:])
        masks = np.stack(
            [
                _rasterize_roi_mask(
                    points,
                    volume_shape,
                    config.source.z_scale_ratio,
                )
                for points in dataset.roi
            ],
            axis=0,
        )
    _save_outputs(config.output_dir, dataset.volumes, dataset.roi, masks)
