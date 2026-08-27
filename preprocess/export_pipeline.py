"""Export prepared raw Image and ROI arrays as launcher-compatible NPY files."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from raw_dataset_loader import (
    RawDatasetConfig,
    iter_transformed_volumes,
    prepare_raw_inputs,
    _transform_rois,
)


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
    """Replace a regular NPY output only after its contents are complete."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("wb") as output_file:
            np.save(output_file, array, allow_pickle=False)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_volume(
    image_output: np.memmap,
    mask_output: np.memmap | None,
    local_t: int,
    source_volume: int,
    expected_source_volume: int,
    volume: np.ndarray,
    roi_points: np.ndarray,
    volume_shape: tuple[int, int, int],
    z_scale_ratio: float,
) -> None:
    """Write one transformed Image volume and optional Labels volume."""

    if source_volume != expected_source_volume:
        raise ValueError(
            f"Unexpected source volume {source_volume} at output index {local_t}; "
            f"expected {expected_source_volume}"
        )
    if volume.shape != volume_shape:
        raise ValueError(
            f"Volume {source_volume} shape {volume.shape} does not match "
            f"{volume_shape}"
        )
    image_output[local_t] = volume
    if mask_output is not None:
        mask_output[local_t] = _rasterize_roi_mask(
            roi_points,
            volume_shape,
            z_scale_ratio,
        )
    # Flush each completed volume so dirty mapped pages do not accumulate
    # across a long export on a local or SMB-backed output volume.
    image_output.flush()
    if mask_output is not None:
        mask_output.flush()


def _commit_staged_outputs(
    staging_dir: Path,
    output_dir: Path,
    save_mask: bool,
) -> None:
    """Replace final outputs only after all staging files are complete."""

    (staging_dir / "volumes.npy").replace(output_dir / "volumes.npy")
    (staging_dir / "neuron_point_tuple.npy").replace(
        output_dir / "neuron_point_tuple.npy"
    )
    mask_path = output_dir / "neuron_mask.npy"
    if save_mask:
        (staging_dir / "neuron_mask.npy").replace(mask_path)
    else:
        mask_path.unlink(missing_ok=True)


def _close_memmap(array: np.memmap | None) -> None:
    """Flush and close a NPY mapping, including on Windows."""

    if array is None:
        return
    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not getattr(mapping, "closed", False):
        mapping.close()


def export_dataset(config: ExportConfig) -> None:
    """Stream raw inputs into launcher-compatible NPY files."""

    selected, roi_data, alignment_by_volume = prepare_raw_inputs(config.source)
    print(f"Source volumes: {selected}")
    print(f"TIFF path:      {config.source.tiff_path}")
    print(f"ROI mode:       {config.source.roi_source_mode}")
    print(f"ROI path:       {config.source.roi_source_path}")
    print(f"Output folder:  {config.output_dir}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter_transformed_volumes(
        config.source,
        selected,
        alignment_by_volume,
    )
    try:
        with TemporaryDirectory(
            prefix=".export-",
            dir=config.output_dir,
        ) as temporary_dir:
            staging_dir = Path(temporary_dir)
            try:
                first_local_t, first_source_volume, first_volume = next(iterator)
            except StopIteration as error:
                raise RuntimeError("No source volumes were read") from error

            volume_shape = tuple(int(size) for size in first_volume.shape)
            image_shape_yx = volume_shape[1:]
            roi = _transform_rois(
                config.source,
                selected,
                roi_data.points,
                roi_data.source_volumes,
                alignment_by_volume,
                image_shape_yx,
            )
            image_shape = (len(selected), *volume_shape)
            image_size_gib = (
                int(np.prod(image_shape, dtype=np.int64))
                * np.dtype(np.float32).itemsize
                / 1024**3
            )
            print(
                f"Streaming Image: shape={image_shape}  dtype=float32  "
                f"size={image_size_gib:.1f} GiB"
            )

            image_output: np.memmap | None = None
            mask_output: np.memmap | None = None
            try:
                image_output = np.lib.format.open_memmap(
                    staging_dir / "volumes.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=image_shape,
                )
                if config.save_mask:
                    mask_output = np.lib.format.open_memmap(
                        staging_dir / "neuron_mask.npy",
                        mode="w+",
                        dtype=np.int16,
                        shape=image_shape,
                    )

                _write_volume(
                    image_output,
                    mask_output,
                    first_local_t,
                    first_source_volume,
                    selected[first_local_t],
                    first_volume,
                    roi[first_local_t],
                    volume_shape,
                    config.source.z_scale_ratio,
                )
                del first_volume

                written_count = 1
                for local_t, source_volume, volume in iterator:
                    _write_volume(
                        image_output,
                        mask_output,
                        local_t,
                        source_volume,
                        selected[local_t],
                        volume,
                        roi[local_t],
                        volume_shape,
                        config.source.z_scale_ratio,
                    )
                    written_count += 1
                    del volume
                if written_count != len(selected):
                    raise RuntimeError(
                        f"Wrote {written_count} volumes; expected {len(selected)}"
                    )
            finally:
                _close_memmap(image_output)
                _close_memmap(mask_output)
                image_output = None
                mask_output = None
                gc.collect()

            save_npy_atomic(staging_dir / "neuron_point_tuple.npy", roi)
            _commit_staged_outputs(
                staging_dir,
                config.output_dir,
                config.save_mask,
            )

            print(
                f"\nSaved Image: {config.output_dir / 'volumes.npy'}  "
                f"shape={image_shape}  dtype=float32"
            )
            print(
                f"Saved ROI: {config.output_dir / 'neuron_point_tuple.npy'}  "
                f"shape={roi.shape}  dtype={roi.dtype}"
            )
            if config.save_mask:
                print(
                    f"Saved optional Labels: "
                    f"{config.output_dir / 'neuron_mask.npy'}  "
                    f"shape={image_shape}  dtype=int16"
                )
            print("Done.")
    finally:
        iterator.close()
