"""Export source ROI points to NPY without image loading or geometry changes.

Edit the settings below, then run ``pixi run preprocess-roi``. Output rows keep
the source columns and coordinate units. Realtime row i represents neuron ID i;
missing volumes and missing neurons remain NaN. No PCA, flips, Z reversal,
coordinate reordering, cropping, or rescaling is performed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import hdf5plugin  # Registers compression filters for source HDF5 files.
import numpy as np

from roi_source import (
    RoiSourceMode,
    _index_realtime_files,
    find_point_dataset,
    load_roi_data,
    point_frames,
)

# ---------------------------------------------------------------------------
# Frequently edited configuration
# ---------------------------------------------------------------------------

# Same mode names as RAW_ROI_SOURCE_MODE in script/launch_datasets.py.
RAW_ROI_SOURCE_MODE: RoiSourceMode = "dynamics" # "dynamics" or "realtime-results"
RAW_ROI_SOURCE_PATH = Path(r"Z:\data5\CBMI_inferred_results\WJH\20260923\w1\dynamics.h5")
OUTPUT_DIR = Path("data/20260923_w1")

# None or [] selects every source volume automatically. Realtime mode uses
# range(0, max_file_volume + 1), including gaps; dynamics uses all point groups.
# Set an explicit range to include missing volumes AFTER the last realtime file,
# and keep the requested volume order identical to the Image time axis.
SELECTED_VOLUMES: Sequence[int] | None = None

# Fallback numbering for nonnumeric dynamics group names; numeric names retain
# their source volume IDs. Ignored in realtime-results mode.
DYNAMICS_FIRST_VOLUME = 0


def _validate_selected_volumes(selected_volumes: Sequence[int]) -> tuple[int, ...]:
    if len(selected_volumes) == 0:
        raise ValueError("SELECTED_VOLUMES must not be empty")
    try:
        selected = tuple(int(volume) for volume in selected_volumes)
        invalid = any(
            number < 0 or number != original
            for number, original in zip(selected, selected_volumes, strict=True)
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("Selected volume numbers must be non-negative integers") from error
    if invalid:
        raise ValueError("Selected volume numbers must be non-negative integers")
    if len(set(selected)) != len(selected):
        raise ValueError("SELECTED_VOLUMES contains duplicates")
    return selected


def _dynamics_names_by_volume(
    dynamics: h5py.File,
    first_volume: int,
) -> dict[int, str]:
    frame_names = point_frames(dynamics)
    if not frame_names:
        raise ValueError(f"No supported neuron point datasets found in {dynamics.filename}")
    names_by_volume: dict[int, str] = {}
    for frame_index, name in enumerate(frame_names):
        volume = int(name) if name.isdigit() else first_volume + frame_index
        if volume in names_by_volume:
            raise ValueError("dynamics.h5 contains duplicate source volume IDs")
        names_by_volume[volume] = name
    return names_by_volume


def _resolve_selected_volumes(
    source_path: Path,
    source_mode: RoiSourceMode,
    selected_volumes: Sequence[int] | None,
    first_volume: int,
) -> tuple[int, ...]:
    if selected_volumes is not None and len(selected_volumes) > 0:
        return _validate_selected_volumes(selected_volumes)
    if source_mode == "realtime-results":
        files_by_volume = _index_realtime_files(source_path)
        return tuple(range(max(files_by_volume) + 1))
    with h5py.File(source_path, "r") as dynamics:
        return _validate_selected_volumes(
            tuple(_dynamics_names_by_volume(dynamics, first_volume))
        )


def _load_dynamics_points(
    source_path: Path,
    selected_volumes: Sequence[int],
    first_volume: int,
) -> np.ndarray:
    """Read only selected point datasets, without requiring center/rot fields."""

    frames: list[np.ndarray] = []
    with h5py.File(source_path, "r") as dynamics:
        names_by_volume = _dynamics_names_by_volume(dynamics, first_volume)
        missing = [volume for volume in selected_volumes if volume not in names_by_volume]
        if missing:
            raise KeyError(f"Volumes missing from dynamics.h5: {missing}")

        expected_shape: tuple[int, ...] | None = None
        for volume in selected_volumes:
            dataset = find_point_dataset(dynamics[names_by_volume[volume]])
            if dataset is None:
                raise RuntimeError(f"Point dataset is missing for dynamics volume {volume}")
            if dataset.ndim != 2 or dataset.shape[1] < 6:
                raise ValueError(
                    f"Point dataset {dataset.name} must have shape (N,K>=6), "
                    f"got {dataset.shape}"
                )
            if expected_shape is None:
                expected_shape = dataset.shape
            elif dataset.shape != expected_shape:
                raise ValueError(
                    f"Point dataset {dataset.name} has shape {dataset.shape}; "
                    f"expected {expected_shape}"
                )
            frames.append(np.asarray(dataset[()], dtype=np.float32))
    return np.stack(frames, axis=0)


def export_roi(
    source_path: Path,
    source_mode: RoiSourceMode,
    selected_volumes: Sequence[int] | None,
    output_path: Path,
    first_volume: int = 0,
) -> None:
    """Export (T,N,K>=6) ROIs; None/empty selection includes all source volumes."""

    if source_mode not in ("dynamics", "realtime-results"):
        raise ValueError("RAW_ROI_SOURCE_MODE must be 'dynamics' or 'realtime-results'")
    if source_mode == "dynamics" and not source_path.is_file():
        raise FileNotFoundError(f"Dynamics source does not exist: {source_path}")
    if source_mode == "realtime-results" and not source_path.is_dir():
        raise FileNotFoundError(f"Realtime results directory does not exist: {source_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}; choose a new OUTPUT_DIR")
    selected = _resolve_selected_volumes(
        source_path, source_mode, selected_volumes, first_volume
    )

    missing_volumes: tuple[int, ...] = ()
    if source_mode == "dynamics":
        points = _load_dynamics_points(source_path, selected, first_volume)
    else:
        # Disabling alignment returns source coordinates directly. Missing files,
        # groups, point datasets, and neuron IDs remain NaN without interpolation.
        dataset = load_roi_data(source_path, source_mode, selected, align_xy=False)
        points = dataset.points
        missing_volumes = dataset.missing_volumes

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".roi-export-", dir=output_path.parent) as temp_dir:
        temporary_path = Path(temp_dir) / output_path.name
        with temporary_path.open("wb") as output_file:
            np.save(output_file, points, allow_pickle=False)
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite {output_path}; choose a new OUTPUT_DIR")
        temporary_path.replace(output_path)

    print(f"ROI source:      {source_mode}  {source_path}")
    print(f"Volumes:         selected={len(selected)}  missing={len(missing_volumes)}")
    print(
        f"Saved ROI:       {output_path}  shape={points.shape}  "
        f"dtype={points.dtype}  size={points.nbytes / 1024**2:.1f} MiB"
    )


def main() -> None:
    export_roi(
        source_path=RAW_ROI_SOURCE_PATH,
        source_mode=RAW_ROI_SOURCE_MODE,
        selected_volumes=SELECTED_VOLUMES,
        output_path=OUTPUT_DIR / "neuron_point_tuple.npy",
        first_volume=DYNAMICS_FIRST_VOLUME,
    )


if __name__ == "__main__":
    main()
