"""Summarize final corrections in Worm Neuron Annotator sidecar JSON files."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

APP_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = APP_DIR.parent

# ---------------------------------------------------------------------------
# Editable settings
# ---------------------------------------------------------------------------

# Each item is (proofread JSON, matching raw ROI NPY or None). The raw NPY
# gives an exact denominator if some (volume, neuron) slots are invalid and is
# required for schema-v1 JSON, which does not contain ``changed_fields``.
PROOFREADING_INPUTS: tuple[tuple[Path, Path | None], ...] = (
    (
        REPOSITORY_DIR
        / "data"
        / "20260304_w3"
        / "neuron_point_tuple.proofread.json",
        None,
    ),
)
OUTPUT_DIR = REPOSITORY_DIR / "data" / "proofreading_statistics"

# Use "complete" only when every raw neuron ID in the selected volume range
# was proofread. "partial" avoids treating unreviewed IDs as correct.
ProofreadingScope = Literal["partial", "complete"]
PROOFREADING_SCOPE: ProofreadingScope = "partial"

# In partial mode, None infers IDs that occur in the sidecar's patches or
# global markers. Explicitly list reviewed-but-unchanged IDs here because an
# unchanged ID leaves no trace in the sparse JSON.
PARTIAL_NEURON_IDS: tuple[int, ...] | None = None

# Optional raw NPY volume-index range [start, stop). None analyzes all volumes.
VOLUME_RANGE: tuple[int, int] | None = (0, 1000)
# One complete raw NPY volume defines the anatomical point cloud. This is
# intentionally independent of VOLUME_RANGE, which only controls error rates.
POINT_CLOUD_VOLUME_INDEX = 551
# Orient the representative cloud before drawing the three 2D views.
POINT_CLOUD_ALIGN_XY = True
POINT_CLOUD_ROTATE_CCW_DEGREES = 90.0
POINT_CLOUD_FLIP_X = True
POINT_CLOUD_FLIP_Y = False
REVIEWED_POINT_SIZE = 8
UNREVIEWED_POINT_SIZE = 6
VERIFY_RAW_SHA256 = True
RAW_COUNT_CHUNK_VOLUMES = 256


@dataclass(frozen=True)
class Patch:
    volume_index: int
    neuron_id: int
    state: str
    fields: tuple[str, ...] | None
    center_zyx: tuple[float, float, float] | None = None
    size_zyx: tuple[float, float, float] | None = None
    implicit_delete_all: bool = False


@dataclass(frozen=True)
class Sidecar:
    path: Path
    schema_version: int
    raw_shape: tuple[int, int, int]
    raw_dtype: str
    raw_sha256: str
    z_divisor: float
    patches: tuple[Patch, ...]
    delete_all_ids: frozenset[int]
    placement_size_ids: frozenset[int]
    added_ids: frozenset[int]
    retired_ids: frozenset[int]

    @property
    def raw_t(self) -> int:
        return self.raw_shape[0]

    @property
    def raw_n(self) -> int:
        return self.raw_shape[1]


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-number list")
    vector = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{name} must contain finite numbers")
    return vector


def _id_set(value: Any, name: str) -> frozenset[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise ValueError(f"{name} must be a list of non-negative integers")
    result = frozenset(value)
    if len(result) != len(value):
        raise ValueError(f"{name} contains duplicate IDs")
    return result


def _load_sidecar(path: Path) -> Sidecar:
    if not path.is_file():
        raise FileNotFoundError(f"proofread JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(
            stream,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    if not isinstance(payload, dict):
        raise ValueError("sidecar root must be an object")

    schema_version = payload.get("schema_version")
    if schema_version not in (1, 2):
        raise ValueError(f"unsupported schema_version: {schema_version!r}")
    raw = payload.get("raw")
    if not isinstance(raw, dict):
        raise ValueError("sidecar raw metadata is missing")
    shape_value = raw.get("shape")
    if (
        not isinstance(shape_value, list)
        or len(shape_value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in shape_value
        )
    ):
        raise ValueError("raw.shape must be [T, N, K]")
    raw_shape = tuple(shape_value)
    if raw_shape[0] <= 0 or raw_shape[1] <= 0 or raw_shape[2] < 6:
        raise ValueError("raw.shape must describe non-empty (T, N, K>=6)")
    z_divisor = float(raw.get("z_divisor"))
    if not math.isfinite(z_divisor) or z_divisor <= 0:
        raise ValueError("raw.z_divisor must be positive and finite")

    added_value = payload.get("added_neurons")
    if not isinstance(added_value, dict):
        raise ValueError("added_neurons must be an object")
    added_ids = _id_set(added_value.get("committed"), "added_neurons.committed")
    retired_ids = _id_set(added_value.get("retired"), "added_neurons.retired")
    delete_all_ids = _id_set(payload.get("delete_all_ids"), "delete_all_ids")
    placement_value = payload.get("placement_size")
    if not isinstance(placement_value, dict):
        raise ValueError("placement_size must be an object")
    placement_size_ids = frozenset(int(key) for key in placement_value)

    records = payload.get("observation_patches")
    if not isinstance(records, list):
        raise ValueError("observation_patches must be a list")
    patches: list[Patch] = []
    seen: set[tuple[int, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each observation patch must be an object")
        volume_index = record.get("volume_index")
        neuron_id = record.get("neuron_id")
        state = record.get("state")
        if (
            isinstance(volume_index, bool)
            or not isinstance(volume_index, int)
            or not 0 <= volume_index < raw_shape[0]
        ):
            raise ValueError(f"invalid patch volume_index: {volume_index!r}")
        if isinstance(neuron_id, bool) or not isinstance(neuron_id, int):
            raise ValueError(f"invalid patch neuron_id: {neuron_id!r}")
        key = (volume_index, neuron_id)
        if key in seen:
            raise ValueError(f"duplicate observation patch: {key}")
        seen.add(key)

        center = size = None
        if state == "present":
            box = record.get("box")
            if not isinstance(box, dict):
                raise ValueError(f"present patch {key} has no box")
            center = _vector3(box.get("center_zyx"), f"patch {key} center")
            size = _vector3(box.get("size_zyx"), f"patch {key} size")
            if any(item <= 0 for item in size):
                raise ValueError(f"patch {key} size must be positive")
        elif state != "deleted":
            raise ValueError(f"unknown patch state at {key}: {state!r}")

        fields = None
        if schema_version == 2:
            fields_value = record.get("changed_fields")
            canonical = [
                field
                for field in ("presence", "center_zyx", "size_zyx")
                if isinstance(fields_value, list) and field in fields_value
            ]
            if not fields_value or canonical != fields_value:
                raise ValueError(f"invalid changed_fields at patch {key}")
            fields = tuple(canonical)
        patches.append(Patch(volume_index, neuron_id, state, fields, center, size))

    return Sidecar(
        path=path,
        schema_version=schema_version,
        raw_shape=raw_shape,
        raw_dtype=str(raw.get("dtype")),
        raw_sha256=str(raw.get("sha256")),
        z_divisor=z_divisor,
        patches=tuple(patches),
        delete_all_ids=delete_all_ids,
        placement_size_ids=placement_size_ids,
        added_ids=added_ids,
        retired_ids=retired_ids,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_raw_roi(path: Path | None, sidecar: Sidecar) -> np.ndarray | None:
    if path is None:
        if sidecar.schema_version == 1:
            raise ValueError(
                "schema-v1 JSON has no changed_fields; configure its raw ROI NPY"
            )
        return None
    raw_roi = np.load(path, mmap_mode="r", allow_pickle=False)
    if raw_roi.shape != sidecar.raw_shape or raw_roi.dtype.str != sidecar.raw_dtype:
        raise ValueError("raw ROI shape or dtype does not match the JSON metadata")
    if VERIFY_RAW_SHA256 and _sha256(path) != sidecar.raw_sha256:
        raise ValueError("raw ROI SHA-256 does not match the JSON metadata")
    return raw_roi


def _raw_geometry(
    raw_roi: np.ndarray, sidecar: Sidecar, volume_index: int, neuron_id: int
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    values = np.asarray(raw_roi[volume_index, neuron_id, :6], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values[3:6] <= 0):
        return None
    x, y, z_scaled, width, height, depth_scaled = values
    return (
        (z_scaled / sidecar.z_divisor, y, x),
        (depth_scaled / sidecar.z_divisor, height, width),
    )


def _resolve_fields(
    sidecar: Sidecar, raw_roi: np.ndarray | None
) -> tuple[Patch, ...]:
    if raw_roi is None:
        return sidecar.patches
    resolved: list[Patch] = []
    for patch in sidecar.patches:
        if patch.state == "deleted":
            derived = ("presence",)
        elif patch.neuron_id >= sidecar.raw_n:
            derived = ("presence",)
        else:
            raw_geometry = _raw_geometry(
                raw_roi, sidecar, patch.volume_index, patch.neuron_id
            )
            if raw_geometry is None:
                derived = ("presence",)
            else:
                raw_center, raw_size = raw_geometry
                fields = []
                if patch.neuron_id in sidecar.delete_all_ids:
                    fields.append("presence")
                if patch.center_zyx != raw_center:
                    fields.append("center_zyx")
                if patch.size_zyx != raw_size:
                    fields.append("size_zyx")
                derived = tuple(fields)
        if patch.fields is not None and patch.fields != derived:
            raise ValueError(
                f"changed_fields disagrees with raw ROI at "
                f"({patch.volume_index}, {patch.neuron_id})"
            )
        if derived:
            resolved.append(replace(patch, fields=derived))
    return tuple(resolved)


def _analysis_neuron_ids(sidecar: Sidecar) -> tuple[int, ...]:
    if PROOFREADING_SCOPE == "complete":
        return tuple(range(sidecar.raw_n))
    if PROOFREADING_SCOPE != "partial":
        raise ValueError("PROOFREADING_SCOPE must be 'partial' or 'complete'")

        neuron_ids = {
            patch.neuron_id
        for patch in sidecar.patches
            if patch.neuron_id < sidecar.raw_n
        }
        neuron_ids.update(
            neuron_id
            for neuron_id in sidecar.delete_all_ids | sidecar.placement_size_ids
            if neuron_id < sidecar.raw_n
        )
    if not neuron_ids:
        raise ValueError(
            "no reviewed raw neuron IDs can be inferred from this sidecar"
        )
    return tuple(sorted(neuron_ids))


def _volume_bounds(raw_t: int) -> tuple[int, int]:
    if VOLUME_RANGE is None:
        return 0, raw_t
    start, stop = VOLUME_RANGE
    if not 0 <= start < stop <= raw_t:
        raise ValueError(f"VOLUME_RANGE must satisfy 0 <= start < stop <= {raw_t}")
    return start, stop


def _eligible_counts(
    raw_roi: np.ndarray | None,
    sidecar: Sidecar,
    neuron_ids: tuple[int, ...],
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    if raw_roi is None:
        per_volume = np.full(stop - start, len(neuron_ids), dtype=np.int64)
        per_neuron = np.zeros(sidecar.raw_n, dtype=np.int64)
        per_neuron[list(neuron_ids)] = stop - start
        return per_volume, per_neuron
    per_volume = np.zeros(stop - start, dtype=np.int64)
    per_neuron = np.zeros(sidecar.raw_n, dtype=np.int64)
    chunk_size = max(1, RAW_COUNT_CHUNK_VOLUMES)
    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, stop)
        values = np.asarray(
            raw_roi[chunk_start:chunk_stop, list(neuron_ids), :6],
            dtype=float,
        )
        valid = np.all(np.isfinite(values), axis=2)
        valid &= np.all(values[:, :, 3:6] > 0, axis=2)
        per_volume[chunk_start - start : chunk_stop - start] = valid.sum(axis=1)
        per_neuron[list(neuron_ids)] += valid.sum(axis=0)
    return per_volume, per_neuron


def _selected_events(
    patches: tuple[Patch, ...],
    sidecar: Sidecar,
    raw_roi: np.ndarray | None,
    neuron_ids: tuple[int, ...],
    start: int,
    stop: int,
) -> list[Patch]:
    included_ids = set(neuron_ids) | set(sidecar.added_ids)
    events = [
        patch
        for patch in patches
        if patch.neuron_id in included_ids
        and start <= patch.volume_index < stop
    ]
    explicit = {(patch.volume_index, patch.neuron_id) for patch in patches}
    for neuron_id in sorted(sidecar.delete_all_ids):
        if neuron_id not in included_ids:
            continue
        for volume_index in range(start, stop):
            if (volume_index, neuron_id) in explicit:
                continue
            if raw_roi is not None and _raw_geometry(
                raw_roi, sidecar, volume_index, neuron_id
            ) is None:
                continue
            events.append(
                Patch(
                    volume_index,
                    neuron_id,
                    "deleted",
                    ("presence",),
                    implicit_delete_all=True,
                )
            )
    return sorted(events, key=lambda event: (event.volume_index, event.neuron_id))


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _accuracy(moved: int, eligible: int) -> float | None:
    rate = _ratio(moved, eligible)
    return None if rate is None else 1.0 - rate


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _report_name(path: Path) -> str:
    text = f"{path.parent.name}__{path.stem}"
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in text)


def _raw_point_cloud(raw_roi: np.ndarray, sidecar: Sidecar) -> np.ndarray:
    """Return XYZ from one complete raw NPY volume without temporal averaging."""
    volume_index = POINT_CLOUD_VOLUME_INDEX
    if not 0 <= volume_index < sidecar.raw_t:
        raise ValueError(
            f"POINT_CLOUD_VOLUME_INDEX must be within [0, {sidecar.raw_t})"
        )
    values = np.asarray(raw_roi[volume_index, :, :6], dtype=float)
    valid = np.all(np.isfinite(values), axis=1)
    valid &= np.all(values[:, 3:6] > 0, axis=1)
    centers_xyz = np.full((sidecar.raw_n, 3), np.nan, dtype=float)
    centers_xyz[valid] = values[valid, :3]
    return centers_xyz


def _rotate_xy(
    points_xy: np.ndarray, center_xy: np.ndarray, degrees_ccw: float
) -> np.ndarray:
    angle = math.radians(float(degrees_ccw))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=float)
    return (points_xy - center_xy) @ rotation.T + center_xy


def _orient_point_cloud_xy(centers_xyz: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply deterministic PCA orientation, rotation, and flips in XY."""
    oriented = np.asarray(centers_xyz, dtype=float).copy()
    valid = np.all(np.isfinite(oriented), axis=1)
    points_xy = oriented[valid, :2]
    if points_xy.shape[0] < 2:
        raise ValueError("at least two finite neuron positions are required")

    center_xy = np.median(points_xy, axis=0)
    pca_angle_degrees: float | None = None
    if POINT_CLOUD_ALIGN_XY:
        centered = points_xy - center_xy
        _, singular_values, right_vectors = np.linalg.svd(
            centered, full_matrices=False
        )
        if singular_values[0] <= np.finfo(np.float32).eps:
            raise ValueError("point-cloud XY coordinates have no PCA direction")
        principal_axis = right_vectors[0]
        # Resolve the PCA sign ambiguity toward the original +Y direction.
        if principal_axis[1] < 0 or (
            np.isclose(principal_axis[1], 0.0) and principal_axis[0] < 0
        ):
            principal_axis = -principal_axis
        source_angle = math.atan2(principal_axis[1], principal_axis[0])
        pca_angle_degrees = math.degrees(math.pi / 2.0 - source_angle)
        points_xy = _rotate_xy(points_xy, center_xy, pca_angle_degrees)

    points_xy = _rotate_xy(
        points_xy, center_xy, POINT_CLOUD_ROTATE_CCW_DEGREES
    )
    if POINT_CLOUD_FLIP_X:
        points_xy[:, 0] = 2.0 * center_xy[0] - points_xy[:, 0]
    if POINT_CLOUD_FLIP_Y:
        points_xy[:, 1] = 2.0 * center_xy[1] - points_xy[:, 1]
    oriented[valid, :2] = points_xy
    return oriented, {
        "pca_aligned": POINT_CLOUD_ALIGN_XY,
        "pca_target_axis": "+Y" if POINT_CLOUD_ALIGN_XY else None,
        "pca_rotation_ccw_degrees": pca_angle_degrees,
        "additional_rotation_ccw_degrees": POINT_CLOUD_ROTATE_CCW_DEGREES,
        "flip_x": POINT_CLOUD_FLIP_X,
        "flip_y": POINT_CLOUD_FLIP_Y,
        "center_xy": [float(value) for value in center_xy],
    }


def _write_error_map_2d_html(
    path: Path,
    sidecar: Sidecar,
    raw_roi: np.ndarray,
    neuron_ids: tuple[int, ...],
    per_neuron: list[dict[str, Any]],
    start: int,
    stop: int,
) -> dict[str, Any]:
    centers_xyz = _raw_point_cloud(raw_roi, sidecar)
    centers_xyz, orientation = _orient_point_cloud_xy(centers_xyz)
    reviewed_ids = set(neuron_ids)
    stats_by_id = {
        int(row["neuron_id"]): row
        for row in per_neuron
        if row["identity_type"] == "raw"
    }
    plotted_ids = [
        neuron_id
        for neuron_id in range(sidecar.raw_n)
        if np.all(np.isfinite(centers_xyz[neuron_id]))
    ]
    reviewed_plotted = [
        neuron_id for neuron_id in plotted_ids if neuron_id in reviewed_ids
    ]
    unreviewed_plotted = [
        neuron_id for neuron_id in plotted_ids if neuron_id not in reviewed_ids
    ]
    if not plotted_ids:
        raise ValueError("no raw neuron has a valid position for the 2D error map")
    plotted_centers = centers_xyz[plotted_ids]
    data_min = np.min(plotted_centers, axis=0)
    data_max = np.max(plotted_centers, axis=0)
    data_span = data_max - data_min
    padding = np.maximum(data_span * 0.05, 1.0)
    axis_ranges = {
        "x": [float(data_min[0] - padding[0]), float(data_max[0] + padding[0])],
        "y": [float(data_max[1] + padding[1]), float(data_min[1] - padding[1])],
        "z": [float(data_min[2] - padding[2]), float(data_max[2] + padding[2])],
    }
    display_span = data_span + 2.0 * padding

    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, None], [{}, {}]],
        subplot_titles=("XOZ", "XOY", "ZY"),
        # Allocate subplot windows in proportion to the actual anatomical
        # spans: XOY dominates, while Z-containing views stay narrow.
        column_widths=(float(display_span[0]), float(display_span[2])),
        row_heights=(float(display_span[2]), float(display_span[1])),
        horizontal_spacing=0.015,
        vertical_spacing=0.025,
    )
    projections = (
        (1, 1, 0, 2, "X", "Z"),
        (2, 1, 0, 1, "X", "Y"),
        (2, 2, 2, 1, "Z", "Y"),
    )
    reviewed_rates = [
        stats_by_id[neuron_id]["move_error_probability"] or 0.0
        for neuron_id in reviewed_plotted
    ]
    if reviewed_rates:
        color_min = min(reviewed_rates)
        color_max = max(reviewed_rates)
        if math.isclose(color_min, color_max):
            color_min = max(0.0, color_min - 0.005)
            color_max = min(1.0, color_max + 0.005)
            if math.isclose(color_min, color_max):
                color_max = color_min + 0.01
    else:
        color_min, color_max = 0.0, 1.0
    reviewed_hover = [
        [
            neuron_id,
            stats_by_id[neuron_id]["moved_observations"],
            stats_by_id[neuron_id]["eligible_raw_observations"],
            stats_by_id[neuron_id]["resized_observations"],
            stats_by_id[neuron_id]["presence_changed_observations"],
        ]
        for neuron_id in reviewed_plotted
    ]
    unreviewed_hover = [[neuron_id] for neuron_id in unreviewed_plotted]

    for projection_index, (
        row,
        column,
        x_index,
        y_index,
        x_label,
        y_label,
    ) in enumerate(projections):
        if unreviewed_plotted:
            coordinates = centers_xyz[unreviewed_plotted]
            figure.add_trace(
                go.Scatter(
                    x=coordinates[:, x_index],
                    y=coordinates[:, y_index],
                    mode="markers",
                    name="Unreviewed",
                    legendrank=2,
                    showlegend=projection_index == 0,
                    customdata=unreviewed_hover,
                    marker={
                        "size": UNREVIEWED_POINT_SIZE,
                        "color": "#9e9e9e",
                        "opacity": 0.65,
                        "line": {"color": "#606060", "width": 0.25},
                    },
                    hovertemplate=(
                        "Neuron %{customdata[0]}<br>"
                        "Status: unreviewed<br>"
                        f"Position source: raw NPY volume {POINT_CLOUD_VOLUME_INDEX}<br>"
                        f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=column,
            )
        if reviewed_plotted:
            coordinates = centers_xyz[reviewed_plotted]
            figure.add_trace(
                go.Scatter(
                    x=coordinates[:, x_index],
                    y=coordinates[:, y_index],
                    mode="markers",
                    name="Reviewed",
                    legendrank=1,
                    showlegend=projection_index == 0,
                    customdata=reviewed_hover,
                    marker={
                        "size": REVIEWED_POINT_SIZE,
                        "color": reviewed_rates,
                        "colorscale": "Viridis",
                        "cmin": color_min,
                        "cmax": color_max,
                        "showscale": projection_index == 2,
                        "colorbar": {
                            "title": "Move error rate",
                            "tickformat": ".0%",
                            "x": 1.02,
                        },
                        "line": {"color": "#202020", "width": 0.25},
                    },
                    hovertemplate=(
                        "Neuron %{customdata[0]}<br>"
                        "Status: reviewed<br>"
                        "Move error rate: %{marker.color:.2%}<br>"
                        "Moved: %{customdata[1]} / %{customdata[2]}<br>"
                        "Resized observations: %{customdata[3]}<br>"
                        "Presence changes: %{customdata[4]}<br>"
                        f"Position source: raw NPY volume {POINT_CLOUD_VOLUME_INDEX}<br>"
                        f"{x_label}: %{{x:.2f}}<br>{y_label}: %{{y:.2f}}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=column,
            )

    x_axes = (
        (1, 1, "X", axis_ranges["x"]),
        (2, 1, "X", axis_ranges["x"]),
        (2, 2, "Z", axis_ranges["z"]),
    )
    y_axes = (
        (1, 1, "Z", axis_ranges["z"], "x"),
        (2, 1, "Y", axis_ranges["y"], "x2"),
        (2, 2, "Y", axis_ranges["y"], "x3"),
    )
    for row, column, title, axis_range in x_axes:
        figure.update_xaxes(
            title_text=title,
            range=axis_range,
            nticks=5,
            zeroline=False,
            fixedrange=False,
            row=row,
            col=column,
        )
    for row, column, title, axis_range, anchor in y_axes:
        figure.update_yaxes(
            title_text=title,
            range=axis_range,
            nticks=5,
            zeroline=False,
            scaleanchor=anchor,
            scaleratio=1,
            fixedrange=False,
            row=row,
            col=column,
        )

    figure.update_layout(
        template="plotly_white",
        height=700,
        hovermode="closest",
        dragmode="zoom",
        # legend={
        #     "orientation": "h",
        #     "x": 0.98,
        #     "xanchor": "right",
        #     "y": 1.16,
        #     "yanchor": "bottom",
        #     "itemsizing": "constant",
        # },
        showlegend=False,
        margin={"l": 60, "r": 100, "b": 50, "t": 105},
    )
    figure.write_html(
        path,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "doubleClick": "reset",
        },
    )
    return {
        "generated": True,
        "path": path.name,
        "plot_type": "three orthographic 2D projections",
        "projections": ["XOZ", "XOY", "ZY"],
        "reviewed_neurons_plotted": len(reviewed_plotted),
        "unreviewed_neurons_plotted": len(unreviewed_plotted),
        "neurons_without_valid_position": sidecar.raw_n - len(plotted_ids),
        "position_definition": (
            "raw neuron_point_tuple.npy XYZ from one complete volume"
        ),
        "point_cloud_volume_index": POINT_CLOUD_VOLUME_INDEX,
        "z_coordinate_units": "raw neuron_point_tuple.npy scaled Z",
        "color_definition": "moved observations / eligible raw observations",
        "color_range": [color_min, color_max],
        "color_range_source": "observed reviewed-neuron error-rate range",
        "marker_sizes": {
            "reviewed": REVIEWED_POINT_SIZE,
            "unreviewed": UNREVIEWED_POINT_SIZE,
        },
        "axis_scale": "equal numeric units in all projections",
        "axis_ranges": axis_ranges,
        "subplot_window_ratio_xyz": [float(value) for value in display_span],
        "orientation": orientation,
    }


def analyze_sidecar(sidecar_path: Path, raw_roi_path: Path | None) -> Path:
    sidecar = _load_sidecar(sidecar_path)
    raw_roi = _load_raw_roi(raw_roi_path, sidecar)
    patches = _resolve_fields(sidecar, raw_roi)
    neuron_ids = _analysis_neuron_ids(sidecar)
    start, stop = _volume_bounds(sidecar.raw_t)
    events = _selected_events(
        patches, sidecar, raw_roi, neuron_ids, start, stop
    )
    volume_eligible, neuron_eligible = _eligible_counts(
        raw_roi, sidecar, neuron_ids, start, stop
    )

    by_neuron: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_volume: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for event in events:
        assert event.fields is not None
        for field in event.fields:
            by_neuron[event.neuron_id][field] += 1
            by_volume[event.volume_index][field] += 1

    all_resize_ids = {
        patch.neuron_id
        for patch in patches
        if patch.fields and "size_zyx" in patch.fields
    }
    global_size_ids = (
        set(sidecar.placement_size_ids)
        - set(sidecar.added_ids)
        - set(sidecar.delete_all_ids)
    )
    analyzed_id_set = set(neuron_ids)
    global_resize_ids = global_size_ids & all_resize_ids & analyzed_id_set
    global_size_noop_ids = (
        global_size_ids - all_resize_ids
    ) & analyzed_id_set

    per_neuron: list[dict[str, Any]] = []
    for neuron_id in list(neuron_ids) + sorted(sidecar.added_ids):
        counts = by_neuron[neuron_id]
        eligible = int(neuron_eligible[neuron_id]) if neuron_id < sidecar.raw_n else 0
        moved = counts["center_zyx"]
        resized = counts["size_zyx"]
        per_neuron.append(
            {
                "neuron_id": neuron_id,
                "identity_type": "raw" if neuron_id < sidecar.raw_n else "added",
                "eligible_raw_observations": eligible,
                "moved_observations": moved,
                "resized_observations": resized,
                "presence_changed_observations": counts["presence"],
                "move_error_probability": _ratio(moved, eligible),
                "inferred_position_accuracy": _accuracy(moved, eligible),
                "has_move": moved > 0,
                "has_any_resize_in_range": resized > 0,
                "has_global_resize": neuron_id in global_resize_ids,
            }
        )

    per_volume: list[dict[str, Any]] = []
    for volume_index in range(start, stop):
        counts = by_volume[volume_index]
        eligible = int(volume_eligible[volume_index - start])
        moved = counts["center_zyx"]
        per_volume.append(
            {
                "volume_index": volume_index,
                "eligible_raw_observations": eligible,
                "moved_observations": moved,
                "resized_observations": counts["size_zyx"],
                "presence_changed_observations": counts["presence"],
                "move_error_probability": _ratio(moved, eligible),
                "inferred_position_accuracy": _accuracy(moved, eligible),
            }
        )

    modified: list[dict[str, Any]] = []
    for event in events:
        assert event.fields is not None
        center = event.center_zyx or (None, None, None)
        size = event.size_zyx or (None, None, None)
        modified.append(
            {
                "volume_index": event.volume_index,
                "neuron_id": event.neuron_id,
                "state": event.state,
                "changed_fields": "|".join(event.fields),
                "moved": "center_zyx" in event.fields,
                "resized": "size_zyx" in event.fields,
                "presence_changed": "presence" in event.fields,
                "implicit_delete_all": event.implicit_delete_all,
                "corrected_center_z": center[0],
                "corrected_center_y": center[1],
                "corrected_center_x": center[2],
                "corrected_size_z": size[0],
                "corrected_size_y": size[1],
                "corrected_size_x": size[2],
            }
        )

    def event_count(field: str) -> int:
        return sum(field in (event.fields or ()) for event in events)

    moved_total = event_count("center_zyx")
    resized_total = event_count("size_zyx")
    presence_total = event_count("presence")
    eligible_total = int(volume_eligible.sum())
    moved_ids = {
        event.neuron_id
        for event in events
        if event.neuron_id in analyzed_id_set
        and event.fields
        and "center_zyx" in event.fields
    }
    resized_ids = {
        event.neuron_id
        for event in events
        if event.neuron_id in analyzed_id_set
        and event.fields
        and "size_zyx" in event.fields
    }

    warnings = [
        "The JSON stores final sparse differences, not edit history; "
        "repeated edit actions cannot be counted.",
        "Accuracy assumes the selected range was fully proofread; "
        "unchanged does not prove reviewed.",
        "Repeated volumes and neurons are correlated; rates are descriptive, "
        "not independent-trial estimates.",
    ]
    if raw_roi is None:
        warnings.append(
            "No raw ROI NPY was supplied, so every raw (volume, neuron) slot was assumed valid."
        )
    if PROOFREADING_SCOPE == "partial":
        warnings.append(
            "Partial-mode reviewed neuron IDs are the raw IDs that occur in "
            "the sidecar observation patches or global markers."
        )

    counts = {
        "raw_neurons_in_dataset": sidecar.raw_n,
        "analyzed_raw_neurons": len(neuron_ids),
        "raw_volumes": sidecar.raw_t,
        "analyzed_volumes": stop - start,
        "eligible_raw_observations": eligible_total,
        "moved_observations": moved_total,
        "resized_observations": resized_total,
        "presence_changed_observations": presence_total,
        "global_resize_neurons": len(global_resize_ids),
        "global_size_application_noop_neurons": len(global_size_noop_ids),
        "added_neurons": len(sidecar.added_ids),
        "retired_added_neurons": len(sidecar.retired_ids),
        "delete_all_neurons": len(sidecar.delete_all_ids & analyzed_id_set),
    }
    rates: dict[str, float | None] = {}
    if PROOFREADING_SCOPE == "complete":
        counts.update(
            {
                "raw_neurons_with_move": len(moved_ids),
                "raw_neurons_with_any_resize_in_range": len(resized_ids),
            }
        )
        rates = {
            "move_error_probability": _ratio(moved_total, eligible_total),
            "inferred_position_accuracy": _accuracy(moved_total, eligible_total),
            "raw_neurons_with_move_fraction": _ratio(
                len(moved_ids), sidecar.raw_n
            ),
            "raw_neurons_without_move_fraction": _accuracy(
                len(moved_ids), sidecar.raw_n
            ),
            "raw_neurons_with_any_resize_in_range_fraction": _ratio(
                len(resized_ids), sidecar.raw_n
            ),
            "global_resize_neuron_fraction": _ratio(
                len(global_resize_ids), sidecar.raw_n
            ),
        }

    report_dir = OUTPUT_DIR / _report_name(sidecar.path)
    report_dir.mkdir(parents=True, exist_ok=True)
    if raw_roi is None:
        error_map = {
            "generated": False,
            "reason": "A matching raw ROI NPY is required for 3D positions.",
        }
    else:
        error_map = _write_error_map_2d_html(
            report_dir / "neuron_error_map_2d.html",
            sidecar,
            raw_roi,
            neuron_ids,
            per_neuron,
            start,
            stop,
        )

    summary = {
        "report_schema_version": 3,
        "input": {
            "sidecar_path": str(sidecar.path.resolve()),
            "sidecar_schema_version": sidecar.schema_version,
            "raw_roi_path": str(raw_roi_path.resolve()) if raw_roi_path else None,
            "raw_shape_tnk": list(sidecar.raw_shape),
            "z_divisor": sidecar.z_divisor,
            "proofreading_scope": PROOFREADING_SCOPE,
            "partial_neuron_id_source": (
                "sidecar_patches_and_global_markers"
                if PROOFREADING_SCOPE == "partial"
                else None
            ),
            "analyzed_volume_range_half_open": [start, stop],
            "denominator_source": (
                "raw_roi_valid_boxes" if raw_roi is not None else "raw_shape_dense_assumption"
            ),
        },
        "definitions": {
            "move": "final center_zyx differs from the raw ROI center",
            "move_error_probability": "moved observations / eligible raw observations",
            "inferred_position_accuracy": "1 - move_error_probability",
            "global_resize_neuron": (
                "raw neuron with placement_size metadata and an effective "
                "size_zyx change"
            ),
        },
        "counts": counts,
        "rates": rates,
        "neuron_ids": {
            "analyzed_raw": list(neuron_ids),
            "with_move_in_range": sorted(moved_ids),
            "with_any_resize_in_range": sorted(resized_ids),
            "with_global_resize": sorted(global_resize_ids),
            "with_global_size_application_but_no_effective_resize": sorted(
                global_size_noop_ids
            ),
            "delete_all": sorted(sidecar.delete_all_ids & analyzed_id_set),
            "added": sorted(sidecar.added_ids),
            "retired_added": sorted(sidecar.retired_ids),
        },
        "visualization": error_map,
        "warnings": warnings,
    }

    summary_path = report_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(report_dir / "per_neuron.csv", per_neuron, list(per_neuron[0]))
    _write_csv(report_dir / "per_volume.csv", per_volume, list(per_volume[0]))
    _write_csv(
        report_dir / "modified_observations.csv",
        modified,
        [
            "volume_index",
            "neuron_id",
            "state",
            "changed_fields",
            "moved",
            "resized",
            "presence_changed",
            "implicit_delete_all",
            "corrected_center_z",
            "corrected_center_y",
            "corrected_center_x",
            "corrected_size_z",
            "corrected_size_y",
            "corrected_size_x",
        ],
    )

    print(f"Analyzed: {sidecar.path}")
    if PROOFREADING_SCOPE == "complete":
        accuracy = rates["inferred_position_accuracy"]
        resize_fraction = rates["global_resize_neuron_fraction"]
        accuracy_text = "n/a" if accuracy is None else f"{100 * accuracy:.4f}%"
        resize_text = (
            "n/a"
            if resize_fraction is None
            else f"{100 * resize_fraction:.4f}%"
        )
        print(
            f"  complete: moved={moved_total}/{eligible_total}; "
            f"inferred position accuracy={accuracy_text}"
        )
        print(
            f"  global resized neurons={len(global_resize_ids)}/"
            f"{sidecar.raw_n}; fraction={resize_text}"
        )
    else:
        print(
            f"  partial: reporting {len(neuron_ids)} neuron IDs found in "
            "the sidecar"
        )
        for row in per_neuron:
            if row["identity_type"] != "raw":
                continue
            accuracy = row["inferred_position_accuracy"]
            accuracy_text = (
                "n/a" if accuracy is None else f"{100 * accuracy:.4f}%"
            )
            print(
                f"  ID {row['neuron_id']}: moved="
                f"{row['moved_observations']}/"
                f"{row['eligible_raw_observations']}; "
                f"position accuracy={accuracy_text}; "
                f"global resize={row['has_global_resize']}"
            )
    print(f"  report={summary_path}")
    if error_map["generated"]:
        print(f"  2D error map={report_dir / error_map['path']}")
    else:
        print(f"  2D error map skipped: {error_map['reason']}")
    return summary_path


def main() -> None:
    if not PROOFREADING_INPUTS:
        raise ValueError("PROOFREADING_INPUTS must contain at least one JSON")
    for sidecar_path, raw_roi_path in PROOFREADING_INPUTS:
        analyze_sidecar(sidecar_path, raw_roi_path)


if __name__ == "__main__":
    main()
