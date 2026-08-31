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
VOLUME_RANGE: tuple[int, int] | None = None
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


def _analysis_neuron_ids(
    sidecar: Sidecar, patches: tuple[Patch, ...]
) -> tuple[int, ...]:
    if PROOFREADING_SCOPE == "complete":
        return tuple(range(sidecar.raw_n))
    if PROOFREADING_SCOPE != "partial":
        raise ValueError("PROOFREADING_SCOPE must be 'partial' or 'complete'")

    if PARTIAL_NEURON_IDS is None:
        neuron_ids = {
            patch.neuron_id
            for patch in patches
            if patch.neuron_id < sidecar.raw_n
        }
        neuron_ids.update(
            neuron_id
            for neuron_id in sidecar.delete_all_ids | sidecar.placement_size_ids
            if neuron_id < sidecar.raw_n
        )
    else:
        neuron_ids = set(PARTIAL_NEURON_IDS)
        if len(neuron_ids) != len(PARTIAL_NEURON_IDS):
            raise ValueError("PARTIAL_NEURON_IDS contains duplicate IDs")
    if any(
        isinstance(neuron_id, bool)
        or not isinstance(neuron_id, int)
        or not 0 <= neuron_id < sidecar.raw_n
        for neuron_id in neuron_ids
    ):
        raise ValueError(
            f"PARTIAL_NEURON_IDS must be within [0, {sidecar.raw_n})"
        )
    if not neuron_ids:
        raise ValueError(
            "no raw neuron IDs can be inferred from this sidecar; set "
            "PARTIAL_NEURON_IDS explicitly"
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


def analyze_sidecar(sidecar_path: Path, raw_roi_path: Path | None) -> Path:
    sidecar = _load_sidecar(sidecar_path)
    raw_roi = _load_raw_roi(raw_roi_path, sidecar)
    patches = _resolve_fields(sidecar, raw_roi)
    neuron_ids = _analysis_neuron_ids(sidecar, patches)
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
    if PROOFREADING_SCOPE == "partial" and PARTIAL_NEURON_IDS is None:
        warnings.append(
            "Partial-mode neuron IDs were inferred from sidecar changes and "
            "global markers. Reviewed but unchanged IDs cannot be inferred "
            "from a sparse sidecar."
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

    summary = {
        "report_schema_version": 2,
        "input": {
            "sidecar_path": str(sidecar.path.resolve()),
            "sidecar_schema_version": sidecar.schema_version,
            "raw_roi_path": str(raw_roi_path.resolve()) if raw_roi_path else None,
            "raw_shape_tnk": list(sidecar.raw_shape),
            "z_divisor": sidecar.z_divisor,
            "proofreading_scope": PROOFREADING_SCOPE,
            "partial_neuron_id_source": (
                "explicit_config"
                if PROOFREADING_SCOPE == "partial"
                and PARTIAL_NEURON_IDS is not None
                else "sidecar_inference"
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
        "warnings": warnings,
    }

    report_dir = OUTPUT_DIR / _report_name(sidecar.path)
    report_dir.mkdir(parents=True, exist_ok=True)
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
            "the sidecar/config"
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
    return summary_path


def main() -> None:
    if not PROOFREADING_INPUTS:
        raise ValueError("PROOFREADING_INPUTS must contain at least one JSON")
    for sidecar_path, raw_roi_path in PROOFREADING_INPUTS:
        analyze_sidecar(sidecar_path, raw_roi_path)


if __name__ == "__main__":
    main()
