"""Merge sparse proofreading sidecars for one neuron dataset.

Inputs are ordered from highest to lowest priority. If two annotators changed
the same observation differently, the earlier input wins. Raw data is never
modified.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable

APP_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = APP_DIR.parent

# ---------------------------------------------------------------------------
# Editable settings
# ---------------------------------------------------------------------------

# Highest priority first. The first path is the primary annotator whose value
# wins when the same annotation key has different values in multiple files.
PROOFREADING_INPUTS: tuple[Path, ...] = (
    Path(r"data\20260304_w3\neuron_point_tuple.proofread_wjh.json"),
    Path(r"data\20260304_w3\neuron_point_tuple.proofread_lyx.json"),
                                        )
OUTPUT_PATH = (
    REPOSITORY_DIR
    / "data"
    / "20260304_w3"
    / "neuron_point_tuple.proofread_all.json"
)
REPORT_PATH = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}.merge_report.json")


@dataclass(frozen=True)
class Source:
    priority: int
    path: Path
    payload: dict[str, Any]
    patches: dict[tuple[int, int], dict[str, Any]]
    delete_all_ids: frozenset[int]
    placement_size: dict[int, list[Any]]
    added_status: dict[int, str]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"proofreading JSON does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"sidecar root must be an object: {path}")
    return payload


def _integer_list(value: Any, name: str, path: Path) -> list[int]:
    if (
        not isinstance(value, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be a list of unique integers: {path}")
    return value


def _parse_source(priority: int, path: Path) -> Source:
    payload = _load_json(path)
    if payload.get("schema_version") not in (1, 2):
        raise ValueError(f"unsupported schema_version in {path}")

    raw = payload.get("raw")
    if not isinstance(raw, dict):
        raise ValueError(f"raw metadata is missing: {path}")
    shape = raw.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in shape)
        or shape[0] <= 0
        or shape[1] <= 0
        or shape[2] < 6
    ):
        raise ValueError(f"raw.shape must be [T, N, K>=6]: {path}")

    records = payload.get("observation_patches")
    if not isinstance(records, list):
        raise ValueError(f"observation_patches must be a list: {path}")
    patches: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"observation patch must be an object: {path}")
        volume_index = record.get("volume_index")
        neuron_id = record.get("neuron_id")
        state = record.get("state")
        if (
            isinstance(volume_index, bool)
            or not isinstance(volume_index, int)
            or not 0 <= volume_index < shape[0]
            or isinstance(neuron_id, bool)
            or not isinstance(neuron_id, int)
            or neuron_id < 0
            or state not in ("present", "deleted")
        ):
            raise ValueError(f"invalid observation patch in {path}: {record!r}")
        key = (volume_index, neuron_id)
        if key in patches:
            raise ValueError(f"duplicate observation patch {key} in {path}")
        patches[key] = record

    delete_all_ids = frozenset(
        _integer_list(payload.get("delete_all_ids"), "delete_all_ids", path)
    )
    placement = payload.get("placement_size")
    if not isinstance(placement, dict):
        raise ValueError(f"placement_size must be an object: {path}")
    placement_size: dict[int, list[Any]] = {}
    for key, value in placement.items():
        try:
            neuron_id = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid placement_size key {key!r}: {path}") from error
        if str(neuron_id) != key or not isinstance(value, list):
            raise ValueError(f"invalid placement_size entry {key!r}: {path}")
        placement_size[neuron_id] = value

    added = payload.get("added_neurons")
    if not isinstance(added, dict):
        raise ValueError(f"added_neurons must be an object: {path}")
    committed = _integer_list(added.get("committed"), "committed IDs", path)
    retired = _integer_list(added.get("retired"), "retired IDs", path)
    if set(committed) & set(retired):
        raise ValueError(f"an added ID is both committed and retired: {path}")
    added_status = {neuron_id: "committed" for neuron_id in committed}
    added_status.update({neuron_id: "retired" for neuron_id in retired})

    return Source(
        priority=priority,
        path=path,
        payload=payload,
        patches=patches,
        delete_all_ids=delete_all_ids,
        placement_size=placement_size,
        added_status=added_status,
    )


def _check_compatible(sources: list[Source]) -> None:
    primary = sources[0]
    for source in sources[1:]:
        if source.payload["raw"] != primary.payload["raw"]:
            raise ValueError(
                f"raw metadata differs between {primary.path} and {source.path}"
            )
        if source.payload.get("image_signature") != primary.payload.get(
            "image_signature"
        ):
            raise ValueError(
                f"image_signature differs between {primary.path} and {source.path}"
            )


def _key_text(key: Hashable) -> str:
    if isinstance(key, tuple):
        return ",".join(str(item) for item in key)
    return str(key)


def _merge_mapping(
    sources: Iterable[Source],
    values: Callable[[Source], dict[Hashable, Any]],
    category: str,
    conflicts: list[dict[str, Any]],
) -> tuple[dict[Hashable, Any], dict[Hashable, Source]]:
    merged: dict[Hashable, Any] = {}
    owners: dict[Hashable, Source] = {}
    for source in sources:
        for key, value in values(source).items():
            if key not in merged:
                merged[key] = value
                owners[key] = source
            elif merged[key] != value:
                conflicts.append(
                    {
                        "category": category,
                        "key": _key_text(key),
                        "winner": str(owners[key].path.resolve()),
                        "discarded": str(source.path.resolve()),
                        "reason": "higher-priority input wins",
                    }
                )
    return merged, owners


def _with_presence_field(record: dict[str, Any]) -> dict[str, Any]:
    fields = record.get("changed_fields")
    if not isinstance(fields, list) or "presence" in fields:
        return record
    updated = dict(record)
    updated["changed_fields"] = [
        field
        for field in ("presence", "center_zyx", "size_zyx")
        if field == "presence" or field in fields
    ]
    return updated


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def merge_sidecars(
    input_paths: tuple[Path, ...], output_path: Path, report_path: Path
) -> tuple[Path, Path]:
    """Merge sidecars and return the merged JSON and conflict-report paths."""
    if not input_paths:
        raise ValueError("PROOFREADING_INPUTS must contain at least one JSON")
    resolved_inputs = [path.resolve() for path in input_paths]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise ValueError("PROOFREADING_INPUTS contains a duplicate path")
    if output_path.resolve() in resolved_inputs or report_path.resolve() in resolved_inputs:
        raise ValueError("output and report paths must not overwrite an input JSON")
    if output_path.resolve() == report_path.resolve():
        raise ValueError("OUTPUT_PATH and REPORT_PATH must be different")

    sources = [_parse_source(index, path) for index, path in enumerate(input_paths)]
    _check_compatible(sources)
    primary = sources[0]
    output_schema_version = min(
        int(source.payload["schema_version"]) for source in sources
    )
    conflicts: list[dict[str, Any]] = []

    # Added IDs are allocated locally by each annotator, so the same numeric ID
    # can denote different new neurons. Keep that ID entirely from the highest-
    # priority source instead of mixing its observations across people.
    added_owners: dict[int, Source] = {}
    for source in sources:
        for neuron_id in source.added_status:
            owner = added_owners.setdefault(neuron_id, source)
            if owner is not source:
                conflicts.append(
                    {
                        "category": "added_neuron_id",
                        "key": str(neuron_id),
                        "winner": str(owner.path.resolve()),
                        "discarded": str(source.path.resolve()),
                        "reason": "added IDs are annotator-local; higher-priority input wins",
                    }
                )

    def owned_patches(source: Source) -> dict[Hashable, Any]:
        result: dict[Hashable, Any] = {}
        for key, value in source.patches.items():
            if key[1] in added_owners and added_owners[key[1]] is not source:
                continue
            if output_schema_version == 1 and "changed_fields" in value:
                value = {
                    field: field_value
                    for field, field_value in value.items()
                    if field != "changed_fields"
                }
            result[key] = value
        return result

    def owned_values(
        source: Source, values: dict[int, Any]
    ) -> dict[Hashable, Any]:
        return {
            key: value
            for key, value in values.items()
            if key not in added_owners or added_owners[key] is source
        }

    patches, patch_owners = _merge_mapping(
        sources, owned_patches, "observation_patch", conflicts
    )
    placement, _ = _merge_mapping(
        sources,
        lambda source: owned_values(source, source.placement_size),
        "placement_size",
        conflicts,
    )
    statuses, _ = _merge_mapping(
        sources,
        lambda source: owned_values(source, source.added_status),
        "added_neuron_status",
        conflicts,
    )

    delete_owners: dict[int, Source] = {}
    for source in sources:
        for neuron_id in source.delete_all_ids:
            if neuron_id in added_owners and added_owners[neuron_id] is not source:
                continue
            delete_owners.setdefault(neuron_id, source)
    delete_all_ids = set(delete_owners)

    raw_n = int(primary.payload["raw"]["shape"][1])
    committed = {key for key, status in statuses.items() if status == "committed"}
    retired = {key for key, status in statuses.items() if status == "retired"}
    added_lineage = committed | retired
    if added_lineage and added_lineage != set(range(raw_n, max(added_lineage) + 1)):
        raise ValueError(
            "merged added-neuron lineage contains an ID gap; added IDs from "
            "different annotators cannot be safely identified automatically"
        )

    known_ids = set(range(raw_n)) | committed
    delete_all_ids &= known_ids
    placement = {
        key: value for key, value in placement.items() if key in known_ids
    }

    final_patches: list[dict[str, Any]] = []
    for key in sorted(patches):
        record = patches[key]
        owner = patch_owners[key]
        neuron_id = key[1]
        if neuron_id not in known_ids:
            continue
        delete_owner = delete_owners.get(neuron_id)
        if delete_owner is not None:
            if record["state"] == "deleted":
                continue  # The global marker already represents this deletion.
            if delete_owner.priority < owner.priority:
                conflicts.append(
                    {
                        "category": "delete_all_vs_observation",
                        "key": _key_text(key),
                        "winner": str(delete_owner.path.resolve()),
                        "discarded": str(owner.path.resolve()),
                        "reason": "higher-priority delete-all marker wins",
                    }
                )
                continue
            if output_schema_version == 2:
                record = _with_presence_field(record)
        final_patches.append(record)

    output: dict[str, Any] = {
        "schema_version": output_schema_version,
        "raw": primary.payload["raw"],
        "observation_patches": final_patches,
        "delete_all_ids": sorted(delete_all_ids),
        "placement_size": {
            str(key): placement[key] for key in sorted(placement)
        },
        "added_neurons": {
            "committed": sorted(committed),
            "retired": sorted(retired),
        },
    }
    if "image_signature" in primary.payload:
        output["image_signature"] = primary.payload["image_signature"]

    conflict_counts = Counter(item["category"] for item in conflicts)
    report = {
        "inputs_in_priority_order": [str(path.resolve()) for path in input_paths],
        "primary_input": str(input_paths[0].resolve()),
        "output": str(output_path.resolve()),
        "output_schema_version": output_schema_version,
        "merged_observation_patches": len(final_patches),
        "conflict_count": len(conflicts),
        "conflict_counts_by_category": dict(sorted(conflict_counts.items())),
        "conflicts": conflicts,
    }
    _write_json_atomic(output_path, output)
    _write_json_atomic(report_path, report)
    return output_path, report_path


def main() -> None:
    output_path, report_path = merge_sidecars(
        PROOFREADING_INPUTS, OUTPUT_PATH, REPORT_PATH
    )
    print(f"Merged {len(PROOFREADING_INPUTS)} proofreading JSON files")
    print(f"  primary={PROOFREADING_INPUTS[0]}")
    print(f"  output={output_path}")
    print(f"  report={report_path}")


if __name__ == "__main__":
    main()
