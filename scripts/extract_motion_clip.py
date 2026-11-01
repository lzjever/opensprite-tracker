#!/usr/bin/env python3
"""Select one complete packed motion without retargeting or recomputing samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opensprite_tracker.motion_contract import validate_motion_contract

FRAME_FIELDS = (
    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
    "body_lin_vel_w", "body_ang_vel_w",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def extract(source, destination, name, manifest, retarget_metadata):
    source, destination = Path(source), Path(destination)
    with np.load(source, allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    names = original["motion_names"].astype(str).tolist()
    if names.count(name) != 1:
        raise ValueError(f"Expected exactly one motion named {name!r}; found {names}")
    index, total = names.index(name), len(original["joint_pos"])
    starts, lengths = original["motion_starts"], original["motion_lengths"]
    if starts.ndim != 1 or starts.shape != lengths.shape or len(starts) != len(names):
        raise ValueError("Inconsistent packed motion boundaries")
    if not np.issubdtype(starts.dtype, np.integer) or not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError("Motion boundaries must be integer arrays")
    start, length = int(starts[index]), int(lengths[index])
    stop = start + length
    if start < 0 or length < 2 or stop > total:
        raise ValueError("Selected motion is not an in-range complete trajectory")
    for key in FRAME_FIELDS:
        if original[key].shape[0] != total or not np.isfinite(original[key]).all():
            raise ValueError(f"Invalid frame field {key}")
    metadata = json.loads(Path(retarget_metadata).read_text())
    if metadata["motion"] != name or metadata["output_frames"] != length:
        raise ValueError("Original retarget report does not describe the selected complete clip")
    selected = {}
    classifications = {}
    for key, value in original.items():
        if key == "motion_starts":
            selected[key] = np.zeros(1, dtype=value.dtype)
            classifications[key] = "single clip start rebased to zero"
        elif key.startswith("motion_") and value.ndim and value.shape[0] == len(names):
            selected[key] = value[index:index + 1].copy()
            classifications[key] = "selected clip metadata, unchanged value/dtype"
        elif value.ndim and value.shape[0] == total:
            selected[key] = value[start:stop].copy()
            classifications[key] = "exact contiguous source slice"
        else:
            selected[key] = value.copy()
            classifications[key] = "unchanged shared metadata"
    if destination.resolve() == source.resolve():
        raise ValueError("Never overwrite the source bank")
    destination.parent.mkdir(parents=True, exist_ok=True)
    existed = destination.exists()
    if existed:
        with np.load(destination, allow_pickle=False) as archive:
            if set(archive.files) != set(selected):
                raise ValueError("Existing destination has different fields; refusing overwrite")
            for key, expected in selected.items():
                actual = archive[key]
                if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                    raise ValueError(f"Existing destination differs in {key}; refusing overwrite")
    else:
        # Exclusive create prevents an existing motion from being overwritten.
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **selected)
    revision = str(selected["dataset_revision"].item())
    contract = validate_motion_contract(destination, revision)
    with np.load(destination, allow_pickle=False) as archive:
        for key, expected in selected.items():
            if archive[key].dtype != expected.dtype or not np.array_equal(archive[key], expected):
                raise RuntimeError(f"Written selection is not exact in {key}")
    fps = float(selected["fps"].item())
    evidence = {
        "schema": 1, "operation": "exact complete-clip extraction; no retarget/resample/FK/velocity recomputation",
        "source": display_path(source), "destination": display_path(destination),
        "existing_destination_verified_without_overwrite": existed,
        "motion_name": name, "source_bank_clip_index": index,
        "source_bank_frames_half_open": [start, stop],
        "destination_frames_half_open": [0, length],
        "source_local_frame_range_inclusive": [0, length - 1],
        "original_bvh_frames_half_open": [0, metadata["source_frames"]],
        "output_frames": length, "expected_transitions": length - 1, "fps": fps,
        "sample_time_range_seconds": [0.0, (length - 1) / fps],
        "frame_count_over_fps_seconds": length / fps,
        "motion_quality": selected["motion_qualities"].astype(str).tolist(),
        "quality_note": "Original needs_review and all original geometry residuals remain; extraction is not a new feasibility validation",
        "contract": contract,
        "source_sha256": file_hash(source), "destination_sha256": file_hash(destination),
        "hash_note": "File SHA256 identifies exact files; array hashes below identify decompressed raw dtype/shape samples independently of NPZ ZIP representation",
        "fields": {key: {
            "shape": list(value.shape), "dtype": value.dtype.str,
            "selection": classifications[key],
            "raw_array_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            "exact_source_equality_verified": key != "motion_starts",
            "selection_rule_verified": True,
        } for key, value in selected.items()},
        "original_retarget_metadata_path": display_path(retarget_metadata),
        "original_retarget_metadata": metadata,
        "reproduce": f".venv/bin/python scripts/extract_motion_clip.py --source {display_path(source)} --output {display_path(destination)} --name {name} --retarget-metadata {display_path(retarget_metadata)} --manifest {display_path(manifest)}",
    }
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest).write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({key: evidence[key] for key in (
        "destination", "motion_name", "output_frames", "expected_transitions",
        "source_bank_frames_half_open", "sample_time_range_seconds", "motion_quality",
        "destination_sha256", "existing_destination_verified_without_overwrite",
    )}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--retarget-metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    extract(args.source, args.output, args.name, args.manifest, args.retarget_metadata)
