"""Validate target and genuine native-UMR identity before allocating a GPU."""

from pathlib import Path
import numpy as np

METHOD = "umr_native_nr_surface_vector_v1"
ROBOT = "opensprite0825"
CORRESPONDENCES = ("geno", "opensprite0825")


def validate_motion_contract(path, dataset_revision, *, allow_diagnostic_motion=False):
    path = Path(path)
    if not path.is_file():
        raise ValueError(
            f"Expected one repository-owned packed NPZ motion library: {path}"
        )
    with np.load(path, allow_pickle=False) as data:
        scalar = lambda key: (
            str(np.asarray(data[key]).reshape(-1)[0]) if key in data.files else None
        )
        method = scalar("retarget_method")
        robot = scalar("robot_model")
        revision = scalar("dataset_revision")
        correspondence = (
            tuple(str(x) for x in np.asarray(data["correspondence_names"]).reshape(-1))
            if "correspondence_names" in data.files
            else ()
        )
    if allow_diagnostic_motion:
        if robot not in (None, ROBOT):
            raise ValueError(
                f"Diagnostic data explicitly targets another robot: {robot}"
            )
        return {
            "diagnostic": True,
            "retarget_method": method,
            "robot_model": robot,
            "dataset_revision": revision,
            "correspondence_names": correspondence,
        }
    if not dataset_revision:
        raise ValueError(
            "Formal training/evaluation requires an explicit --dataset-revision"
        )
    expected = {
        "retarget_method": METHOD,
        "robot_model": ROBOT,
        "dataset_revision": dataset_revision,
        "correspondence_names": CORRESPONDENCES,
    }
    actual = {
        "retarget_method": method,
        "robot_model": robot,
        "dataset_revision": revision,
        "correspondence_names": correspondence,
    }
    for name, value in expected.items():
        if actual[name] != value:
            raise ValueError(
                f"Motion contract {name}: expected {value!r}, found {actual[name]!r}; "
                "prototype IK or another robot is not a formal OpenSprite UMR motion"
            )
    return {"diagnostic": False, **actual}
