#!/usr/bin/env python3
"""Focused asset parity and external-PD standing smoke check, on CPU."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from opensprite_tracker.robot import BODY_JOINT_NAMES, JOINT_CONTROL, ROBOT_XML


def validate(seconds=10., lag_seconds=0.):
    if seconds <= 0 or lag_seconds < 0:
        raise ValueError("seconds must be positive and lag_seconds nonnegative")
    official = mujoco.MjModel.from_xml_path(str(
        ROOT / "assets/opensprite/vendor/sprite0825/mujoco/sprite0825_external_pd.xml"))
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, .05], friction=[.8, .02, .001])
    model = spec.compile()
    assert (model.nq, model.nv, model.nu) == (38, 37, 0)
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat",
                  "jnt_range", "jnt_axis", "dof_armature"):
        np.testing.assert_array_equal(getattr(model, field), getattr(official, field))
    ids = [model.joint(n).id for n in BODY_JOINT_NAMES]
    assert tuple(model.joint(i).name for i in range(1, model.njnt)) == BODY_JOINT_NAMES
    controls = [JOINT_CONTROL[n] for n in BODY_JOINT_NAMES]
    qids, vids = model.jnt_qposadr[ids], model.jnt_dofadr[ids]
    nominal = np.array([c.default_pos_rad for c in controls])
    kp, kd, effort = (np.array([getattr(c, key) for c in controls])
                      for key in ("kp", "kd", "command_effort_nm"))
    np.testing.assert_array_equal(effort, model.jnt_actfrcrange[ids, 1])
    data = mujoco.MjData(model)
    data.qpos[qids] = nominal
    mujoco.mj_forward(model, data)
    initial_non_ground = [i for i, c in enumerate(data.contact)
                          if model.geom_bodyid[c.geom[0]] and model.geom_bodyid[c.geom[1]]]
    assert not initial_non_ground, "nominal configuration has nonlocal self contact"
    minimum_height = data.qpos[2]
    filtered = np.zeros(31)
    maximum_effort = 0.
    for _ in range(round(seconds / model.opt.timestep)):
        torque = np.clip(kp * (nominal-data.qpos[qids]) - kd*data.qvel[vids], -effort, effort)
        if lag_seconds:
            filtered += -np.expm1(-model.opt.timestep/lag_seconds) * (torque-filtered)
        else:
            filtered[:] = torque
        data.qfrc_applied[vids] = filtered
        maximum_effort = max(maximum_effort, float(np.max(np.abs(filtered/effort))))
        mujoco.mj_step(model, data)
        minimum_height = min(minimum_height, data.qpos[2])
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert minimum_height > .40, f"PD stand fell: minimum pelvis z={minimum_height}"
    assert maximum_effort <= 1. + 1e-12
    print(f"PASS physical parity; 31-joint external PD {seconds:g}s with {lag_seconds*1000:g}ms lag; "
          f"pelvis min/final z={minimum_height:.4f}/{data.qpos[2]:.4f}m; "
          f"maximum effort fraction={maximum_effort:.3f}; mass={model.body_mass.sum():.6f}kg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--lag-seconds", type=float, default=0.)
    args = parser.parse_args()
    validate(args.seconds, args.lag_seconds)
