"""Sprite's CAD-derived full-body primitives and original vendor box soles.

The local asset generator owns geometry, inertials and contact exclusions.
This adapter adds no shells or physical constants from another robot.
"""
import re
import mujoco
import numpy as np
from mjlab.utils.spec_config import CollisionCfg
from opensprite_tracker.robot import ROBOT_XML, FULL_BODY_COLLISION_PATTERN

GROUND_PATTERN = FULL_BODY_COLLISION_PATTERN


def ground_collision_cfg():
    """Enable actual body-ground and nonlocal self contacts of Sprite."""
    return CollisionCfg(geom_names_expr=(GROUND_PATTERN,), contype=1,
                        conaffinity=1, condim=3, priority=1, friction=(.8,))


def full_body_spec():
    return mujoco.MjSpec.from_file(str(ROBOT_XML))


def ground_geom_ids(model):
    return np.asarray([g for g in range(model.ngeom)
        if re.fullmatch(GROUND_PATTERN, (model.geom(g).name or "").rsplit("/", 1)[-1])
        and model.geom_bodyid[g] > 0], dtype=np.int32)


def ground_bottom_z(model, data):
    """Exact support height of the compiled Sprite box/capsule/sphere shapes."""
    ids=ground_geom_ids(model)
    output=data.geom_xpos[ids,2].copy()
    for i,gid in enumerate(ids):
        kind,size=model.geom_type[gid],model.geom_size[gid]
        rz=data.geom_xmat[gid].reshape(3,3)[2]
        if kind==mujoco.mjtGeom.mjGEOM_BOX:
            output[i]-=np.abs(rz)@size
        elif kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
            output[i]-=size[0]+size[1]*abs(rz[2])
        elif kind==mujoco.mjtGeom.mjGEOM_SPHERE:
            output[i]-=size[0]
        else:
            raise ValueError(f"Unsupported Sprite collision geometry: {kind}")
    return output
