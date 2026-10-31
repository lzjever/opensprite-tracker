#!/usr/bin/env python3
"""Build a training MJCF from the unmodified public Sprite0825 release.

Inertials, kinematic transforms, joint limits and armatures remain official.
Mesh visuals are retained. Collisions use reproducible inscribed primitives
to make full-body contact practical in MuJoCo Warp; the original foot boxes
are retained exactly. This is a simulation approximation, not vendor geometry.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "assets/opensprite/vendor/sprite0825"
OUTPUT = ROOT / "assets/opensprite/opensprite.xml"


def numbers(values):
    return " ".join(f"{float(x):.10g}" for x in values)


def build():
    source = VENDOR / "mujoco/sprite0825_external_pd.xml"
    model = mujoco.MjModel.from_xml_path(str(source))
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "opensprite0825_tracker")
    root.find("compiler").set("meshdir", "vendor/sprite0825")
    ET.SubElement(root, "option", timestep="0.005", integrator="implicitfast",
                  gravity="0 0 -9.81", iterations="20", ls_iterations="10")
    contact = ET.SubElement(root, "contact")
    bodies = {b.get("name"): b for b in root.findall(".//body")}
    for name, body in bodies.items():
        for index, geom in enumerate(body.findall("geom")):
            if geom.get("type") == "mesh":
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("group", "2")
                geom.set("name", f"{name}_visual_{index}")
            else:
                # Preserve the public 104 x 44 x 10 mm sole box exactly.
                side = name.split("_")[0]
                geom.set("name", f"{side}_foot_collision")
                geom.set("friction", "0.8 0.02 0.001")
                geom.set("condim", "3")
                geom.set("group", "3")
        if name.endswith("ankle_roll_link"):
            continue  # Original foot box already provides the collision.
        bid = model.body(name).id
        gid = next(g for g in range(model.ngeom)
                   if model.geom_bodyid[g] == bid and
                   model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
        mesh = model.geom_dataid[gid]
        adr, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        verts = model.mesh_vert[adr:adr + count]
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, model.geom_quat[gid])
        verts = verts @ matrix.reshape(3, 3).T + model.geom_pos[gid]
        low, high = verts.min(0), verts.max(0)
        center, half = (low + high) / 2, (high - low) / 2
        common = dict(name=f"{name}_primitive_collision", group="3",
                      friction="0.8 0.02 0.001", condim="3",
                      rgba="0.35 0.55 0.75 0.35")
        if name in ("pelvis_link", "waist_yaw_link"):
            ET.SubElement(body, "geom", type="box", pos=numbers(center),
                          size=numbers(.9 * half), **common)
        else:
            axis = int(np.argmax(half))
            radius = .85 * float(np.min(half))
            length = max(float(half[axis]) - radius, 0)
            delta = np.eye(3)[axis] * length
            ET.SubElement(body, "geom", type="capsule", size=numbers([radius]),
                          fromto=numbers(np.r_[center - delta, center + delta]),
                          **common)
    # Near joint housings overlap by design, as in the CAD model. Suppress
    # parent/grandparent contacts only; nonlocal body contacts remain enabled.
    for bid in range(1, model.nbody):
        parent = int(model.body_parentid[bid])
        for _ in range(2):
            if parent == 0:
                break
            ET.SubElement(contact, "exclude", body1=model.body(bid).name,
                          body2=model.body(parent).name)
            parent = int(model.body_parentid[parent])
    # Nominal geometric root height places the public sole at ground z=0.
    # Actual episode resets come from retargeted references, not this value.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    foot_z = data.xpos[model.body("left_ankle_roll_link").id, 2] - .039
    bodies["pelvis_link"].set("pos", numbers([0, 0, .65 - foot_z]))
    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="unicode")
    result = mujoco.MjModel.from_xml_path(str(OUTPUT))
    assert (result.nq, result.nv, result.nu) == (38, 37, 0)
    assert np.allclose(result.body_mass, model.body_mass, rtol=0, atol=0)
    assert np.allclose(result.body_inertia, model.body_inertia, rtol=0, atol=0)
    assert np.allclose(result.jnt_range, model.jnt_range, rtol=0, atol=0)
    assert np.allclose(result.dof_armature, model.dof_armature, rtol=0, atol=0)
    print(f"Built {OUTPUT.relative_to(ROOT)}: nq={result.nq}, nv={result.nv}, "
          f"mass={result.body_mass.sum():.6f} kg, geoms={result.ngeom}")


if __name__ == "__main__":
    build()
