#!/usr/bin/env python3
"""Validate Sprite's actual CAD T pose and native UMR surface sampling.

Use .venv-umr. All imports/data are repository-owned. This does not substitute
an IK solver or hand-selected keypoints for the native surface correspondence.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/umr/scripts"))
import mujoco
import numpy as np
from opensprite_tracker.robot import ROBOT_XML, BODY_JOINT_NAMES, TRACKED_BODY_NAMES


def validate(output, build_sample=False):
    config_path = ROOT / "configs/umr/opensprite.json"
    config = json.loads(config_path.read_text())
    robot = config["robot"]
    model_path = (config_path.parent / robot["xml"]).resolve()
    assert model_path == ROOT / "assets/opensprite/retarget_scene.xml"
    assert robot["name"] == robot["slot_name"] == "opensprite0825"
    assert robot["point_cloud_center"] == robot["sample_point_cloud_center"] == "body:pelvis_link"
    assert config["correspondence"]["dataset"]["smpl_center"] == "pelvis"
    assert config["correspondence"]["smpl_name"] == "geno"
    m = mujoco.MjModel.from_xml_path(str(model_path))
    bare = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    for field in ("body_mass", "body_inertia", "body_pos", "body_quat", "jnt_axis", "jnt_range", "dof_armature"):
        np.testing.assert_array_equal(getattr(m, field), getattr(bare, field))
    assert (m.nq, m.nv, m.nu, m.nbody) == (bare.nq, bare.nv, bare.nu, bare.nbody)
    assert int(np.sum(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)) == 1
    d = mujoco.MjData(m)
    for name, angle in robot["tpose_qpos"].items():
        d.qpos[m.joint(name).qposadr] = angle
    mujoco.mj_forward(m, d)
    assert tuple(m.body(i).name for i in range(1, m.nbody)) == TRACKED_BODY_NAMES
    assert tuple(m.joint(i).name for i in range(1, m.njnt)) == BODY_JOINT_NAMES
    assert np.all(d.qpos[7:] >= m.jnt_range[1:, 0]) and np.all(d.qpos[7:] <= m.jnt_range[1:, 1])
    root = d.xpos[m.body("pelvis_link").id]
    points = {name: (d.xpos[m.body(name).id]-root).tolist() for name in (
        "waist_roll_link", "waist_yaw_link", "left_shoulder_pitch_link", "left_elbow_link",
        "left_wrist_roll_link", "right_shoulder_pitch_link", "right_elbow_link",
        "right_wrist_roll_link", "left_ankle_roll_link", "right_ankle_roll_link",
        "head_pitch_link", "head_roll_link", "head_yaw_link")}
    # Robot +Y is anatomical left; native UMR converts to human +X left.
    assert points["left_wrist_roll_link"][1] > .5
    assert points["right_wrist_roll_link"][1] < -.5
    assert abs(points["left_wrist_roll_link"][2] - points["left_shoulder_pitch_link"][2]) < .02
    assert abs(points["right_wrist_roll_link"][2] - points["right_shoulder_pitch_link"][2]) < .02
    transform = np.asarray(robot["frame_transform"]["smpl_to_robot_root_matrix"], float)
    np.testing.assert_allclose(transform @ transform.T, np.eye(3), atol=0)
    assert np.linalg.det(transform) == 1
    visual = np.flatnonzero((m.geom_type == mujoco.mjtGeom.mjGEOM_MESH) &
                           (m.geom_contype == 0) & (m.geom_conaffinity == 0))
    assert len(visual) == 32
    assert set(m.geom_bodyid[visual]) == set(range(1, m.nbody))
    result = dict(robot=robot["name"], body_count=32, hinge_count=31,
                  official_visual_mesh_count=len(visual), explicit_ground_plane_count=1,
                  tpose_relative_body_positions=points,
                  source_center="Geno Hips", target_center="Sprite pelvis_link",
                  actuator_groups="31 independent joints; user-requested delay/torque-lag model",
                  all_physical_limits_pass=True)
    if build_sample:
        from build_correspondence_ae_dataset import build_robot_sample, visual_mesh_geom_ids
        from mujoco_geom_surface import geom_local_mesh
        sample = build_robot_sample(model_path, robot["slot_name"], 4096, 9000, 8, 0., 1.,
            to_smpl_frame=True, pose="tpose", exterior_surface=True,
            exterior_method="first_hit", exterior_ray_distance=0.,
            point_cloud_center_name=robot["sample_point_cloud_center"],
            tpose_qpos=robot["tpose_qpos"])
        assert sample["points"].shape == (4096, 3)
        assert np.isfinite(sample["points"]).all()
        assert sample["points"][:, 0].min() < -.5 and sample["points"][:, 0].max() > .5
        all_geoms = visual_mesh_geom_ids(m)
        assert np.array_equal(all_geoms, visual)
        face_counts = [len(geom_local_mesh(m, int(g))[1]) for g in all_geoms]
        starts = np.r_[0, np.cumsum(face_counts)]
        selected = np.searchsorted(starts[1:], sample["sample_face_ids"], side="right")
        counts = np.bincount(selected, minlength=len(all_geoms))
        result["sampled_points_per_body"] = {
            m.body(m.geom_bodyid[g]).name: int(counts[i]) for i,g in enumerate(all_geoms)}
        result["full_mesh_vertices"] = len(sample["vertices"])
        result["full_mesh_triangles"] = len(sample["faces"])
        result["sample_bounds_human_xyz_m"] = [sample["points"].min(0).tolist(), sample["points"].max(0).tolist()]
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/"opensprite_surface_probe.npz", **sample)
    output.mkdir(parents=True, exist_ok=True)
    (output/"opensprite_surface_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/umr-asset-validation")
    parser.add_argument("--build-sample", action="store_true")
    args = parser.parse_args()
    validate(args.output, args.build_sample)
