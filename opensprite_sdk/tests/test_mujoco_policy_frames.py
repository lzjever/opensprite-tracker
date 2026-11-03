"""Independent MuJoCo cross-check of sensor-only pelvis-to-torso reconstruction."""
import numpy as np

from opensprite_sdk.backends.mujoco import MujocoBackend
from opensprite_sdk.constants import JOINT_NAMES
from opensprite_sdk.policy import remap_imu


def test_pelvis_to_torso_gyro_and_gravity_match_mujoco_for_moving_waist():
    rng = np.random.default_rng(23)
    with MujocoBackend() as backend:
        model = backend.model
        root = model.body("pelvis_link").id
        chain = []
        for body_name in ("waist_roll_link", "waist_yaw_link"):
            body = model.body(body_name).id
            joint = model.body_jntadr[body]
            chain.append({"joint": model.joint(joint).name,
                "body_quat_wxyz": model.body_quat[body].tolist(),
                "axis": model.jnt_axis[joint].tolist(),
                "qref": float(model.qpos0[model.jnt_qposadr[joint]])})
        for _ in range(30):
            qpos = model.qpos0.copy()
            qpos[3:7] = rng.normal(size=4)
            qpos[3:7] /= np.linalg.norm(qpos[3:7])
            qpos[backend.qids] = rng.uniform(-.2, .2, 31)
            qvel = rng.normal(size=37)
            backend.reset(qpos=qpos, qvel=qvel)
            rotation = backend.data.xmat[root].reshape(3, 3)
            # cvel angular components are world-aligned; mj_objectVelocity BODY local uses the inertial frame.
            pelvis_gyro = rotation.T @ backend.data.cvel[root, :3]
            gravity = -rotation[2, :]
            gyro, projected = remap_imu(qpos[backend.qids], qvel[backend.vids], pelvis_gyro, gravity, chain)
            actual = backend.read().imu_state
            np.testing.assert_allclose(gyro, actual.gyroscope, atol=1e-12)
            np.testing.assert_allclose(projected, actual.projected_gravity, atol=1e-12)
