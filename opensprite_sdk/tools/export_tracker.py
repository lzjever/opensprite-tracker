#!/usr/bin/env python3
"""Export a trusted goal-sim2real checkpoint to a standalone CPU ONNX bundle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from opensprite_sdk.constants import JOINT_NAMES
from opensprite_sdk.policy import quaternion_matrix


class FrozenActor(nn.Module):
    def __init__(self, weights):
        super().__init__()
        for prefix in ("enc", "dec"):
            keys = sorted((k for k in weights if k.startswith(prefix+".") and k.endswith(".weight")),
                          key=lambda k: int(k.split(".")[1]))
            layers = []
            for index, key in enumerate(keys):
                out_dim, in_dim = weights[key].shape
                layer = nn.Linear(in_dim, out_dim)
                layer.weight.data.copy_(weights[key])
                layer.bias.data.copy_(weights[key.replace("weight", "bias")])
                layers.append(layer)
                if index < len(keys)-1:
                    layers.append(nn.ELU())
            setattr(self, prefix, nn.Sequential(*layers))
        for name in ("obs", "goal"):
            self.register_buffer(name+"_mean", weights[name+"_norm.mean"].clone())
            self.register_buffer(name+"_std", (weights[name+"_norm.var"]+1e-4).sqrt())

    def forward(self, observation_history, goal):
        z = torch.tanh(self.enc((goal-self.goal_mean)/self.goal_std))
        observations = ((observation_history-self.obs_mean)/self.obs_std).flatten(-2)
        return self.dec(torch.cat((z, observations), -1))


def reference_arrays(motion, goal_frames=10):
    with np.load(motion, allow_pickle=False) as data:
        if not np.isclose(float(data["fps"].reshape(-1)[0]), 50):
            raise ValueError("reference must be 50Hz")
        joint_names = data["joint_names"].astype(str).tolist()
        ids = [joint_names.index(n) for n in JOINT_NAMES]
        root = data["body_names"].astype(str).tolist().index("pelvis_link")
        q, dq = data["joint_pos"][:, ids], data["joint_vel"][:, ids]
        p, quat = data["body_pos_w"][:, root], data["body_quat_w"][:, root]
        v, w = data["body_lin_vel_w"][:, root], data["body_ang_vel_w"][:, root]
        if "motion_lengths" in data and (len(data["motion_lengths"]) != 1 or int(data["motion_lengths"][0]) != len(q)):
            raise ValueError("export one complete reference clip at a time")
    rotation = quaternion_matrix(quat).astype(np.float32)
    fx, fy = rotation[:, 0, 0], rotation[:, 1, 0]
    front = fx*fx+fy*fy >= 1e-8
    fx, fy = np.where(front, fx, rotation[:, 1, 1]), np.where(front, fy, -rotation[:, 0, 1])
    norm = np.sqrt(fx*fx+fy*fy)
    yaw = np.zeros_like(rotation)
    yaw[:, 0, 0] = yaw[:, 1, 1] = fx/norm
    yaw[:, 1, 0], yaw[:, 0, 1], yaw[:, 2, 2] = fy/norm, -fy/norm, 1
    ht = yaw.transpose(0, 2, 1)
    rt = rotation.transpose(0, 2, 1)
    local_v = np.einsum("nij,nj->ni", rt, v)
    local_w = np.einsum("nij,nj->ni", rt, w)
    future = np.minimum(np.arange(len(q))[:, None]+np.arange(1, goal_frames+1), len(q)-1)
    dp = np.einsum("nij,nkj->nki", ht, p[future]-p[:, None])
    rel = np.einsum("nij,nkjl->nkil", ht, rotation[future])[..., :2].reshape(len(q), goal_frames, 6)
    goals = np.concatenate((dp, rel, local_v[future], local_w[future], q[future], .1*dq[future]), -1).reshape(len(q), -1).astype(np.float32)
    qpos = np.concatenate((p, quat, q), -1).astype(np.float64)
    # MuJoCo free-joint translation is world-frame, angular velocity is local.
    qvel = np.concatenate((v, local_w, dq), -1).astype(np.float64)
    return {"goals": goals, "qpos": qpos, "qvel": qvel,
            "root_rotation": rotation, "joint_names": np.asarray(JOINT_NAMES)}


def imu_chain(model_path):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(model_path))
    body, root = model.body("waist_yaw_link").id, model.body("pelvis_link").id
    chain = []
    while body != root:
        if body <= 0 or model.body_jntnum[body] != 1:
            raise ValueError("unsupported IMU kinematic chain")
        j = int(model.body_jntadr[body])
        chain.append({"joint":model.joint(j).name, "body_quat_wxyz":model.body_quat[body].tolist(),
                      "axis":model.jnt_axis[j].tolist(), "qref":float(model.qpos0[model.jnt_qposadr[j]])})
        body = int(model.body_parentid[body])
    return {"source_link":"pelvis_link", "target_link":"waist_yaw_link", "chain":chain[::-1]}


def export(checkpoint, motion, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("choose a new bundle directory")
    # PyTorch pickle checkpoints are trusted local inputs, never remote messages.
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights, robot = ck["pi"], ck["robot_contract"]
    args = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
    variant = ck.get("training_variant", args.get("training_variant"))
    if (variant != "goal-sim2real" or weights["obs_norm.mean"].numel() != 99
            or weights["goal_norm.mean"].numel() != 770 or int(args.get("hist", 10)) != 10
            or tuple(robot["joints"]) != JOINT_NAMES or ck.get("torque_lag_enabled", True)
            or robot["physics_dt"] != .005 or robot["policy_dt"] != .02):
        raise ValueError("export requires the supported no-lag legacy_v4 goal-sim2real checkpoint")
    from opensprite_sdk.backends.mujoco import load_profile, PACKAGE_ROOT
    profile = load_profile("legacy_v4")
    control = dict(robot["control"])
    for key, field in (("default_q", "default_pos_rad"), ("kp", "kp"), ("kd", "kd"), ("effort_limit", "command_effort_nm")):
        np.testing.assert_allclose(profile[key], [control[n][field] for n in JOINT_NAMES], rtol=0, atol=1e-12)
    actor = FrozenActor(weights).eval().requires_grad_(False)
    reference = reference_arrays(motion)
    rng = np.random.default_rng(42)
    history = torch.tensor(rng.normal(0, .2, (4, 10, 99)), dtype=torch.float32)
    goal = torch.tensor(reference["goals"][[0, 100, len(reference["goals"])//2, -1]])
    with torch.inference_mode():
        expected = actor(history, goal).numpy()
    output.mkdir(parents=True)
    torch.onnx.export(actor, (history, goal), str(output/"actor.onnx"), input_names=["observation_history", "goal"],
                      output_names=["action"], dynamic_axes={"observation_history":{0:"batch"}, "goal":{0:"batch"}, "action":{0:"batch"}},
                      opset_version=17, dynamo=False)
    np.savez_compressed(output/"reference.npz", **reference)
    np.savez_compressed(output/"parity.npz", observation_history=history.numpy(), goal=goal.numpy(), action=expected)
    contract = {"schema":"opensprite-policy-v1", "training_variant":"goal-sim2real", "profile_id":"legacy_v4",
                "joint_names":list(JOINT_NAMES), "obs_dim":99, "goal_dim":770, "hist":10, "policy_hz":50,
                "frames":len(reference["goals"]), "duration_seconds":(len(reference["goals"])-1)/50,
                "imu_link":"waist_yaw_link", "imu_remap":imu_chain(PACKAGE_ROOT/profile["robot_model"]),
                "default_q":profile["default_q"], "kp":profile["kp"], "kd":profile["kd"],
                "action_scale":float(robot["action_scale"]), "action_clip":20, "checkpoint_iteration":ck.get("iteration"),
                "source_checkpoint_sha256":hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                "source_motion_sha256":hashlib.sha256(Path(motion).read_bytes()).hexdigest(),
                "hardware_qualified":False, "files":{name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                    for name in ("actor.onnx", "reference.npz", "parity.npz")}}
    (output/"contract.json").write_text(json.dumps(contract, indent=2)+"\n")
    print(json.dumps({"bundle":str(output), "frames":contract["frames"], "iteration":contract["checkpoint_iteration"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    export(args.checkpoint, args.motion, args.output)
