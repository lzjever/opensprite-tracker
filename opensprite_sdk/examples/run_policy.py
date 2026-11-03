#!/usr/bin/env python3
"""Run a standalone dance actor over the same SDK channels in sim or hardware."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from opensprite_sdk.client import LowLevelClient
from opensprite_sdk.core.channel import ChannelFactoryInitialize
from opensprite_sdk.policy import PolicyBundle, quaternion_matrix
from opensprite_sdk.service import BackendService


def run(args):
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration<.02):
        raise ValueError("duration must include at least one 20ms policy step")
    bundle = PolicyBundle(args.bundle)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("choose a new output directory")
    if args.backend == "mujoco":
        from opensprite_sdk.backends.mujoco import MujocoBackend
        backend = MujocoBackend(profile=bundle.contract["profile_id"], seed=args.seed)
        backend.reset(qpos=bundle.reference["qpos"][0], qvel=bundle.reference["qvel"][0])
        # Simulation-only placement: remove initial floor penetration without
        # using this privileged information in any actor observation.
        floor=backend.model.geom("sdk_ground").id
        distances = [c.dist for c in backend.data.contact if floor in c.geom]
        lift = max(0., .001-min(distances)) if distances else 0.
        if lift:
            qpos = backend.data.qpos.copy()
            qpos[2] += lift
            backend.reset(qpos=qpos, qvel=bundle.reference["qvel"][0])
        backend.enable()
        steps = round(backend.policy_dt/backend.dt)
    else:
        if args.video:
            raise ValueError("hardware recording uses external cameras; SDK truth rendering is sim only")
        if not args.hardware_config:
            raise ValueError("hardware requires --hardware-config")
        if args.actuate and bundle.contract.get("hardware_qualified", False) is not True:
            raise ValueError("this policy bundle has no hardware qualification; use shadow first")
        from opensprite_sdk.backends.hardware import HardwareBackend
        backend = HardwareBackend(args.hardware_config, allow_actuation=args.actuate)
        if bundle.contract["profile_id"] != backend.profile_id:
            raise ValueError("policy/hardware revision mismatch; retrain or qualify the matching profile")
        steps = 0
        lift = 0.
    output.mkdir(parents=True)
    max_transitions = bundle.contract["frames"]-1
    if args.duration is not None:
        max_transitions = min(max_transitions, int(args.duration*50))
    ChannelFactoryInitialize(21, transport="memory")
    records, first_failure, writer = [], None, None
    start_wall = time.monotonic()
    summary = {"backend":args.backend, "profile":bundle.contract["profile_id"],
               "checkpoint_iteration":bundle.contract.get("checkpoint_iteration"),
               "source_frames":bundle.contract["frames"], "first_failure_seconds":None,
               "initial_lift_m":lift, "hardware_tested":False}
    try:
        shadow=args.backend == "hardware" and not args.actuate
        with BackendService(backend, shadow=shadow) as service, LowLevelClient() as client:
            service.publish_state()
            if args.backend == "hardware" and args.actuate:
                backend.enable()
                service.publish_state()
            for frame in range(max_transitions):
                state = client.Read()
                command = bundle.command(state, frame,
                    now_ns=time.monotonic_ns() if state.clock_domain == "monotonic" else state.stamp_ns)
                client.Write(command)
                # In-process memory delivery is synchronous; the exact same
                # command and state classes are used by the separate Unix demo.
                if not service.tick(steps=steps):
                    raise RuntimeError("policy command was not delivered")
                record = {"frame":frame+1, "q_target":[j.q for j in command.joint_cmd],
                          "action":bundle.previous_action.tolist(), "state_sequence":state.seq}
                if args.backend == "mujoco":
                    truth = backend.debug_truth()
                    arrived = frame+1
                    ref = bundle.reference["qpos"][arrived]
                    rotation = quaternion_matrix(truth["qpos"][3:7])
                    height = abs(truth["qpos"][2]-ref[2])
                    gravity = float(np.sum((rotation[2]-bundle.reference["root_rotation"][arrived, 2])**2))
                    failed = height>.17 or gravity>1.6
                    if failed and first_failure is None:
                        first_failure = arrived/50
                    record.update(qpos=truth["qpos"].tolist(), qvel=truth["qvel"].tolist(),
                                  height_error=float(height), gravity_error_sq=gravity,
                                  scored=first_failure is None)
                    if args.video and frame % 2 == 0:
                        import imageio.v2 as imageio
                        from PIL import Image, ImageDraw
                        if writer is None:
                            writer=imageio.get_writer(str(output/"simulation.mp4"), fps=25, codec="libx264", quality=8)
                        image=Image.fromarray(backend.render(width=960, height=540))
                        draw=ImageDraw.Draw(image)
                        draw.rectangle((0, 0, 960, 55), fill=(15, 20, 30))
                        draw.text((12, 9), f"OpenSprite SDK | {bundle.contract['profile_id']} | t={arrived/50:.2f}s | model {bundle.contract.get('checkpoint_iteration')}", fill="white")
                        draw.text((12, 31), f"height error {height:.3f} m | gravity sq {gravity:.3f} | "+("POST-FAILURE / UNSCORED" if first_failure is not None else "scored"), fill="orange" if first_failure is not None else "white")
                        writer.append_data(np.asarray(image))
                records.append(record)
                if first_failure is not None and not args.continue_after_failure:
                    break
                if args.realtime or args.backend == "hardware":
                    deadline = start_wall+(frame+1)/50
                    remaining = deadline-time.monotonic()
                    if args.backend == "hardware" and remaining < -.04:
                        raise RuntimeError("policy missed two periods; refusing phase drift")
                    time.sleep(max(0., remaining))
        summary.update(transitions=len(records), simulated_seconds=len(records)/50,
                       first_failure_seconds=first_failure,
                       full_reference_completed=len(records)==bundle.contract["frames"]-1,
                       standard_full_success=(args.backend=="mujoco" and first_failure is None
                                              and len(records)==bundle.contract["frames"]-1),
                       wall_seconds=time.monotonic()-start_wall)
    except BaseException as error:
        summary.update(error=f"{type(error).__name__}: {error}", transitions=len(records),
                       first_failure_seconds=first_failure, standard_full_success=False)
        raise
    finally:
        if writer is not None:
            writer.close()
        (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        if records:
            np.savez_compressed(output/"rollout.npz", **{key:np.asarray([r[key] for r in records]) for key in records[0]})
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--backend", choices=("mujoco", "hardware"), default="mujoco")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--continue-after-failure", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--hardware-config", type=Path)
    parser.add_argument("--actuate", action="store_true")
    args=parser.parse_args()
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration<=0):
        parser.error("duration must be finite and positive")
    print(json.dumps(run(args), indent=2))
