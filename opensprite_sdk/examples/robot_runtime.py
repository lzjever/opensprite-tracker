#!/usr/bin/env python3
"""Run a robot endpoint; policy programs connect through the same SDK messages."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path

from opensprite_sdk.core.channel import ChannelFactoryInitialize, ChannelServer
from opensprite_sdk.runtime import RobotRuntime


def run(args):
    if not math.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("duration must be finite and positive")
    profile = args.profile or ("legacy_v4" if args.backend == "mujoco" else "physical_v5")
    bundle = None
    diagnostics = {"first_failure_seconds":None, "max_height_error_m":0.,
                   "max_gravity_error_sq":0., "full_reference_completed":False}
    if args.backend == "mujoco":
        if args.actuate or args.acknowledge_actuation:
            raise ValueError("hardware actuation flags do not apply to simulation")
        from opensprite_sdk.backends.mujoco import MujocoBackend
        backend = MujocoBackend(profile=profile, seed=args.seed)
        if args.bundle:
            from opensprite_sdk.policy import PolicyBundle
            bundle = PolicyBundle(args.bundle)
            if bundle.contract["profile_id"] != backend.profile_id:
                raise ValueError("initial reference and simulator profile differ")
            reference = bundle.reference
            backend.reset(qpos=reference["qpos"][0], qvel=reference["qvel"][0])
            floor = backend.model.geom("sdk_ground").id
            distances = [c.dist for c in backend.data.contact if floor in c.geom]
            lift = max(0., .001-min(distances)) if distances else 0.
            if lift:
                qpos = backend.data.qpos.copy()
                qpos[2] += lift
                backend.reset(qpos=qpos, qvel=reference["qvel"][0])
        allowed, shadow = args.enable_sim, False
    else:
        if args.enable_sim or args.bundle or args.viewer:
            raise ValueError("simulation initialization/viewer flags cannot be used on hardware")
        if not args.hardware_config:
            raise ValueError("hardware requires --hardware-config")
        if args.duration > 10:
            raise ValueError("initial hardware commissioning duration must not exceed10 seconds")
        if args.actuate and not args.acknowledge_actuation:
            raise ValueError("actuation requires --actuate and --acknowledge-actuation")
        from opensprite_sdk.backends.hardware import HardwareBackend
        from opensprite_sdk.backends.hardware_config import HardwareConfig
        hardware = HardwareConfig.load(args.hardware_config)
        if profile != hardware.data.get("profile_id", "physical_v5"):
            raise ValueError("hardware profile must match the native robot configuration")
        backend = HardwareBackend(hardware, profile_id=profile,
                                  allow_actuation=args.actuate)
        # CAN and IMU processes must already have been explicitly provisioned.
        # A shadow endpoint can grant only validation, never physical enable.
        allowed, shadow = True, not args.actuate
    server = (ChannelServer(args.endpoint, domain_id=args.domain)
              if args.transport == "unix" else nullcontext())
    try:
        with server:
            ChannelFactoryInitialize(args.domain, interface=args.interface, transport=args.transport,
                endpoint=args.endpoint if args.transport == "unix" else None)
            runtime = RobotRuntime(backend, allow_enable=allowed, shadow=shadow)
            def evaluate(current):
                if bundle is None or not current.started_motion:
                    return
                import numpy as np
                from opensprite_sdk.policy import quaternion_matrix
                truth = backend.debug_truth()
                frame = min(round(truth["time"]*50), bundle.contract["frames"]-1)
                height = float(abs(truth["qpos"][2]-bundle.reference["qpos"][frame, 2]))
                gravity = float(np.sum((quaternion_matrix(truth["qpos"][3:7])[2]
                    -bundle.reference["root_rotation"][frame, 2])**2))
                diagnostics["max_height_error_m"] = max(diagnostics["max_height_error_m"], height)
                diagnostics["max_gravity_error_sq"] = max(diagnostics["max_gravity_error_sq"], gravity)
                if diagnostics["first_failure_seconds"] is None and (height > .17 or gravity > 1.6):
                    diagnostics["first_failure_seconds"] = frame/50.
                diagnostics["full_reference_completed"] = frame >= bundle.contract["frames"]-1
            result = runtime.run(duration=args.duration, viewer=args.viewer, on_state=evaluate)
            if bundle is not None:
                result.update(diagnostics)
                result["standard_full_success"] = (diagnostics["full_reference_completed"]
                    and diagnostics["first_failure_seconds"] is None and result["fault"] is None)
    except BaseException:
        try:
            backend.close()
        except BaseException:
            pass
        raise
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mujoco", "hardware"), default="mujoco")
    parser.add_argument("--transport", choices=("dds", "unix", "memory"), default="dds")
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--interface", help="DDS NIC name, for example lo or eth0")
    parser.add_argument("--endpoint", default="/tmp/opensprite-sdk-runtime.sock")
    parser.add_argument("--profile", help="defaults to legacy_v4 in sim, physical_v5 on hardware")
    parser.add_argument("--bundle", type=Path, help="simulation-only explicit first reference pose")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-sim", action="store_true")
    parser.add_argument("--hardware-config", type=Path)
    parser.add_argument("--actuate", action="store_true")
    parser.add_argument("--acknowledge-actuation", action="store_true")
    parser.add_argument("--duration", type=float, default=240.)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path)
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2))
    if result["fault"] is not None:
        raise SystemExit(1)
