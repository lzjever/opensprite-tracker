#!/usr/bin/env python3
"""Same channel client for simulation or an explicitly armed hardware service."""
import argparse
import json
import math
import time
from importlib.resources import files

from opensprite_sdk.client import LowLevelClient, joint_command
from opensprite_sdk.core.channel import ChannelFactoryInitialize


def run(args):
    ChannelFactoryInitialize(args.domain, transport="unix", endpoint=args.endpoint)
    with LowLevelClient() as client:
        first = client.Read()
        origin = [j.q for j in first.joint_state]
        if not all(j.valid for j in first.joint_state):
            raise RuntimeError("incomplete joint feedback")
        kp, kd = [args.kp]*31, [args.kd]*31
        if args.gains_profile:
            if args.gains_profile != first.profile_id:
                raise ValueError("gains profile differs from robot profile")
            profile=json.loads((files("opensprite_sdk")/"profiles"/(args.gains_profile+".json")).read_text())
            kp,kd=profile["kp"],profile["kd"]
        if args.gains:
            data=json.loads(args.gains.read_text())
            if data.get("profile_id") != first.profile_id:
                raise ValueError("reviewed gains must name the current profile")
            kp,kd=data["kp"],data["kd"]
        start = time.monotonic()
        state = first
        count = 0
        while time.monotonic()-start < args.duration:
            elapsed = time.monotonic()-start
            target = origin.copy()
            # A bounded, slow, single joint experiment; amplitude defaults zero.
            target[args.joint] += args.amplitude*math.sin(2*math.pi*.2*elapsed)
            command = joint_command(state, target, kp, kd)
            client.Write(command)
            count += 1
            time.sleep(.02)
            state = client.Read()
            if state.faults:
                raise RuntimeError(str(state.faults))
        disable=joint_command(state, origin, [0.]*31, [0.]*31)
        for joint in disable.joint_cmd:
            joint.mode=0
        client.Write(disable)
        if state.clock_domain == "sim":
            # Publisher acceptance is not execution acknowledgement. A final
            # enabled snapshot may already be queued when mode0 is sent.
            source_sequence, source_session = state.seq, state.session_id
            deadline=time.monotonic()+1.
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    raise TimeoutError("simulator did not acknowledge disable")
                state=client.Read(timeout=remaining)
                if state.session_id!=source_session or state.faults:
                    raise RuntimeError("simulator session changed or faulted while disabling")
                if state.seq>source_sequence and state.status=="disabled":
                    break
        return {"commands":count, "last_state_sequence":state.seq, "profile":state.profile_id,
                "clock_domain":state.clock_domain, "status":state.status}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="/tmp/opensprite-sdk.sock")
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--duration", type=float, default=2.)
    parser.add_argument("--kp", type=float, default=2.)
    parser.add_argument("--kd", type=float, default=.12)
    parser.add_argument("--joint", type=int, choices=range(31), default=30)
    parser.add_argument("--amplitude", type=float, default=0.)
    parser.add_argument("--gains-profile", choices=("legacy_v4", "runtime_timing_v4"))
    from pathlib import Path
    parser.add_argument("--gains", type=Path, help="reviewed JSON profile_id/kp/kd arrays")
    args=parser.parse_args()
    if not all(math.isfinite(v) for v in (args.duration,args.kp,args.kd,args.amplitude)) or args.duration<=0 or args.kp<0 or args.kd<0 or abs(args.amplitude)>.05:
        parser.error("finite positive duration/nonnegative gains and amplitude <= .05rad required")
    print(json.dumps(run(args), indent=2))
