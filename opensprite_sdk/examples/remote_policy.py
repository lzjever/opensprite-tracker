#!/usr/bin/env python3
"""Run the CPU ONNX actor against an independent sim or hardware robot endpoint."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import sys

from opensprite_sdk.core.channel import ChannelFactoryInitialize
from opensprite_sdk.policy import PolicyBundle
from opensprite_sdk.runtime import ManagementClient


def next_state(client, previous, *, timeout=1.):
    """Ignore idle discovery packets and require actual robot-time progress."""
    deadline = time.monotonic()+timeout
    while True:
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError("robot state time did not advance")
        state = client.Read(timeout=remaining)
        if state.session_id != previous.session_id or state.clock_domain != previous.clock_domain:
            raise RuntimeError("robot session/clock changed; explicit policy restart required")
        if state.seq > previous.seq and state.stamp_ns > previous.stamp_ns:
            delta = state.stamp_ns-previous.stamp_ns
            if state.clock_domain == "sim" and delta != 20_000_000:
                raise RuntimeError("missed policy observation / simulation phase drift; expected20ms")
            if state.clock_domain == "monotonic" and (
                    state.seq != previous.seq+1 or delta > 40_000_000):
                raise RuntimeError("missed native observation / hardware policy deadline")
            return state


def run(args):
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration < .02):
        raise ValueError("duration must include at least one20ms policy step")
    bundle = PolicyBundle(args.bundle)
    transitions = bundle.contract["frames"]-1
    if args.duration is not None:
        transitions = min(transitions, int(args.duration*50))
    ChannelFactoryInitialize(args.domain, interface=args.interface, transport=args.transport,
        endpoint=args.endpoint if args.transport == "unix" else None)
    began = time.monotonic()
    frames = 0
    enabled = False
    disable_reply = None
    with ManagementClient(controller_session=bundle.session_id) as client:
        first = client.Read(timeout=10.)
        if first.profile_id != bundle.contract["profile_id"]:
            raise ValueError("policy/robot profile mismatch")
        # Current bundles have no hardware qualification. A remote monotonic
        # endpoint is allowed only for an explicitly selected shadow run.
        if first.clock_domain != "sim" and first.status != "shadow":
            if bundle.contract.get("hardware_qualified") is not True:
                raise ValueError("policy bundle is not qualified for hardware actuation")
        claimed = False
        failure = None
        try:
            client.request("claim", state=first)
            claimed = True
            client.request("enable")
            enabled = True
            state = client.Read(timeout=3.)
            for frame in range(transitions):
                # Sensor stamps belong to the backend clock. Never compare
                # remote robot monotonic time with this machine's monotonic.
                command = bundle.command(state, frame, now_ns=state.stamp_ns)
                client.Write(command)
                frames += 1
                if frame+1 < transitions:
                    state = next_state(client, state)
            # Backend acceptance is explicit, and simulation must also finish
            # the last20ms policy window. This is never a measured-motor ACK.
            deadline = time.monotonic()+1.
            minimum_stamp = command.stamp_ns+(20_000_000 if state.clock_domain == "sim" else 0)
            while True:
                state = next_state(client, state, timeout=max(.001, deadline-time.monotonic()))
                if (state.accepted_controller_session == bundle.session_id
                        and state.accepted_command_seq >= command.seq
                        and state.stamp_ns >= minimum_stamp):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("final command has no matching backend acceptance")
            client.state = state
        except BaseException as error:
            failure = error
            raise
        finally:
            try:
                if enabled:
                    disable_reply = client.request("disable", timeout=3.)
                elif claimed:
                    client.request("release", timeout=1.)
            except BaseException as cleanup_error:
                if failure is None:
                    raise
                print(json.dumps({"cleanup_error":str(cleanup_error),
                    "original_error":str(failure)}), file=sys.stderr, flush=True)
    result = {"profile_id":bundle.contract["profile_id"],
        "checkpoint_iteration":bundle.contract.get("checkpoint_iteration"),
        "transitions_sent":frames, "reference_seconds_sent":frames/50.,
        "full_reference_sent":frames == bundle.contract["frames"]-1,
        "last_backend_accepted_command_seq":state.accepted_command_seq,
        "tracking_success_evaluated":False,
        "disable_request_accepted":disable_reply.accepted if disable_reply else False,
        "disable_execution_verified":disable_reply.execution_verified if disable_reply else False,
        "wall_seconds":time.monotonic()-began, "video_recorded":False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--transport", choices=("dds", "unix"), default="dds")
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--interface", help="DDS NIC name, for example lo or eth0")
    parser.add_argument("--endpoint", default="/tmp/opensprite-sdk-runtime.sock")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--output", type=Path)
    print(json.dumps(run(parser.parse_args()), indent=2))
