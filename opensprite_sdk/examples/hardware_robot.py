#!/usr/bin/env python3
"""Expose an explicitly launched Jetson native controller through SDK channels."""
import argparse
import json
import math
from pathlib import Path
import time

from opensprite_sdk.backends.hardware import HardwareBackend
from opensprite_sdk.core.channel import ChannelFactoryInitialize, ChannelServer
from opensprite_sdk.service import BackendService


def run(args):
    if args.actuate and not args.acknowledge_actuation:
        raise ValueError("actuation requires --acknowledge-actuation and an explicitly powered native supervisor")
    # This process never launches CAN or changes device privileges. Native and
    # IMU processes are provisioned explicitly using docs/hardware.md.
    backend=HardwareBackend(args.hardware, allow_actuation=args.actuate)
    Path(args.endpoint).parent.mkdir(parents=True, exist_ok=True)
    count=0
    with ChannelServer(args.endpoint, domain_id=args.domain):
        ChannelFactoryInitialize(args.domain, transport="unix", endpoint=args.endpoint)
        with BackendService(backend, shadow=not args.actuate) as service:
            service.publish_state()
            if args.actuate:
                backend.enable()
                service.publish_state()
            started=time.monotonic()
            while time.monotonic()-started < args.duration:
                service.tick()
                count+=1
                if service.stop_requested:
                    break
    return {"state_publications":count, "actuation_requested":args.actuate,
            "motor_disable_verification":"inspect native completion report"}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hardware", required=True, type=Path)
    p.add_argument("--endpoint", default="/tmp/opensprite-sdk-real.sock")
    p.add_argument("--domain", type=int, default=2)
    p.add_argument("--duration", type=float, default=5.)
    p.add_argument("--actuate", action="store_true")
    p.add_argument("--acknowledge-actuation", action="store_true")
    args=p.parse_args()
    if not math.isfinite(args.duration) or not 0<args.duration<=10:
        p.error("duration must be in (0,10] seconds for commissioning")
    print(json.dumps(run(args), indent=2))
