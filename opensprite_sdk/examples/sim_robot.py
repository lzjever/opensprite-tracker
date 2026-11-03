#!/usr/bin/env python3
"""Start a separate MuJoCo robot process for Unitree-style SDK programs."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="legacy_v4")
    parser.add_argument("--endpoint", default="/tmp/opensprite-sdk.sock")
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--duration", type=float, default=10.)
    parser.add_argument("--enable-sim", action="store_true")
    parser.add_argument("--accelerated", action="store_true")
    args = parser.parse_args()
    from opensprite_sdk.service import run_sim_server
    run_sim_server(profile=args.profile, endpoint=args.endpoint, domain_id=args.domain,
                   realtime=not args.accelerated, duration=args.duration,
                   enable=args.enable_sim)


if __name__ == "__main__":
    main()
