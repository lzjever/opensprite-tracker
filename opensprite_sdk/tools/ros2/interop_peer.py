#!/usr/bin/env python3
"""ROS 2 side of the native DDS interoperability acceptance test.

Run inside a ROS environment after building ros2/opensprite_msgs. This process
uses generated ROS type support and rclpy, never the SDK's DDS implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.convert import message_to_ordereddict
from opensprite_msgs import msg as ros_messages


def construct(name, values):
    result = getattr(ros_messages, name)()
    for field, value in values.items():
        if field == "joint_cmd":
            value = [construct("JointCmd", item) for item in value]
        elif field == "joint_state":
            value = [construct("JointState", item) for item in value]
        elif field == "imu_state":
            value = construct("IMUState", value)
        setattr(result, field, value)
    return result


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--duration", type=float, default=20.)
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    rclpy.init()
    node = Node("opensprite_sdk_interop_acceptance")
    received, errors, counts = {}, {}, {}
    publishers, subscriptions = [], []
    report = {"schema": "opensprite-ros2-interop-peer-v1", "received": received,
              "counts": counts, "errors": errors, "passed": False}
    try:
        for name, envelope in fixtures.items():
            cls = getattr(ros_messages, name)
            qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                durability=DurabilityPolicy.VOLATILE,
                reliability=(ReliabilityPolicy.RELIABLE if name.startswith("Control")
                             else ReliabilityPolicy.BEST_EFFORT))
            publishers.append((node.create_publisher(cls, f"/{args.prefix}/ros/{name}", qos),
                               construct(name, envelope["data"])))

            def receive(message, name=name, expected=envelope["data"]):
                plain = json.loads(json.dumps(message_to_ordereddict(message)))
                received[name] = plain
                counts[name] = counts.get(name, 0) + 1
                if plain != expected:
                    errors[name] = {"expected": expected, "actual": plain}
                report["passed"] = set(received) == set(fixtures) and not errors
                atomic_json(args.output, report)

            subscriptions.append(node.create_subscription(
                cls, f"/{args.prefix}/native/{name}", receive, qos))
        print(json.dumps({"event": "ros2_peer_ready", "types": list(fixtures)}), flush=True)
        deadline = time.monotonic() + args.duration
        next_publish = time.monotonic()
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.01)
            if time.monotonic() >= next_publish:
                for publisher, message in publishers:
                    publisher.publish(message)
                next_publish = time.monotonic() + .05
        report["passed"] = set(received) == set(fixtures) and not errors
        atomic_json(args.output, report)
        print(json.dumps({"event": "ros2_peer_finished", "passed": report["passed"]}), flush=True)
        return 0 if report["passed"] else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
