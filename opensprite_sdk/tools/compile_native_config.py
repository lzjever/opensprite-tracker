#!/usr/bin/env python3
"""Compile measured hardware calibration for the Python-free device gateway.

This is an offline configuration step. The resulting file contains no policy,
reference motion, model weights or code-loading hooks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from opensprite_sdk.backends.hardware_config import HardwareConfig
from opensprite_sdk.constants import JOINT_NAMES, JOINT_ORDER_HASH


def compile_hardware(source, output):
    source, output = Path(source), Path(output)
    config = HardwareConfig.load(source)
    data = config.data
    result = {
        "schema": "opensprite-native-hardware-v1",
        "hardware_identity": data["hardware_identity"],
        "configured": True,
        "estop_configured": data["estop"]["configured"],
        "synthetic": bool(data.get("synthetic", False)),
        "profile_id": data.get("profile_id", "physical_v5"),
        "joint_names": list(JOINT_NAMES),
        "joint_order_hash": JOINT_ORDER_HASH,
        "motor_names": list(config.motor_names),
        "motor_order_hash": config.motor_order_hash,
        "joint_to_motor": config.matrix.tolist(),
        "motor_to_joint": config.inverse.tolist(),
        "motor_offset": config.offset.tolist(),
        "joint_limits": [data["joint_limits"][name] for name in JOINT_NAMES],
        "native_socket": data["native"]["socket_path"],
        "imu_socket": data["imu"]["socket_path"],
        "imu_link": data["imu"]["attached_link"],
        "sensor_to_body_rotation": data["imu"]["sensor_to_body_rotation"],
        "imu_max_age_ns": int(data["imu"]["max_age_ms"] * 1e6),
        "native_max_age_ns": int(data["controller"].get("native_state_max_age_ms", 60) * 1e6),
        "motor_max_age_ns": 10_000_000,
        "source_max_age_ns": 60_000_000,
        "command_timeout_ns": 100_000_000,
        "joint_execution_hz": 500,
        "state_hz": 500,
        "can_wire_budget": config.can_wire_budget(),
        "source_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return {"output": str(output), "profile_id": result["profile_id"],
            "joint_execution_hz": 500, "state_hz": 500,
            "synthetic": result["synthetic"], "hardware_accessed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compile_hardware(args.hardware_config, args.output), indent=2))
