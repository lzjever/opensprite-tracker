"""Offline calibration compilation and native preflight without device access."""
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from opensprite_sdk.constants import JOINT_NAMES, JOINT_ORDER_HASH
from test_hardware_config import measured_fixture

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compile_native_hardware", ROOT/"tools/compile_native_config.py")
COMPILER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPILER)


@pytest.fixture
def compiled(tmp_path):
    source, output = tmp_path/"measured.json", tmp_path/"native.json"
    source.write_text(json.dumps(measured_fixture()))
    result = COMPILER.compile_hardware(source, output)
    return output, json.loads(output.read_text()), result


def test_offline_config_contains_measured_mapping_and_no_policy(compiled):
    path, config, report = compiled
    assert path.is_file() and report["hardware_accessed"] is False
    assert config["synthetic"] is True
    assert config["joint_names"] == list(JOINT_NAMES)
    assert config["joint_order_hash"] == JOINT_ORDER_HASH
    assert config["joint_execution_hz"] == config["state_hz"] == 500
    np.testing.assert_allclose(np.array(config["joint_to_motor"])@np.array(config["motor_to_joint"]), np.eye(31), atol=1e-12)
    assert config["motor_max_age_ns"] == 10_000_000
    assert config["imu_max_age_ns"] <= 30_000_000
    assert not {"actor", "checkpoint", "onnx", "reference", "model"}.intersection(config)


def native_binary():
    path=ROOT/"native-runtime/build/opensprite_robot"
    if not path.is_file():
        pytest.skip("build the optional native robot runtime with tools/build_robot.sh")
    return str(path)


def fail_preflight(config, tmp_path, extra=()):
    report=tmp_path/"report.json"
    command=[native_binary(), "--backend", "hardware", "--config", str(config),
             "--duration", "1", "--report", str(report), *extra]
    completed=subprocess.run(command, text=True, capture_output=True, timeout=3)
    assert completed.returncode != 0
    body=json.loads(report.read_text())
    assert body["fault"] and body["hardware_tested"] is False
    return body["fault"]


def test_synthetic_config_cannot_actuate_before_device_connect(compiled, tmp_path):
    config, _, _=compiled
    error=fail_preflight(config,tmp_path,("--actuate","--acknowledge-actuation","--allow-enable"))
    assert "synthetic" in error


def test_native_matrix_tampering_rejected_before_device_connect(compiled, tmp_path):
    path, config, _=compiled
    config["motor_to_joint"][0][0] += 1
    path.write_text(json.dumps(config))
    assert "inverse" in fail_preflight(path,tmp_path)


def test_hardware_rejects_privileged_truth_output_before_device_connect(compiled, tmp_path):
    path, _, _=compiled
    assert "simulation-only" in fail_preflight(path,tmp_path,("--truth-output",str(tmp_path/"trace.jsonl")))


@pytest.mark.parametrize("argument", ["4294967296", "-1", "1junk"])
def test_native_domain_parser_does_not_wrap_or_ignore_suffix(compiled, tmp_path, argument):
    path, _, _=compiled
    assert "domain" in fail_preflight(path,tmp_path,("--domain",argument)).lower()
