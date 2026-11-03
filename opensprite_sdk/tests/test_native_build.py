"""Compile and exercise the real C++ deployment ABI without opening devices."""
from pathlib import Path
import shutil
import subprocess
import pytest
from opensprite_sdk.backends.native_ipc import StatePacket
from opensprite_sdk.backends.hardware_config import HardwareConfig
from test_hardware_config import measured_fixture

ROOT=Path(__file__).resolve().parents[1]


def test_cpp_wire_and_motor_freshness(tmp_path):
    compiler=shutil.which('c++')
    if compiler is None:pytest.skip('C++ compiler is required for the optional native contract test')
    binary=tmp_path/'contract';wire=tmp_path/'state.bin'
    subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Wpedantic','-Werror','-pthread',str(ROOT/'native/tests/contract_test.cpp'),'-o',str(binary)],check=True,capture_output=True)
    files=HardwareConfig.from_dict(measured_fixture()).generate(tmp_path/'native-config')
    result=subprocess.run([str(binary),str(wire),files['motors'],files['kinematics'],files['joint_safety']],check=True,capture_output=True,text=True)
    assert 'PASS' in result.stdout
    packet=StatePacket.unpack(wire.read_bytes(),expected_hash=123,previous_sequence=0,now=1_006_000_000)
    assert packet.position[30]==3. and packet.host_receive_ns[30]==1_000_000_000
