# OpenSprite SDK

[简体中文](README.md)

An independent SDK for OpenSprite controllers, simulation and Jetson deployment. The interface follows Unitree SDK2 conventions: ChannelFactory, Publisher, Subscriber, LowCmd and LowState. Messages use OpenSprite's own31-joint order and mechanism mapping; they do not share Unitree robot wire layouts.

The native C++ `opensprite_robot` process receives typed Fast DDS messages and selects a MuJoCo or hardware backend. C++, Python and ROS 2 clients use the same protocol. The robot process loads no Actor, ONNX or training code. Hardware connects through binary local IPC to independent CAN and IMU programs; networking stays outside the CAN scheduling loop.

| Clock | Contract |
|---|---|
| Joint command dispatch | All31 joints at500Hz, every2ms |
| MuJoCo physics and simulated sensors | 200Hz, every5ms |
| Dance Actor | 50Hz, every20ms |
| CAN transport scheduler | 2kHz phases; each motor command/poll target500Hz |

Command dispatch and physics have separate clocks. Sensor timestamps retain actual sample times. These configured rates do not establish measured motor feedback coverage or hard real-time performance.

## Build

From this directory, with Python3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[sim,deploy,test]'
sudo apt-get install cmake g++ libfastrtps-dev libfastcdr-dev nlohmann-json3-dev
bash tools/build_dds.sh
bash tools/build_robot.sh
```

The build script locates MuJoCo C headers/library in the active environment; `MUJOCO_ROOT` can select another installation. The resulting executable needs no Python interpreter. `bash tools/build_robot.sh --hardware-only` builds the Jetson-facing endpoint without MuJoCo.

## Robot endpoint and SDK clients

Prepare an actor-free, portable robot scene and run the endpoint:

```bash
python tools/prepare_sim_scene.py --output outputs/standing-scene
native-runtime/build/opensprite_robot --config outputs/standing-scene/scene.json \
  --domain 42 --interface lo --allow-enable --duration 240 \
  --report outputs/robot-report.json
```

A controller claims ownership and enables separately. The endpoint validates source states, sessions, ordering, freshness and timeouts. DDS write success is not motor execution confirmation.

The C++ examples build against the installed SDK:

```bash
cmake --install native-dds/build --prefix "$PWD/outputs/sdk-install"
cmake -S examples/cpp -B build/cpp -DCMAKE_PREFIX_PATH="$PWD/outputs/sdk-install"
cmake --build build/cpp --parallel 2
build/cpp/opensprite_low_level --domain 42 --duration 2
```

The example reads state by default. `--hold --rate-hz 500` explicitly claims/enables and sends a measured-pose target; its low gains demonstrate the interface, not free-standing behavior. See the [C++ SDK](docs/cpp-sdk.md) and [native runtime](native-runtime/README.md).

Python read-only example:

```python
from opensprite_sdk.core.channel import ChannelFactoryInitialize
from opensprite_sdk.runtime import ManagementClient

ChannelFactoryInitialize(42, "lo", transport="dds")
with ManagementClient() as robot:
    state = robot.Read(timeout=5)
    q = [joint.q for joint in state.joint_state]
```

DDS defaults to loopback. Select an explicit interface for networking and a distinct domain per robot. Canonical ROS 2 definitions are in `ros2/opensprite_msgs`; SDK deployment does not require ROS 2. See the [wire and time contracts](docs/dds.md).

## Dance policy

Export a trusted checkpoint and50Hz retargeted motion:

```bash
python -m pip install -e '.[export]'
python tools/export_tracker.py --checkpoint /path/to/model.pt \
  --motion /path/to/lafan_dance.npz --output outputs/dance-bundle
python tools/prepare_sim_scene.py --motion /path/to/lafan_dance.npz \
  --output outputs/dance-scene
```

In the parent demo repository, explicit preparation inputs are `../artifacts/checkpoints/goal-sim2real_model_024000.pt` and `../data/motions/lafan_dance.npz`. Prepared scenes and bundles can be relocated independently.

Start the endpoint with the dance scene, then start the policy in another process:

```bash
# Robot: no policy weights or Actor imports.
native-runtime/build/opensprite_robot --config outputs/dance-scene/scene.json \
  --domain 42 --allow-enable --duration 240 --report outputs/dance-robot.json
# Client: CPU ONNX Actor with sensor-only observations.
python examples/remote_policy.py --bundle outputs/dance-bundle --domain 42
```

The Actor retains99 observations,10 history frames,770 reference-only goal values, normalization and action clipping. It selects50Hz windows from real observations using bounded history, without fabricating sample timestamps. Missing windows fail explicitly. Tracking failure itself does not reset or stop the full reference.

Run a complete separate-process check with external truth evaluation, without recording:

```bash
python tools/check_native_policy.py --scene outputs/dance-scene/scene.json \
  --bundle outputs/dance-bundle --output outputs/native-acceptance/full-dance.json
```

An independent live viewer reads the robot's optional `--truth-output outputs/truth.jsonl` diagnostics:

```bash
MUJOCO_GL=glfw python examples/view_native.py --scene outputs/dance-scene/scene.json \
  --truth outputs/truth.jsonl
```

The viewer renders copied poses without controlling physics or recording. Python memory/Unix and simulation examples remain available for compatibility; the native executable is the deployment entry point.

## Hardware and validation

The target is NVIDIA Jetson Orin Nano Super Developer Kit. See the [hardware guide](docs/hardware.md) for per-device calibration, mechanisms, independent CAN/IMU setup and native report checks. Hardware defaults to shadow and keeps explicit local actuation permission. No system services are created.

The current v4 dance policy is not qualified for the newer shoulder revision. Jetson ARM64, actual USB IMU protocol, CAN/USB latency and powered motion still require device validation.

```bash
python -m pytest tests -q
python tools/check_native_acceptance.py --ros2 --output outputs/native-acceptance/acceptance.json
python tools/check_standalone.py --robot
```

The [plan](docs/plan/native-control-sdk-500hz-20261104.md) records team decisions. The [validation report](docs/validation-native-20261104.md) records tests and the complete dance run. This folder contains its code, robot assets, native executors, messages, configuration and licenses. `.reference`, `outputs`, parent code and local calibration are not deployment dependencies. License: AGPL-3.0-only, with [third-party notices](THIRD_PARTY_NOTICES.md).
