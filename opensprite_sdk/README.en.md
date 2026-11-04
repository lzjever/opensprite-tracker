# OpenSprite SDK

[简体中文](README.md)

A self-contained SDK for OpenSprite simulation and Jetson deployment. It follows the SDK2 publishing/subscription style: `ChannelFactoryInitialize`, `ChannelPublisher`, `ChannelSubscriber`, `LowCmd_`, and `LowState_`. The same joint commands drive a MuJoCo backend or a native hardware backend.

All public arrays use OpenSprite's31-joint order. `motor_cmd` and `motor_state` are spelling aliases for joint-space data. Physical motor mapping, both ankle differentials, and the head pitch/roll differential belong to the hardware backend. Transports include in-process, local Unix and native Fast DDS. The ROS 2 message package is included; OpenSprite messages do not share Unitree robot wire layouts. See the [DDS protocol and build guide](docs/dds.md).

Controllers subscribe to state and publish commands asynchronously, following the SDK2 idiom. A successful `Write()` means transport acceptance, not execution. Slow diagnostics do not block publishers. Commands retain their actual source snapshot; source ages and current execution feedback are checked separately. See the [review fixes](docs/plan/review-fixes-20261103.md).

Models, meshes, configuration templates and native sources are included. The directory can be extracted into its own repository. Research checkouts and generated output are ignored. See [licenses and provenance](THIRD_PARTY_NOTICES.md).

## Install and run

Python3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[sim,deploy,test]'
python -m pytest tests -q
```

Start the simulator and client in separate terminals:

```bash
python examples/sim_robot.py --endpoint /tmp/opensprite-sdk.sock \
  --domain 1 --duration 10 --enable-sim

python examples/low_level.py --endpoint /tmp/opensprite-sdk.sock \
  --domain 1 --duration 2 --gains-profile legacy_v4
```

The client requests whole-body disable on completion. Physics waits for the first command, then advances normally with watchdog checks. Omitting `--gains-profile` selects a low-gain interface demonstration, not free-standing gains.

## Shared Fast DDS runtime

The robot runtime owns command validation, controller ownership and the device backend. A separate policy process receives the same sensor messages and sends the same31-joint targets in simulation and on hardware.

```bash
sudo apt-get install cmake g++ libfastrtps-dev libfastcdr-dev nlohmann-json3-dev
bash tools/build_dds.sh

# Terminal one, after exporting the bundle below. This opens a live window.
MUJOCO_GL=glfw python examples/robot_runtime.py --backend mujoco \
  --domain 42 --interface lo --bundle outputs/dance-bundle \
  --enable-sim --duration 240 --viewer

# Terminal two: independent CPU ONNX policy, full reference, no recording.
python examples/remote_policy.py --domain 42 --interface lo \
  --bundle outputs/dance-bundle
```

Remove `--viewer` for headless operation. Prefix the viewer command with `PYGLFW_LIBRARY_VARIANT=x11` if needed on Wayland. DDS defaults to loopback; select an explicit interface/address for network use and a separate domain per robot. The ROS 2 package in `ros2/opensprite_msgs` is the authoritative wire schema. SDK deployment does not require ROS 2.

Claim and enable are separate operations. State reports which controller command the backend accepted; this does not confirm physical motor execution. Robot timestamps retain the robot clock, and source freshness is checked on the robot. The hardware runtime connects to explicitly provisioned native CAN/IMU processes and defaults to shadow mode. The current dance policy remains unqualified for the newer hardware profile.

Run `python tools/check_dds_acceptance.py --ros2 --output outputs/dds-acceptance/acceptance.json` for actual DDS and container-based ROS 2 checks. `python tools/check_standalone.py --dds` rebuilds and installs from relocated source, blocks parent imports and runs a separate runtime/client through real DDS. See the [acceptance harness](tools/ros2/README.md).

## Dance sim2sim

Export a trusted `goal-sim2real` checkpoint and its50Hz reference. Deployment uses CPU ONNX Runtime and does not need the training package, world model or PyTorch.

```bash
python -m pip install -e '.[export]'
python tools/export_tracker.py --checkpoint /path/to/model.pt \
  --motion /path/to/lafan_dance.npz --output outputs/dance-bundle
python tools/check_bundle.py outputs/dance-bundle
python tools/benchmark_policy.py outputs/dance-bundle
python examples/run_policy.py --bundle outputs/dance-bundle \
  --backend mujoco --continue-after-failure --output outputs/full-dance
```

The supported export contract is the legacy v4 dance actor:99 observations,10 history frames,770 reference-only goal values and31 actions. Normalization and±20 action clipping are retained. The reference advances once per policy step without looping or resetting. Full completion after a failure is recorded separately from tracking success.

For this parent repository, the explicit export inputs are `../artifacts/checkpoints/goal-sim2real_model_024000.pt` and `../data/motions/lafan_dance.npz`. They are optional export inputs, not SDK runtime dependencies.

Open an interactive real-time MuJoCo window without recording:

```bash
MUJOCO_GL=glfw python examples/view_policy.py --bundle outputs/dance-bundle
```

Space pauses/resumes, `R` starts a new run from the first frame, and `Esc` closes. The translucent blue robot shows the reference. Policy and physics continue after failure with a visible first-failure marker; the window stays open at the end. Use `--paused` to start paused. On Wayland desktops with window issues, prefix the command with `PYGLFW_LIBRARY_VARIANT=x11` to use Xwayland.

CPU headless video, with the system `libosmesa6` library installed:

```bash
MUJOCO_GL=osmesa python examples/run_policy.py --bundle outputs/dance-bundle \
  --backend mujoco --duration 2 --video --continue-after-failure \
  --output outputs/dance-video
```

## Jetson deployment

The target is NVIDIA Jetson Orin Nano Super Developer Kit with the official four-bus CAN FD layout. See the [hardware guide](docs/hardware.md) for native builds, per-robot measurements, zero-gain policy shadow, explicit actuation and completion reports.

```bash
sudo apt-get install build-essential cmake
bash tools/build_native.sh
ctest --test-dir build/native --output-on-failure
cp config/hardware.jetson.example.json config/hardware.local.json
```

Use `examples/deploy_jetson.py` for a complete supervised finite run, including preflight and native completion-report checks. `examples/hardware_robot.py` exposes an explicitly launched native controller through the same SDK channels. `examples/run_policy.py --backend hardware` provides the matching-policy deployment path. The supplied legacy dance bundle is not qualified for the newer J4340P shoulder revision. Profile mismatches and unqualified powered deployment are rejected.

The public BOM describes USB Type-C IMU connectivity; the published, tested runtime reader uses SPI/GPIO. This SDK includes that SPI implementation and the explicit IPC contract. USB-module protocol support has not been validated.

## Validation

See the [implementation plan](docs/plan/implementation.md), [API contract](docs/api.md), and [validation record](docs/validation.md). Tests cover real CPU physics, cross-process channels, source/ONNX parity, a complete225.66-second dance, native compilation, fault injection and fake hardware peers. Jetson ARM64, physical CAN/IMU, powered operation and target latency remain untested.

Version0.2.0 passes250 Python tests,3 native CTests, actual bidirectional ROS 2 interoperability, relocated installation and the full225.66-second DDS policy path. The first tracking failure remains193.76s; completion of the communication test does not imply full-motion tracking success. See the [latest acceptance record](docs/validation-dds-20261104.md).

Run `python tools/check_standalone.py` before extracting this directory into a new repository. It builds a wheel, isolates imports, checks packaged assets and exercises relocated examples and native sources.
