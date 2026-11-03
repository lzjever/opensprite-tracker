# OpenSprite SDK

[简体中文](README.md)

A self-contained SDK for OpenSprite simulation and Jetson deployment. It follows the SDK2 publishing/subscription style: `ChannelFactoryInitialize`, `ChannelPublisher`, `ChannelSubscriber`, `LowCmd_`, and `LowState_`. The same joint commands drive a MuJoCo backend or a native hardware backend.

All public arrays use OpenSprite's31-joint order. `motor_cmd` and `motor_state` are spelling aliases for joint-space data. Physical motor mapping, both ankle differentials, and the head pitch/roll differential belong to the hardware backend. This release supports in-process and local Unix channels; it is **not Unitree DDS wire compatible and does not provide remote DDS transport**.

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

Run `python tools/check_standalone.py` before extracting this directory into a new repository. It builds a wheel, isolates imports, checks packaged assets and exercises relocated examples and native sources.
