# OpenSprite SDK

[English](README.en.md)

这是一套给 OpenSprite 开发控制器、仿真和 Jetson 部署的独立 SDK。接口参考 Unitree SDK2 的 ChannelFactory、Publisher、Subscriber、LowCmd 和 LowState；消息使用 OpenSprite 自己的31关节顺序和机构映射，不与 Unitree 机器人的消息布局互换。

原生 C++ `opensprite_robot` 独立接收 Fast DDS 命令，选择 MuJoCo 或硬件后端。上层 C++、Python、ROS 2 控制器使用同一组消息。机器人进程不加载 Actor、ONNX 或训练代码；硬件模式通过本机二进制 IPC 连接独立的 CAN/IMU 程序。网络处理不进入 CAN 调度线程。

| 周期 | 当前约定 |
|---|---|
| 关节目标下发 | 全身31关节500Hz，每2ms更新 |
| MuJoCo物理及仿真传感器 | 200Hz，每5ms更新 |
| 舞蹈Actor | 50Hz，每20ms生成目标 |
| CAN分相调度 | 2kHz；每个电机发送与轮询目标500Hz |

500Hz指执行器命令，不改变仿真物理。状态保留实际采样时间；IMU采样周期单独声明。配置频率不等于已经测得实际电机反馈或硬实时性能。

## 安装和构建

在本目录执行，Python3.10或以上：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[sim,deploy,test]'
sudo apt-get install cmake g++ libfastrtps-dev libfastcdr-dev nlohmann-json3-dev
bash tools/build_dds.sh
bash tools/build_robot.sh
```

Python仅用于上层客户端或离线工具。原生程序运行时不需要Python解释器。构建脚本从当前环境定位MuJoCo C库，也可设置 `MUJOCO_ROOT`。Jetson硬件版本可用 `bash tools/build_robot.sh --hardware-only`，不链接MuJoCo。

## 独立仿真

场景工具准备本地机器人、网格和初始姿态，不读取策略。执行进程等待独立控制器申请控制权并使能：

```bash
python tools/prepare_sim_scene.py --output outputs/standing-scene
native-runtime/build/opensprite_robot --backend mujoco \
  --config outputs/standing-scene/scene.json --domain 42 --interface lo \
  --allow-enable --duration 240 --report outputs/robot-report.json
```

另一个终端可连接Python或C++控制器。C++客户端仅依赖安装后的SDK：

```bash
cmake --install native-dds/build --prefix "$PWD/outputs/sdk-install"
cmake -S examples/cpp -B build/cpp -DCMAKE_PREFIX_PATH="$PWD/outputs/sdk-install"
cmake --build build/cpp --parallel 2
build/cpp/opensprite_low_level --domain 42 --duration 2
```

该例程默认只读状态。`--hold --rate-hz 500` 显式申请控制权并按500Hz发送测得姿态目标；低增益保持是接口示例，不是自由站立配方。原生入口、安装和后端参数见 [原生运控说明](native-runtime/README.md)，类型化接口见 [C++ SDK](docs/cpp-sdk.md)。

Python只读示例：

```python
from opensprite_sdk.core.channel import ChannelFactoryInitialize
from opensprite_sdk.runtime import ManagementClient

ChannelFactoryInitialize(42, "lo", transport="dds")
with ManagementClient() as robot:
    state = robot.Read(timeout=5)
    q = [joint.q for joint in state.joint_state]
```

claim和enable是独立操作。命令来源、控制权、旧包和超时由执行端校验；发送成功不表示电机已经执行。默认loopback，每台机器人使用独立domain。ROS 2定义在 `ros2/opensprite_msgs`，SDK运行不要求安装ROS 2。[DDS协议](docs/dds.md)说明Topic、QoS、时钟和管理应答。

## 舞蹈策略与完整验证

策略导出接受显式提供的 `goal-sim2real/model24000` checkpoint 和50Hz动作文件。部署使用CPU ONNX Runtime，不运行世界模型：

```bash
python -m pip install -e '.[export]'
python tools/export_tracker.py --checkpoint /path/to/model.pt \
  --motion /path/to/lafan_dance.npz --output outputs/dance-bundle
python tools/check_bundle.py outputs/dance-bundle
python tools/prepare_sim_scene.py --motion /path/to/lafan_dance.npz \
  --output outputs/dance-scene
```

在父项目中，输入可选用 `../artifacts/checkpoints/goal-sim2real_model_024000.pt` 和 `../data/motions/lafan_dance.npz`。这些仅是离线准备输入；生成后的场景和bundle可独立搬迁。

机器人端使用舞蹈场景启动，另一个进程运行策略：

```bash
# 终端一：不接触模型权重。
native-runtime/build/opensprite_robot --config outputs/dance-scene/scene.json \
  --domain 42 --allow-enable --duration 240 --report outputs/dance-robot.json
# 终端二：Actor只读取本体传感器消息。
python examples/remote_policy.py --bundle outputs/dance-bundle --domain 42
```

50Hz Actor从真实状态流中选择20ms观测窗口，使用有界历史，既不按每条200/500Hz状态推进目标，也不改写时间。窗口真正缺失时明确退出；失败姿态本身不终止舞蹈。模型仍保留99维本体观测、10帧历史、770维参考目标及原始归一化和动作限幅。

完整复验由独立工具启动两进程并单独评估诊断真值，不录制视频：

```bash
python tools/check_native_policy.py --scene outputs/dance-scene/scene.json \
  --bundle outputs/dance-bundle --output outputs/native-acceptance/full-dance.json
```

实时观察窗口也独立于执行进程。机器人端加 `--truth-output outputs/truth.jsonl`，另一个终端运行：

```bash
MUJOCO_GL=glfw python examples/view_native.py --scene outputs/dance-scene/scene.json \
  --truth outputs/truth.jsonl
```

窗口只读取诊断姿态，不控制物理或发送策略观测；不录像。Wayland下可加 `PYGLFW_LIBRARY_VARIANT=x11`。

原有Python仿真和本地memory/Unix例程保留兼容，可见 `examples/run_policy.py`、`examples/view_policy.py`。新的部署入口使用原生执行进程。

## Jetson部署与验收

主控为 NVIDIA Jetson Orin Nano Super Developer Kit。逐机电机、耦合机构、软硬限位和IMU配置见 [硬件说明](docs/hardware.md)。离线编译校准后，原生硬件gateway连接已经显式启动的CAN/IMU进程，默认shadow；不自动启动或接管设备，不创建系统服务。

当前舞蹈bundle对应旧版v4，不能直接用于新版四肩电机配置上电。Jetson ARM64、USB IMU协议、实际CAN/USB时延和上电动作仍需设备验收。SDK提供入口和假设备测试，不把它们写成实机验证。

```bash
python -m pytest tests -q
python tools/check_native_acceptance.py --ros2 --output outputs/native-acceptance/acceptance.json
python tools/check_standalone.py --robot
```

[实施计划](docs/plan/native-control-sdk-500hz-20261104.md)记录team共识，[验收记录](docs/validation-native-20261104.md)列出测试和完整舞蹈结果。代码、机器人、原生执行器、消息、配置和许可都在本目录，`.reference/`、`outputs/`及本机标定不参与发布或运行。本工程使用AGPL-3.0-only，第三方许可见 [来源说明](THIRD_PARTY_NOTICES.md)。
