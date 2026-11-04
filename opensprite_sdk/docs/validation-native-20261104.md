# 原生运控与 C++ SDK 验收

2026-11-04，版本0.3.0，Linux x86_64开发机。三个实施与验证单元完成交叉审查，未留下阻断项。本轮没有训练、录像或操作真实电机。

## 执行架构

上层C++/Python SDK发送统一DDS消息，独立C++ `opensprite_robot` 管理控制权、命令缓存和设备后端。机器人端不加载Actor、ONNX或Python。MuJoCo与硬件共用消息和管理状态机，分别连接物理仿真与原生CAN/IMU进程。

| 时钟 | 频率与行为 |
|---|---|
| 全身31关节命令下发 | 500Hz，每2ms更新目标；两次Actor更新之间保持目标 |
| MuJoCo物理与仿真传感器 | 200Hz，每5ms步进，保留真实采样时间 |
| Actor推理与参考推进 | 50Hz，每20ms选择真实观测窗口 |
| 硬件CAN调度 | 2kHz四槽错相，每个电机500Hz发指令及轮询反馈 |

指令下发频率与物理频率独立。仿真不生成额外的500Hz传感器样本。命令延迟仍按5ms物理tick计算，一阶力矩迟滞关闭。

## 软件结果

| 检查 | 结果 |
|---|---|
| Python全量单元与集成测试 | 292项通过，无跳过 |
| 原生独立验收 | 26项通过，覆盖类型SDK、真实DDS、二进制假设备、控制权、失联与退出 |
| 安装后的C++消费者 | 仅通过安装头文件与 `OpenSpriteSDK::dds` 构建，四种类型消息与真实ROS 2 Humble互通 |
| Fast DDS 2.6 / GCC 11 | 编译、安装、类型往返与64条接收历史通过 |
| 500Hz C++客户端 | 1秒发送500条，收到201条真实状态；500次执行、200步物理 |
| 全31电机调度 | 逐电机计数500Hz；校验实际FD模式、速率与槽位预算 |
| 真实DDS接收历史 | 显式64缓存保留真实样本；默认latest1；窗口真正丢失时退出，不插值或改时间戳 |
| 诊断读者暂停 | 已匹配订阅者暂停后，健康客户端与执行循环继续，保护边界保持有效 |
| 原生程序依赖 | 无Python/ONNX；硬件专用构建不链接MuJoCo |
| 生成文件与发行 | DDS生成检查通过；独立搬迁、wheel、C++安装与相对RPATH通过 |
| 实时MuJoCo窗口 | 独立读取诊断姿态，仅渲染；退出通过，不录像 |

显式历史读者关闭Fast DDS Data-sharing，使用独立接收历史，避免共享writer的单条历史限制。机器人端发布仍采用有界latest状态；接收缓存不能恢复真实网络丢包。

## 完整舞蹈

独立Actor使用已发布的 `goal-sim2real / model24000`，从源首帧连续执行，不重置、不循环参考。机器人端只接收命令，真值由外部工具评分。

| 指标 | 结果 |
|---|---|
| 参考转移与策略命令 | 11283 |
| 仿真时长 | 225.66秒 |
| 500Hz关节下发 | 112830次 |
| 200Hz物理 | 45132步 |
| 首次失败 | 193.76秒 |
| 全程跟踪成功 | 否；失败后继续真实策略与物理至轨迹结束 |
| 执行端故障、诊断丢失/写入错误 | 均为0 |
| 开发机调度deadline miss | 6次；不构成硬实时保证 |

原始证据在本地忽略目录 `outputs/native-acceptance/full-dance-release.json`。程序完成与模型跟踪成功分别记录。受主机负载影响而超出来源新鲜度限制时，执行端仍拒绝命令并失能，验收没有放宽30ms IMU或10ms硬件反馈边界。

## 复验入口与实机范围

在子工程目录执行：

```bash
python -m pytest tests -q
python tools/generate_dds_schema.py --check
python tools/check_native_acceptance.py --ros2 --output outputs/native-acceptance/recheck.json
python tools/check_standalone.py --robot
python tools/check_native_policy.py --scene outputs/dance-scene/scene.json \
  --bundle outputs/dance-bundle --output outputs/native-acceptance/full-dance.json
```

场景与bundle的准备入口见 [README](../README.md)。独立发行检查排除 `.reference`、`outputs`、本机标定和构建目录，并阻断父项目导入。测试复用已安装的第三方依赖。

Jetson ARM64、实际USB/CAN时延、IMU协议、电机500Hz反馈能力及带电动作仍需实机验收。1M/5M总线在8电机一发一收时的保守理论负载为71.68%，最坏槽位358.4微秒；这是预算与准入条件，不能替代设备测量。当前舞蹈模型对应v4机构，不直接用于新版肩部上电。
