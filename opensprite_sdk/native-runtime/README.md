# OpenSprite 原生运控入口

`opensprite_robot` 是独立 C++ 执行进程。上层通过统一 Fast DDS 消息发送关节目标；运行进程选择 MuJoCo 或硬件后端。Actor、训练代码和模型权重不进入此进程。Python仅用于可选的离线配置工具或上层客户端。

仿真与真机的关节命令执行目标均为500Hz；MuJoCo物理固定200Hz；舞蹈Actor保持50Hz。执行与物理采用独立时钟，仿真只发布真实物理采样，不伪造500Hz传感器观测。CAN程序维持2kHz分相调度，对31个电机按500Hz发送与轮询。实际设备时延和反馈覆盖需要目标机测量。

## 构建

在 SDK 根目录执行，依赖及原生DDS构建见上级README：

```bash
python -m pip install -e '.[sim,deploy,test]'
bash tools/build_robot.sh
```

构建脚本从当前Python环境定位MuJoCo的C库和头文件；也可显式设置 `MUJOCO_ROOT`。运行二进制不需要Python解释器。Jetson硬件端可以完全不链接MuJoCo：

```bash
bash tools/build_robot.sh --hardware-only
# 也可直接CMake配置，构建本身不要求Python。
cmake -S native-runtime -B native-runtime/build-hardware \
  -DOPENSPRITE_WITH_MUJOCO=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build native-runtime/build-hardware --parallel 2
```

安装开发机版本会把MuJoCo共享库和许可证放入安装目录。可移动安装前缀内 `bin` 和 `lib`，程序通过相对RPATH查找库；系统仍需安装Fast DDS等第三方动态依赖。

```bash
cmake --install native-runtime/build --prefix /path/to/install
```

## 仿真

离线工具生成可独立搬迁的机器人场景，不读取Actor：

```bash
python tools/prepare_sim_scene.py --profile legacy_v4 --output outputs/standing-scene
native-runtime/build/opensprite_robot --backend mujoco \
  --config outputs/standing-scene/scene.json --domain 42 --interface lo \
  --allow-enable --duration 240 --report outputs/robot-report.json
```

场景包含本地机器人资产、配置和初始姿态。可选 `--motion /path/to/lafan_dance.npz` 只从动作数据提取初始状态。参考轨迹和策略属于上层程序；运行进程没有 `--bundle` 或checkpoint入口。

`--truth-output /path/to/truth.jsonl` 仅供仿真评估，按20ms机器人时间输出诊断姿态，经独立有界worker写盘。丢弃或写入错误进入报告并令验收失败。真值不发布到Actor使用的LowState，硬件模式拒绝此参数。不录制视频。

## 真机后端

先准备逐机校准，并按硬件部署流程显式启动原生CAN和IMU程序。编译校准不是设备探测，也不发送电机指令：

```bash
python tools/compile_native_config.py --hardware-config config/hardware.local.json \
  --output outputs/hardware-native.json
native-runtime/build-hardware/opensprite_robot --backend hardware \
  --config outputs/hardware-native.json --domain 42 --interface lo \
  --allow-enable --duration 10 --report outputs/hardware-gateway-report.json
```

默认shadow只接受已经证明为零增益的原生对端。实际执行需要本地 `--actuate --acknowledge-actuation --allow-enable`，同时保留原生程序的显式上电许可。synthetic配置不能上电。SDK不自动启动设备、不创建系统服务，首次硬件例程仍限制10秒。

消息管理遵循claim、enable、disable和release。claim不使能；execute权限不靠DDS发现授予。命令来源缓存、controller序号、机器人epoch、传感器时间和关节包络分别检查。公开状态头使用机器人融合完成时刻，原生包及各传感器采样时间保留并独立检验；跨机器不直接比较两台主机的monotonic时钟。

`accepted_command_seq` 表示后端接受目标。硬件disable应答只确认请求路径，实际电机失能以原生报告和设备验收为准。C++进程、真实DDS与假设备测试不等于Jetson、CAN/IMU和上电实测。
