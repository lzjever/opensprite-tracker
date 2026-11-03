# Jetson 真机后端

本后端面向 NVIDIA Jetson Orin Nano Super Developer Kit 和 Kunhong 四路 USB CAN FD。公共命令始终采用 SDK 的31关节坐标；CAN电机编号、方向、传动比、零位和三个耦合机构由硬件配置处理。

已在开发机完成 Linux x86_64 编译、原生离线测试和假 IPC 对端测试。本项目尚未在 Jetson 或实际机器人上完成验证，编译成功不表示驱动时延或动作已通过实机验收。

## 构建

在本子工程目录中执行：

```bash
sudo apt-get install build-essential cmake
python -m pip install -e '.[test]'
bash tools/build_native.sh
ctest --test-dir build/native --output-on-failure
python -m pytest tests/test_hardware_config.py tests/test_hardware_backend.py tests/test_native_ipc.py tests/test_native_build.py
```

`build/native/sprite_can_shadow` 是自包含的 CAN 执行程序。`build/native/icm45686/sprite_icm45686` 是 SPI ICM45686 采集和六轴估计程序。构建不安装系统服务、不修改 CAN、GPIO 或电机配置。原生进程持有本机唯一CAN owner锁，拒绝并行执行器；已有IPC路径不会被自动覆盖。异常退出后须确认没有执行器运行，再清理已失效socket。

## 逐机配置

```bash
cp config/hardware.jetson.example.json config/hardware.local.json
mkdir -p outputs/native
```

模板保留官方电机型号和 CAN 分配，所有逐机测量项留空，`configured=false`。本地配置不提交 Git。需要填写：

- 本机身份、四个实际 SocketCAN 接口名、控制器核号、独立断电链配置。
- 每个电机的固件身份、编码器方向、零位、传动比、软硬限位、MIT 寄存器读回范围、电流/温度限制及命令和反馈力矩上限。
- 左踝、右踝、头部 pitch/roll 三个耦合矩阵及零位，严格按列出的关节和电机顺序标定。矩阵条件数上限为100。
- `joint_limits` 的31个关节范围、速度、Kp/Kd和广义力矩上限。电机力矩与关节广义力矩分开配置。
- IMU 的传输程序、IPC 路径、安装旋转、所在 link 和实测确认。公共 IMU 坐标采用所附着 link 的前/左/上轴。

四个脚踝电机以500Hz接收主机计算的前馈力矩，电机内 Kp/Kd为零；其余27电机以50Hz接收 MIT 位置/Kp/Kd。2kHz是调度频率。运行时分别检查脚踝10ms、其他电机60ms的主机收包新鲜度，以及IMU 30ms的读取起始时刻。这里的时间戳在应用层从SocketCAN取帧时记录，不等于驱动内部采样时刻，也不包含内核排队延迟。

原生 MIT 编码器采用位置16bit、速度/Kp/Kd/力矩12bit。Kp量化范围固定0–500、Kd固定0–5；不同的配置会被拒绝，不会静默使用其他比例。位置/速度/力矩范围来自每个驱动的寄存器读回。

模板中的电流限制是必须记录的驱动配置。CAN MIT反馈没有独立电流测量字段，本SDK不伪造电流监控。驱动内部过流保护和独立断电链仍由实际硬件实现。

## IMU 接口

官方 BOM 标注 ICM45686 模块通过 USB Type-C 连接，而公开 runtime 的原生实测采集程序使用 Orin 的 SPI/GPIO DRDY。接口名称和同一传感器型号不足以确定USB串口报文；本项目没有虚构USB模块协议。

提供的SPI程序适用于与官方runtime相同的接线和安装方向。先确认设备节点和GPIO映射，使用该程序的帮助获取参数：

```bash
build/native/icm45686/sprite_icm45686 --help
```

已确认SPI接线的设备可按该接口采集：

```bash
mkdir -p outputs/imu
ICM_IPC_SOCKET=/tmp/opensprite-sdk-imu.sock \
  build/native/icm45686/sprite_icm45686 capture \
  /dev/spidev0.0 /dev/gpiochip0 106 800 120 \
  outputs/imu/session.csv X_DOWN_Z_BACK CONFIGURE_ICM45686_RAW_NO_CAN
```

上面的设备节点和GPIO偏移仅为参数示例，程序会核对 Orin 的GPIO标识并拒绝不匹配。采集包含15秒静止偏置估计，不控制电机。若实际模块使用USB，需要它的读取程序提供相同的 `SPIMU003` 接口；这项USB适配尚未验证。

IMU服务通过 Unix datagram 接收 `GETIMU01`，返回192字节 `SPIMU003` 包。SDK校验原始计数、偏置、单位、四元数和重力的一致性后，再把gyro、加速度及重力方向旋转到声明的link坐标。全局位置、全局heading不会进入公共状态。

## 完整策略监督例程

`examples/deploy_jetson.py` 是完整有限时流程的首选入口。它先检查bundle和逐机配置、预热ONNX并确认已经校准的IMU服务就绪，然后才启动原生CAN进程。默认运行真实策略shadow：

```bash
python examples/deploy_jetson.py \
  --bundle outputs/bundle-physical-v5 \
  --hardware config/hardware.local.json \
  --output outputs/deploy-shadow-001
```

bundle的profile必须与硬件配置完全一致。当前发布的 `legacy_v4` 舞蹈bundle用于原训练模型；`physical_v5`硬件配置需要匹配的bundle，示例会在启动CAN之前拒绝不匹配。

已完成该bundle实际硬件资格验收后，契约中的 `hardware_qualified` 应记录为true。上电执行同时要求两个显式参数：

```bash
python examples/deploy_jetson.py \
  --bundle outputs/bundle-physical-v5-qualified \
  --hardware config/hardware.local.json \
  --output outputs/deploy-powered-001 \
  --actuate --acknowledge-actuation
```

例程根据硬件配置的native时长运行，初始上限10秒。它不会因为策略循环先结束就主动关IPC，并且会等待原生进程自行完成有限时流程。EOF本身不算成功：原生退出码必须为0，完整报告必须 `passed=true`。上电模式还检查全部31电机的 `final_disabled=true`；shadow检查没有使能和非零控制发送，并确认策略IPC测试通过。

结果保存在输出目录的 `summary.json`、`native/native-report.json`、启动参数和stdout/stderr日志中。程序异常会记录app_error并触发退出失能路径；失败不会被改写成成功。

上电首个当前姿态零增益目标会被原生启动gate固定下来，排队的后续目标在该帧完成验证前不会覆盖它。开发机离线测试覆盖了这项队列竞态。

`hardware_robot.py` 和 `run_policy.py --backend hardware` 是面向已手工启动进程的高级接口示例；它们不代替这里的原生完成报告核验。手动流程提前退出可能触发故障失能，不能作为正常完整运行的验收依据。

## 生成并启动执行器进程

先根据适配器驱动配置四路SocketCAN。官方链路使用1Mbit/s仲裁、5Mbit/s数据和BRS，接口名称由本地配置指定。适配器驱动安装和引脚接线应按官方硬件文档完成。

完成本机测量后，生成输入和启动参数：

```bash
python tools/prepare_hardware.py \
  --hardware config/hardware.local.json --output outputs/native/shadow \
  > outputs/native-shadow-launch.json
```

此命令仅生成三个TSV并输出 `native_argv`，不打开CAN。模板未填写时立即失败。该启动参数对应电机失能下的完整策略shadow，建立策略IPC并发送CAN零增益回显。SDK读取真实关节和IMU，计算并提交完整策略目标；原生进程计算电机命令但只发送零增益回显，不发送拟议动作。

shadow客户端采用默认不使能构造，并使用 `preview()`，不能调用 `enable()`：

```python
from opensprite_sdk.backends.hardware import HardwareBackend

with HardwareBackend("config/hardware.local.json") as robot:
    state = robot.read()
    # 按策略构建完整LowCmd，填写源state序号/session和controller session。
    # robot.preview(command) 以50Hz重复，执行器只发送失能零增益回显。
```

原生state会声明kind=3。`preview()`拒绝kind=1的上电执行进程，避免把客户端上的shadow标志误当成设备侧限制。首次shadow目标需在首个state发布后100ms内到达；应在连接前完成策略载入和IMU偏置估计。

需要上电执行时，另行显式生成：

```bash
python tools/prepare_hardware.py \
  --hardware config/hardware.local.json --output outputs/native/actuation \
  --actuate --acknowledge-actuation > outputs/native-actuation-launch.json
```

生成的 `native_argv` 可以由同一主机的监督程序直接传给 `subprocess.Popen`，无需拼接shell命令。它保留了原生的全电机已失能确认、有限时长和动作使能检查。第一版SDK生成器将上电时间限制在10秒以内，未开放绕过验收的长时间选项。

```python
import json
import subprocess

launch = json.load(open("outputs/native-actuation-launch.json"))
controller = subprocess.Popen(launch["native_argv"])
```

原生进程先轮询完整关节反馈，最多等待2秒接收首个当前姿态、零增益目标。IMU必须已完成偏置估计后再启动它。随后由SDK显式接管：

```python
from opensprite_sdk.backends.hardware import HardwareBackend

with HardwareBackend("config/hardware.local.json", allow_actuation=True) as robot:
    state = robot.read()
    robot.enable()
    # 控制程序按50Hz提供完整31关节命令；采用最新state序号和session。
    # 关闭IPC会触发原生进程的watchdog/失能退出路径。
```

`start()`仅连接已明确启动的原生进程，不启动电机；`enable()`才释放第一帧零增益启动目标。之后的 `write()` 要求统一全身mode、唯一控制者session、递增序号、匹配的源状态session/序号及新鲜传感器。禁用命令也通过同样的身份检查。首次动作应从实测姿态逐渐增加增益，不能直接切到舞蹈首帧。

原生state ABI2为776字节，kind=1声明上电执行进程，kind=3声明只发零增益的shadow进程，保留31个电机的实际主机接收时刻；原生target沿用ABI1的1280字节。原版528字节state不会被当成可部署状态接收。SDK对外JSON消息与这套二进制ABI相互独立。

`LowState.status`为 `ready` 或 `shadow`；本接口没有把实际使能状态和力矩反馈放进原生state，因此不宣称状态已测得enabled，`tau_est_valid=false`。退出后还要读取生成的 `native-report.json`，确认执行结果与最终失能反馈。SDK进程关闭本身不是失能成功的测量证据。正常有限时流程应让原生进程运行到自己的截止时刻并检查报告；提前断开走故障失能路径，报告可能标记未通过，不能当作成功运行。

## 配方与机构范围

当前舞蹈checkpoint的IMU观察附着于 `waist_yaw_link`，本真机模板的IMU位于 `pelvis_link`。部署策略适配器必须处理腰部的动态运动学或明确拒绝frame不匹配；修改一个静态安装旋转不能消除这个差异。

仿真的 `legacy_v4` 参数复现现有训练。实际BOM的四个肩部pitch/roll电机采用J4340P，与旧模型不同。统一SDK并不自动解决质量、惯量、力矩速度包络、传感器时延和机械标定差异。硬件部署使用单独测量配置，不覆盖旧训练配置。
