# Fast DDS 与 ROS 2

SDK 使用原生 Fast DDS 传输 OpenSprite 类型消息。Python 的初始化、Publisher、Subscriber 和 LowLevelClient 接口保持一致。内存与 Unix 传输继续用于本地程序和测试。线上协议采用 OpenSprite 的31关节顺序，不与 Unitree 机器人的消息布局互换。

## 构建与接口

Ubuntu 开发机安装依赖并构建：

```bash
sudo apt-get install cmake g++ libfastrtps-dev libfastcdr-dev nlohmann-json3-dev
bash tools/build_dds.sh
```

生成的类型源码随 SDK 提供。修改消息后需要 `fastddsgen` 和 Java，并运行 `tools/generate_dds_schema.py`。构建和重新生成的参数见相应工具。

```python
from opensprite_sdk.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from opensprite_sdk.messages import LowState_

ChannelFactoryInitialize(42, "lo", transport="dds")
subscriber = ChannelSubscriber("rt/lowstate", LowState_)
subscriber.Init(queueLen=1)
try:
    if not subscriber.WaitForMatch(5):
        raise TimeoutError("没有发现状态发布者")
    state = subscriber.Read(timeout=1)
finally:
    subscriber.Close()
```

默认仅使用本机 loopback。跨机器显式选择网卡或其 IPv4 地址，并保持 domain 一致。不同机器人实例使用不同 domain。发现匹配只说明 DDS 端点相互发现，不表示控制权、使能或命令执行成功。

## 消息来源

权威定义在 `ros2/opensprite_msgs/msg`。工具从固定、有界的 `.msg` 子集生成 IDL 和转换代码，Fast DDS-Gen 生成 C++ 序列化类型。Python 对象在进程内转换为原生类型，DDS 网络上传输类型化 CDR，不传 JSON 字符串。

| ROS 2 Topic | DDS Topic | 消息 |
|---|---|---|
| `/lowcmd` | `rt/lowcmd` | `opensprite_msgs/msg/LowCmd` |
| `/lowstate` | `rt/lowstate` | `opensprite_msgs/msg/LowState` |
| `/control_request` | `rt/control_request` | `opensprite_msgs/msg/ControlRequest` |
| `/control_reply` | `rt/control_reply` | `opensprite_msgs/msg/ControlReply` |

DDS 类型名采用 ROS 2 的 `opensprite_msgs::msg::dds_::<消息名>_`。消息包可以由 colcon 单独构建。关节、IMU、时间和 source session 的含义见 [接口契约](api.md)。字符串上限按 UTF-8 字节检查。

高频命令和状态使用 BEST_EFFORT、VOLATILE、KEEP_LAST(1)，控制程序持续发送新目标。管理请求和应答使用有界 RELIABLE 队列与有限写入等待。管理操作有本地超时和重试期限。

## 执行端边界

统一执行进程选择 MuJoCo 或硬件后端，控制程序只读状态并发布统一命令。claim 授予一个 controller session，enable 单独处理。DDS 本身允许多个发布者，控制权检查在执行端进行。session 是运行时关联标识，不是身份认证；联网部署需要受控网络，不能用 session 字符串代替 DDS Security。

命令保留其来源状态的机器人 session 和序号。时间戳属于机器人声明的 sim 或 monotonic 时钟，不能在远程控制机上直接用本机 monotonic 相减。执行端使用自身缓存和自身时钟判断来源状态及传感器新鲜度。客户端使用本地接收超时处理断流，策略适配器以收到的机器人时刻检查各项观测的相对年龄。

管理应答包含 `accepted` 和 `execution_verified`。请求已接收、仿真已失能、真机失能已请求、真机电机状态已验证是不同结果。当前硬件状态没有完整电机使能反馈；原生退出报告仍是终止验收的一部分。

严格舞蹈客户端要求仿真观测间隔为20ms；漏掉窗口会明确退出并请求失能，避免目标相位悄悄落后。当前客户端没有把丢包后的观测历史插值为有效历史，也不宣称适用于未经时延验收的网络。

消息回调与执行调度分开。执行端取最新目标，原生 CAN 调度维持自身频率和保护。网络断开、控制者消失或反馈过期时触发本地保护。逐电机、IMU、机构映射、profile 和逐机标定检查保持有效。

## 实测范围

验收记录包括原生 Fast DDS 版本、ROS 2 容器版本、跨进程通信、故障注入和完整 MuJoCo 策略执行。计划见 [实施计划](plan/fastdds-unified-runtime-20261104.md)。Jetson ARM64、真实网络/CAN/IMU、调度抖动和电机上电需要设备验收；开发机通信测试不能替代这些测量。

完整策略链路验收可运行：

```bash
python tools/check_dds_policy.py --bundle outputs/dance-bundle \
  --output outputs/dds-acceptance/full-dance.json
```

它启动独立 MuJoCo 和 CPU ONNX 进程，核对命令接受、参考时长与最终失能，并分别保存首次失败和完整执行结果，不录制视频。
