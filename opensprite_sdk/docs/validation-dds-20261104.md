# Fast DDS 统一执行层验收

SDK 0.2.0 已完成协议、执行层和验证三个单元的实施与独立交叉审查。本次没有启动训练、录制视频或访问真实电机。实施计划见 [plan](plan/fastdds-unified-runtime-20261104.md)。

## 实际结果

| 检查 | 结果 |
|---|---|
| SDK 全量 Python 测试 | 250项通过，无跳过，10.96秒 |
| 原生 C++ 构建与 CTest | 搬迁后重新构建，3项通过 |
| 真实 DDS 定向测试 | 12项通信/控制测试加1项假硬件集成，共13项通过 |
| ROS 2 双向互通 | 官方 Humble 容器独立编译消息，四种消息双向逐字段一致，ROS侧每种收到558条 |
| 大整数与嵌套字段 | 超过2**53的整数、31关节、IMU、中文、命令接受关联均保留 |
| DDS 版本 | Ubuntu 24.04 x86_64 的 Fast DDS2.11.2/Fast CDR1.1.0；Humble Ubuntu22.04容器的2.6.12/1.0.29均构建并实际收发 |
| MuJoCo 实时窗口 | 独立DDS策略进程执行50条命令、200物理tick、1.00秒，正常关闭，未录像 |
| 完整舞蹈 | 独立DDS runtime与CPU ONNX：11283条命令全部接受，45132物理tick，225.66秒，无重置或循环 |
| 末帧与关闭 | 最后接受序号11283，完成最后20ms控制窗口后请求失能；仿真失能验证通过 |
| 独立发行 | 搬迁SDK、构建wheel和DDS库、安装后阻断父项目import，真实DDS双进程执行20条命令、80物理tick、0.40秒，通过 |

完整舞蹈使用已发布的 `goal-sim2real/model24000`。严格20ms观测窗口复验通过。首次失败仍为193.76秒，失败后继续完整执行；`standard_full_success=false`。最终复验耗时约225.86秒，调度超过一个5ms tick的延后记录5次，没有静默跳过物理步或观测窗口。软件链路完成不表示整段动作跟踪成功。

ROS 2 容器镜像ID为 `sha256:1813d3c85d7f96ff7d3012d865204583255740182db5d0065f8f8cd029a83138`。互通使用 `rmw_fastrtps_cpp`，不声称其他RMW已经验收。独立安装复用本机已安装的第三方依赖，没有把参考checkout或父项目作为运行依赖。

## Review 关闭记录

| 发现 | 修复和验证 |
|---|---|
| Fast DDS bool返回值被误作错误码 | 修正重载判断，真实DDS与ROS 2收发通过 |
| 重复创建topic及读取锁饥饿 | 引用计数复用topic；生命周期保护允许匹配查询与读取并行 |
| 最终命令可能被失能抢先处理 | 增加后端接受序号/控制者关联；等待末窗口后再失能 |
| 初始发布相位及漏帧后目标漂移 | 按物理步数发布；严格检测20ms窗口，漏帧明确失败并请求失能 |
| 较晚IMU被误判来自未来 | 头时间采用观测汇合时刻；原生包与传感器时间分别保留和校验 |
| 应答或硬件读取阻塞控制 | 有界独立应答worker；硬件阻塞读取在命令锁外，原子提交快照 |
| 关闭竞争与清理错误 | 局部endpoint引用；优先关闭后端，逐项清理并保留原始异常 |
| GUI共享模型可能修改物理 | 使用独立模型和数据，只复制诊断姿态 |

测试覆盖本地使能许可、控制权、篡改重试、缓存淘汰后的重放、旧session、断流watchdog和控制进程被SIGTERM。暂停真实DDS诊断进程时，健康读者和发布者继续运行。

假硬件测试使用实际 HardwareBackend、二进制native/IMU协议和真实DDS，验证三条shadow目标及来源序号、原始传感器时间、99维远程Actor观测和实际IPC断连。失能应答 `execution_verified=false`，没有用软件断连冒充电机反馈；反馈和应答线程均退出。

## 重复验收

在 SDK 根目录及已安装依赖的环境执行：

```bash
bash tools/build_dds.sh
python -m pytest tests -q
python tools/check_dds_acceptance.py --ros2 --output outputs/dds-acceptance/acceptance.json
python tools/check_standalone.py --dds
python tools/check_dds_policy.py --bundle outputs/dance-bundle \
  --output outputs/dds-acceptance/full-dance.json
```

完整证据保存在忽略的 `outputs/dds-acceptance/`，单元审查保存在 `outputs/dds-review/`。发行文档提供结果摘要和复现入口，这些输出不参与安装或运行。

## 实机范围

Jetson ARM64、真实USB IMU协议、CAN接线、网络端到端延迟与上电动作尚未实测。原生聚合反馈仍为50Hz，脚踝执行反馈仍要求10ms；晚到命令可能等待下一份新鲜快照。没有放宽门槛，也没有把该链路宣称为训练中的5ms延迟。当前旧版舞蹈bundle不能静默换成新版真机profile。

Session用于协作式控制权，不是网络身份认证。默认loopback；每台机器人独立domain。联网和上电需要配置与设备验收。
