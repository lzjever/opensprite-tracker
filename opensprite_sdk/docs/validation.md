# 软件验收记录

2026-11-03，Linux x86_64开发机。所有验证在CPU上执行，未访问电机、CAN或IMU设备。没有更改tracker训练配置。

| 检查 | 实际结果 |
|---|---|
| Python单元和集成测试 | 199项通过，包含消息、跨进程、物理、策略与硬件假对端 |
| 原生C++完整构建 | CAN和ICM采集器构建成功；使用隔离安装的CMake，不修改系统环境 |
| CTest | 3项通过：native ABI/失联、ICM估计器、ICM包 |
| 原actor与独立导出实现 | 32组输入，最大绝对误差0 |
| 原参考目标与独立目标生成 | 141帧含尾部，最大误差9.5367e-7 |
| ONNX CPU与导出PyTorch | 最大误差1.0729e-6 |
| 开发机CPU推理基准 | 100次预热、1000次batch1；p99约0.354ms，最大约0.412ms；不含传感器/CAN时延 |
| 骨盆到上身IMU转换 | 30组随机姿态和运动腰关节，与独立MuJoCo运动学核对通过 |
| 完整舞蹈SDK部署 | 11283转移、225.66秒；没有重置或循环；首次失败193.76秒，后续继续且不评分 |
| 独立仿真录像 | OSMesa CPU渲染两秒，通过目视检查 |
| 子目录搬迁 | 独立复制、wheel安装、阻断父项目import、42份资产SHA、真实站立与两进程例程通过 |
| 搬迁后native源码 | 5个目标直接cc/c++编译，3项离线测试通过 |

完整舞蹈验证使用已发布的 `goal-sim2real / model24000`，CPU ONNX Runtime执行。首次失败与原完整录像一致；这不是完整跟踪成功，也不证明不同执行后端逐浮点状态完全相同。

已测主要版本：Python3.12、NumPy2.5.3、MuJoCo3.11.0、ONNX Runtime1.31.1；导出使用PyTorch2.8.0，仅CPU执行。独立迁移测试复用已安装的第三方依赖，不声称测试了全新机器的网络安装。

可重复执行：

```bash
python -m pytest tests -q
bash tools/build_native.sh
ctest --test-dir build/native --output-on-failure
python tools/check_standalone.py
python tools/check_bundle.py outputs/dance-bundle
python examples/run_policy.py --bundle outputs/dance-bundle \
  --backend mujoco --continue-after-failure --output outputs/recheck
```

本地详细证据由上述命令保存在忽略的 `outputs/acceptance/`；审查修复的定向证据保存在 `outputs/fix-review-20261103/`。发行文档保留结果摘要。参考源码、设备标定和测试临时文件不提交。

## 实机验收边界

尚未执行Jetson ARM64构建、目标机推理时延、USB IMU报文、真实CAN驱动和上电动作验收。软件交付包含这些操作的代码和配置入口，不能写成已经实测通过。

逐电机时间戳是应用层从SocketCAN取帧时记录的monotonic时间，用于识别某电机停止提供反馈；它不包含驱动内部采样时刻，也不能单独测得内核排队延迟。真实时延需要在Jetson上联合测量。

原生层提供总力矩cap、速度硬限、温度/RMS及失能检查，当前没有实现完整连续的电机力矩速度曲线。MIT反馈中的力矩是估计量，SDK当前state ABI尚不包含该字段。电流阈值记录到配置，但没有伪造CAN电流测量。

正常部署需要监督程序等待native有限时流程结束并检查报告。提前关闭SDK会触发故障失能路径，不能将这种退出当作native验收成功。持续225.66秒的真机舞蹈不在首版10秒受限上电例程的已验证范围内。

## 0.1.1 交付审查修复

原154项测试未覆盖交付后的定向反例。本轮按[修复计划](plan/review-fixes-20261103.md)补充45项Python回归，并扩展真实C++回归：

- 非脚踝完整PD超关节effort时拒绝；传动比、头部耦合、量化及held命令均核验；实际发送顺序的每个中间向量都要通过。
- 故障总线不再阻断健康总线的三轮失能和验证，整体报告仍保留故障，不能误报通过。
- 50Hz跨进程控制可以引用仍新鲜的上一帧，发送的source序号不被重签；未知、过期或其他epoch仍拒绝。
- 暂停普通SDK诊断进程时，200次有截止时间的发布和健康订阅者均可继续；恢复后序号保序，最终失能状态可正常交付。
- 失能序号高水位、固定1/3tick延迟、重启epoch、发布接受与执行确认的区别都有反例回归。
- 独立迁移检查排除本地标定，源码归档保留.gitignore/.gitattributes。

完整舞蹈使用相同bundle重跑11283转移，首次失败仍为193.76秒；没有重置或循环。该结果验证软件修复没有改变默认仿真配方，不表示真机或整段跟踪已经成功。原生检查基于最新反馈估算，仍不提供连续瞬时力矩测量保证。
