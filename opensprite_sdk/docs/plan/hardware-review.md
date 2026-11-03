# 硬件实施与交叉审查

研究来源为官方 runtime `4515206c24404e18a2d5c168eccdb7468aea6ab7`、组织公开BOM及31关节模型。原生代码和TDK驱动许可保留于 `native/`。本次没有访问CAN、IMU或机器人，也没有改动tracker训练。

## 已落实

- 原生CAN层增加跨进程唯一owner锁，并拒绝覆盖已有IPC路径。
- 单文件C++17 CAN层自包含编译，保留原生启动确认、总力矩包络、温度、退出失能和有限时长行为。
- 发现原生seen标志不能证明当前反馈新鲜。增加逐电机真实host receive timestamp与500/50Hz对应的10/60ms失联检查。
- state新增ABI2，kind=1上电执行/kind=3只发零增益shadow，SDK按设备声明允许write或preview，不能仅靠客户端标志推断；31个收包时间附于原state主体，长度776字节；target保持ABI1、1280字节。Python与实际C++输出交叉验证。
- SDK公共状态转换为关节顺序；motor_order_hash和joint_order_hash分别计算。三处耦合用A转换位置/速度、A^-T转换力矩，测试功率守恒。
- 逐机参数未填写则拒绝真实后端。官方现场标定没有作为本机标定发布。CPU假数据带synthetic标签，不能生成上电启动命令或在未注入假IPC时打开设备连接。
- 原生ICM45686采集器、六轴估计与TDK基础驱动纳入子工程，SPI采集和CAN无相互运行依赖。

## 交叉质询及修正

接口agent指出控制session可被换名以及读后缓存继续老化；仿真agent独立发现同一owner缺口。现执行与失能命令都核对唯一owner、递增序号、源state session，enable/write/发送前重查电机与IMU时间，避免重新打时间掩盖旧反馈。

仿真agent审查确认直接关节signed ratio、三个耦合矩阵、offset及力矩逆转置关系正确。新增标定矩阵和整体变换条件数上限100，拒绝近奇异映射。

接口agent指出原生MIT编解码器固定Kp/Kd量化范围，配置不能声称其他范围。现只接受0–500/0–5协议范围。

仿真agent发现训练IMU link为waist_yaw_link而硬件为pelvis_link。公共状态显式携带attached_link，策略部署必须处理动态腰部运动学；不能静态改名。

BOM使用USB Type-C描述IMU连接，公开runtime reader使用SPI/GPIO。当前提供可构建SPI实现和明确SPIMU003契约，没有编造USB模块协议或宣称USB即插即用。

## 已执行验证

- Python单元测试覆盖配置拒绝、矩阵往返/功率、CAN状态ABI、目标布局、ICM原始计数/偏置/姿态一致性、假SEQPACKET/DGRAM对端、ownership与缓存老化。
- `tools/build_native.sh` 完整Linux x86_64构建成功。
- CTest 3项通过：native真实序列化/逐电机失联、ICM估计器、ICM包。
- 尚未执行Jetson ARM64构建、真实适配器、上电、完整舞蹈或时延验收。测试通过不替代这些实机结果。

## 有限时监督与首次目标审查

新增 `examples/deploy_jetson.py`，配置/资格/IMU与ONNX预热全部先于CAN启动。shadow和上电使用同一bundle接口，主程序等待native自己的有限时退出及完整报告；EOF、提前close和假对端都不会独立被认作硬件验收。上电报告需要全部31电机最终失能确认。

原生启动gate原先一次drain可能被排队第二目标覆盖首zero目标。现首包收集模式固定第一包，重复收集不会覆盖，验证后再恢复正常drain；真实C++队列回归覆盖zero与非zero同时排队。

原生正常失能验证需要约240ms，原客户端200ms recv等待会在正常尾部误超时。客户端等待改为500ms，电机10/60ms、IMU30ms及目标100ms有效期没有放宽；配置显式拒绝扩大这些时限。
