# OpenSprite 原版 SuperMimic 迁移计划与结果

## 最终采用的已成功来源

用户指明本地 `beyondmimic-umr` 最新 SuperMimic 效果好，要求尽量直接复刻网络和训练逻辑。最终代码从 `exp/agibot-x2-supermimic-fullbody-joint-2h-20261022` @`32fc15e2d2a2f455cea23df2d1f6a96e3f31fab5` 的 `supermimic_x2` 直接复制为 `opensprite_tracker/supermimic`；完整研究快照在 `.reference/beyondmimic-supermimic-latest`，不参与运行。实际成功 model22000 的launch记录训练代码 `1605c558`；已对照权重核实网络尺寸。原文件MIT许可证在本包保留。

本地 model22000 的真实完整评估记录：枪火2285/2285、同条fallAndGetUp3 5108/5108转移，nominal seed42成功率均1.0，body平均误差3.43/3.95cm，root姿态误差.101/.096rad。全局root位置均误差.242/.590m，说明旧配方优先相对动作形态，不可冒称全球定位准确。OpenSprite最终将如实报告root drift、body/EE/joint errors和完整成功。

## 直接保留的网络与训练

- plain MLP actor：参考770→1024→512→256→32 tanh latent；10帧105维历史与latent拼接→1024→512→256→31动作。只把源29joint对应的obs99/goal730/output29改成真实Sprite31joint的105/770/31，未增加MoE、FSQ或attention。
- WM 768×3 SiLU/LayerNorm、局部状态增量和真实数据归一化、8步自回归监督；critic 256/128；bf16线性层，物理积分float32。
- 原buffer、engine、actor BPTT H32、critic lambda-return、Adam、5epochs×4critic minibatches、源return-adaptive sampler直接复用。一个共享policy/WM/critic支持三个动作，按clip等质量/各环境按ID分组，片段内困难阶段采样；不拼接相邻动作。默认500transition episode，可在真实源结尾提前结束。
- 保留源时序 `s_t/ref_t → a_t → s_(t+1)/ref_(t+1)`。N帧源有N-1个转移，最后参考帧为真实terminal状态；末尾未来token夹到本clip尾，不会跳到下一动作。
- `legacy_full`原21项奖励按实际Sprite几何移植，full-body impact为可选第22项；真实四个5ms子步监督速度、requested/applied torque及foot/body impact，双预测head用于模型内成本，不拿预测伪装真实传感器。

## OpenSprite必须独立适配的物理

- 官方Sprite0825全部31joint、32body（两腰/三头整链），默认物理模型含全身地面与合法非局部自碰撞；只用新机器人惯量/axis/hard limits/effort/velocity/armature/frictionloss/PD。
- 真实与可微FK从本repo MJCF读取；虚拟IMU明确为waist_yaw_link/identity，不假装存在X2的IMU site；单boxsole精确计算foot height/heel/toe，不保留Capsule7。
- 用户明确要求与X2相同的延迟/力矩迟滞：命令0或1个**5ms物理tick**，每4tick(20ms)到分相更新时以0.1概率重新抽样、0.9保持；31joint独立tau Uniform5–10ms，first-request初始化与低通更新沿用源。声明这是用户指定的仿真模型，非Sprite电机实测。
- memory66为归一化filtered torque31、前目标31、lag/phase/init/valid；context221依实际Sprite顺序记录foot friction、encoder bias、torso COM、kp/kd/effort、armature/frictionloss和tau。所有值来自真实启动物理参数。WM预测全部memory，源算法对未来随机lag作条件预测，不声称复放未来RNG。
- 原DR机制保留但参数基准来自Sprite：PD×.9–1.1、effort×.85–1、官方armature×.7–1.3、官方frictionloss×.5–1.5（0仍为0）；torso COM半宽用Sprite chest mesh bbox半长10%约[.00588,.01225,.01102]m。未知硬件误差不冒称标定。
- reference-relative完整gravity²>1.6与pelvis高度差>.17m终止，共享真实和模型内语义；lying/falling本身合法，参考支撑资格依据实际Sprite壳体。

## 数据、预算和交付

正式入口在GPU初始化前核对`retarget_method=umr_native_nr_surface_vector_v1`、`robot_model=opensprite0825`、显式dataset_revision和`correspondence_names=(geno,opensprite0825)`。普通position-IK原型不得标UMR，不得续入正式。原型general tracker代码和轨迹已移研究/诊断区，不作为第二套可选训练交付。

CLI：`.venv/bin/python -m opensprite_tracker.supermimic.train --output DIR --motion data/motions/lafan_three.npz --dataset-revision opensprite-umr-geno-v1-20261031 --iterations 1000000 --max-wall-seconds R --num-envs 768 --wm-width 768 --wm-batch 1024 --pi-batch 256 --horizon 32 --amp bf16 --compile`。最终并行数以显存/真实吞吐确定，768可三组均分。R为扣除全部先前GPU任务与180秒保存预留后的campaign预算；resume累计max(saved clock,supervisor floor)，不会重新得到R秒。

checkpoint含wm/pi/critic、全部optimizers、双cost heads、采样器和时间；同目录临时写完atomic rename。每1000iter或600秒保存，SIGTERM仅在完整更新边界保存停止。每小时检查真实分动作回报/termination/joint errors、WM valid fraction/误差、成本head正尾监督以及GPU/NaN/进展，必要时保留检查点纠正。

最终逐完整动作至少3seed、默认nominal physics但保留用户指定delay/lag；eval不reset，不循环参考，短max-steps诊断不得称完整成功。源eval+render工具保存真实rollout及reference ghost mp4。旧rootXYZ≤.1m等人为quality门槛已撤销，不阻碍按源效果交付，但完整物理失败与各项误差必须如实报告。

## 已完成的最短相关验证

- 资产成员32body随机pose FK/boxsole/allprimitive支撑与MuJoCo最大误差2.2e-13；obs105/metrics534→97或98；21项真实/模型内奖励、双head监督、matching lying reference/gradient有限。
- 新原版pipeline在真实native UMR三小段bank上完成3次actor/WM/critic/cost-head更新（bf16），actor gradient非零，三clip active fraction各1/3，atomic保存；sameoutput resume再完成第4次，实际runtime floor20保持。它仅证明新接口和训练链路，并非模型学会长动作。
- 4 focused CPU checks通过：source精确网络结构、实际单5ms tick命令延迟+低通tau、WM包含并监督实际delay memory、formal UMR身份gate。

最终H32真实规模检查完成：768env、WM768×3/batch1024×10/K8、actor256/H32/bf16/compile；首个完整actor/WM/value/双cost-head更新成功，actor gradient .886，峰值allocated1.843GiB。首次图编译及setup共362.188579秒，safe boundary保存退出，不能把首次编译时间当steady throughput。站立sensor实测Fz=-71.90N/foot，正总载143.808N与模型重量143.730N吻合；root z .4592m。

正式完整UMR bank已由相同MotionLib实际加载：24589×31 joint、24589×32 body，三个source长度11284/5109/8196及四项metadata正确，全部闭区间关节限位通过。tracker全部pilots1008.9845805秒、UMR AE22.5869186秒，合计1031.5714991秒；扣180秒保存预留，正式campaign上限R=41988.4285009秒。账在outputs/smoke/budget.json及outputs/umr/correspondence_gpu_time.json，产生新训练段需继续累计。
