# OpenSprite 硬件约束

31关节的effort、velocity、角度范围和轴向来自随仓库提供的官方[URDF](../assets/opensprite/vendor/sprite0825/sprite0825_sanitized_v4.urdf)与[外部PD MJCF](../assets/opensprite/vendor/sprite0825/mujoco/sprite0825_external_pd.xml)。这两个文件与[官方提交35afc212](https://github.com/BeijingDynamics/open_sprite/tree/35afc2126fc31fa4cf21a2e6061c8bfa6c69726a/models/sprite0825)逐字节一致。训练转换保留质量、惯量、COM、armature、damping及frictionloss。

| 关节组 | 数量 | 标称effort（Nm） | 标称velocity（rad/s） |
|---|---:|---:|---:|
| 双侧髋、膝 | 8 | 12 | 10 |
| 双侧踝pitch/roll | 4 | 6 | 12.56 |
| 腰roll | 1 | 30 | 3.8 |
| 腰yaw | 1 | 12 | 3.8 |
| 双肩、双肘、双腕yaw | 10 | 3 | 3.8 |
| 双腕pitch/roll、头部 | 7 | 0.8 | 3.8 |

[参数读取](../opensprite_tracker/robot.py)按关节名构建；[执行器](../opensprite_tracker/task.py)逐关节硬限幅力矩，gear/gain为1。训练将可用力矩随机化为标称值的85%–100%，奖励、成本预测与评估采用对应实际限值。31关节对照及真实执行器CPU边界测试未发现导入、映射或单位错误。

速度上限用于参考动作约束、奖励和诊断，实际仿真没有硬速度封顶，也没有随转速变化的力矩包络，真实轨迹可能超速。当前参考的最大速度约为URDF上限98%，位置无超限；参考合规不保证实际跟踪或动力学可行。

位置范围由MuJoCo约束求解，可能出现小量穿透。动作目标未按关节范围裁剪：policy的±20范围经0.25缩放后可到默认角±5rad，再减编码器偏差。评估中的高度/重力成功标准不能代替完整硬件合规检查。

PD增益是仿真设定，并非官方辨识参数：髋/膝kp=60、kd=2，踝60/1.5，腰20/0.7，肩/肘/腕yaw为8/0.3，腕pitch/roll及头部为2/0.12；训练对kp/kd做±10%随机化。两套配方均关闭一阶力矩迟滞，保留0/1个5ms tick命令延迟。

踝pitch/roll是双耦合电机的广义坐标。当前采用独立关节限幅，未实现电机侧差动映射与联合可行域，也没有实测的热、电流及转速—力矩曲线。`goal-sim2real`限定Actor的传感器输入，不代表已完成真机能力验证。
