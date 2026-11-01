# 单舞蹈八小时CLI审查

先读AGENTS.md、configs/training.json和当前run的supervisor、launch、interface及最近hourly报告。唯一run为outputs/train/opensprite-dance-no-torque-lag-cold-8h-20261101，唯一motion为data/motions/lafan_dance.npz，clip0=dance2_subject1，225.66秒。工作目录始终为本repo，不联系第三方、不影响其他项目。

固定配方为actor1024/512/256、latent32、WM2048×3、critic256/128、768env、CPU replay1000/env、WM4×1024/K8、actor512/H32、critic5×4、lr2e-4、gamma.99、lambda.95、500outer WM-only。tau=0、alpha=1，保留0/1个5ms tick命令延迟、hold .9、每4tick刷新。保持官方模型、当前数据、奖励、solver50/25、高度误差阈值.17m和gravity平方误差1.6。不得因指标不佳放宽判定或修改参考。

预算hard28800秒、soft28620秒、prior0；设置、编译、预热和恢复计入。普通resume保持本run累计时钟。只使用已有CLI后台runner，不创建service/timer/cron，不启动第二个trainer或审查runner。

每小时执行：
1. 核对唯一trainer/supervisor的PID、argv、cwd、固定source、checkpoint增长、预算、NaN/Inf/OOM和GPU状态。明确故障先保留checkpoint，交叉审查后在原预算内恢复。
2. 用CPU比较最近日志窗口的真实reward、joint误差、termination、episode长度、WM1/8步误差及有效样本、成本head、actor梯度和critic loss。24个WM优化留出环境仍参与归一化，不声称验证完全独立。
3. 需要判定闭环能力且有新checkpoint时，先搜索并复用同checkpoint/clip/seed42/N3完整请求。必要时串行一次完整评估，不设置max-steps，不重置或循环。全体首次失败可早停，仍记失败。不得每小时重复prefix矩阵。
4. 同时检查body-relative/foot/joint/root世界漂移、实际施加力矩、速度和接触。局部姿态相似或短窗模型误差小不能当作完整成功。现有needs_review数据状态如实保留。
5. 先读outputs/interventions/ACTIVE.json，避免与前台GPU工作冲突。视频已由前台交付，不另开录像。训练与评估重叠墙钟不双计，独占评估额外耗时单独记录。
6. 有team工具时安排独立数值与物理CPU审查并交叉质询；没有时如实写单审。不要为正常波动改配方，不自行增加预算。

每次报告写RUN/hourly/STAMP/analysis-report.md及必要JSON，更新RUN/hourly/latest.md。公开docs保持当前单舞蹈的简练说明，不把原始日志、内部路径或历史实验写入发布文档。允许提交当前说明，禁止移动固定训练分支。推送由前台统一处理。

到预算后确认supervisor finished、returncode和累计时间、trainer退出及最终checkpoint。提前signal/crash不能称完整预算结束，不加训。固定最终checkpoint，对clip0分别用seed42/43/44各N3请求全源；复用已有同协议结果，报告首次失败和完整成功率。最终评估计时单列。交付至outputs/delivery/opensprite_dance_no_torque_lag_20261101，包含checkpoint、动作、summary、必要rollout、README和复现命令；videos=[]，引用已交付视频即可。

确认文件及三个seed结果齐备后，原子写RUN/hourly/delivery-complete.json，包含status=complete、completed_at、quality_pass真实布尔、quality_notes、delivery_dir、checkpoint、readme、evaluation_reports和videos=[]。预算耗尽不等于质量达标。
