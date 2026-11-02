# goal-sim2real 单舞蹈八小时CLI审查

先读AGENTS.md、本次runner传入配置、对应run的supervisor/launch/interface及最近hourly报告。从传入配置读取run_dir、source_root、source_commit、delivery_dir和预算，不默认使用configs/training.json或其他run。审查runner可在主仓库执行，trainer/supervisor在source_root指定的固定worktree执行；二者cwd不必相同。未配置source_root时才使用runner根目录。唯一motion为固定源码目录内的data/motions/lafan_dance.npz，clip0=dance2_subject1，225.66秒。不联系第三方、不影响其他项目。

核对training-variant=goal-sim2real、Actor99维与reference-only goal770、Critic独立参考编码器/解码器1024/512/256及latent32、WM2048×3。Critic可用global特权，WM训练专用。保持768env、WM4×1024/K8、Actor512/H32、Critic5×4、lr2e-4、gamma.99、lambda.95、500outer WM及成本head预热。保留官方模型、单舞蹈、当前奖励、solver50/25、参考高度阈值.17m和gravity平方误差1.6。tau=0、alpha=1，命令延迟0/1个5ms tick、hold .9、每4tick刷新。不得因指标不佳放宽判定。

累计预算hard28800秒、soft28620秒。启动前GPU验收、编译、预热和恢复计入本次账本；prior_training_seconds从配置读取，active_seconds已含prior，不得重复相加。正式trainer的max-wall-seconds为soft预算减prior；checkpoint.runtime_seconds不含prior，恢复floor为active_seconds减prior。正式训练从零开始，不加载验收checkpoint，保留500outer预热。普通resume保留累计时钟。使用已有CLI后台runner，不创建service/timer/cron，不启动第二个trainer或审查runner。

每小时以CPU执行：
1. 核对唯一trainer/supervisor的PID、argv、cwd、固定source、checkpoint增长、预算、NaN/Inf/OOM及GPU遥测。用source_root检查HEAD等于source_commit及已跟踪源码干净，不能以主仓库HEAD替代。trainer的/proc/PID/cwd须匹配source_root，argv须匹配supervisor记录；不能因它不同于审查runner目录而判定进程消失。主仓库内的绝对run_dir/delivery_dir与worktree源码分开，Python可使用主仓库.venv的绝对路径。明确故障先保留现场，在原预算内处理，不移动固定训练分支。
2. 对照上次报告，比较真实reward、joint/foot/body/root误差、termination、episode长度、WM1/8步误差及有效样本、成本head、Actor梯度、Critic loss和吞吐。24个WM优化留出env仍参与归一化，不称完全独立验证。Critic扩大导致吞吐变化应如实报告，不自动改变逻辑batch或超参。
3. 核对checkpoint中的Actor接口、Critic结构与执行器模式。实际全局位置/heading不输入Actor；训练奖励及参考失稳判定使用真值，不能将两者混淆。
4. 如需闭环评估，先复用同checkpoint/clip/seed42/N3完整请求。新增GPU评估必须与其他GPU工作串行协调，先读主仓库outputs/interventions/ACTIVE.json；不在CPU审查中自行并行启动评估。评估使用source_root中的固定源码和机器人/动作资产，checkpoint及输出使用明确绝对路径；不要从可变主仓库误导入另一版本代码。必要时请求全源，不设max-steps，不重置或循环。全体首次失败可早停，仍记失败。独占评估耗时单列，不双计重叠墙钟。
5. 有team工具时安排独立数值与物理CPU审查并交叉质询；没有时如实写单审。健康时不改训练源码/配置，不重复smoke，不为正常波动增加预算。局部姿态或短窗模型误差小不能当完整成功。

每次报告写RUN/hourly/STAMP/analysis-report.md及必要JSON，更新RUN/hourly/latest.md。公开文档仅保留当前单舞蹈配方说明，不加入原始日志、机器路径或无关实验。推送由前台统一处理。

预算结束后核对supervisor状态、returncode、累计时间、trainer退出和最终checkpoint。提前signal/crash不能称完整预算结束。固定最终checkpoint，对clip0用seed42/43/44各N3请求全源，复用同协议结果，记录首次失败和完整成功率；最终评估耗时单列。向配置delivery_dir交付checkpoint、动作、summary、必要rollout、README及复现命令，videos=[]，不自动录像。

文件与三个seed结果齐备后，原子写RUN/hourly/delivery-complete.json，含status=complete、completed_at、quality_pass真实布尔、quality_notes、delivery_dir、checkpoint、readme、evaluation_reports和videos=[]。预算结束不等于质量通过。
