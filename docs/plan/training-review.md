# OpenSprite 三动作统一模型：训练与审查

## 正式运行

运行目录：`outputs/train/opensprite-umr-lafan3-supermimic-20261031`。
固定源码分支：`train/opensprite-umr-lafan3-supermimic-20261031`。
配置与完整 argv：`configs/training.json`。

一个共享actor、critic、世界模型及双子步成本head，直接移植本地成功的 `beyondmimic-umr` SuperMimic `32fc15e`/训练源码`1605c558`。plain MLP enc/dec1024–512–256、latent32、WM768×3、critic256–128；OpenSprite31joint/32body，768环境均分三组，BF16，K8/H32、WM batch1024×10、actor batch256、critic5×4，Adam2e-4。名义参数及DR基准来自OpenSprite；命令0/5ms延迟、0.9保持/20ms抽签时序、一阶tau5–10ms为用户明确要求。

真实UMR库：`data/motions/lafan_three.npz`，revision `opensprite-umr-geno-v1-20261031`。三源长度11284/5109/8196、起点0/11284/16393，合计24589帧，50Hz。每条单独求FK/速度，未跨源拼接。全部hard limits/velocity和最终ground检查通过。保留needs_review：舞蹈14帧自穿深度>3cm（最深43.7mm）；起身/步行最深19.6/26.1mm，无>3cm。该残差明确披露，不能把参考当作几何完美。

## 12小时累计预算

| 项目 | GPU训练墙钟秒 |
| --- | ---: |
| 全部tracker pilots/编译/恢复测试 | 1008.984581 |
| 全新Geno→OpenSprite UMR correspondence训练 | 22.586919 |
| 已用合计 | 1031.571499 |
| 正式训练上限 | 41988.428501 |
| 停止/保存预留 | 180 |
| 总上限 | 43200 |

主监控累计计费，普通恢复以max(checkpoint时钟,监控记录)计，不重置预算。每1000更新或600秒原子保存，SIGTERM在完整更新边界保存退出；异常退出最多自动按相同配置恢复一次，仍失败则团队定位修复。只控制本任务自己的进程组，避免重复launcher。

启动：`tmux new-session -d -s opensprite-train '.venv/bin/python scripts/monitor_training.py --config configs/training.json > outputs/train-supervisor-console.log 2>&1'`。

审查读取`supervisor.json`、`train-segment-*.log`、`tensorboard/`、`reviews/`与checkpoint。每小时由主代理/团队实际分析分动作真实回报、终止/误差、世界模型有效样本与误差、成本head、显存/温度/NaN和推进；自动遥测文件只是证据，不替代分析。需要修正时保留旧快照并记录改动与恢复。

## 训练前最短验证

- 最新真实UMR三动作实际载入，31joint/32body和完整长度核对通过；原型IK不得过正式数据身份gate。
- 新plain架构真实更新、actor/critic/WM/双cost-head梯度与atomic checkpoint、sameoutput恢复通过；真实H32/768环境更新通过，峰值allocated1.843GiB。
- 新版8项focused CPU检查通过；匹配倒地参考合法，真实5ms命令延迟/低通及memory监督核对。
- 随机临时目录隔离测试：只复制仓库代码/资产，无`.reference`或相邻checkout，tracker CPU factory和原生UMR模型配置加载通过。retarget成员另作clean-checkout真实UMR短段验证，不阻碍已授权训练。

## 小时审查与最终评估

正式启动后按时间追加。最终用同一个checkpoint逐clip0/1/2从第一帧运行至真实结尾，无reset或循环；至少seed42/43/44，nominal physics保留用户要求的随机delay/lag。报告完整成功率、失败帧、body/EE/joint姿态误差、root drift及真实速度/力矩/接触峰值，保存reference ghost视频。短诊断不得称完整成功，不以平均奖励替代动作表现。

### 2026-10-31 22:48:32 +08:00：正式启动

源码commit `344566f8181f43d54bb0cf2305d3e73c204aef71`，训练分支已固定。本机tmux `opensprite-train` 中主监控已拉起唯一训练进程；GPU真实初始化与Warp缓存加载正常。所有数据和代码已提交，启动前工作区干净。本次新增文档记录不改变固定训练源码。

最终独立数据QA：三源bank逐段逐值一致；ground≥0.5mm；起身无全身悬浮，舞蹈跳步最长全身离地0.54s。数据残差见上文和`outputs/umr/final-review`。主代理临时目录隔离CPU加载通过，retarget成员在clean checkout补做真实原生UMR短段（只CPU）。tracker成员负责首轮预热到actor启动的独立监控。小时审查以campaign累计训练时钟每3600s为准，首轮在正式启动约42分钟后，避免pilot预算漏算。

### 2026-10-31 22:54 +08:00：首次联合更新

预热500次完成；550次已包含50次actor/critic联合更新，actor梯度范数3.52（更新前，源算法裁剪至1）且全部标量有限。三clip各1/3；真实归一化回报从预热末-1.279改善到-0.975，关节RMSE约0.548/0.545/0.524rad，仍处于早期学习而非效果验收。WM loss约0.259。首次联合图编译已结束。无需纠错，继续原配置。clean checkout的真正UMR CPU3帧验证也已通过，所有源/FBX/slots/MJCF/Three均来自clone内，无.reference依赖。

### 2026-10-31 23:05 +08:00：checkpoint1000首次实际前缀观察

资产成员用同一1000检查点对三clip各3环境、max_steps500做只读前缀诊断（不是完整成功）。舞蹈首次失败2.06–3.20s，起身3.60–4.92s，步行1.56–1.66s；均为尚未收敛的高度误差越过0.17m，非重力或数值异常。初始q逐值匹配参考，root初差≤0.5mm，实际effort≤额定、脚载荷/体重单位一致，未见DoF/坐标/接触符号错位。body相对误差均值0.216/0.140/0.194m，明确不能交付为已学会。继续原训练，未改任何源码或门槛。证据`early-eval-1000/`，评估已退出且未暂停训练。

### 2026-10-31 15:31:59 UTC：小时审计 20261031T152850.063332Z

结论：训练正常推进，早期存活明显改善，但共享策略仍未达到完整动作交付质量。保持唯一现有trainer运行；本次没有停止、恢复、额外训练、改变训练源码/配置或移动固定train分支。

读取了AGENTS、training.json、既有review、本次runner snapshot及动态supervisor。`hourly/latest.md`尚不存在，这是首份runner小时报告；比较基线使用上面的23:05 +08记录及真实`early-eval-1000`三份summary。trainer PID1946390、supervisor PID1946383的完整argv及cwd均核实属于本repo，restarts=0，没有第二个本任务trainer。固定分支仍为`344566f8181f43d54bb0cf2305d3e73c204aef71`；当前普通文档分支为master，runtime相对固定快照无diff。

预算截至15:31:59 UTC：supervisor于15:31:57更新，campaign=3637.188秒（1.0103小时，**已含prior1031.571499秒**）；正式累计=2605.616秒；至soft43020余39382.812秒（10.9397小时），至hard43200余39562.812秒，二者相差180秒停止预留。未重复加prior，未改cap。checkpoint1000/3000/4000的正式时钟分别581.862/1660.827/2193.306秒；4000对应campaign3224.877秒，早于现有时钟，顺序一致。独立评估没有梯度更新，也没有另计一笔训练时间；trainer持续运行的时段仍由原supervisor照常累计。

进度由runner snapshot的4400推进至4700，CSV、分段日志和TensorBoard实际继续更新；最新保存且本次冻结评估的是`model_004000.pt`。全部已读CSV标量和checkpoint4000（含优化器）张量有限，无NaN/Inf/OOM/traceback/异常退出。GPU抽样利用率77–86%、总显存4311MiB/16303MiB、温度55–59°C；训练进程显存3364MiB，训练allocated峰值1.914GiB。另一个既有GPU进程独立存在，未发送信号或作改动。

最近两个各10个记录点窗口：3500–3950 → 4000–4450（表中termination/reset为每真实步概率，RMSE为rad；不是从首帧的评估结果）。

| 动作 | reward | termination | reset频率 | joint RMSE | active fraction |
| --- | --- | --- | --- | --- | --- |
| clip0 dance2_subject1 | .1264 → .1522 | .004669 → .003563 | .006397 → .006056 | .4231 → .4356 | 各1/3 |
| clip1 fallAndGetUp3_subject1 | .2782 → .3033 | .004738 → .003425 | .010441 → .009959 | .3632 → .3810 | 各1/3 |
| clip2 walk4_subject1 | .4954 → .5144 | .001619 → .000931 | .003678 → .005294 | .2228 → .2597 | 各1/3 |

三动作没有被采样饿死；终止率均下降，但近期joint误差均反弹，步行reset频率亦上升，不能只看奖励。全局真实reward .3000→.3233；已结束episode平均长度147.64→140.45步，归一化episode return -.2253→-.4091，近期表现并非全面改善。现有训练日志没有逐clip episode长度，不能把reset倒数冒充实测长度；下面提供各clip实际前缀的连续500步证据。对比更早900–1100窗口，整体termination .02784→.00264、episode长度34.29→140.45，长期学习有效。

WM loss .11442→.11093，每有效转移误差 .11702→.11352，末步误差 .15963→.15536；末步valid fraction .95573→.95426，仅轻微下降，且明显高于900–1100的.80402。本次WM改善不能归因于大量早失败缩小有效样本，但WM精度仍不代表真实完整成功。双成本head loss .001129→.001061，disagreement .000540→.000517；这些是监督拟合证据，不能保证峰值成本预测准确。actor裁剪前梯度均值6.33（源码裁剪范数1、逐值.1）；critic loss .01760→.01574。critic梯度范数未输出遥测，不能虚报数值；源码每次critic更新检查有限梯度并裁剪至1，且checkpoint3000→4000的critic权重/偏置L2变化26.95、Adam步数至70000，有限且非零的优化器状态支持实际更新。actor、WM及成本head也均有有限非零更新，证据`checkpoint-update-evidence.json`。imagined survival .8794→.9030、imagined reward .3943→.4170，高于真实reward .3233；仍需警惕模型内乐观和速度/力矩代价。

本次只对未曾评估的同一checkpoint4000运行三clip各一次：seed42、num-envs3、nominal physics保留随机command delay和torque lag、无reset/循环、max-steps500；三个请求均退出0，全部实例连续完成500转移/10秒，没有首次失败。**均为prefix诊断**，summary的complete_clip=false、success_rate=null，不能称全长成功。完整源分别为11283/5108/8195转移（225.66/102.16/163.90秒）。

| clip | 1000首次失败秒范围 | 4000前缀存活 | body相对误差均值 | foot相对误差均值 | joint绝对误差均值 | pelvis全局漂移均值/最大 | waist root姿态误差均值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 舞蹈 | 2.06–3.20 | 3/3至10秒 | 21.62→11.80cm | 6.31cm | .216rad | .205/.421m | .308rad |
| 1 起身 | 3.60–4.92 | 3/3至10秒 | 14.00→10.58cm | 11.15cm | .237rad | .729/1.732m | .661rad |
| 2 步行 | 1.56–1.66 | 3/3至10秒 | 19.37→7.98cm | 3.01cm | .190rad | .0688/.142m | .351rad |

不同checkpoint的有效时间覆盖不同，表中误差均值不能视作严格同覆盖配对改善。body相对误差采用legacy waist anchor；body世界坐标RMSE均值.258/.800/.113m。pelvis漂移直接从rollout root_pos与ref_root_pos计算，区别于summary中以waist为root的指标。起身10秒末三个环境pelvis漂移1.643/1.699/1.672m，尽管不触发当前物理终止，也明显不能称准确完成。三个clip最大速度/官方速度限值比4.60/4.52/4.11，raw torque请求/官方effort比4.13/3.30/3.57；实际施加torque比均≤1。足接触峰值599/409/411N，非足body接触峰值0/78.1/0N。保留合法倒地接触；这些峰值提示跟踪和控制代价尚待改善，不能通过新增禁地面接触来掩盖。

证据目录：`outputs/train/opensprite-umr-lafan3-supermimic-20261031/hourly/20261031T152850.063332Z/`，包含`eval-clip0/1/2`各summary/rollout及日志、`analysis.json`和checkpoint更新证据；旁侧`.telemetry.json`是本次TensorBoard抽样。三个评估均已结束，没有残留诊断进程。

下一小时关注：三clip joint RMSE反弹是否持续，起身全局漂移/姿态误差、步行reset频率、速度与raw请求峰值，以及WM有效样本和真实收益是否同步。新checkpoint再做不重复的prefix；当前body约8–12cm、root姿态.31–.66rad仍远高于来源model22000约3.4/4.0cm和.10rad的解释尺度，尚不满足“稳定且误差接近”的全长成熟评估触发条件。不恢复rootXYZ≤.1m门槛。后续进入该阶段即请求完整源，最终结束仍须同一共享policy的3clip×3seed完整请求和ghost交付。参考needs_review自穿/局部壳体残差仍如实保留。本次训练未结束，故未创建delivery-complete marker，也未关闭小时timer。

审计收尾15:33:28 UTC复核：训练已到4850，唯一trainer PID1946390仍与supervisor完整argv/cwd匹配，三个诊断进程均退出，summary与非空rollout均核实为同一4000检查点的500步前缀。最新supervisor 15:33:27的campaign=3727.196秒（1.0353小时），正式=2695.624秒，soft余39292.804秒（10.9147小时）、hard余39472.804秒；180秒停止预留不变。未创建交付完成标记。


### 2026-11-01：调度方式更正为 CLI 后台任务

用户明确当前为 CLI，并授权后台任务每小时审查训练数值。废弃 Desktop Scheduled 方案；旧 system service/timer 已移除。新增独立 tmux 后台 runner，每 3600 秒串行运行真实 `codex exec` 审计，保留独占锁、逐次证据/报告、失败重试和已验证交付后的自动结束。独立训练监督器、固定训练快照及原 12 小时累计预算保持原值。真实启动验证另记，不以此计划说明冒充已启用。
