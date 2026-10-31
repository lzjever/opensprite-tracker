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


### 2026-10-31T16:38:29.456781+00:00：CLI小时审计 20261031T162303.586759Z

结论：训练正常持续推进，未见需恢复的程序/数值故障，保持现有唯一trainer运行。checkpoint10000的步行在seed42的3个环境完成全源；舞蹈完整请求全部提前失败，起身只验证了10秒前缀。三动作共享策略整体尚未通过交付质量。本次未修改训练源码/配置、启动额外训练、重置预算或移动固定训练分支。

审计标识 `20261031T162303.586759Z`，结论时刻 `2026-10-31T16:38:29.456781+00:00`，supervisor动态状态更新于 `2026-10-31T16:38:28.619170+00:00`。从上一报告4850推进至本次收尾11950更新；restarts=0。trainer PID1946390、supervisor PID1946383完整argv/cwd与本repo匹配；独占supervisor锁正常，无第二个本任务trainer。固定源码仍为 `344566f8181f43d54bb0cf2305d3e73c204aef71`，当前普通文档分支master；runtime、资产、动作库和training.json相对固定训练快照无diff。评估元数据记录master文档/编排commit，训练源码以该固定快照为准。

预算：campaign=7627.977秒（2.1189小时，supervisor.active_seconds已经含prior1031.571499秒）；正式累计=6596.406秒；至soft43020余35392.023秒（9.8311小时），至hard43200余35572.023秒，停止保存预留180秒不变。没有再次加prior。checkpoint4000/9000/10000/11000正式runtime为2193.306/4876.102/5420.363/5995.056秒，分别加一次prior后均早于当前campaign，时钟顺序一致。评估没有梯度更新；训练同时运行的时间继续由既有supervisor累计，没有另外收费一笔评估时间或恢复新预算。

对照上次使用的4000–4450窗口，最新11200–11650窗口的三clip reward由.152/.303/.514升至0.329/0.461/0.647，joint RMSE由.436/.381/.260降至0.348/0.345/0.228，长期改善明确。审计开始时9500–9950→10000–10450的起身reward .392→.381、termination .00378→.00481曾退化；动态复查显示随后恢复，不能把早期快照当最终状态。

最近两个相邻各10记录点窗口10700–11150 → 11200–11650如下（termination/reset为每真实步频率，RMSE为rad）：

| 动作 | reward | termination | reset频率 | joint RMSE | active fraction |
| --- | --- | --- | --- | --- | --- |
| clip0 舞蹈 | 0.294152 → 0.328998 | 0.002247 → 0.001991 | 0.004053 → 0.004131 | 0.404778 → 0.348379 | 各1/3 |
| clip1 起身 | 0.411382 → 0.461403 | 0.003378 → 0.002559 | 0.006400 → 0.008059 | 0.374613 → 0.345079 | 各1/3 |
| clip2 步行 | 0.664219 → 0.647459 | 0.000413 → 0.000647 | 0.002672 → 0.002841 | 0.225622 → 0.228352 | 各1/3 |

三个clip采样始终各1/3，没有饿死。近期舞蹈/起身reward、termination、关节误差改善，但步行reward下降、termination上升、关节误差略升；起身reset频率亦升高。全局真实reward 0.4566→0.4793，episode平均长度229.89→199.73步，归一化episode return 0.0443→-0.0981，故不是全面改善。训练日志未输出逐clip episode长度，不能以reset倒数冒充实测长度；逐clip实际连续长度见前缀/完整请求。

WM loss 0.12986→0.12062，每有效转移误差0.13202→0.12265，末步误差0.18661→0.17469，末步valid fraction 0.97179→0.96846。有效比例仅降约.0033且仍约97%，没有大量早失败导致有效样本坍缩的证据；这些是混合模型误差，不能解释为厘米或证明完整跟踪成功。相对上次每有效转移.11352仍较高，需同时观察数据分布与真实表现。双成本head loss 0.0009222→0.0008918，分歧0.0004429→0.0004444（略升）；不能从均值保证峰值成本准确。

actor裁剪前梯度均值5.253→5.325，critic loss 0.01632→0.01701。critic梯度范数没有遥测，未虚报；源码每次检查有限并裁剪至1。9000→10000的actor/critic/WM/成本head权重偏置L2变化分别7.34/25.62/12.51/11.09，优化器状态有限且步数增长，支持实际更新。imagined survival 0.9302→0.9383，imagined reward 0.5395→0.5543，仍高于真实reward 0.4793；模型内收益不能替代真实评估，特别是episode回报近期下降。

本次只对未做过该诊断的同一 `model_010000.pt` 各clip做一次seed42、num-envs3、max-steps500的prefix；nominal physics保留随机command delay/一阶torque lag，无reset或参考循环。全部9个实例完成10秒，无首次失败；summary均complete_clip=false、success_rate=null，不能称完整成功。与4000比较同一500步覆盖：

| clip | 本次prefix | body相对误差均值cm（4000→10000） | waist root姿态rad（4000→10000） | foot相对误差cm | joint MAE rad | pelvis漂移均值/最大m |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3/3、500步/10秒 | 11.80 → 4.73 | 0.308 → 0.094 | 3.45 | 0.191 | 0.073/0.272 |
| 1 | 3/3、500步/10秒 | 10.58 → 6.42 | 0.661 → 0.195 | 8.35 | 0.232 | 0.266/0.634 |
| 2 | 3/3、500步/10秒 | 7.98 → 4.52 | 0.351 → 0.100 | 2.91 | 0.198 | 0.050/0.103 |

body/foot相对量以legacy waist_yaw_link锚定，pelvis全局漂移另从rollout计算；joint MAE区别于训练RMSE。起身最后1秒body误差均值.236m、root姿态.955rad，10秒末pelvis漂移.608/.527/.499m；前缀均值掩盖尾段恶化。三个prefix速度峰值/官方限值2.95/2.29/2.06，raw torque请求/额定2.07/3.82/1.97，实际施加比例均≤1；足接触峰值365/483/190N，非足body接触峰值均0N。未改变合法倒地接触或引入upright终止。

舞蹈与步行前缀连续两次存活、误差接近来源解释尺度后，实际对同一10000候选新增两个**不传max-steps**的完整源请求，仍为seed42、3环境、nominal delay/lag、无reset/参考循环。这不是最终3clip×3seed验收，也没有按动作换checkpoint：

- 舞蹈：请求11283转移/225.66秒；实际至1132转移后全失败，三个首次失败21.88/21.76/22.64秒，success_rate=0、complete_clip=false。失败高度误差.180/.171/.189m，超过既有.17m；重力向量平方误差.847/.799/1.073未越1.6。有效段body相对误差10.13cm、joint MAE .267rad、root姿态.124rad；pelvis漂移均值.138m/最大.696m，raw torque比峰值6.20、速度比2.95、足载荷峰值667N、施加torque比≤1，非足接触0N。完整请求失败不能叫完整成功。
- 步行：请求并执行8195转移/163.90秒，3/3全程存活，complete_clip=true、success_rate=1.0；这只覆盖seed42。body相对误差均值3.46cm、foot 2.36cm、joint MAE .199rad、root姿态.116rad；body世界坐标RMSE均值.558m。pelvis漂移均值.554m/最大1.746m，末帧1.494/1.572/1.746m，不能隐瞒。速度比峰值2.06、raw torque比3.24、实际施加比≤1、足载荷峰值494N、非足接触0N；body线/角速度RMSE均值.179m/s、1.046rad/s。
- 起身：本小时仅完成500步prefix；尾段误差仍大，未宣称全长成功。

来源model22000 body约3.4/4.0cm、root姿态约.10rad仅作解释尺度，不是OpenSprite保证验收线；不恢复已撤销rootXYZ≤.1m硬门槛，但照实报告drift。参考needs_review仍保留：舞蹈14帧自穿>3cm，起身/步行有局部壳体残差。当前没有三动作共享policy通过的证据。

数值/设备：所有已读CSV标量、TensorBoard抽样及4000/9000/10000/11000（含优化器）张量有限，无NaN/Inf/OOM/traceback；CSV、分段日志、TensorBoard实际继续更新，全部五个评估进程均已退出。训练allocated峰值1.914GiB；无评估时总显存4311MiB/16303MiB，评估时5063MiB，GPU观察49–96%、温度54–59°C。另一个既有GPU进程未作任何操作。收尾GPU：49 %, 4311 MiB, 54。

后台：既有tmux `opensprite-hourly-audit`真实runner PID2070952持有 `hourly/audit.lock`，CLI子进程2070955/2070962是本次审计父链，argv/cwd已核验。runner命令没有-m/--model覆盖，未递归启动Codex、第二runner或系统service/timer/cron。实际启动16:23:03.586776 UTC，默认interval3600、串行且跳过错过周期；依该进程的启动时刻/参数，下次目标17:23:03.586776 UTC（北京时间11月1日01:23:03）。本审计尚在其子进程内，父runner在本次退出后才写next_run_at，因此这项是已核实调度规则推导的目标，未伪称已观察到等待状态。机器需保持开机；tmux退出/重启后需重启同脚本，预算不重置。

采取动作仅为读取、CPU验证和真实无梯度评估、保存证据/文档；未做训练纠错。下一小时关注舞蹈约22秒的失败段、起身9–10秒尾段、步行近期termination/关节反弹与全程drift，核查WM有效样本/成本head分歧和真实episode回报是否同步改善。后续新checkpoint作不重复诊断，合适时继续完整源；最终必须同一个冻结共享checkpoint的3clip×3seed完整请求、ghost视频和独立交付核验。本次训练未结束，未创建delivery-complete marker，未终止父runner。

证据目录：`outputs/train/opensprite-umr-lafan3-supermimic-20261031/hourly/20261031T162303.586759Z/`，包含analysis.json、closing-trends.json、closing-verification.json、evaluation-comparison.json、runner-verification.json、最新checkpoint验证、三个prefix及两个完整请求的summary/rollout/log。完整分析另保存在该目录 `analysis-report.md`；已追加 `docs/plan/training-review.md`。

收尾复核 `2026-10-31T16:39:23.576542+00:00`：唯一trainer PID1946390完整argv/cwd仍匹配，推进至12050；新增model_012000.pt（正式runtime 6602.374秒）的213个checkpoint/优化器张量全部有限。supervisor `2026-10-31T16:39:18.623309+00:00` 的campaign=7677.981秒（已含prior）、正式=6646.410秒，soft余35342.019秒、hard余35522.019秒，预留180秒不变。本次真实仿真评估仍全部固定10000，未重复诊断12000。


### 2026-11-01：按论文规模重开，用户明确新给12小时

旧轮安全停止，model015551成功保存，旧campaign9508.822220秒作为历史保留。用户明确重新给12h；新轮prior0、soft43020/hard43200、reserve180，从新网络随机初始化，不扣旧轮。新run为opensprite-umr-lafan3-paper-20261101。执行方案和论文对照见paper-scale-restart.md；WM2048×3、policy2048/1024/512/256、4096env/replay1000、WM10×2048/K8、pi4096/H32、TF32。

16GB实现只改变存储与逻辑batch内执行分块；CPU等价验证WM/actor整批mean与clip/Adam一次、critic固定λ-target及5×4更新。CPU replay约10.18GiB，未缩buffer或采样范围。交叉审查额外修复源末帧未来价值泄漏，保留arrival reward。训练前检查不涉及GPU；真实初始化/编译和首次完整更新将作为新正式campaign计费。旧后台审计已停，待新训练首轮验证后改指新run启动。


### 2026-11-01 01:23 +08：论文规模真实首完整更新通过

新固定源码`119c2660e1dfd6d35ac40e5f04b17342bce1dea6`，分支`train/opensprite-umr-lafan3-paper-20261101`不再移动。01:17:11+08启动tmux `opensprite-paper-train`，唯一trainer PID2133080；初始化/编译已进入新campaign累计时钟。实际launch.json确认all_networks_random、4096env、1000replay、WM2048×3、10×2048/K8、PI4096/H32、policy2048/1024/512/256、ampnone/TF32。

真实首轮保存model_000001.pt，235733901bytes；runtime297.274秒（含初始化/编译）。CPU重新读盘核实229个checkpoint/optimizer张量全有限；主干9498764参数，actor9321791参数。Adam状态精确为WM10、PI1、critic20、双costhead10，证明microbatch没有变成额外Adam更新。证据`RUN/first-update-validation.json`。三clip实测active_fraction=.333496/.333252/.333252，符合1366/1365/1365。首轮真实reward -1.204、WMloss3.822、每有效转移误差3.372、末步valid .839、actor梯度.612、criticloss1.217均有限；新随机模型尚未学会，不拿旧轮优秀指标冒充本轮质量。首轮速率包含编译，不作为steady吞吐。

CPU replay分配/真实4096×64启动采集与CUDA采样路径已通过，无OOM或设备索引错误；trainer主存RSS约14–15GiB，当前GPU总显存约5.3GiB，GPU高负载、温度55°C。旧训练与旧审计均未重启。

新纯CLI小时runner于01:23:07+08启动tmux `opensprite-paper-hourly-audit`，PID2164748、CLI2164752，状态RUN/hourly/runner-state.json，真实agent已执行读取。立即首审后每3600秒串行，不创建service/timer/cron。当前新预算hard预计13:17:11+08，soft预计13:14:11+08（无停机条件下），普通恢复不刷新。


### 2026-11-01 01:34 +08：较大执行批量验证通过

固定source快照c71b9d1，恢复同一model29及累计floor604.447193秒；新12h未刷新。新的WM2048物理整批、PI512分8块（logical4096）、critic8192已实际完成update30并原子保存model_000030.pt。读盘验证229个张量有限，Adam步数WM300/PI30/critic600/costhead300，与累计30次逻辑outer一致；wm/actor逻辑batch仍2048/4096。runtime905.372秒已包含前阶段和新形状编译，GPU无OOM，PyTorch峰2.753GiB、GPU总约7GiB。证据RUN/batch-tuning-validation.json。新CLI小时runner2204241/agent2204246于01:33:38+08启动并真实读取当前证据，旧runner均已停止；仍无systemd/cron。


### 2026-11-01 01:36 +08：论文规模批量恢复后首审（20261031T173338.998615Z）

本次审计：训练持续推进，尚未收敛；未发现需要停训修复的数值或进程故障，保持现有训练运行。

审计标识 `20261031T173338.998615Z`。2026-10-31 17:36:16 UTC（北京时间11月1日01:36:16）核实：日志推进至60更新，最新atomic检查点为 `model_000030.pt`。campaign累计1084.556秒，prior=0，正式累计同为1084.556秒；soft43020余41935.444秒（11.649小时），hard43200余42115.444秒，180秒保存预留不变。未计入旧campaign9508.822秒，未重置恢复预算。

本轮 `hourly/latest.md` 原先不存在：17:23首审在批量调优维护时中断，未形成最终报告。本次以其已保存analysis.json、三份model1评估及训练记录作对照，不把旧小网络campaign成绩当本轮成绩。当前训练源码为固定 `c71b9d1ead9780080b39958c577e7181a5ba090d`；实际resume launch记录一致，runtime/配置/资产/动作相对该快照无diff。保留初始119c266与旧训练分支，没有移动固定分支。

下表均为同一个共享policy的真实训练统计，箭头依次表示29→40→60更新；termination/reset为每真实步频率，joint为RMSE rad。

| 动作 | reward | termination | reset | joint RMSE |
| --- | --- | --- | --- | --- |
| clip0 舞蹈 | -1.459→-1.737→-1.410 | .02612→.02753→.02981 | .02629→.02782→.03297 | .5354→.5315→.5405 |
| clip1 起身 | -1.353→-1.345→-.802 | .02920→.02872→.02567 | .02904→.03320→.04700 | .4516→.4265→.3982 |
| clip2 步行 | -1.138→-1.217→-.974 | .02771→.02837→.02901 | .02744→.02810→.02892 | .5215→.5322→.5370 |

30/40/50/60的全局真实reward为-1.456/-1.433/-1.217/-1.062，episode长度25.25/33.03/28.33/27.57步，归一化episode return -1.933/-1.695/-1.535/-1.495。近期回报回升，但episode仍短；起身reward/joint改善同时reset升高，舞蹈/步行termination与关节误差相对29升高，不能称全面改善。日志无逐clip实测episode长度，未用reset倒数冒充。active fraction始终.333496/.333252/.333252，对应1366/1365/1365环境，无采样饿死。

对未诊断过的同一 `model_000030.pt`，串行执行三clip各一次seed42、num-envs3、max-steps500的真实prefix。nominal physics保留随机command delay/一阶torque lag，无reset或参考循环。全部9实例提前物理失败：

| clip | 实际转移数/请求500 | 三实例首次失败秒 | 有效段body/foot均值cm | joint MAE rad | pelvis漂移均值/最大m |
| --- | --- | --- | --- | --- | --- |
| 0 | 49 | .98/.98/.98 | 22.02/14.38 | .2174 | .131/.451 |
| 1 | 50 | .98/1.00/.98 | 20.73/14.34 | .2097 | .127/.459 |
| 2 | 47 | .94/.94/.94 | 20.61/14.92 | .1883 | .129/.441 |

首检查点1的舞蹈/步行曾3/3存活500步，起身均在8.88秒失败；本次显著退化，不能把不同覆盖长度的joint/drift均值下降解释为改善。对齐前47转移后，三clip body误差19.00/17.89/17.33→21.27/19.87/20.61cm，foot误差1.70/2.44/1.70→13.03/12.79/14.92cm，root姿态.0646/.0541/.0677→.3562/.3272/.3707rad，也支持真实退化。失败时pelvis高度误差.173–.193m，超过既有.17m阈值；未修改终止定义。body/foot相对指标锚定waist_yaw_link，pelvis全局漂移另算。三个summary均complete_clip=false、success_rate=null；这是prefix失败证据，没有做全源成功声明，也未满足成熟后安排完整评估的条件。

本次三个prefix速度/官方限值峰值3.37/3.30/3.47，raw torque/额定峰值4.23/3.83/3.72，施加torque比均≤1；足接触峰274/276/247N，非足body接触峰0N。短段误差、超速和力矩饱和风险明显，不能以无NaN替代质量验收。参考needs_review仍保留：舞蹈14帧自穿>3cm，起身/步行亦有局部壳体残差。来源model22000的厘米误差只作解释尺度，不是本轮达标证据，不恢复已撤销rootXYZ门槛。

WM在29→40→60的loss .58610→.50585→.38731，每有效转移误差.63698→.54270→.42388，末步误差.89019→.75911→0.57928，末步valid .82227→.82466→.80610。相对首轮valid .83887也下降；有效样本没有坍缩，但当前约19%窗口末步已无效，且真实episode短、prefix退化，低loss不能独立视为好消息。双成本head loss .019945→.016168→.013332，分歧.000797→.000770→.000927近期反弹，不能保证峰值成本准确。上述loss采用本repo统计归一化，不与官方nominal误差作倍数比较。

actor裁剪前梯度29/40/50/60为3.75/4.79/11.29/7.19，critic loss .0652/.0720/.0677/.0399；critic梯度范数无遥测，源码有有限检查并裁剪至1，未虚报其数值。imagined survival .1346→.1173→.1535，imagined reward .00444→-.01725→-.02065，远好于真实负回报，提示早期模型偏差/优化风险，尚不能据此认定具体程序bug。

CPU逐张量核实model1/29/30及优化器各229张量均有限。其WM/actor/critic/成本head Adam步数分别10/1/20/10、290/29/580/290、300/30/600/300，证明恢复延续且logical更新次数未被microbatch放大。checkpoint正式runtime297.274/598.480/905.372秒；resume floor604.447秒，未丢失早先attempt时长。未重复训练smoke或新增GPU pilot。

CSV、当前TensorBoard标量及分段日志无NaN/Inf/OOM/Traceback；核实CSV从8576增至10897字节、日志38658→44777、当前TensorBoard7766→15444，旧事件文件保持不动符合换进程恢复。恢复后40/50/60吞吐.198/.187/.193更新每秒，30的低值含编译，不作稳态比较。GPU观察26–53%、7003MiB/16303MiB、52–58°C；训练allocated峰2.753GiB。唯一trainer PID2171568、supervisor2171561的完整argv/cwd/进程组与本repo匹配；另一个既有GPU进程未操作，三个评估进程均已正常退出。审计辅助读取曾遇Path序列化与summary字段名错误，已纠正并完整保存证据；均非训练故障。

后台runner PID2204241持有本RUN的audit.lock，CLI父子PID2204246/2204253已核实，命令无-m/--model覆盖。将同一个既有tmux会话名从opensprite-paper-hourly-audit改为用户指定 `opensprite-hourly-audit`，pane PID仍2204241；没有启动第二runner/Codex或系统调度。supervisor.lock由2171561独占。当前审计仍在runner子进程内，依据实际启动17:33:38.998633 UTC与3600秒串行规则，下次目标18:33:38.998633 UTC（北京时间02:33:39）；next_run_at须本次退出后由父runner写入，未伪称已观察到waiting。机器关机或tmux退出后须重启同脚本，预算不重置。

本次仅评估、保存证据/文档及校正审计tmux名称；未改训练源码/配置、机器人/UMR/奖励/网络语义、预算或固定train分支。下一小时重点看三动作前缀能否恢复、舞蹈/步行误差和termination、起身高reset、WM末步有效比例与成本head分歧、actor梯度及真实episode是否同步改善。仅对新检查点诊断，稳定后再做完整源；最终仍需同一冻结共享检查点的3clip×3seed完整请求及ghost交付。本次训练尚在运行，未创建delivery-complete marker，也未终止父runner。

报告：`outputs/train/opensprite-umr-lafan3-paper-20261101/hourly/20261031T173338.998615Z.md`；同内容保存为 `hourly/latest.md`、证据目录内 `analysis-report.md`，并追加 `docs/plan/training-review.md`。证据目录 `hourly/20261031T173338.998615Z/` 含verification.json、closing-verification.json、evaluation-comparison.json及三份prefix summary/rollout/log。

收尾动态复核 2026-10-31T17:37:56.461544+00:00：唯一trainer仍运行，推进至80，campaign=1184.565秒，soft余41835.435秒，hard余42015.435秒。三clip reward -1.2188/-1.2500/-0.6635，joint RMSE 0.5121/0.4427/0.5098；起身相对60出现反弹，不能用较早改善快照掩盖。WM每有效转移0.3758346438407898、末步valid0.7981933951377869，详见final-progress.json。本次三份实际评估仍固定30，未重复诊断。


### 2026-11-01：团队交叉质询后恢复WM预热并重训

用户明确批量/更新频率由team决定，不强制相同。已停paper轮model289，returncode0，累计2244.928498502355s。新run warmstart阶段继承此为prior，cap43020/43200不刷新，最早9508s不计。48项CPU检查与model289实际CPU初始化预校验通过。具体交叉质询、配方与预算见warmstart-training.md：env768、WM1024×4、PI512、critic5×4，大网络TF32不变；只WM/head+norm/Adam继承2890步，actor/critic全新；先500outer=2000新增WM更新，末端4890，留出24env做1/8步物理误差/false-alive审查后再joint。sourceend/bootstrap/delay/lag/UMR身份保持已有修正和严格适配。小时CLI审计在joint实际通过后恢复，并已更新有证据的学习问题干预授权。


### 2026-11-01 02:17 +08：WM预热500审核通过，准备joint

WMonly500×4已实际完成，model500验证WM/head4890步、PI/V优化0步、actor训练权重与model1相同、critic全状态相同、张量有限。heldout8step root坐标RMSE9.04→6.64mm、q.0448→.0341rad；末次falsealive7/47，近三次25/130仍有乐观风险。tracker/assets独立conditionalGO，同原余额恢复联合训练；不宣称H32准确或三动作成功。详见warmstart-training.md与RUN/warmup-checkpoint-review.json。


### 2026-11-01 02:21 +08：joint实际更新与小时审计恢复

新joint固定85acdaf，trainer2257110。model501实际CPU核对WM/head4894、PI1、critic20，229张量全有限；到outer700=actor200更新，约1.44outer/s，GPU4.8GiB，未发现OOM/NaN。真实回报和episode仍处早期，不能因吞吐/有限性宣称控制质量。CLI小时runner2259231/agent2259237于02:19:46启动并实际读取本run证据，正常时继续，发现有依据的学习/程序故障可安全干预并在同余额续训。完整命令/边界/审查记录见warmstart-training.md。
