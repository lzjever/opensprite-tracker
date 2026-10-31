# SuperMimic 论文规模重新训练（历史阶段）

最新执行方案已由用户授权调整batch/频率并先预热WM，见[warmstart-training.md](warmstart-training.md)。以下是已停止并保留的论文规模实验记录。

## 用户授权与新预算

2026-11-01用户要求“现在就改。训练规模包括批量也按照论文来改。做完之后重新开训”，随后明确“重新给12小时预算”。本轮是全新campaign，从头初始化WM、actor、critic和优化器；不对不兼容旧权重做静默加载。旧campaign于01:09 +08停止，末模型`outputs/train/opensprite-umr-lafan3-supermimic-20261031/model_015551.pt`成功保存，累计9508.822220秒（含其旧prior），作为历史保留，不再扣新预算。

新run：`outputs/train/opensprite-umr-lafan3-paper-20261101`。新prior初始0，硬上限43200秒，supervisor/CLI软上限43020秒，180秒停止预留。初始化/编译/所有本轮GPU验证、失败与恢复均计入；不因普通重启再次刷新。正式启动记录真实时间与固定commit，预计结束时间从实际启动算；不沿用旧10:28结束预期。

## 官方依据与目标

[官方论文](https://arxiv.org/pdf/2610.28480)，Appendix C TableA.4第15页、Appendix D TableA.5第16页；精度TF32见第8页Figure5(a)图注及[官方项目](https://supermimic.github.io/)脚注4。

| 项目 | 新配置 |
| --- | --- |
| WM | 2048/2048/2048，Sprite393输入/140输出，9498764参数 |
| Motion encoder/shared decoder | 均2048/1024/512/256，latent32+tanh；Sprite合计9321791参数 |
| Critic | 256/128 |
| 仿真环境 | 4096；env_id modulo3，clip0/1/2为1366/1365/1365 |
| Replay | 每环境1000步，保留全部字段和reset边界 |
| WM | 每轮10个独立Adam更新，每批2048个8步自回归窗口 |
| Policy | 每次4096条32步想象轨迹，整批一次Adam，β=.9/.95 |
| Critic | 固定同一批λ-return，5epochs，每epoch4个逻辑minibatch |
| 学习率/γ/λ/UTD | 2e-4 / .99 / .95 / .4 |
| LR计划 | 250000次cosine至.03倍，之后constant |
| 迭代上限 | 320000；12h预算可能先到达，不能保证原论文硬件吞吐 |
| 数据噪声 | action noise std.1，保持物理domain randomization |
| 精度 | TF32矩阵乘，FP32状态积分与解析头，关闭BF16 autocast |

原代码500次WM-only预训练不在论文TableA.5中，改为0；开始前64真实步用于bootstrap replay/normalization，然后按算法每轮共同更新。64步bootstrap是本地启动细节，不冒称论文给出的参数。

## 16GB实现与语义约束

当前执行参数已在首轮实测后调为WM microbatch2048（一次处理完整WM批）、PI microbatch512；下文256/128是最初验证配置，调优记录见文末。网络、逻辑批量、优化器更新次数及新12h累计上限不变。

论文硬件H100 80GB，本机RTX5080Laptop16GB。不能把逻辑batch、网络尺寸或环境数缩小后仍声称按论文规模运行。

- Replay含664个float32及f/ep_step/done，4096×1000约10.18GiB，放本机CPU RAM；按原分布在存储device抽取，样本传训练GPU。保留s1预reset、terminal transition有效、后续mask和episode历史截断语义。默认不pin整个10GB RAM。
- WM逻辑2048窗口分microbatch256，loss分母保持完整2048，整批末才裁剪/Adam一次；外层仍做10次。
- Actor逻辑4096条分microbatch128，各自完整H32可微滚动，按完整4096求均值累计梯度，整批末clip/Adam一次。WM/critic权重冻结但对状态/动作的导数不断。
- Critic想象特征/目标可存CPU，逻辑每轮5×4次Adam不变；单逻辑minibatch内部microbatch8192累计。RunningNorm必须使用同一完整想象批次，λ-target固定后epochs中不重算。
- 这会改变浮点求和顺序和随机数消费顺序，不声称与一次物理整批bitwise相同；同一logical batch目标/mean、优化器次数和采样范围必须一致。

## Sprite与动作保持

机器人仍为31joint/32body OpenSprite，官方模型物理参数、新Geno/OpenSprite correspondence、完整UMR三源保持。用户要求的5ms command delay与5–10ms torque lag保留，未替换成G1/X2物理或改变延迟以追论文误差。一个共享policy支持三动作，不从来源参考偷用已学权重。

当前legacy_full奖励、额外两成本head、Sprite终止/接触适配延续用户已接受的本地SuperMimic训练逻辑；本次对齐的是论文网络与训练规模，不冒称所有机器人/奖励/监督实现与G1实验逐值相同。

## 验证与启动

团队分工：tracker实现engine逻辑批量累积与critic目标；assets实现CPU replay并验证边界/字段；retarget独立检查论文/环境分配/精度与恢复；主代理配置、预算、停止/启动、固定快照、文档和集成验证。

启动前CPU检查新网络尺寸、CLI、replay和梯度累积等价性。随后真实4096环境、WM2048逻辑batch、PI4096/H32的首次完整更新作为本轮正式训练验证，全部初始化和编译均由supervisor计时。必须看到有限WM/actor/critic更新及checkpoint后才报告稳定开训；OOM/失败保留日志和累计时钟，不静默降规模。

通过后重启纯CLI每小时后台审计，明确指向新run；旧runner已停止，不建系统service/timer。审计需关注微批逻辑步数、主存/GPU显存、吞吐、三动作采样与完整实际表现。最终仍为同一共享checkpoint的3clip×3seed完整请求和3ghost视频，诚实披露未收敛项。


## 交叉审查与启动前验证记录

CPU整批对microbatch测试验证不等长分块仍保留WM/actor梯度、损失和最终参数（仅容许浮点舍入），WM/actor各1次Adam，critic5×4=20次；全批RunningNorm和固定λ-target一致。Replay4项测试验证环形覆盖、末转移有效及后续mask、episode/最旧历史边界、跨目的device接口和2669byte/transition容量。机器人delay/lag与原生UMR身份测试继续通过。

独立审查确认4096按modulo3分配1366/1365/1365无需降低为4095；想象起点拒绝H步越界会产生轻微clip比例变化，不能声称每个想象batch严格均分。

另修复审查发现的旧末帧bootstrap：H步允许到inclusive clip_end，但该步reward之后应无未来价值；现在到源末帧的imagined death=1，保留该步reward，tail梯度0。新增CPU测试验证，不改合法倒地接触或物理终止阈值。

为真实首轮验证，训练第一完整outer更新额外写日志和atomic checkpoint；以后每10次日志、1000次或600秒保存。首次实际完整更新即验证论文4096env/2048WM/4096actor/H32规模，初始化和编译计入本轮时钟。


### 2026-11-01 01:23 +08：论文规模真实首完整更新通过

新固定源码`119c2660e1dfd6d35ac40e5f04b17342bce1dea6`，分支`train/opensprite-umr-lafan3-paper-20261101`不再移动。01:17:11+08启动tmux `opensprite-paper-train`，唯一trainer PID2133080；初始化/编译已进入新campaign累计时钟。实际launch.json确认all_networks_random、4096env、1000replay、WM2048×3、10×2048/K8、PI4096/H32、policy2048/1024/512/256、ampnone/TF32。

真实首轮保存model_000001.pt，235733901bytes；runtime297.274秒（含初始化/编译）。CPU重新读盘核实229个checkpoint/optimizer张量全有限；主干9498764参数，actor9321791参数。Adam状态精确为WM10、PI1、critic20、双costhead10，证明microbatch没有变成额外Adam更新。证据`RUN/first-update-validation.json`。三clip实测active_fraction=.333496/.333252/.333252，符合1366/1365/1365。首轮真实reward -1.204、WMloss3.822、每有效转移误差3.372、末步valid .839、actor梯度.612、criticloss1.217均有限；新随机模型尚未学会，不拿旧轮优秀指标冒充本轮质量。首轮速率包含编译，不作为steady吞吐。

CPU replay分配/真实4096×64启动采集与CUDA采样路径已通过，无OOM或设备索引错误；trainer主存RSS约14–15GiB，当前GPU总显存约5.3GiB，GPU高负载、温度55°C。旧训练与旧审计均未重启。

新纯CLI小时runner于01:23:07+08启动tmux `opensprite-paper-hourly-audit`，PID2164748、CLI2164752，状态RUN/hourly/runner-state.json，真实agent已执行读取。立即首审后每3600秒串行，不创建service/timer/cron。当前新预算hard预计13:17:11+08，soft预计13:14:11+08（无停机条件下），普通恢复不刷新。


## 首轮实测后提高执行批量

首20更新约.09–.096次/s，GPU总显存约6.1GiB/16GiB、PyTorch峰约1.037GiB，存在较多余量。团队只读评估确认MLP/LayerNorm/FK无batch平方项，WM与actor图顺序执行。为减少不必要分块开销，安全停止于model_000029.pt后，将WM执行批256→2048（完整论文2048窗口一起执行），PI执行批128→512（每logical4096分8块）。critic执行批8192不变。

停止返回0，supervisor累计604.447193秒保留，prior仍0、soft43020/hard43200不变，不再给12h；恢复floor≥604.447193秒。旧固定source119c266分支保留，新增batch快照，所有权重/Adam状态从同一29检查点恢复，无随机重开或架构变化。重新填充replay与新形状编译仍计时；若大形状OOM只减执行分块，不能减论文logical batch。旧小时审计暂停防止维护竞态，验证后恢复同一CLI runner。


### 2026-11-01 01:34 +08：较大执行批量验证通过

固定source快照c71b9d1，恢复同一model29及累计floor604.447193秒；新12h未刷新。新的WM2048物理整批、PI512分8块（logical4096）、critic8192已实际完成update30并原子保存model_000030.pt。读盘验证229个张量有限，Adam步数WM300/PI30/critic600/costhead300，与累计30次逻辑outer一致；wm/actor逻辑batch仍2048/4096。runtime905.372秒已包含前阶段和新形状编译，GPU无OOM，PyTorch峰2.753GiB、GPU总约7GiB。证据RUN/batch-tuning-validation.json。新CLI小时runner2204241/agent2204246于01:33:38+08启动并真实读取当前证据，旧runner均已停止；仍无systemd/cron。


## 世界模型10:1优化器计数的证据边界

用户追问WM更新是否为actor十倍。当前本地实现明确每outer：采10个WM minibatch（每批2048个8步窗口），逐批各做一次Adam；actor一次logical4096/H32优化；critic5×4=20次Adam。满窗口请求数为WM10×2048×8=163840转移监督、actor4096×32=131072想象转移，真实WM有效数仍取决于reset mask，不能将optimizer计数比直接当样本量或计算量比。

官方p16 TableA.5明确写“10 batches of 2048 windows per update”，另写policy updates per world-model update=1；p15 AlgorithmA.1只给拟合WM阶段与更新policy阶段，没有展开十批逐批step还是先累积十批再step。当前10次WM Adam是沿用本地成功源码、将十批按minibatch优化的实现解释，不能冒称官方已公开代码验证了10:1 Adam；也没有证据断言官方必须十批累积一次。本次只澄清证据，不凭歧义改动正在运行的健康训练。

较大执行批量恢复后已推进到60次更新，无OOM/NaN；最近日志steady吞吐约.193次/s，相比小执行分块.09–.096次/s约2倍，PyTorch峰2.753GiB。WMloss约.387，模型刚开始学习；不能从训练指标宣称三动作完整成功。小时CLI审计保持运行。
