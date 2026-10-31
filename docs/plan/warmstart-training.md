# 大世界模型预热后重新训练：团队审查与执行方案

## 最新授权与问题

2026-11-01用户要求团队认真分析、交叉质询、修正并重训，先预热世界模型；随后明确batch/更新频率可由team决定，不强制相等。继续每小时实际检查，发现问题及时干预续训。保留大网络、Sprite真实物理、UMR三动作、一共享policy、command delay与torque lag。新方案为本机和三动作调整，不再声称训练规模逐项等同论文。

此前大方案checkpoint147的真实3env前缀：舞蹈.96/1.04/1.00s、起身1.26/1.26/1.28s、步行1.40/1.42/1.38s均高度误差失败，没有完整动作成功。身体世界误差约.255/.247/.237m，joint RMSE .434/.403/.374rad；证据paper轮on-demand/20261101-0149/review147.json。它证明当前策略差，不单独证明具体算法bug。

旧成功小方案约1.85–1.9outer/s；大方案.195/s。WM窗口吞吐从18944–19456/s降至3994/s，actor轨迹吞吐反而从474–486/s升至799/s，但actor优化步数少近10倍。WM/PI网络、精度、环境与batch同时改变，不能只归因batch大。取消原500outer×10=5000次WM预热是额外未充分验证变量，需纠正；不能把500outer误称500次WM Adam。

## 团队交叉质询与结论

- tracker质疑简单1:1:1：32步后的bootstrap权重gamma^32约.725，critic从20次降为1次可能欠拟合tail。保留原5epochs×4minibatches。
- assets质疑只看混合WMloss：近半输出为actuator memory且未来随机delay有固有不确定性；fresh actor几乎零动作，预热采集可能偏向被动/失败及饱和域。预热应检查真实留出窗1/8步物理量、动作/接触分布与false-alive，不能按随机actor摔倒判预热失败。
- retarget质疑环境/监督比例：env4096配WM1024×4，每新realtransition只有.4个WM窗口；env768为2.133，接近原大方案2.0，还减少仿真和CPU搬运。三源随机相位下768=3×256足够构成有覆盖的起点池，但不保证所有相位已学会。
- 三方认可仅继承同Sprite/dataset的WM+成本head+norm/Adam，actor/critic和采样器重置；坏策略产生的真实物理转移仍是合法动力学监督。warm start不能代替验证。严禁只继承权重却清空norm，也不能用旧initialize-from偷偷加载旧actor。
- 主代理要求全部候选schema/Adam预验证后才修改模块，防止后半校验失败造成半初始化；通过21项专属测试和实际model289 CPU预校验。

## 最终训练配方

| 项目 | 当前值 |
| --- | --- |
| WM / policy | 2048×3 / enc与dec均2048-1024-512-256，latent32 |
| 物理/精度 | Sprite31joint/32body，TF32+FP32积分，原delay/lag/DR |
| 环境 / replay | 768，3clip各256；CPU1000steps/env约1.91GiB |
| WM训练 / actor训练 | 每outer4批×1024个K8窗口 / 每outer1批×512条H32轨迹 |
| critic | 同一想象批的固定lambda-return，5epochs×4全局逻辑minibatch |
| 执行分块 | WM1024、PI512均完整单批，critic4096；不误乘Adam步数 |
| optimizer | 原Adam及2e-4，policy beta=.9/.95；gamma.99、lambda.95、UTD.4 |
| 验证 | 末24环境（每clip8）仅作为WM留出窗；训练用744env（每clip248） |

WM窗口/全采集新转移=4096/(2.5×768)=2.133；对WM训练pool为4096/(2.5×744)=2.202。actor历史仍可用全部env，WM训练窗口不会取末24；采集的RunningNorm仍观察全部env，故不声称normalization完全隔离。验证采样使用独立Generator，不消耗训练RNG。

## 初始化、预热与阶段审核

paper轮已安全停止于model289，全部权重/日志保留。本run复制独立source文件`outputs/train/opensprite-umr-lafan3-warmstart-20261101/world-model-source.pt`，不依赖邻居或.reference。显式--world-model-from只加载WM、双成本head、各自norm与Adam，保留2890次优化；检查method、robot/data、collision、模块key/shape/dtype、Adam组/矩尺寸/finite。actor/critic权重与optimizer及sampler全部fresh。actor obs/goal norm在真实collect中正常更新；“冻结actor”指学习参数与Adam，不是停止观察统计。

第一阶段配置phase=world_model_warmup，iterations=wm_only_iters=500。额外500outer×4=2000次WM/成本head Adam，末端WM总4890，actor/critic Adam仍0；此阶段结束时保存，主代理人工审核后才改phase joint/iterations320000并resume同run，不新给预算。每50outer及启动时在1536个留出K8窗做eager/no_grad诊断：全体/各clip的1/8步rootpos/rotation/q/qd/速度与memory归一化及原单位误差、validcount、物理false-alive。无真实failure分母的rate记null，不冒称0误差或数值故障。

预热审核不采用单一混合loss硬阈值：检查各clip有效coverage、1→8步误差增长、米/rad物理误差相对既有.17m/1.6重力阈值的余量、真实failure的false-alive、costhead分歧。若仍明显偏乐观或关键物理误差过大，在同剩余预算内继续WM-only并重新审核；不能只因到500就宣称准备好。joint后再验证真实prefix；预热actor仍摔倒不说明WM无法预测跌倒。

## 预算与源码

本次不刷新此前用户新给12h。最早Oct31轮9508.822220秒不计；已停Nov1 paper轮2244.928498502355秒必须作为本子run prior。supervisor软cap43020/hard43200，180秒停止预留；本段CLI max-wall-seconds=40775.07150149765。active_seconds已含prior，resume-floor仅active-prior，不二次加prior。所有新GPU初始化/编译、预热、续训算本段；CPU研究/停机不伪算训练。固定源码各阶段另建分支、旧分支不移动。

训练前48项CPU检查通过：原Sprite/UMR/delay合同、micro梯度/optimizer次数、source末帧bootstrap、WM-only严格初始化及源无别名、留出窗口与RNG隔离、诊断不改norm/Adam、resume移除初始化flag且不重置预算。实际source289 CPU预校验确认WM/head步数2890。

## 每小时审查与干预

联合训练确认实际更新/检查点后，恢复纯CLI tmux小时审计，不使用系统service/timer/cron。每轮看逐clip真实return/终止/joint误差、episode、有效WM误差及留出分项/false-alive、critic/actor/model内外差距、数据采样/速度/力矩/资源及真实checkpoint prefix。数值有限不能代替策略收敛。

用户已授权必要干预与续训：若持续真实退化、明显模型乐观或覆盖不足，经证据复核可先冻结actor补WM、调整batch/更新比率等可逆训练参数；每次安全停止、保留checkpoint/日志、做最小验证、新固定快照、保持累计时钟，不用单个噪声点反复改算法。机器人/UMR/延迟/力矩迟滞不得被偷偷简化。最终仍需同一共享模型的3clip×3seed完整请求、3ghost视频及诚实质量报告。


## 实际预热审核：有条件进入joint

固定warmup快照c869ce7，02:13:19+08正式启动。48CPU检查通过；500outer在约169.96秒内完成（初始化/编译计入），model500保存。读盘WM/head优化步数4890，actor/critic optimizer均无step；actor的enc/dec参数与model1逐值相等，critic全state逐值相等，所有张量有限。actor观察norm正常积累，未误当其为参数训练。证据warmup-checkpoint-review.json。

最后8步留出误差.44885→.28949；root位置坐标RMSE9.04→6.64mm、旋转向量分量RMSE.0591→.0432rad、q RMSE.0448→.0341rad；有效1249/1536=.813，各clip410以上有效。末次falsealive7/47，但最近400/450/500合计25/130=.192；H8只有.16s，不能据此宣称H32=.64s已经准确。

tracker与assets独立审查并交叉质询后均给conditionalGO：所有clip的原单位误差改善，继续冻结窄动作域的边际收益有限，应开始joint收集新分布；保留每50outer留出诊断，并看真实prefix与imagined收益差距。不是宣称策略已可用。改phase joint/iterations320000，同run从model500完整resume；prior、maxwall、cap不变，新固定joint分支，旧warmup分支保留。


## 联合训练实际启动与检查

固定joint快照85acdaf，02:17:59+08从model500 resume同run，预算/时钟不变。model501读盘确认WM/head4894、actor1、critic20，229个张量均有限；proof为first-joint-update-validation.json。到outer700（actor200次）零自动故障重启，吞吐约1.44outer/s，比上一大批量.19/s约7倍；GPU约4.8GiB，Torch峰2.51GiB。速度改善不等于策略质量已通过，真实reward/episode仍处早期且有反弹，后续由实际前缀和完整评估判断。

纯CLI小时runner于02:19:46+08在tmux opensprite-hourly-audit启动，PID2259231、CLI2259237；已实际读取训练/留出诊断，立即首审后每3600秒串行。异常安全恢复继续使用prior2244.9284985+本段累计时钟，不能刷新12h。详细状态在本run/hourly/runner-state.json，结论在latest.md。

运行本repo已有的world-model-source.pt时使用配置中的显式WMonly初始化参数；此文件是本repo本地训练产物副本，不依赖.reference或外部checkout。公开代码不附带LAFAN派生模型：没有兼容种子时移除--world-model-from，从随机WM开始，增加预热并执行同一留出验证；不声称可直接复现继承模型的初始统计或训练成绩。最终checkpoint完全包含运行参数和权重，推理不需要源初始化文件。
