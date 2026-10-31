# OpenSprite LAFAN 原生 UMR 方案与交叉审查

目标是**一个共享 tracker 学习三条独立完整动作**。正式动作必须由原生 UMR 的蒙皮表面、学习 correspondence、法线与约束优化生成。骨架 task-space IK 只作已隔离的诊断 baseline（`outputs/diagnostics/`），生产入口不导入它，不可将它称作 UMR。

## 源和完整动作

作者仓库 https://github.com/orangeduck/lafan1-resolved ，BVH 下载 https://theorangeduck.com/media/uploads/Geno/lafan1-resolved/bvh.zip 。这是由原C3D重新解算、Geno75关节/蒙皮的60Hz版本，BVH实际Frame Time=0.016667；不是官方22关节30Hz版本，不能套其SMPL-X适配器。`.reference/orangeduck/lafan1-resolved`保留仓库和完整原始BVH包，运行只用`data/sources/lafan1-resolved`的三源及`assets/geno`原生FBX。

| 动作 | 源帧数 | 源N×dt时长 | 选择证据 |
|---|---:|---:|---|
| dance2_subject1 | 13541 | 225.687847s | 长舞蹈；源根倾斜99百分位33.7°，比dance1的69.6°更温和 |
| fallAndGetUp3_subject1 | 6131 | 102.185377s | YY 20261007 `render_representatives.py`明确的历史倒地起身测试条；此次完整保留，不只取历史展示前35s |
| walk4_subject1 | 9835 | 163.919445s | 较慢长步行，根速度95百分位0.89m/s，比walk1约1.64m/s更简单 |

全部源保持原样、真实时轴、原始帧序，不裁短、不循环、不补尾帧。50Hz采样只覆盖最后一个真实源采样时刻之前。倒地起身含接触与快速过渡，不能称简单或保证12小时内全部掌握。

## 真正 UMR 链

`third_party/umr`来自官方 https://github.com/hanyang9/UMR 的YY固定版本`e24fc070`。保留算法与MIT notices；移除了YY memory-lifecycle补丁对外部`yy_retargeter.umr_memory`的依赖，其余本体未改。`.reference/UMR`用于源码对照。`scripts/retarget_lafan.py`是正式native入口。

1. 原生Node/owned Three FBX exporter读取仓库内`assets/geno/Geno.fbx`；75骨架BVH驱动真实蒙皮surface，不用SOMA代理或X2动作。
2. 新`configs/umr/opensprite.json`只定义Sprite0825：32官方STL visual meshes（不是primitive代理面）、31官方关节，Tpose仅左右肩roll±π/2，其余0。源center是Geno Hips，机器人center是`pelvis_link`，都明确按各自真实身高归一；不用X2 waist/G1 bbox ratio/1.32m身高。
3. **从零训练Geno/OpenSprite新correspondence**：4096点、原生500epoch、两样本all_samples，身份严格为`['geno','opensprite0825']`。已完成，日志实测身高1.711344m/0.924985m；GPU含启动保守墙钟22.586919s，计入12小时预算。`assets/umr_correspondence/geno_opensprite0825/`中保留dataset和learned slots；不借任何X2 slots。
4. 使用原生NR surface-vector optimizer和法线transport。NR本身**没有SMPL-X55/19语义分段**：source_slot_part_ids全0、uniform surface slots/cost，原算法会禁用semantic self_contact_map。不能把defaults里同名配置宣传成实际启用的语义接触项。仍有真实surface point/normal、source slot ground contact/anchor、机器人几何self-penetration。
5. joint hard ranges由Sprite MJCF读，后处理速度边界从同一官方URDF读。没有X2 pose、head数量、motor envelope或feet配置fallback。

## 地面与模型适配的实际审查

首轮native NR默认surface_slots_all短段已暴露采样漏sole corners：walk最低-6.96mm，getup25–30s -12.68mm，74–79s -25.36mm。原算法已经支持`ground_penetration_hard_constraint_mode=mujoco_collision`，所以直接用这个现成模式约束实际32collision geoms，而不是改算法或套IK根高度修补。

该模式需要真实world plane；裸机器人XML没有地面，若ground_geoms=0会静默无约束，源码审查已捕获。独立`assets/opensprite/retarget_scene.xml`仅include原机器人并加z=0平面，32body/31joint的质量、惯量、轴、限位、armature全部逐值一致。采样跳过world plane，复建dataset必须验证与先前点云逐位一致，再复用刚学slots。

新型号初始收敛显式选`pose_init_iters=160`、每帧`iters=3`。短段和完整baseline都证明只在后处理限速会破坏地面几何（74s片段需8.84cm、全长89.34s需19.69cm补偿，均被3cm cap拒绝），因此通过原算法已有`robot.dof_max_dq_box`在native QP内同步约束：每迭代box=官方URDF速度×真实BVH dt×0.95/iters。native直接以dt=1积分dq，global_step_size是L2锥半径而非更新乘数；故每帧合计不超过官方速度的95%，root6DOF不添加虚构硬件限。初始160迭代只建立首姿，不消费动作时间。其他源目标沿用原生NR。self-penetration仍为native hard/slack cost机制，**有slack不等于无穿透保证**；内部slack未导出时，必须报告实际几何signed-distance残差，不能称全硬通过。躺卧和真实身体接触有效。

## 输出、验证和训练隔离

正式转换模块`retarget/convert_umr.py`只接受完整native NR轨迹、正确robot/source身份、官方31关节顺序、全部源frame_ids。共享`retarget/processing.py`仅做严格float32边界、YY物理时间acceleration/velocity QP、纯FK和逐源速度，不包含旧IK标定或目标解算。

最终`data/motions/lafan_three.npz`保留三源names/starts/lengths、31joint与32body完整world pose/velocity、fps50。生产guard为：

- `retarget_method=umr_native_nr_surface_vector_v1`
- `robot_model=opensprite0825`
- `dataset_revision=opensprite-umr-geno-v1-20261031`
- `correspondence_names=['geno','opensprite0825']`

独立审查新correspondence左右/前后/上下、法线、脚底覆盖；short UMR覆盖0–5s站走、25s倒地、74s支撑起身。只有短段模型/几何正确后才算完整三源。native→50Hz反混叠与速度QP可能重新引入几何误差，最终float32重新检查真实32geometry ground/self contact、全身离地时长、限位、速度及代表段渲染。需要review的轨迹明确标`needs_review`，仍纳入同一共享policy，不能隐藏质量失败。

合库前每源独立FK和求导，不跨动作边界；训练连续窗口只在单源内部，完整源评估从首帧开始且不能reset或wrap。

## 独立环境与预算

UMR环境`.venv-umr`使用`requirements-umr.txt`：torch2.8cu128适配本机RTX5080，numpy1.26/MuJoCo3.3.7兼容原算法；Node≥18解析FBX，所有exporter源码/Three/FBX/slots在本repo。无外部checkout、`.reference`或symlink运行依赖。

原生NR蒙皮和优化走CPU，不占GPU训练预算。correspondence GPU训练以attempt累计，失败、重试也在finally结算，台账`outputs/umr/correspondence_gpu_time.json`。tracker训练团队将其与pilots、正式训练累计到12小时上限。

## 全长最终实测（2026-10-31）

原生full_velocity分支全部源成功，最终bank 24589帧，motion_starts=[0,11284,16393]、motion_lengths=[11284,5109,8196]。全部官方closed hard ranges/velocity严格通过，全部32 primitive地面最低点≥+0.500mm。小幅root-Z残差补偿硬上限30mm，实际如下。

| 动作 | 50Hz帧数 | N/fps时长 | root-Z最大/平均补偿 | 最深自接触穿透 | >30mm帧占比 |
|---|---:|---:|---:|---:|---:|
| dance2_subject1 | 11284 | 225.68s | 13.32/3.29mm | 43.70mm | 0.1241%（14帧） |
| fallAndGetUp3_subject1 | 5109 | 102.18s | 15.85/4.12mm | 19.57mm | 0 |
| walk4_subject1 | 8196 | 163.92s | 13.01/5.39mm | 26.11mm | 0 |

表中N/fps表示帧extent；最后一个采样timestamp是(N−1)/fps，文件保存精确time，不补末帧。舞蹈全body最低点>2cm累计3.36s，最长.54s（包含源跳跃）；getup无全body离地>2cm。各源均标needs_review并纳入共享policy，不隐藏self slack。

独立审查：新slots左右错侧0，normalized xyz相关系数.9947/.9991/.9165，足底34/41向下slots均绑定正确Foot/Toe，法线/bary误差约1e-7。形态差异仍影响细节：Sprite臂展/身高1.286，Geno约.98，height-normalized几何correspondence会把一些人体手slots配到机器人前臂。getup手末端root-local方向的左右median角差16.1°/16.9°，95百分位约61°/56°；整体手撑/倒地/起身趋势保留，复杂接触tail保真有限。这些是新embodiment的实际残余，不是旧机器人映射复用。

正式重建：`.venv-umr/bin/python scripts/retarget_lafan.py --stage all`；已完成raw native在`outputs/umr/full_velocity/`，仅重做转换：`.venv-umr/bin/python -m retarget.convert_umr --input-dir outputs/umr/full_velocity`。核心算法没有改成IK，所有重采样、末端FK和小几何补偿均在显式后处理层。
