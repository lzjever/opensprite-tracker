# OpenSprite 真实策略录像与高画质展示

用户要求参考本地beyondmimic-umr的录制脚本、格子地面、高画质与信息叠加。参考脚本存档在.reference/beyondmimic-recording-20261101，仅供研究；实际renderer完全在本repo，不import相邻项目。不移植X2机器人或其录制脚本中的关闭termination设置。

## 本次固定模型与真实轨迹

在2026-11-01 02:51开始时，最新已保存检查点为model_003000.pt（包括500轮WM预热、actor2500更新），三个动作统一固定该模型。seed42、预先指定env0、不挑选最好实例；请求各源前1500转移/30秒，nominal保留随机command delay与torque lag，无reset或参考循环。只读评估不追加梯度训练，现有trainer与12h ledger继续。

| 动作 | 实际记录 | 结果 | 末端pelvis漂移 |
| --- | --- | --- | --- |
| dance2_subject1 | 30.00秒 | 未物理失败，但不是完整225.66秒源成功 | 3.089m |
| fallAndGetUp3_subject1 | 10.06秒 | 姿态失败：gravity-reference²1.600183超1.6，高度差.03449m未越.17m | 1.534m |
| walk4_subject1 | 30.00秒 | 未物理失败，但不是完整163.90秒源成功 | 1.817m |

root/body误差保留世界与相对坐标语义；腿部瞬时速度仍可超官方限值，实际施加torque均限幅。summary/rollout位于outputs/recordings/model_003000-20261101-0251/clipC/eval，不把短录像称为全长完成。素材仅本机保存，不推送LAFAN派生视频到公开仓库。

## 视觉展示

renderer：opensprite_tracker/supermimic/render.py；只对已记录姿态做CPU MuJoCo FK和EGL渲染，不步进物理。保留官方32body网格与真实世界坐标，不做逐帧位置对齐隐藏漂移。白红为策略实体，蓝色半透明为参考。

- 1920×1080，50fps，MSAA4，4096阴影贴图、棋盘地面、双光源。
- 相机按两组全部link的位置与透视深度自动适配，平滑追踪；HUD上下安全区保留，漂移达到3m仍显示完整双模型。
- 中文HUD：动作名、checkpoint、seed/env、帧率、局部时间/源长度、请求/实际录制长度、LIVE/FAILED、pelvis世界位置漂移、joint RMSE、20ms内逐关节施加力矩/额定峰比及速度/限值峰比。
- 状态t展示刚完成转移t-1测量；初始状态无转移测量显示破折号，绝不使用未来信息。RMS直接来自记录q-ref；速度比是周期峰值，不是endpoint值。
- 起身首次失败之后不继续模拟；仅将最终真实画面停留1.5秒，并明确标为视频停帧回看。render.json记录这75个额外编码帧。
- H264 CRF18、yuv420p、faststart；Pillow优先使用系统NotoSansCJK，支持DejaVu降级。

```bash
.venv/bin/python -m opensprite_tracker.supermimic.render \
  --rollout outputs/recordings/model_003000-20261101-0251/clip0/eval/rollout.npz \
  --output outputs/recordings/example-new-video \
  --width 1920 --height 1080 --fps 50
```

--preview-frame750只出预览；--max-render-frames可做少量帧视觉QA。--model允许移动checkout后的repo内MJCF覆盖。所有输出目录新建，防止覆盖原始记录。

## 实际验收

已真实渲染并目视检查初始、15秒、30秒极端漂移及getup失败帧；修复了透视深度下参考脚部被HUD遮挡的问题。成片解码抽检舞蹈29.5s、起身停帧11s、步行15s，文字与模型可见。ffprobe核对三段1080p50fps帧数1501/579/1501，时长30.02/11.58/30.02秒；实际物理时长30/10.06/30秒，差异来自初始帧与明示停帧。合集3581帧、71.62秒，带三段chapter。video-validation.json保存帧数/来源，未生成摘要清单。

合辑：outputs/recordings/model_003000-20261101-0251/opensprite_tracker_3000_overview_1080p.mp4。

本次只升级离线render.py与文档；训练加载的env/engine/网络/奖励/actuator代码及固定train分支不变。后台审查发现renderer相对训练快照有diff时，应按离线展示更新记录，不误判为训练物理在运行中被更改。
