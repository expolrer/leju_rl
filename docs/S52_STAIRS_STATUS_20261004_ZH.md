# S52 楼梯训练状态（2026-10-04 整理）

本记录根据 6 服务器 `/home/zzx23457/hhw/LejuLab-Train` 的训练档案核对。服务器训练状态文件 `analysis/s52_transfer/s52_transfer_state.json` 仅更新到 v56，不能单独视为最新结果；按实际运行目录和归档时间，v68 是最新完成的短预检，v67 是最新完成候选物理联评的轮次。目前未发现楼梯训练进程。

## 受保护模型与完成情况

| 模型 | 来源与 SHA-256 | 当前用途 |
| --- | --- | --- |
| S53 `model_92099.pt` | `logs/rsl_rl/kuavoS53_stairs_tgmp_terrain_conditioned_tracking/2026-08-22_19-01-28_s53_tgmp_terrain_warm90100_v7/model_92099.pt`；`6483ff66456f1e218713f228114a89b6bd5688d94d4c2612ebc247375812f0f4` | 只读 S53 teacher |
| S52 v22 `model_92150.pt` | `logs/rsl_rl/kuavoS52_stairs_adaptive_kl_guard/s52_model92150_warmstart_v22_20260909/model_92150.pt`；`93c457de92f9102125d7c88f8721e25a35e83caa03e4112d7a01074c78e50f56` | 受保护的 S52 warm-start；尚未通过多 seed 楼梯终点及冲击联合验收 |

S53 正式轮还有一个**不同**的 `model_92150.pt`，SHA-256 为 `1a4902952951da68a80d23522a5d3b7004bb0cc46f844f54f90313c6bc2f7b0f`。不同本体、任务、目录的同名权重不能互换。S52 已完成 27 个策略关节与 29 个仿真关节的映射、148 维旧 actor 观测及 27 维动作检查，Lab/MuJoCo 站立和行走已验证；S52 MuJoCo 完整楼梯、正式部署及域随机化尚未验收。

## 最新完整候选联评：v67 FutureContactResidual

v67 冻结原 148 维 actor，新增未来接触目标的 154 维残差模块。零残差 checkpoint 在固定 seed=131 的 1351 步物理 rollout 中与原模型逐帧动作相同。`32×2` 冒烟和 `128×8` 预检完成；五个候选均完成上楼、平台、四级正向下楼与落地，零重置，但未达到终点误差 `<0.25 m` 门。

| checkpoint | 终点误差 m | 滑移 p95 m/s | 峰值足力 N | 在线最坏净空 mm |
| --- | ---: | ---: | ---: | ---: |
| 92150 | 0.3361 | 0.2106 | 1360.7 | 0.0058 |
| 92151 | 0.3181 | 0.2302 | 1457.2 | 0.1030 |
| 92153 | 0.3077 | 0.2198 | 1305.6 | 0.2117 |
| 92155 | 0.2957 | 0.2905 | 2142.2 | 0.1011 |
| 92157 | 0.3565 | 0.2361 | 1652.6 | 0.0077 |

本轮否决，未采用新权重。完整报告见 `/home/zzx23457/hhw/LejuLab-Train/analysis/s52_transfer/training_records/v67_future_contact_residual_preflight128x8_20260914`。仅 8 次 PPO 更新，短预检失败不等于条件化路线无效。

## v68 短预检与最终物理回放

v68 `FutureContactPhaseResidual` 的 `128×8` 训练达到目标，总计 24,576 timesteps；最终 checkpoint 为 `model_92157.pt`。原归档分析命令缺少 `--rollout-dir` 参数，已用现有分析器补跑，生成完整 TensorBoard 标量 CSV/JSON、奖励诊断图和回放比较文件。运行目录：`logs/rsl_rl/kuavoS52_stairs_future_contact_residual/2026-09-14_01-11-54_s52_future_contact_phase_residual_preflight128x8_v68_20260914`；归档目录：`analysis/s52_transfer/training_records/v68_future_contact_phase_residual_preflight128x8_20260914`（均在 6 服务器项目根目录下）。

训练 mean reward 从 6.6024 到 13.4502，在 step 92153 降至 -7.6290 后恢复；episode length 从 22 到 119.8。末点奖励提升不等于物理安全提升。已对最终 `model_92157.pt` 运行固定 seed=131、1351 步、对应 S52 Play 任务的无界面 Isaac/PhysX rollout；动作参考推进到最后一帧，零失败重置，但接触滑移 p95 为 0.2026 m/s、在线最小距离约 **0.0649 mm**、峰值足部接触力约 **2359.7 N**。终点绝对路线误差尚需用项目既有路线分析器核准，不能把“零重置”写成完整任务验收。现有参考适配分析器的 root RMSE 因首帧世界/局部坐标不一致而被污染，不作为验收依据。

因此 v68 最终权重**未采用**，也未进入 MuJoCo 楼梯部署。v67 的候选联评仍是最近一次完整终点误差表；S52 v22 权重仍为受保护 warm-start，而非已通过多 seed 的楼梯模型。v68 最终回放的 NPZ/JSON 和曲线已保存在服务器归档；按项目约定，不发布失败模型视频。

## 发布与保留规则

公开仓库保留任务与机器人必要源文件、配置 YAML、一键训练及评测/部署脚本、受保护且经用途核实的模型，以及各轮实际奖励配置、训练曲线和失败原因。失败权重清理须先核对配置、曲线、报告、SHA-256 和保护名单；不删除 S53 teacher、S52 受保护基线、尚未联评的 v68 候选或活动进程 checkpoint。只有 Lab 固定几何多 seed 联合达标后，才推进 MuJoCo 楼梯验证与分阶段域随机化。

历史 README 中的“在线足部采样净空”“离线脚尖净空”和“同源刚体 cost”口径不同，不应直接比较阈值。
