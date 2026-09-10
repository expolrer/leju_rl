# S52 v34 TerminalPredictiveTubeInstantImpact 失败分析

## 结论

v34 回到 v32 的保守采样分布，从受保护 S52 v22 `model_92150.pt` 开始，只新增最后一级到终点区间的预测路线管道。`32x2` 与 `128x60` 正常完成，无 Traceback、NaN、OOM 或 shape 错误；新增信号可观测且参与优化，但七个固定 seed42 候选均未通过终点门，最终模型三个 seed 也全部失败。因此 v34 否决，不进入正式训练、MuJoCo 或域随机化。

## 奖励设计

在 frame 1120--1340 计算：

`predicted_error_xy = current_route_error_xy + 0.25 s * relative_planar_velocity`

预测距离超过 `0.18 m` 后使用 scale `0.08 m` 的 pseudo-Huber 屏障，权重 `-0.75`，并在 frame 1120--1190 线性启用。其余奖励、瞬时冲击 CMDP、teacher trust、固定几何和 v32 采样分布保持不变。

## 曲线

- mean reward：`-8.2999 -> 79.1898`，最终即峰值，末 7 点均值 `77.1427`。
- mean episode length：`16.05 -> 578.34`，末 7 点均值 `555.18`。
- policy KL：最终 `0.000427`；teacher action RMSE 最终 `0.000192`。
- constraint raw step cost：最终 `0.00525`，末 7 点均值 `0.00748`；dual multiplier 最终 `0.2287`。
- predictive route error：最终 `0.3133 m`，最低 `0.0354 m @ model_92199`，末 7 点均值 `0.3430 m`。
- current terminal route error：最终 `0.1907 m`，最低 `0.0114 m @ model_92204`，末 7 点均值 `0.2171 m`。
- predictive-tube reward：最终 `-0.06370`，最低 `-0.2030 @ model_92159`，末 7 点均值 `-0.03399`。
- 训练观测瞬时足力峰值约 `6133 N`；anchor termination 最终 `0.5`。

训练统计显示预测量比当前误差更早暴露风险，但 `model_92200` 附近的统计低点没有转化为闭环安全，说明固定 0.25 秒外推跨越摆腿、触地和双支撑切换时模型不成立。

## 固定 seed=42 候选

| checkpoint | 完整通过 | 终点误差 (m) | 滑移 p95 (m/s) | 峰值足力 (N) |
|---|---:|---:|---:|---:|
| model_92150 | 否 | 0.2992 | 0.2584 | 1298.1 |
| model_92160 | 否 | 0.2503 | 0.2484 | 2387.8 |
| model_92170 | 否 | 0.2666 | 0.2627 | 1709.4 |
| model_92180 | 否 | 0.3277 | 0.1969 | 1870.5 |
| model_92190 | 否 | 0.3146 | 0.2754 | 1676.7 |
| model_92200 | 否 | 0.3417 | 0.2717 | 2077.8 |
| model_92209 | 否 | 0.3103 | 0.2297 | 1495.2 |

`model_92160` 仅比门槛高约 `0.3 mm`，但冲击达到约 `2388 N`，不能按单一终点指标采用。最终 `model_92209` 在 seed 7/42/131 的终点误差为 `0.2861/0.3103/0.3290 m`，滑移 p95 为 `0.2923/0.2297/0.2430 m/s`，峰值足力为 `1849.2/1495.2/1192.3 N`。

## 下一方向

Contact-conditioned locomotion 的原方法用未来接触切换表达非周期行为；Safe-Stop/PRISM 则把停止安全看作依赖当前状态的可停止性问题。工程上，v35 不再使用跨接触的固定时域预测，而在末级落地后的双脚支撑状态中计算平面路线管道，并只惩罚超出管道的位置和沿误差方向继续外漂的速度。这样把信用放在可实际制动的接触状态，且不侵入空中摆腿阶段。

- Contact-conditioned learning of locomotion policies：https://arxiv.org/abs/2408.00776
- Humanoid Safe Stop via Learned Stoppability Value：https://arxiv.org/abs/2609.02358
- Learning Safe-Stoppability Monitors for Humanoid Robots：https://arxiv.org/abs/2603.22703

双视图 MP4 仅保存在服务器：`analysis/s52_transfer/training_records/v34_terminal_predictive_tube_instant_impact_preflight128x60_20260911/model_92209_seed42_model29999_style.mp4`。
