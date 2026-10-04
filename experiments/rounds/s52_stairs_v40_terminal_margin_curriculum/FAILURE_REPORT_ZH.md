# S52 v40 TerminalMarginCurriculum 失败分析

## 结论

v40 从受保护的 S52 v22 `model_92150.pt` warm-start，保持 v39 的课程、学习率、摆腿净空、路线与冲击 cost 不变，只将终点容差收紧约 `20 mm`：`allowed_ahead 0.02 -> 0.00 m`、`allowed_route_error 0.18 -> 0.16 m`、`error_threshold 0.24 -> 0.22 m`。`32x2` 冒烟在补齐新 experiment 的受保护 checkpoint 符号链接后通过，`128x15` 短预检正常结束，无 NaN、OOM 或训练异常。

seed=131 下六个 checkpoint 均完成上楼、平台、四级正向下楼和落地，零 reset，但终点误差为 `0.3015--0.3597 m`，比 v39 最好候选的 `0.2571 m` 明显退化。说明在当前三个重叠终点项中同时压缩阈值，产生了与 teacher 保形、双支撑和路线制动相冲突的梯度，并没有实现“多走 20 mm”的直观效果。v40 否决，不进入正式训练、MuJoCo 台阶验收或域随机化。

## 曲线

- mean reward：`-6.3333 -> 7.6349`，最低 `-15.7367`，最终即峰值。
- mean episode length：`16.0 -> 160.33`，训练分布仍处于恢复期。
- teacher KL 最终 `0.001353`；teacher action RMSE 最终 `2.002e-4`。
- value loss 最终 `7.3222`；surrogate loss 最终 `8.524e-4`。
- raw step cost 最终 `0.01073`；dual multiplier 最终 `0.1718`。
- 训练观测瞬时足力峰值为 `1721.8 N`，终端支撑地面路线误差最终 `0.1719 m`。
- 训练平均终端误差看似较小，但固定物理 rollout 全部大于 `0.30 m`，证明 batch 均值不能代替完整路线尾部验收。

## seed=131 候选

| checkpoint | 完整路线 / reset | 终点误差 (m) | 滑移 p95 (m/s) | 峰值足力 (N) |
| --- | --- | ---: | ---: | ---: |
| `92151` | 是 / 0 | 0.3015 | 0.2524 | 2051.9 |
| `92154` | 是 / 0 | 0.3181 | 0.2359 | 1523.5 |
| `92157` | 是 / 0 | 0.3164 | 0.2230 | 1442.3 |
| `92160` | 是 / 0 | 0.3537 | 0.2663 | 1299.0 |
| `92163` | 是 / 0 | 0.3597 | 0.2351 | 1096.0 |
| `92164` | 是 / 0 | 0.3082 | 0.2300 | 2151.1 |

## 下一步

v40 已否定“从安全 teacher 出发，同时收紧三个终点阈值”这条路线。v41 将保留 v40 的局部终点余量，但以 v39 近门槛的 `model_92163.pt` 作为学生初始化，仍以受保护 `model_92150.pt` 作 teacher，学习率降到 `2.5e-8`，只做 9 次更新并每 3 次保存。目标是检验“近可行策略局部抛光”是否能消除 `7--11 mm` 尾部误差，不将 v39 候选标记为安全基线。

服务器最新真实双视图 MP4：

```text
/home/zzx23457/hhw/LejuLab-Train/analysis/s52_transfer/training_records/v40_terminal_margin_curriculum_preflight128x15_20260912/model_92164_seed131_model29999_style.mp4
```
