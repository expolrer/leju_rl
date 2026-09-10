# S52 v31 TailNormInstantImpact 失败分析

## 结论

v31 正常完成 `32x2` 冒烟和 `128x60` 短预检，但没有更新后 checkpoint 通过固定 seed=42 的完整 Lab 台阶终点门，多 seed 最终模型也失败。本轮不进入正式训练、MuJoCo 或域随机化，安全基线仍为受保护的 S52 v22 `model_92150.pt`。

## 设计与曲线

v31 保留 v30 的终点 top-20% 有效样本归一化，上限仍为 `16`；撤销片段 running-peak 回放，恢复 v29 的逐步瞬时冲击 CMDP cost、`1400 N` 下降阈值和 `1800 N` 全程应急阈值。

- mean reward：`-8.2999 -> 89.5474`，末 7 点均值 `81.2484`。
- mean episode length：`16.05 -> 712.65`，末 7 点均值 `661.37`。
- policy KL：最终 `0.000473`；teacher action RMSE：最终 `0.000186`。
- constraint raw step cost：`0.03387 -> 0.00939`，末 7 点均值 `0.00648`。
- dual multiplier：`0.7050 -> 0.3471`，末 7 点均值 `0.2592`。
- 终点 tail route error 末 7 点均值 `0.4166 m`；anchor position termination 最终 `0.4167`。
- 训练传感器单帧足力最高约 `9849.9 N @ 92165`，说明瞬时接触读数仍有明显尖峰，但该尖峰不再被跨片段回填。

v31 相比 v30 恢复了 episode 长度并降低了 anchor 终止率，证明撤销 running-peak 是正确方向；但终点尾部归一化上限 16 仍使少量终点样本产生过强梯度，更新后策略继续出现终点路线漂移。

## 固定 seed=42 联评

| checkpoint | 完整通过 | 终点误差中位数 (m) | 滑移 p95 (m/s) | 峰值足力 (N) |
|---|---:|---:|---:|---:|
| model_92150 | 否 | 0.3002 | 0.2584 | 1298.1 |
| model_92160 | 否 | 0.2871 | 0.2087 | 1753.9 |
| model_92170 | 否 | 0.2755 | 0.2294 | 1821.7 |
| model_92180 | 否 | 0.2791 | 0.2567 | 1917.3 |
| model_92190 | 否 | 0.2529 | 0.2257 | 1376.7 |
| model_92200 | 否 | 0.3422 | 0.2185 | 2070.2 |
| model_92209 | 否 | 0.3399 | 0.2499 | 2232.7 |

所有 rollout 都完成上楼、平台、四级正向下楼并零 reset，但没有 checkpoint 通过 `0.25 m` 终点误差门。最接近的 `model_92190` 仍超出 `2.9 mm`。

最终 `model_92209` 在 seed 7/42/131 的终点误差为 `0.1918/0.3399/0.3187 m`，只有 seed7 通过；峰值足力约为 `2018.8/2232.7/1917.9 N`。多 seed 路线和冲击均不合格。

## 下一版

v32 继续从受保护 S52 v22 快照重新开始，保持瞬时冲击定义、teacher trust、固定几何和全部基础奖励不变，只将终点尾部有效样本归一化上限由 `16` 降到 `4`。这是一次单变量消融，用于确认 v31 的路线漂移是否由过强 tail 梯度造成。若仍无法通过，则停止乘法归一化，改为按终点 active 子批次单独优化或使用终点状态回放缓冲区。

参考：

- SCPO: https://arxiv.org/abs/2306.12594
- ASCPO: https://arxiv.org/abs/2410.01212
- PPO: https://arxiv.org/abs/1707.06347
- legged_gym: https://github.com/leggedrobotics/legged_gym

双视图 MP4 仅保存在服务器：`analysis/s52_transfer/training_records/v31_tail_norm_instant_impact_preflight128x60_20260911/model_92209_seed42_model29999_style.mp4`。
