# S52 v30 SegmentPeakTailNorm 失败分析

## 结论

v30 的 `32x2` 冒烟和 `128x60` 短预检均正常结束，没有 Traceback、NaN 或 OOM，但不满足进入正式训练、MuJoCo 或域随机化的门槛。安全基线继续使用 S52 v22 `model_92150.pt`，不采用 v30 的任何更新后权重。

## 本轮设计

- 从受保护的 S52 v22 `model_92150.pt` warm-start，重置优化器。
- 终点 frame 1260--1340 内选择误差最大的 20% 活动环境，并按 `min(16, num_envs / tail_count)` 归一化 pseudo-Huber 尾部惩罚，避免稀疏终点样本被 128 环境 batch 均值稀释。
- 将四个下降片段内的最大足端接触力保留到片段结束，形成 running-peak CMDP cost。
- 下降冲击阈值由 1400 N 收紧到 1300 N，全程应急阈值由 1800 N 收紧到 1600 N。

## 曲线结果

- mean reward：`-8.2999 -> 81.2783`，最终也是峰值；末 7 点均值 `79.4776`。
- mean episode length：`16.05 -> 630.74`，末 7 点均值 `610.64`。
- policy KL：`0.001383 -> 0.000507`；teacher action RMSE：`0.000154 -> 0.000198`。
- constraint raw step cost：`0.26498 -> 0.15142`，末 7 点均值 `0.18546`。
- dual multiplier：`0.9940 -> 1.0382`，峰值 `1.1603 @ 92190`。
- segment peak cost 达到上限 `4.0 @ 92162`；训练观测瞬时足力峰值约 `5672.9 N @ 92191`。
- 终点 tail 归一化恒为 `16`，但末 7 点 tail active 为 `0`，终点路线误差末 7 点均值仍为 `0.4695 m`。
- 最终一轮 `anchor_pos` termination 为 `1.0`。总回报和 episode length 上升并不代表终点路线安全。

## 真实物理联评

固定 seed=42 的七个 checkpoint 都完成了 1351 步 Isaac/PhysX rollout 且无运行错误，但只有未更新起点 `model_92150` 通过完整 Lab 台阶门：

| checkpoint | 完整通过 | 终点误差中位数 (m) | 滑移 p95 (m/s) | 峰值足力 (N) |
|---|---:|---:|---:|---:|
| model_92150 | 是 | 0.2144 | 0.2159 | 1664.8 |
| model_92160 | 否 | 0.2551 | 0.2286 | 3025.6 |
| model_92170 | 否 | 0.3199 | 0.2642 | 2119.2 |
| model_92180 | 否 | 0.2746 | 0.2242 | 1486.4 |
| model_92190 | 否 | 0.2843 | 0.2184 | 2014.3 |
| model_92200 | 否 | 0.3145 | 0.2178 | 2510.3 |
| model_92209 | 否 | 0.3085 | 0.2421 | 2415.4 |

最终 `model_92209` 的 seed 7/42/131 终点误差分别为 `0.2414/0.3085/0.3318 m`，只有 seed 7 通过；峰值足力分别约 `1702.5/2415.4/1392.9 N`。多 seed 泛化失败。

## 失败原因

1. running-peak cost 在整个下降片段持续回放单次冲击，放大了代价的时间占比。策略更容易通过偏离终段路线来减少受约束状态，而不是学习更柔和的接触。
2. 冲击阈值同步收紧，使第一批 PPO 更新就受到较强 cost 梯度；终点 tail 虽被归一化，但只在少量终点活动环境出现，无法和持续的片段 cost 抗衡。
3. 训练中的 GPU 三角网格接触力峰值存在噪声风险。`legged_gym` 官方文档明确提醒 GPU triangle mesh 的 net contact force 可能不可靠，因此不应把单次噪声峰值长时间回填给 actor。
4. SCPO/ASCPO 的核心是约束危险状态或高概率最坏状态，并不等价于把一个观测峰值复制到后续所有时步。v30 的工程映射过强，破坏了已验证路线能力。

## 下一版方案

v31 从同一 S52 v22 安全快照重新开始：保留 v30 的终点尾部有效样本归一化，但撤销下降片段 running-peak 回放，恢复逐步、局部、封顶的瞬时冲击 cost 和 v29 的阈值；冲击只在真实接触状态产生梯度。继续使用固定几何、严格 teacher trust region 和多 seed 真实物理验收。若 v31 仍出现路线漂移，则下一步不再提高接触力权重，而改用经过接触滤波的 touchdown 短窗口 GRF/垂直速度联合项。

## 一手资料

- SCPO: https://arxiv.org/abs/2306.12594
- ASCPO: https://arxiv.org/abs/2410.01212
- Contact-conditioned locomotion: https://arxiv.org/abs/2408.00776
- legged_gym 官方实现与接触力说明: https://github.com/leggedrobotics/legged_gym

双视图 MP4 仅保存在服务器：`analysis/s52_transfer/training_records/v30_segment_peak_tail_norm_preflight128x60_20260911/model_92209_seed42_model29999_style.mp4`。
