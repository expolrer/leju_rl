# S52 v29 DecoupledImpactTail 失败分析

## 结论

v29 的 `32x2` 冒烟和 `128x60` 短预检正常结束，没有 Traceback、NaN、OOM 或 shape 错误。该版本把 constrained PPO 的唯一 cost 限定为下降/应急冲击，并把终点路线误差改为 batch 中终点活动环境 top 20% 的独立尾部奖励。

结果仍不满足 Lab 多 seed 验收，不能进入正式训练、MuJoCo 或域随机化：

- seed42 只有 `model_92160` 和 `model_92200` 通过 `0.25 m` 终点门，但峰值足力分别约 `2155.6 N` 和 `2272.5 N`。
- 冲击较低的 `model_92190` 峰值约 `1586.7 N`，终点误差却为 `0.353528 m`。
- 最终 `model_92209` 在 seed 7/42/131 的终点误差为 `0.236710/0.291240/0.300723 m`，峰值足力约 `1557.2/2459.9/2237.2 N`；只有 seed7 通过终点门。
- 所有已测试候选均完成上楼、平台和四级正向下楼，且第一周期零 reset，但没有候选同时通过终点和冲击门。

## 曲线

- mean reward：`-8.2999 -> 82.9913`，末 7 点均值约 `78.9143`。
- episode length：`16.05 -> 631.69`，末 7 点均值约 `588.16`。
- policy KL 最终 `5.17e-4`；teacher action RMSE 最终 `1.94e-4`。
- impact 原始 step cost 最终 `0.01228`，末 7 点均值约 `0.00902`。
- dual multiplier 最终 `0.2694`；impact active 末 7 点均值约 `0.7381`。
- terminal tail 奖励最终 `-0.01404`，末 7 点均值仅 `-0.00540`；tail active 末 7 点均值仅 `4.76%`。
- 训练中的单帧足力最大值仍达到约 `9447 N`。

## 为什么失败

解耦后冲击 cost 不再被终点 cost 直接覆盖，但终点 top-20% 奖励在混合起始帧 batch 中被严重稀释：只有处于 frame 1260--1340 的环境先进入 active 集，再从其中选 top 20%，最后 RewardManager 仍对全部 128 个环境求均值。因而 tail active 的末段平均只有 `4.76%`，实际尾部梯度过弱。

同时 impact CMDP 虽有较高 active 比例，但风险仍以少量单帧尖峰出现；平均 return 和单个 PID 乘子未把 seed42/131 的 2.2--2.5 kN 极值压下来。v29 证明“通道解耦”方向正确，但终点尾部奖励必须按有效样本数归一化，冲击必须按每个下降片段的 peak/top-k 而不是普通时步均值保留信用。

## 下一步

本次自动执行已经连续完成 v28、v29 两版短预检，达到单次唤醒上限，因此不立即启动 v30。下一次从受保护 S52 v22 `model_92150.pt` 重新开始：

1. 对 terminal tail 奖励按 `num_envs / tail_count` 做有界归一化，避免终点活动样本在 batch 均值中消失；保留 top 20% 选择，不放大普通成功环境。
2. 将 impact cost 改为每个下降片段内 running peak/top-k 的稠密回填，片段结束后清零；全程应急冲击仍保留。
3. 降低冲击阈值并采用分段软屏障，但保持受保护策略的严格 teacher trust，防止为了降冲击破坏完整下楼。
4. 固定几何和现有 148/27 接口不变；仍执行 `32x2 -> 128x60 -> seed 7/42/131`。

Mind Your Steps 和 Walk the PLANC 都通过明确的未来落脚目标减少终点和落脚不确定性；QuietWalk 直接针对逐脚 GRF；SCPO/ASCPO 强调最坏/高概率状态而不是平均安全。v30 的归一化 top-k 与 per-segment peak 是这些思想在现有框架中的工程实现推断。

- Mind Your Steps: https://arxiv.org/abs/2606.08253
- Walk the PLANC: https://arxiv.org/abs/2601.06286
- QuietWalk: https://arxiv.org/abs/2604.23702
- SCPO: https://arxiv.org/abs/2306.12594
- ASCPO: https://arxiv.org/abs/2410.01212

## 产物

- 服务器归档：`analysis/s52_transfer/training_records/v29_decoupled_impact_tail_preflight128x60_20260910`
- 最终真实双视图 MP4 只保存在服务器。
- 安全快照仍为 v22 `model_92150.pt`，SHA-256 `93c457de92f9102125d7c88f8721e25a35e83caa03e4112d7a01074c78e50f56`。
