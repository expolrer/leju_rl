# S52 v28 BroadLandingBalancedCMDP 失败分析

## 结论

v28 的 `32x2` 冒烟和 `128x60` 短预检均正常结束，没有 Traceback、NaN、OOM 或 shape 错误。训练与 Play 使用同一任务物理定义，最终模型也完成了 1351 步真实 Isaac/PhysX rollout 和 model_29999 双视图视频。

本轮不能采用，也不能进入正式训练、MuJoCo 或域随机化。原因不是机器人不会上下楼，而是没有任何候选同时满足多 seed 终点误差与冲击门：

- `model_92170`：seed 7/42/131 终点误差分别为 `0.253714/0.241630/0.307843 m`，峰值足力约 `1601.7/1761.1/1699.0 N`。
- `model_92209`：seed 7/42/131 终点误差分别为 `0.305370/0.225725/0.309343 m`，峰值足力约 `2461.8/3062.4/1852.2 N`。
- 两组均完整完成上楼、平台和四级正向下楼，且第一周期零 reset；但终点门要求三个 seed 均不高于 `0.25 m`，最终模型的冲击也明显退化。

## 曲线

- mean reward：`-8.2999 -> 88.7980`，末 7 点均值约 `81.8119`。
- episode length：`16.05 -> 620.99`，末 7 点均值约 `577.11`。
- policy KL：最终 `5.03e-4`；teacher action RMSE 最终 `1.94e-4`。
- 原始 constraint step cost：最终 `0.06663`；segment replay 后 step cost 最终 `0.09889`。
- dual multiplier：最终 `0.4361`；约束链路已实际参与更新。
- `s52_balanced_constraint_active` 末 7 点均值约 `0.4375`，不再是 v27 的狭窄漏检。
- 训练中的单帧 `s52_segment_peak_contact_force` 最高达到 `8571 N`，说明高冲击尾部仍存在。

## 为什么失败

v28 扩宽下降冲击窗口后，`model_92170` 的多 seed 峰值足力已经明显低于最终模型，证明广域冲击 cost 有效；但终点误差在 seed 131 仍为 `0.307843 m`。继续训练到 `model_92209` 后，seed42 终点变好，另外两个 seed 仍失败，同时冲击升至 `3062 N`。

根因是“下降冲击”和“终点路线误差”虽然做了量纲归一化和分段回填，仍共用一个标量 cost、一个 critic 和一个 PID 乘子。两种风险发生在不同阶段、频率不同，`max()` 只把当步较大者送入更新；它无法保证每个约束分别满足，也不能针对少数高误差环境形成 CVaR 式尾部梯度。训练均值持续改善，因此不能代表三 seed 最坏表现。

## 下一版方向

v29 从受保护 S52 v22 `model_92150.pt` 重新开始，不继承 v28 候选：

1. constrained PPO 的唯一 cost 只保留完整下降段和全程应急足力冲击，避免终点项争夺同一乘子。
2. 终点路线误差改成独立的环境尾部惩罚：仅对终点活动环境中误差最高的 20% 施加额外 pseudo-Huber/CVaR 近似梯度，普通终点制动与双支撑奖励继续保留。
3. 保持 148 维观测、27 维动作、S52 MJCF/PD/力矩限制、固定几何和严格 S52 快照 teacher；不增加域随机化。
4. 仍先执行 `32x2` 冒烟和 `128x60` 短预检，再以 seed 7/42/131 做完整 1351 步真实物理验收。

Mind Your Steps 用显式三维落脚目标减少速度策略的落脚不确定性；Walk the PLANC 用结构化动态可行落脚规划目标引导 RL；QuietWalk 直接把逐脚 GRF 估计接入训练以抑制冲击。基于这些一手方法，本轮采用“阶段解耦的冲击约束 + 终点尾部路线目标”，而不是继续放大混合标量 cost。这里的 CVaR/top-k 实现是针对当前代码的工程推断，不是上述论文的原样复现。

- Mind Your Steps: https://arxiv.org/abs/2606.08253
- Walk the PLANC: https://arxiv.org/abs/2601.06286
- QuietWalk: https://arxiv.org/abs/2604.23702
- SCPO: https://arxiv.org/abs/2306.12594
- ASCPO: https://arxiv.org/abs/2410.01212

## 产物

- 服务器归档：`analysis/s52_transfer/training_records/v28_broad_landing_balanced_cmdp_preflight128x60_20260910`
- 最终模型双视图 MP4 仅保存在服务器归档目录。
- 安全快照仍为 v22 `model_92150.pt`，SHA-256 `93c457de92f9102125d7c88f8721e25a35e83caa03e4112d7a01074c78e50f56`。
