# S52 v53 WorstStateComposite 短预检失败报告

## 结论

v53 的 `32x2` 冒烟和 `128x6` 短预检均正常结束，无 Traceback、NaN、OOM、Vulkan 或驱动错误。全部固定 `seed=131` 候选都完成上楼、平台、四级正向下楼和落地，且首次周期零重置；但没有候选同时通过路线、同源净空、滑移和冲击门，因此 v53 否决，安全基线仍为受保护的 `model_92150.pt`。

## 实际设计

- warm-start/teacher：受保护 S52 `model_92150.pt`，未从零训练。
- 保留 v52 的 `instant impact + 0.25 * shared clearance` 瞬时复合风险。
- 在 motion frame `800--1340` 内按环境维护 running maximum，并把该最坏值保持到下降片段结束。
- PPO 约束读取 held worst-state cost；数值惰性 reward 权重为 `-1e-8`，不新增直接 actor 奖励。
- 学习率 `1e-9`，训练 6 iteration，每次保存 checkpoint。

## 训练曲线

- mean reward：`6.6024 -> 9.5964`，最低 `-7.7937 @ 92153`，最终点也是峰值。
- mean episode length：`22.0 -> 91.10`。
- policy KL：`4.285e-4 -> 5.276e-4`；teacher KL：`4.298e-5 -> 7.426e-5`。
- value loss：`12.384 -> 3.530`；surrogate loss：`0.001104 -> 0.001273`。
- cost-value loss：`77.325 -> 73.540`；constraint error：`0.26937 -> 0.26796`。
- worst-state held cost 在 `92153` 达到 `3.6099`，active fraction 达到 `0.9583`，但最终更新又回到 0。说明信号已变稠密，却被单个未归一化极端事件主导，尺度与采样状态之间仍强烈振荡。

完整曲线和原始标量位于服务器：

`analysis/s52_transfer/training_records/v53_worst_state_composite_preflight128x6_20260913/curves/`

## 固定 seed=131 真实 PhysX 联评

| checkpoint | 终点路线误差 m | 下楼完成误差 m | 滑移 p95 m/s | 峰值足力 N | 同源最小净空 mm | 同源平均 cost | reset |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline 92150 | 0.342815 | 0.294998 | 0.220741 | 1291.1 | 0.0649 | 0.009489 | 0 |
| 92151 | 0.365078 | 0.327362 | 0.229658 | 1692.0 | 0.0610 | 0.019745 | 0 |
| 92152 | 0.277122 | 0.238567 | 0.212363 | 1618.8 | 0.0115 | 0.009517 | 0 |
| 92153 | 0.372191 | 0.309482 | 0.218056 | 3095.6 | 0.0248 | 0.002409 | 0 |
| 92154 | 0.304310 | 0.217559 | 0.248202 | 1552.6 | 0.0286 | 0.016922 | 0 |
| 92155 | 0.358187 | 0.260034 | 0.258632 | 2018.6 | 1.1559 | 0.000790 | 0 |

`model_92152` 改善路线和滑移，却把同源最小净空压到约 `0.0115 mm`；`model_92155` 提高净空并降低平均 clearance cost，却以更差路线、滑移和约 `2019 N` 冲击为代价。最终 checkpoint 不是最优模型，也不能替换安全基线。

## 失败原因

1. 突缘净空 cost 与冲击 cost 未先归一化，最大值操作会优先锁存数值尺度更大的事件。
2. running maximum 将一次异常冲击复制到整个剩余下降片段，信用虽稠密，但梯度过度保守且缺少时间定位。
3. PID/Lagrangian 的输入尺度剧烈变化，造成 cost-value loss 高且约束误差下降很小。
4. 在线八点 shared clearance 与历史离线 `2.4109 mm` 脚尖验收定义不同；本表只作同一 v53 Play 任务内部比较，不能混用门槛。

## 文献依据与下一版

- PID Lagrangian 明确讨论了乘子更新的振荡、超调及 reward/cost 相对尺度不变性；v54 应先归一化/裁剪各风险分量，再调 PID，而不是继续增大 held-max 权重：https://arxiv.org/abs/2007.03964
- SCPO 用 Maximum MDP 处理 state-wise 最坏违反；本项目 v53 是其工程启发式映射，并非论文算法的完整复现：https://arxiv.org/abs/2306.12594
- Contact-conditioned locomotion 用下一接触位置和剩余接触时间消除非周期动作歧义；若归一化尾部风险仍失败，下一步应给 S52 增加 future-contact 条件，而不是叠加更多终点奖励：https://arxiv.org/abs/2408.00776
- Mind Your Steps 使用三维落脚点序列和动态目标采样，并指出短踏面容易发生突缘碰撞，适合作为后续显式 foothold 接口依据：https://arxiv.org/abs/2606.08253

v54 采用局部归一化、上界裁剪和有限记忆的尾部风险聚合：分别把 clearance 与 impact 映射到可比的 `[0, 1]` 风险，再在下降窗口使用衰减最大值或 top-tail 聚合，避免一次异常支配整个 episode。仍保持 148 维 actor 接口、固定几何和受保护 teacher。

## 回放

最新模型真实双视图 MP4 仅保存在服务器：

`analysis/s52_transfer/training_records/v53_worst_state_composite_preflight128x6_20260913/model_92155_seed131_model29999_style.mp4`

媒体校验：`1820x910`、`50 FPS`、`1352` 帧、`27.04 s`。
