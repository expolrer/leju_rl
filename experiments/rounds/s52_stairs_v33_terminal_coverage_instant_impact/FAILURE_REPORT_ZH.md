# S52 v33 TerminalCoverageInstantImpact 失败分析

## 结论

v33 从受保护的 S52 v22 `model_92150.pt` 重新开始，扩大终点状态采样覆盖，而不继续增大稀疏尾部损失倍率。`32x2` 冒烟与 `128x60` 短预检均正常结束，无 Traceback、NaN、OOM 或 shape 错误；但所有学习后的 checkpoint 都未通过固定 `seed=42` 的终点保持门，最终模型在三个 seed 上也全部失败。因此 v33 不替换安全快照、不进入正式训练、MuJoCo 或域随机化。

## 修改

- `zero_start_fraction: 0.35 -> 0.25`。
- `terrain_focus_fraction: 0.60 -> 0.70`。
- 在原有终点焦点帧基础上加入 `1240/1280/1320`。
- `tail_fraction: 0.20 -> 0.50`，归一化固定为 `1.0`。
- 保留逐时步瞬时冲击 CMDP、固定几何、S52 snapshot teacher 和严格策略信赖域。

## 曲线

- mean reward：`-10.0119 -> 74.0892`，峰值 `77.3154 @ model_92204`，末 7 点均值 `74.7565`。
- mean episode length：最终 `613.77`，末 7 点均值 `590.94`。
- policy KL：最终 `0.000436`。
- constraint raw step cost：最终 `0.00818`，末 7 点均值 `0.00892`。
- terminal tail active：峰值 `0.9167 @ model_92192`，最终及末 7 点均为 `0`。
- terminal tail route error：最低 `0.1223 m @ model_92169`，最终 `0.3679 m`，末 7 点均值 `0.4279 m`。
- anchor position termination：最终 `0.7083`，末 7 点均值 `1.0893`。

终点状态覆盖比 v32 稠密，但曲线最低点没有转化成闭环稳定停止。`model_92169` 附近训练统计很好，固定路线 rollout 的 `model_92170` 仍前冲到 `0.2917 m`。这说明问题不是单纯缺终点样本，而是最后一级落地后的动量没有在越界前获得足够直接的信用。

## 固定 seed=42 候选

| checkpoint | 完整通过 | 终点误差 (m) | 滑移 p95 (m/s) | 峰值足力 (N) |
|---|---:|---:|---:|---:|
| model_92150 | 是 | 0.2414 | 0.2144 | 1614.4 |
| model_92160 | 否 | 0.3428 | 0.2190 | 2062.3 |
| model_92170 | 否 | 0.2917 | 0.2478 | 2209.8 |
| model_92180 | 否 | 0.2981 | 0.2353 | 1744.8 |
| model_92190 | 否 | 0.2734 | 0.2334 | 1448.1 |
| model_92200 | 否 | 0.2947 | 0.2382 | 1322.7 |
| model_92209 | 否 | 0.2774 | 0.2452 | 2576.1 |

最终 `model_92209` 在 seed 7/42/131 的终点误差为 `0.3240/0.2774/0.3083 m`，滑移 p95 为 `0.2628/0.2452/0.2273 m/s`，峰值足力为 `1548.2/2576.1/1903.7 N`。三次均完成上楼、平台、四级正向下楼且零 reset，但都没有稳定停入终点路线门。

## 相关方法与下一版

Safe-Stop 把人形机器人停止形式化为 reach-avoid，并显式估计从当前状态能否安全停入最低风险状态；PRISM 用重要性采样聚焦少见但安全关键的可停止边界。Contact-conditioned locomotion 则表明未来接触切换比平均速度更适合表达非周期和切换动作。论文并没有直接给出本任务的奖励公式。

基于这些原方法做工程推断：v34 回退到 v32 的保守起始分布和 tail 设置，并只在最后一级下降末段加入预测停止屏障。屏障使用当前世界系前向路线误差与相对前向速度，预测短时间后的越界量，在真实位置越过 `0.24 m` 之前就提供制动梯度；窗口必须局限在最后一级落地到终点，不侵入前三个下楼摆腿窗口。

- Humanoid Safe Stop via Learned Stoppability Value：https://arxiv.org/abs/2609.02358
- Learning Safe-Stoppability Monitors for Humanoid Robots：https://arxiv.org/abs/2603.22703
- Contact-conditioned learning of locomotion policies：https://arxiv.org/abs/2408.00776

双视图 MP4 仅保存在服务器：`analysis/s52_transfer/training_records/v33_terminal_coverage_instant_impact_preflight128x60_20260911/model_92209_seed42_model29999_style.mp4`。
