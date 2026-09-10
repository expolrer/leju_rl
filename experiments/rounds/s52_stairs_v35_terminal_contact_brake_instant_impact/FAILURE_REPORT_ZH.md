# S52 v35 TerminalContactBrakeInstantImpact 失败分析

## 结论

v35 从受保护的 S52 v22 `model_92150.pt` 开始，只在末级落地后的终段引入双脚接触条件制动，保持 v32 的固定场景、teacher trust 和瞬时冲击 CMDP。`32x2` 冒烟及 `128x60` 短预检均正常完成，无 Traceback、NaN、OOM 或维度错误。七个 seed=42 候选中只有最终 `model_92209.pt` 通过完整路线判据；最终模型在 seed=7 和 seed=131 均失败，且冲击显著增大。因此 v35 整轮否决，不进入正式训练、MuJoCo 或域随机化。

## 奖励设计

在 frame 1160--1340 计算终点世界系路线误差及沿误差方向的向外速度。位置超过 `0.18 m` 后使用 scale `0.08 m` 的 pseudo-Huber 损失，向外速度使用 scale `0.12 m/s`，二者合成后乘以双脚支撑强度，奖励权重为 `-0.75`。双脚支撑强度由左右脚接触力相对 `80 N` 的归一化最小值给出。其余奖励和约束保持不变。

## 训练曲线

- mean reward：最终 `100.1366`，峰值 `100.7667 @ 92207`，末 7 点均值 `96.6768`。
- mean episode length：最终 `700.07`，末 7 点均值 `663.77`。
- policy KL：最终 `0.000469`；teacher action RMSE 最终 `0.000188`。
- constraint raw step cost：最终 `0.01132`，末 7 点均值 `0.00823`；dual multiplier 最终 `0.2473`。
- contact route error：最终 `0.4597 m`，最低 `0.0412 m @ 92204`，末 7 点均值 `0.2382 m`。
- outward velocity：最终 `0.7246 m/s`，末 7 点均值 `0.4787 m/s`。
- double-support 指标：最终 `0.2413`，末 7 点均值 `0.5547`。
- 新制动项激活率：最大 `1.0`，最终 `0`，末 7 点均值仅 `0.00506`；新增奖励末 7 点均值 `-0.02861`。
- terminal tail route error：最终 `0.5252 m`，最低 `0.1597 m @ 92172`，末 7 点均值 `0.4609 m`。
- 训练观测足力峰值约 `4304 N`；anchor termination 最终 `0.667`，末 7 点均值 `0.869`。

总回报和 episode length 的上升不能视为安全改善。乘法接触门让策略可以通过减少目标窗口内的双脚支撑来降低惩罚，新项在末段只有约 `0.5%` 激活，安全信用过于稀疏。

## 真实物理联评

固定 seed=42：

| checkpoint | 完整通过 | 终点误差 (m) | 滑移 p95 (m/s) | 峰值足力 (N) | 终段双脚接触 |
|---|---:|---:|---:|---:|---:|
| model_92150 | 否 | 0.2711 | 0.2825 | 1580.4 | 1.000 |
| model_92160 | 否 | 0.2567 | 0.2474 | 1777.3 | 1.000 |
| model_92170 | 否 | 0.3144 | 0.2160 | 1293.0 | 1.000 |
| model_92180 | 否 | 0.3423 | 0.2382 | 2416.3 | 1.000 |
| model_92190 | 否 | 0.3006 | 0.2011 | 1613.3 | 1.000 |
| model_92200 | 否 | 0.2636 | 0.2320 | 1701.0 | 1.000 |
| model_92209 | 是 | 0.2382 | 0.2186 | 2493.8 | 1.000 |

最终 `model_92209.pt` 多种子结果：

| seed | 完整通过 | 终点误差 (m) | 滑移 p95 (m/s) | 峰值足力 (N) | 终段双脚接触 |
|---:|---:|---:|---:|---:|---:|
| 7 | 否 | 0.2660 | 0.2592 | 1727.9 | 1.000 |
| 42 | 是 | 0.2382 | 0.2186 | 2493.8 | 1.000 |
| 131 | 否 | 0.4218 | 0.2953 | 2265.1 | 0.440 |

所有 rollout 均完成上楼、平台、四级正向下楼且首周期零 reset，但只有一个种子满足终点门。seed=131 的接触比例下降与训练激活率塌缩相互印证；seed=42 即使通过终点门，峰值足力也远高于受保护基线，不能采用。

## 下一方向

v36 不再用可被策略关闭的纯乘法接触门。终端路线制动损失使用 `0.25 + 0.75 * double_support` 的软混合，保证任何接触状态都有非零信用；在后段另加缺失双脚支撑的独立、封顶、局部惩罚。两项只作用于下楼完成后的终端保持段，不侵入摆腿净空窗口，也不削弱原瞬时冲击 CMDP。预检仍从受保护 `model_92150.pt` 开始。

该方案是基于 contact-conditioned locomotion 与 learned stoppability 方法的工程推断：接触状态用于调节制动强度，但不能成为关闭安全损失的开关。

- Contact-conditioned learning of locomotion policies: https://arxiv.org/abs/2408.00776
- Humanoid Safe Stop via Learned Stoppability Value: https://arxiv.org/abs/2609.02358
- Learning Safe-Stoppability Monitors for Humanoid Robots: https://arxiv.org/abs/2603.22703

双视图 MP4 仅保存在服务器：`analysis/s52_transfer/training_records/v35_terminal_contact_brake_instant_impact_preflight128x60_20260911/model_92209_seed42_model29999_style.mp4`。
