# S52 v36 TerminalSupportFloorInstantImpact 失败分析

## 结论

v36 从受保护 S52 v22 `model_92150.pt` 开始，将 v35 可被关闭的双脚接触乘法门改成带 `0.25` 常开底座的软混合，并增加终端缺失双脚支撑惩罚。`32x2` 与 `128x60` 均正常完成，无 Traceback、NaN、OOM 或维度错误。新制动信号明显变稠密，但七个 seed=42 候选全部未通过 `0.25 m` 终点路线门；最终模型三个 seed 也全部失败且冲击偏高。因此 v36 否决，不进入正式训练、MuJoCo 或域随机化。

## 奖励设计

- 路线误差和沿误差方向的向外速度损失继续只作用于 frame 1160--1340。
- 接触混合系数改为 `0.25 + 0.75 * double_support`，无完整双支撑时仍保留 25% 梯度。
- frame 1260--1340 另加缺失双脚支撑惩罚，frame 1280 达到全权重，权重 `-0.30`。
- 原有终点路线制动、tail loss、teacher trust、瞬时冲击 CMDP、固定几何与其余动作奖励保持不变。

## 训练曲线

- mean reward：最终及峰值 `96.9260`，末 7 点均值 `91.2277`。
- mean episode length：最终及峰值 `688.78`，末 7 点均值 `639.03`。
- policy KL：最终 `0.000477`；teacher action RMSE 最终 `0.000199`。
- constraint raw step cost：最终 `0.00663`，末 7 点均值 `0.00654`；dual multiplier 最终 `0.2159`。
- support-floor route error：最终 `0.3023 m`，最低 `0.0224 m @ 92178`，末 7 点均值 `0.4382 m`。
- outward velocity：最终 `0.6241 m/s`，末 7 点均值 `0.6649 m/s`。
- support blend：最终 `0.8739`，最低值恰为常开底座 `0.25`，末 7 点均值 `0.6936`。
- support-floor brake active：最大 `0.9167`，末 7 点均值 `0.0778`，比 v35 的约 `0.0051` 明显变稠密。
- 新路线制动奖励末 7 点均值 `-0.04057`。
- 缺失双脚支撑指标最大 `0.3391`，但末 7 点均值为 `0`；其奖励末 7 点均值仅 `-0.0000715`。
- terminal tail route error：最终 `0.4383 m`，末 7 点均值 `0.5831 m`。
- 训练观测足力峰值约 `5318 N`；anchor termination 最终 `0.458`，末 7 点均值 `0.708`。

v36 修复了“安全损失可被接触状态关掉”的结构缺陷，但没有解决终端状态采样稀疏。常开制动项参与优化后，平均终端误差仍变大；缺失双支撑项在末段几乎没有有效样本。继续放大权重只会增加与 teacher 和原终点奖励的竞争。

## 真实物理联评

固定 seed=42：

| checkpoint | 完整通过 | 终点误差 (m) | 滑移 p95 (m/s) | 峰值足力 (N) | 终段双脚接触 |
|---|---:|---:|---:|---:|---:|
| model_92150 | 否 | 0.3030 | 0.2426 | 1754.0 | 1.000 |
| model_92160 | 否 | 0.3194 | 0.2153 | 2258.5 | 1.000 |
| model_92170 | 否 | 0.3498 | 0.2726 | 2004.6 | 1.000 |
| model_92180 | 否 | 0.3226 | 0.2400 | 1262.0 | 1.000 |
| model_92190 | 否 | 0.3139 | 0.2254 | 2317.7 | 1.000 |
| model_92200 | 否 | 0.2993 | 0.2147 | 1445.3 | 1.000 |
| model_92209 | 否 | 0.3481 | 0.2781 | 2340.6 | 1.000 |

最终 `model_92209.pt` 多种子结果：

| seed | 完整通过 | 终点误差 (m) | 滑移 p95 (m/s) | 峰值足力 (N) | 终段双脚接触 |
|---:|---:|---:|---:|---:|---:|
| 7 | 否 | 0.2668 | 0.2639 | 2851.7 | 1.000 |
| 42 | 否 | 0.3481 | 0.2781 | 2340.6 | 1.000 |
| 131 | 否 | 0.3239 | 0.2436 | 2212.8 | 1.000 |

所有候选均能执行完整上楼、平台与四级正向下楼，首周期零 reset，但没有候选满足终点路线门，多种子冲击也明显高于当前保守基线。v36 不作为 warm-start 或安全基线。

## 下一方向

v37 保留 v36 的非零制动底座与独立双支撑信号，但不再增大奖励权重。改为终端课程采样：提高 frame 1160--1330 的起始状态覆盖率，尤其增加下楼完成、双脚重新接触和终点保持段样本，同时降低与当前瓶颈无关的早段 focus 比例。先验证新奖励激活率、episode length 和 teacher trust 是否稳定，再做相同多 checkpoint 真实联评。

这属于基于 contact-conditioned locomotion、safe-stoppability 和课程学习思想的工程推断：先保证安全目标在训练分布中被充分观察，再讨论更强的约束或恢复策略。

- Contact-conditioned learning of locomotion policies: https://arxiv.org/abs/2408.00776
- Humanoid Safe Stop via Learned Stoppability Value: https://arxiv.org/abs/2609.02358
- Learning Safe-Stoppability Monitors for Humanoid Robots: https://arxiv.org/abs/2603.22703

双视图 MP4 仅保存在服务器：`analysis/s52_transfer/training_records/v36_terminal_support_floor_instant_impact_preflight128x60_20260911/model_92209_seed42_model29999_style.mp4`。
