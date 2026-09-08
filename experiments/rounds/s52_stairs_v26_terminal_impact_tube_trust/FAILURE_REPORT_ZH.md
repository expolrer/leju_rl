# S52 v26 TerminalImpactTubeTrust 短预检失败报告

## 运行信息

- 任务：`Tracking-Stairs-TerminalImpactTubeTrust-KuavoS52`
- Play：`Tracking-Stairs-TerminalImpactTubeTrust-KuavoS52-Play`
- 实验：`kuavoS52_stairs_terminal_impact_tube_trust`
- run：`2026-09-09_05-04-31_s52_terminal_impact_tube_trust_preflight128x60_v26_20260909`
- warm-start / 固定教师：受保护的 v22 S52 `model_92150.pt`
- warm-start SHA-256：`93c457de92f9102125d7c88f8721e25a35e83caa03e4112d7a01074c78e50f56`
- 最终 checkpoint：`model_92209.pt`
- 最终 SHA-256：`660f5bd4910507c42658233bfcac03ea4cce4fb73f79305f93fd57c1a636e309`
- 规模：`32x2` 冒烟通过；`128x60` 短预检约 51 秒完成
- 运行健康：无 Traceback、NaN、OOM；policy/critic 维度 148/280

## 实际修改

1. 保持 v25 的 S52 快照 teacher、完整路线和终点焦点采样。
2. 终点前向允许超调由 0.04 m 收紧至 0.02 m；终点双支撑位置奖励由 3.0 提至 3.5、位置 std 由 0.18 m 收紧至 0.16 m；terminal brake 权重由 -0.75 调至 -0.90。
3. frame 600--1340 新增足端峰值接触力平方屏障：1400 N 阈值、500 N 尺度、上限 16、权重 -0.10；原接触力项由 -0.002 调至 -0.003。
4. 学习率上界降至 `5e-8`，teacher 均值动作硬投影从 `5e-4` 收紧至 `2e-4`，teacher/action-tail 系数同步提高。

## 曲线结论

- mean reward：`-8.3515 -> 92.2138`，末轮为峰值；末 7 点均值 `85.0814`。
- episode length：`16.05 -> 701.85`，末轮为峰值；末 7 点均值 `652.31`。
- policy KL：`0.001349 -> 0.000463`，最低 `0.000417 @ 92154`；仍高于目标 `5e-5`。
- learning rate：首轮日志 `9.877e-9`，`92152` 起到 `1e-9` 下限。
- teacher action RMSE：`1.519e-4 -> 1.869e-4`，最大 `2.129e-4 @ 92158`。投影前期激活，末 7 点激活率约 10.7%。
- terminal route brake：末轮 `-0.2102`，最小 `-0.4153 @ 92187`，终点信用已足够稠密。
- 新 peak-contact 屏障：末轮仅 `-0.001221`，最小 `-0.003266 @ 92187`，末 7 点均值 `-0.000558`。相对 80 以上总回报仍过于稀疏，无法代表 rollout 的单帧极值。
- 原 contact-force 项末轮 `-0.1716`，但候选真实峰值仍可达 3.35 kN，说明普通 episode/batch 平均惩罚与最坏冲击脱节。
- anchor position error 末轮 `0.4111`，末 7 点均值 `0.4739`；平均回报不能替代世界坐标门。

## 固定物理联评

seed 42 的所有候选都完成上楼、平台、四级正向下楼和末端双脚支撑，首周期零 reset。

| checkpoint / seed | 终点误差中位数 | 严格通关 | 滑移 p95 | 峰值足部力 |
| --- | ---: | --- | ---: | ---: |
| 92150 / 42 | 0.264198 m | 否 | 0.21385 m/s | 1358 N |
| 92160 / 42 | 0.269350 m | 否 | 0.23059 m/s | 2296 N |
| 92170 / 42 | 0.266994 m | 否 | 0.21340 m/s | 1745 N |
| 92180 / 42 | 0.210402 m | 是 | 0.23844 m/s | 1805 N |
| 92180 / 7 | 0.216910 m | 是 | 0.25410 m/s | 2286 N |
| 92180 / 131 | 0.335987 m | 否 | 0.24443 m/s | 1204 N |
| 92190 / 42 | 0.287006 m | 否 | 0.20359 m/s | 2335 N |
| 92200 / 42 | 0.230919 m | 是 | 0.24432 m/s | 3354 N |
| 92209 / 42 | 0.267190 m | 否 | 0.27177 m/s | 2160 N |

`model_92180` 在 seed 42/7 通过，却在 seed 131 退化到 0.336 m；`model_92200` 虽在 seed 42 通过，峰值冲击达到 3.35 kN，不值得继续多 seed。v26 因此整轮否决，不替换 v22 S52 安全快照，不进入正式训练、MuJoCo 或域随机化。

## 原因与下一方向

1. 终点项已稠密，当前瓶颈不是“奖励没出现”，而是 seed 相关闭环尾部风险。
2. 峰值接触屏障只在极少数时刻激活，经 episode 和 batch 平均后量级约 `1e-3`；继续单纯增加权重容易让策略绕开接触或破坏下楼，而不能保证最坏状态。
3. SCPO/ASCPO 强调最坏/高概率状态约束；`Not Only Rewards But Also Constraints` 在多种本体越障中把工程意图写成独立约束，均支持把接触峰值从普通奖励迁移为 cost/约束。
4. Contact-conditioned locomotion 用未来接触位置和切换时间条件化策略。工程上，下一版应按每次下降摆腿/落地片段聚合最大冲击和终点误差，而不是跨整批次求平均。

一手资料：

- SCPO：https://arxiv.org/abs/2306.12594
- ASCPO：https://arxiv.org/abs/2410.01212
- Not Only Rewards But Also Constraints：https://arxiv.org/abs/2308.12517
- Contact-conditioned locomotion：https://arxiv.org/abs/2408.00776
- 官方 legged_gym 接触力实现：https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py

下一版应从 v22 S52 安全快照重新开始，保留 v26 的终点采样，但将下降/落地片段的最大冲击和终点世界误差作为独立 constrained PPO cost；先做只读 cost 分布校准，再实现 32x2 和 128x60。当前自动执行已达到本次最多两版短预检的上限，因此没有启动第三版。
