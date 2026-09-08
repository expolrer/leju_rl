# S52 v24 AdaptiveKLGuard 短预检失败报告

## 1. 运行信息

- 日期：2026-09-09
- 任务：`Tracking-Stairs-AdaptiveKLGuard-KuavoS52`
- Play 任务：`Tracking-Stairs-AdaptiveKLGuard-KuavoS52-Play`
- 实验：`kuavoS52_stairs_adaptive_kl_guard`
- 运行目录：`2026-09-09_03-33-08_s52_adaptive_kl_guard_preflight128x60_v24_20260909`
- 起点：v22 第一更新点 `model_92150.pt`，SHA-256 `93c457de92f9102125d7c88f8721e25a35e83caa03e4112d7a01074c78e50f56`
- 最终：`model_92209.pt`，SHA-256 `9e203ef866d0e956680310ac4aa7355df88e65e6392c49ae16a64af81bd75f9f`
- 规模：`32x2` 冒烟后运行 `128x60`，优化器重置
- 状态：正常结束，无 Traceback、NaN 或 OOM；Lab 验收失败

## 2. 本轮修改

保持 v22 的平台相位桥、四级正向下楼时序、终端路线制动和全部物理参数，仅修改 PPO 更新：

- `schedule: fixed -> adaptive`
- `num_learning_epochs: 3 -> 1`
- `desired_kl: 2e-4`
- 自适应学习率范围：`1e-8 .. 1e-6`
- 初始学习率：`7.5e-7`

自定义 teacher PPO 的学习率上下限改为可配置项，旧任务默认值保持不变。

## 3. 奖励与优化曲线

- mean reward：`-10.6323 -> 95.7294`，峰值 `99.3305 @ 92207`，末 7 点均值 `94.6365`
- mean episode length：`17.29 -> 717.16`，峰值 `730.25 @ 92207`
- policy KL：`0.005016 -> 0.000731`，最小 `0.000529 @ 92168`，末 7 点均值 `0.000678`
- learning rate：首个日志值 `1.481e-7`，在 `92152` 降到下限 `1e-8`
- value loss：`49.378 -> 6.496`，最小 `1.590 @ 92154`
- surrogate loss：`0.01782 -> 0.00244`
- teacher action RMSE：`0.002896 -> 0.002485`
- teacher KL：`0.1953 -> 0.1422`
- terminal route brake：`0 -> -0.03731`，最强 `-0.15947 @ 92205`
- anchor position error：`0.1620 -> 0.6772`，末 7 点均值 `0.4873`

曲线说明 adaptive 分支已经生效并显著压低了更新内 KL，但总奖励和 episode length 上升仍未转化为世界坐标终端精度。

## 4. 固定物理 rollout

所有更新后候选均到达 frame 1340，完成上楼、平台、四级正向下楼和末端双脚支撑，且首次完整周期零 reset。

| 候选 | 终端误差中位数 | 滑移 p95 | 峰值足部力 | Lab 门 |
|---|---:|---:|---:|---|
| 更新前 warm-start 92150 | 0.24597 m | 0.22787 m/s | 1325 N | seed42 通过 |
| 更新后 92150 | 0.32223 m | 0.22960 m/s | 2011 N | 否决 |
| 92160 | 0.27968 m | 0.22802 m/s | 2383 N | 否决 |
| 92170 | 0.32575 m | 0.26617 m/s | 2167 N | 否决 |
| 92180 | 0.25594 m | 0.23062 m/s | 2252 N | 否决，距门槛 5.94 mm |
| 92190 | 0.28233 m | 0.25961 m/s | 2021 N | 否决 |
| 92200 | 0.29900 m | 0.24689 m/s | 2120 N | 否决 |
| 92209 | 0.36713 m | 0.25858 m/s | 2048 N | 否决 |

由于 seed42 已全部失败，不再为更新后候选运行 seed7/131。最终模型真实双视图 MP4 仅保存在服务器训练记录目录。

## 5. 失败原因

1. 当前 adaptive PPO 在每个 minibatch 更新前测量 KL，然后调学习率；它不会撤销已经发生的越界更新。第一轮日志 KL 已达目标的约 25 倍，第一次保存点的终端误差随即退化 76 mm。
2. 第一 minibatch 的新旧策略相同，低 KL 分支会先提高学习率，随后才观察到更新后的 KL 偏大。虽然同一轮后续 minibatch 将学习率降到 `1.48e-7`，损伤已经发生。
3. teacher 仍是原始 S53 `model_92099`，而 warm-start 是经过 S52 动力学适配的 `model_92150`。teacher loss 会把策略拉向跨本体旧策略，不能保护 S52 已验证闭环。
4. 终端保持帧不在主要 terrain-focus 列表中，terminal reward 大多要等长 episode 才激活，信用分配晚于上楼、平台和下楼存活回报。

## 6. 一手资料与下一步

- RSL-RL 配置和 PPO 源码说明 `desired_kl` 只驱动 adaptive 学习率，且该机制是步长调节而非参数回滚：
  - https://github.com/leggedrobotics/rsl_rl/blob/main/docs/guide/configuration.rst
  - https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py
- PPO 原论文以 clipped surrogate 近似信赖域，但不保证每次更新都满足严格 KL 约束：https://arxiv.org/abs/1707.06347
- Truly PPO 指出普通 PPO 不能严格限制 likelihood ratio 或保证信赖域，并提出 rollback clipping：https://arxiv.org/abs/1903.07940
- TRPO 直接以平均 KL 约束策略更新，提供更严格的单调改进出发点：https://arxiv.org/abs/1502.05477

v25 采用低风险的工程映射：保持 v22 奖励和路线，增加 frame 1220/1260/1300/1330 的终端起始采样；把已验证的 S52 warm-start 固定为快照 teacher；加入 teacher action tail 和硬投影，防止首轮离开 S52 闭环；自适应学习率只允许 `1e-9 .. 1e-7`。先做 `32x2` 和 `128x60`，仍以固定物理 rollout 决定是否采用。
