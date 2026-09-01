# Kuavo S52 楼梯迁移 v10 ContactRelease 失败报告

日期：2026-09-01

## 结论

v10 完成 `32x2` 冒烟和 `128x80` 短预检，无 Traceback、NaN 或 OOM。新加入的非饱和
L1/对数奖励在训练中持续可观测，teacher KL 也远低于 v9；但六个固定 `seed=42` 的完整
frame-0 真实 Isaac/PhysX rollout 仍全部在 frame `261--263` 首次失败、重置 5 次，没有完成
首段上楼。因此全部 v10 checkpoint 否决，不进入正式训练、MuJoCo 或域随机化。

## 实际配置

- Task：`Tracking-Stairs-ContactRelease-KuavoS52`
- Play：`Tracking-Stairs-ContactRelease-KuavoS52-Play`
- Experiment：`kuavoS52_stairs_contact_release`
- Run：`2026-09-01_07-07-45_s52_contact_release_preflight128x80_v10_20260901`
- Warm-start/teacher：只读 S53 `model_92099.pt`，SHA-256
  `6483ff66456f1e218713f228114a89b6bd5688d94d4c2612ebc247375812f0f4`
- 采样：55% frame 0、35% frame 180--240 关键摆腿邻域、10% adaptive；
- 新信号：非饱和摆腿路径 L1、未来落脚点 L1、对数接触力；
- teacher action/KL `4.0/0.20`，学习率 `3e-6`，固定 S52 名义物理。

## 曲线分析

- mean reward：`-3.215 -> 62.137`，7 点平滑峰值 `71.353 @ 92177`；
- mean episode length：`1.82 -> 326.25`，7 点平滑峰值 `354.10 @ 92177`；
- teacher action RMSE：`0.00134 -> 0.00332`，平滑最大约 `0.00381`；
- teacher KL：`0.01133 -> 0.06005`，平滑最大约 `0.08685`；
- 新摆腿路径 L1 末值 `-0.1558`，未来落脚点 L1 `-0.0308`，对数接触力 `-0.0849`；
- 上楼高度不足末值 `-0.5262`，anchor position termination 末值 `1.0`。

新信号解决了 v9 的数值饱和，但训练均值包含从理想关键帧 reset 的 episode。它证明策略可在
理想局部状态附近获得更长回报，却不能证明从 frame 0 实际走来的状态分布也被修复。

## 候选真实物理联评

| checkpoint | 首次失败 frame | 首周期完成率 | 滑移 p95 (m/s) | 峰值足部冲击 (N) |
| --- | ---: | ---: | ---: | ---: |
| 92100 | 263 | 19.61% | 0.2542 | 886.8 |
| 92110 | 261 | 19.46% | 0.2613 | 811.4 |
| 92140 | 262 | 19.54% | 0.2608 | 924.6 |
| 92160 | 263 | 19.61% | 0.2629 | 799.8 |
| 92170 | 262 | 19.54% | 0.2605 | 946.0 |
| 92178 | 263 | 19.61% | 0.2629 | 1125.6 |

六个候选与原始 teacher/v8/v9 的失败帧几乎相同。v10 没有在 frame-0 闭环中改变第二次上楼
摆腿的离地时序；最高训练 episode length 是局部 reset 分布的改善，不是完整上楼改善。

## 根因与下一步

当前主要问题是目标域 covariate shift。局部课程把机器人直接放在理想参考的 frame 160--220
附近，而 frame-0 rollout 到达同一时刻时已经累积足端、骨盆、接触和关节状态偏差。策略没有
在自己真实访问的失败前状态上学习恢复，因此继续提高局部奖励权重不会解决问题。

[ASAP](https://agile.human2humanoid.com/) 的明确流程是：在目标系统采集执行轨迹，学习 delta
action 补偿源/目标动力学差异，冻结补偿器后再微调原策略。其官方代码要求 delta-action motion
文件包含实际 `action`。本项目的工程映射为：

1. 用只读 `model_92099` 在 S52 PhysX 采集 frame 0 到失败的实际 `(q, qd, base, contact,
   observation, action)`；
2. 训练 reset 必须回放这些实际访问状态，而 command 仍使用理想 S52 reference；
3. 先在 frame 160--240 学一个小 residual/adapter，主 actor 保持冻结或强 trust；
4. 用 frame-0 DAgger rollout 反复补充新访问状态，直到首段上楼真实通过；
5. 名义 Lab 全任务通过后才进入 S52 MuJoCo 和窄范围域随机化。

这与 Any2Any 的“运动学对齐后只适配动力学敏感模块”一致，也避免 v9 全 actor 漂移与 v10
理想状态过拟合。下一版应先实现只读 actual-state replay 数据检查和 reset 同源探针，再启动
短预检；不再盲目增加 contact/path 权重。

## 归档

- 服务器记录：`analysis/s52_transfer/training_records/stairs_contact_release_v10_preflight128x80_20260901`
- 服务器双视图：`video/model_92178_model29999_style_seed42.mp4`（仅服务器）
- 本地报告：`F:\桌面\20260521\S52_TRANSFER_20260830\STAIRS_CONTACT_RELEASE_V10_FAILURE_REPORT_ZH.md`
