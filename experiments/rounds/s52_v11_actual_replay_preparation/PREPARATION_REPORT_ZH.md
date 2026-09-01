# Kuavo S52 楼梯迁移 v11 ActualReplay 准备与探针报告

日期：2026-09-01

## 当前结论

S52 的 27 关节/148 维策略接口、Lab 站立与行走、MuJoCo 站立与行走均已验证；完整楼梯仍未
通过。v9/v10 所有固定 `seed=42` 候选都在首段上楼约 frame 260--266 重置，说明问题已经从
“奖励是否有梯度”收敛为“理想局部 reset 与 frame-0 实际访问状态不一致”。

v11 已完成源码、teacher 目标域数据、同源重放探针、py_compile 和任务注册准备，但没有在本次
执行中启动第三轮短预检或正式训练。

## 数据来源与保护

- teacher/warm-start：只读 S53 `model_92099.pt`；
- checkpoint SHA-256：`6483ff66456f1e218713f228114a89b6bd5688d94d4c2612ebc247375812f0f4`；
- S52 目标域轨迹：原 teacher 在 v10 S52 Play 任务、固定 `seed=42` 下的 1351 步 PhysX
  rollout；SHA-256：`bc20a9fa6d16d9f8246f4a7ae4ade9d5765cb0dd0d2fdc73da124105948118c2`；
- 265 行前缀 replay 数据：`kuavoS52_model92099_actual_prefix_replay_seed42.npz`；
  SHA-256：`05a88e08de93ba32830a53576ab79dd7efea442e56cbe5558e1744ae8c6c1f19`。

理想 S52 retarget 继续作为 148 维 command。目标域 PhysX rollout 只提供实际根状态、29 关节
状态、上一动作和恢复训练起点，二者没有混用。

## 同源重放探针

探针在 frame `160/180/207/220/240` 写回 teacher 的实际状态，重建 command 与上一动作，再
执行原始下一动作并对照下一帧：

- 写回后 148 维观测最大误差：`7.2e-7`；
- 写回后全身刚体位置最大误差：`9.6e-7 m`；
- 单步后全身刚体位置误差：约 `1.3e-5--1.0e-4 m`；
- 单步观测最大误差：`0.013--0.443`，主要来自关节速度；
- 接触力最大差：frame 207 约 `43.5 N`，frame 240 约 `204.3 N`。

根状态、关节顺序和观测重建已经通过；接触求解器 warm-start/cache 不在 NPZ 中，因此 v11
不能直接从强承重或冲击帧开始。训练起点选择 frame `145--200`，给第二次摆腿前保留
`7--62` 个控制步的接触预热走廊。

## v11 实际配置

- Task：`Tracking-Stairs-ActualReplay-KuavoS52`；
- Play：`Tracking-Stairs-ActualReplay-KuavoS52-Play`；
- Experiment：`kuavoS52_stairs_actual_replay`；
- reset 分布：55% frame 0，45% teacher actual rows
  `145/155/165/175/185/195/200`；
- 奖励：保持 v10 的非饱和 swing path L1、future foothold L1、log contact force，不新增奖励；
- 优化：学习率 `2e-6`，clip `0.03`，desired KL `7e-4`，teacher action/KL
  `6.0/0.30`，重置优化器；
- 固定 S52 名义资产、PD、力矩、接触和楼梯，不启用域随机化。

## 方法依据

- [ASAP](https://agile.human2humanoid.com/)：在目标系统采集实际轨迹，以状态差训练 delta
  dynamics/action，冻结补偿器后微调源策略；
- [Any2Any](https://arxiv.org/abs/2605.23733)：运动学对齐后只适配动力学敏感模块；
- [Mind Your Steps](https://montenegroalessandro.github.io/mind-your-steps/)：未来 3D
  foothold 与完整上楼、平台、下楼足步序列；
- [Contact-conditioned locomotion](https://arxiv.org/abs/2408.00776)：未来接触位置、模式和
  切换时间共同条件化策略；
- [Humanoid-Gym](https://github.com/roboterax/humanoid-gym)：Isaac 训练后以 MuJoCo sim2sim
  独立验收；
- [Dynamics Randomization Revisited](https://www.pair.toronto.edu/understanding-dr/)：DR
  不是名义模型或控制接口错误的替代品。

把 S52 PhysX 当作当前目标系统、把实际状态 reset 用作 DAgger/ASAP 式信用分配，是本项目的
工程映射，不是上述论文直接给出的 Kuavo S52 超参数。

## 下一执行门

1. 先完整归档 v9，再按失败权重滚动规则删除 v9 run 内未受保护的 `model_*.pt`；v10 保留为
   最新失败对照，`model_92099` 永久保护；
2. 运行 v11 `32x2` 冒烟和 `128x60` 短预检；
3. 比较早期、奖励峰值、安全峰值、中段、退化点和最终 checkpoint 的 frame-0 真实 PhysX
   rollout；
4. 只有完整上楼、平台、四级正向下楼、落地且零重置，才进入 S52 MuJoCo 楼梯；
5. MuJoCo 固定参数通过后，再逐项启用质量/质心、摩擦、PD/motor、延迟和噪声的窄范围 DR。
