# Kuavo S52 楼梯迁移 v9 AscentPrefix 失败报告

日期：2026-09-01

## 结论

v9 完成 `32x2` 冒烟和 `128x120` 短预检，训练本身无 Traceback、NaN 或 OOM。六个固定
`seed=42` 的 1351 步真实 Isaac/PhysX rollout 均在首段上楼的 frame `260--266` 首次失败，
每个 rollout 重置 5 次，只完成全任务约 `19.39%--19.84%`，未进入平台和下楼阶段。因此
v9 全部 checkpoint 否决，不能进入正式训练、MuJoCo 或域随机化。

## 实际配置

- Task：`Tracking-Stairs-AscentPrefix-KuavoS52`
- Play：`Tracking-Stairs-AscentPrefix-KuavoS52-Play`
- Experiment：`kuavoS52_stairs_ascent_prefix`
- Run：`2026-09-01_06-44-04_s52_ascent_prefix_preflight128x120_v9_20260901`
- Warm-start/teacher：只读 S53 `model_92099.pt`，SHA-256
  `6483ff66456f1e218713f228114a89b6bd5688d94d4c2612ebc247375812f0f4`
- 固定 S52 物理、128 environments、120 iterations、seed 42、学习率 `8e-6`
- 所有训练 episode 从 frame 0 开始；上楼摆腿、未来落脚点、落脚接触、对侧支撑和骨盆高度
  均使用离线固定相位表。

第一次冒烟因父类自适应采样器在全零失败直方图上收到零概率分布而触发 CUDA assert；保留
`adaptive_uniform_ratio=0.10` 后修复。第二次启动因新 experiment 尚无受保护 warm-start
目录而停止；建立指向原始只读 teacher 的符号链接后修复。这两项均发生在训练迭代前，最终
冒烟和短预检正常结束。

## 奖励曲线

- mean reward：`0.743 -> 59.554`，7 点平滑峰值 `60.671 @ 92117`；
- mean episode length：`2.00 -> 288.02`，7 点平滑峰值 `290.01 @ 92133`；
- teacher action RMSE：`0.00231 -> 0.01261`，最大 `0.02611 @ 92131`；
- teacher KL：`0.03570 -> 0.90068`，最大 `3.954 @ 92131`，说明策略明显偏离 teacher；
- anchor/body/joint position error 末值约 `1.116/0.433/2.419`；
- 上楼摆腿路径奖励末值 `0.1633`，提前接触惩罚 `-0.3754`，摆腿高度不足惩罚
  `-0.4900`；anchor position termination 末值 `2.625`，time-out 为 0。

总回报和 episode length 上升没有转化为首段上楼通过。上楼指数奖励在大误差区接近零，
截断平方接触惩罚在高接触力区饱和，正好失去对失败状态的细粒度信用分配。

## 候选真实物理联评

| checkpoint | 首次失败 frame | 首周期完成率 | 滑移 p95 (m/s) | 峰值足部冲击 (N) |
| --- | ---: | ---: | ---: | ---: |
| 92100 | 264 | 19.69% | 0.2615 | 1068.8 |
| 92110 | 260 | 19.39% | 0.2628 | 886.2 |
| 92120 | 262 | 19.54% | 0.2555 | 852.6 |
| 92130 | 262 | 19.54% | 0.2588 | 945.7 |
| 92170 | 265 | 19.76% | 0.2697 | 852.4 |
| 92218 | 266 | 19.84% | 0.2509 | 1281.9 |

最终模型的右脚计划在 frame `207--267` 摆动，但 frame `207/220/240` 实际接触力仍约
`324/736/212 N`，说明它没有完成离地。在 frame 240，右脚相对参考纵向超前约 `28.7 cm`、
高度低约 `11.6 cm`；frame 266 骨盆低于参考约 `41.2 cm` 后终止。它不是在平台或下楼阶段
失败，而是第二个上楼摆腿支撑转换失败。

## 一手方法与下一版取舍

- FastStair 使用并行可行落脚规划器引导 RL，先以高权重落脚跟踪预训练安全基线，再做速度
  专家和 LoRA；其摆脚参考由起点、Bezier apex 和目标落脚点构造。工程采用：把 S52 首段
  摆腿改成非饱和局部走廊和分阶段课程，不先追求速度。
- Mind Your Steps 使用支撑脚坐标系中的 3D foothold 和动态 goal sampler。工程采用：固定
  目标在一个摆腿片段内不漂移，增加线性未来落脚点误差。
- Contact-conditioned locomotion 使用未来接触切换及其时刻作为目标。工程采用：重点监督
  frame 207 的右脚离地和对侧支撑，不再只靠全局逐帧模仿。
- Any2Any 先运动学对齐，再仅适配动力学敏感模块。27/148 对齐已经完成；v10 先用更小学习率
  和更强 teacher trust 验证局部信号，仍失败时再实现 residual/adapter，避免全 actor 漂移。
- Dynamics Randomization 和 Humanoid-Gym 的 sim2sim/DR 只用于名义模型通过后的鲁棒化。
  当前固定 S52 台阶都未通过，不能用随机化掩盖接触模型或奖励错误。

## v10 计划

v10 从原始 `model_92099.pt` 重新 warm-start，不接续 v9。训练采样设为约 55% frame 0、35%
关键右摆腿邻域、10% 自适应；新增不截断的对数接触力代价、非饱和 L1 摆腿走廊和未来落脚点
误差。全局 teacher action/KL 恢复为 `4.0/0.20`，学习率降为 `3e-6`。先做 `32x2` 冒烟和
`128x80` 短预检，仍以完整 frame-0 真实 rollout 作为唯一通过判据。

## 归档

- 服务器记录：`analysis/s52_transfer/training_records/stairs_ascent_prefix_v9_preflight128x120_20260901`
- 服务器最新双视图：`video/model_92218_model29999_style_seed42.mp4`，27.04 s，1820x910，
  50 fps，1352 frames；MP4 不同步本地。
- 本地报告：`F:\桌面\20260521\S52_TRANSFER_20260830\STAIRS_ASCENT_PREFIX_V9_FAILURE_REPORT_ZH.md`
