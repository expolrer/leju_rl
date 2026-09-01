# Kuavo S52/S53 差异与基础能力双仿真验收

日期：2026-08-30

## 1. 执行原则

本阶段先验证 S52 本体本身，不把楼梯策略失败混入模型审计：

1. 自动解析 S52/S53 URDF，逐关节、逐连杆比较运动学、质量、惯量、限位和碰撞。
2. 在 LejuLab/Isaac Lab 中只做平地站立与低速直行。
3. 导出同一 S52 策略，在官方 S52 MuJoCo MJCF 中复验相同任务。
4. Lab 与 MuJoCo 的基础站立、行走都通过后，才允许把 `model_92099.pt` 作为 S52 楼梯 teacher/warm-start。

所有执行均为无界面模式；未启动或修改 Docker、ROS、`ROS_MASTER_URI` 或真机链路。

## 2. S52 与 S53 完整差异审计摘要

自动审计脚本：

`/home/zzx23457/hhw/LejuLab-Train/scripts/model_tools/compare_kuavo_s52_s53.py`

服务器完整结果：

- `analysis/s52_transfer/KUAVO_S52_S53_MODEL_DIFF.json`
- `analysis/s52_transfer/KUAVO_S52_S53_MODEL_DIFF_ZH.md`

JSON 保留每个共同连杆的质量、质心、惯量、碰撞体，以及每个共同关节的安装位姿、轴、角度范围、力矩和速度限制。关键结论如下。

S52 同时存在“带 dummy root 的原始资产”和“移除 dummy root 的 Deploy/Lab 规范化资产”，两者物理本体相同，但顶层计数相差一个固定根连杆/关节。报告同时保留两种口径：

| 项目 | S52 原始资产 | S52 Deploy/Lab 规范化资产 | S53 | 结论 |
| --- | ---: | ---: | ---: | --- |
| URDF 连杆数 | 47 | 46 | 47 | 规范化 S52 仅移除了 dummy root |
| URDF 关节数 | 46 | 45 | 46 | 同上，少一个固定根关节 |
| 非固定关节 | 29 | 29 | 27 | S52 多 `zhead_1/2_joint` 两个自由头部关节 |
| 策略受控关节 | 27 | 27 | 27 | 名称和顺序完全一致 |
| URDF 总质量 | 62.945627 kg | 62.945627 kg | 68.433410 kg | S53 重 5.487783 kg，约 8.7% |
| 左腿安装链长度 | 约 0.95944 m | 约 0.95944 m | 约 0.95945 m | 名义腿长非常接近，但惯量和局部几何仍不同 |
| 头部 | 两个非策略自由关节 | 两个非策略自由关节 | 固定 | 部署时必须明确锁定 S52 头部，不能让其进入 27 维策略接口 |
| 足底碰撞 | 足底 box 加 6 个固定球/脚 | 六球挂在固定子 link 上，`leg_*6_link` 本身无 collision | 脚 link 上 13 个 box/capsule | 不能用直接 collision 数量判断接触等价性 |
| 官方 S52 MuJoCo 足底 | 6 个球/脚 | 6 个球/脚 | 不适用 | Lab 派生 URDF 保留六球，与 MuJoCo 对齐 |

27 个策略关节均按以下固定顺序控制：左腿 6、右腿 6、腰 1、左臂 7、右臂 7。S52 与 S53 虽可共享动作张量形状，但不能直接共享质量、惯量、执行器、PD、力矩限制和接触参数，因此“27 维相同”不等于“权重可直接部署”。本地 `KUAVO_S52_S53_MODEL_DIFF.*` 使用当前 Deploy/Lab 规范化 S52；服务器首次审计文件使用原始 S52，二者差异仅为上述根节点/碰撞挂载口径。

## 3. S52 Lab 模型构建

官方 S52 模型源：

`/home/zzx23457/hhw/LejuLab-Deploy/src/leju_assets/models/biped_s52`

LejuLab 资产：

- 原始官方 URDF：`source/leju_robot/leju_robot/assets/robots/kuavos52/urdf/biped_s52.urdf`
- Lab 派生 URDF：`source/leju_robot/leju_robot/assets/robots/kuavos52/urdf/biped_s52_lab.urdf`
- 资产与执行器配置：`source/leju_robot/leju_robot/assets/leju.py`
- 速度任务：`source/leju_robot/leju_robot/tasks/locomotion/velocity/config/kuavoS52/`

Lab 派生 URDF 只做仿真接口修正：移除 dummy root、以 `base_link` 为根、修正 mesh 相对路径，并将足底碰撞改为与官方 MuJoCo 相同的六球模型。官方原文件保持不变。

执行器使用 S52 自身的 PD、力矩上限、armature、摩擦补偿和速度相关力矩削弱；策略动作缩放为 `0.25`，控制周期为 `20 ms`。

基础速度策略使用 27 维动作和 450 维 actor 观测：每帧 90 维，包含基座角速度、投影重力、速度指令、27 维相对关节角、27 维关节速度和 27 维上一动作，按 5 帧 term-major 历史拼接。该 450 维平地策略接口与楼梯 `model_92099` 的 148 维接口用途不同，不能混用。

## 4. Lab 站立验收

采用 S52 独立站立任务训练，最终选择 `model_1199.pt`，执行 60 秒确定性无界面物理 rollout：

| 指标 | 结果 |
| --- | ---: |
| 失败重置 | 0 |
| 最低基座高度 | 0.9470 m |
| roll 绝对值 p95 | 0.063 deg |
| pitch 绝对值 p95 | 0.441 deg |
| roll/pitch 最大值 | 0.689 deg |
| 平面终点漂移 | 0.00356 m |
| 速度 RMSE | 0.00267 m/s |
| 左/右脚接触占比 | 1.0 / 1.0 |
| 接触时足端滑动 p95 | 0.00165 m/s |
| 力矩限幅比 p95 / 最大值 | 0.1448 / 0.371 |

结论：S52 Lab 模型具备稳定静态动力学、重力平衡、关节方向和 PD 闭环能力。

数据：`analysis/s52_transfer/lab_base_validation/stand_clean/model_1199_full60s.npz`

导出：

- `analysis/s52_transfer/exported/s52_stand_model1199/policy.pt`
- `analysis/s52_transfer/exported/s52_stand_model1199/policy.onnx`

## 5. Lab 平地行走验收

前三版分别暴露站立投机、滑行跌倒和周期步态跌倒，均已归档。随后以已有行走 teacher 只初始化网络，再用 S52 动力学训练；v5 的 `model_998.pt` 在 60 秒确定性 Lab rollout 中通过：

| 指标 | 结果 |
| --- | ---: |
| 失败重置 | 0 |
| 最低基座高度 | 0.95393 m |
| roll 绝对值 p95 | 3.683 deg |
| pitch 绝对值 p95 | 3.084 deg |
| roll/pitch 最大值 | 4.376 deg |
| 前进距离 | 15.7845 m |
| 指令速度 | 0.25 m/s |
| 平均前向速度 | 0.2630 m/s |
| 速度跟踪 RMSE | 0.0560 m/s |
| 左/右脚接触占比 | 0.739 / 0.676 |
| 双支撑占比 | 0.415 |
| 腾空占比 | 0 |
| 接触滑动 p95 | 0.1038 m/s |
| 峰值足端接触力 | 1633 N |
| 力矩限幅比 p95 / 最大值 | 0.2816 / 0.8776 |

结论：S52 Lab 模型已具备周期接触、交替支撑、持续低速直行和有限姿态摆动，基础运动学与动力学能力通过。

数据：`analysis/s52_transfer/lab_base_validation/walk_v5_preflight/model_998_full60s.npz`

导出：

- `analysis/s52_transfer/exported/s52_walk_model998/policy.pt`
- `analysis/s52_transfer/exported/s52_walk_model998/policy.onnx`

## 6. Lab 与 MuJoCo 接口一致性

MuJoCo 使用官方文件：

`/home/zzx23457/hhw/LejuLab-Deploy/src/leju_assets/models/biped_s52/xml/scene.xml`

模型维度为 `nq=36`、`nv=35`、`nu=29`，仿真步长 `0.002 s`。验证器显式处理 27 个策略执行器和两个锁定头部执行器，并复现相同默认关节角、动作缩放、5 帧历史观测、PD、摩擦补偿、armature 和速度相关力矩裁剪。

初始状态数值门：

| 比较项 | 最大绝对误差 |
| --- | ---: |
| 450 维初始 actor 观测 | 4.77e-7 |
| 首步 27 维策略动作 | 8.64e-7 |

所有观测分段、关节顺序和动作方向均通过，因此后续差异属于物理闭环，不是数组错位。

## 7. MuJoCo 站立验收

官方 MJCF 默认接触时间常数约 `20 ms`，策略在约 26 秒后漂移跌倒。MJCF 注释中给出的刚性接触方向为 `solref=0.005 1`；扫描后使用 `4 ms` 时间常数，原 XML 不落盘修改，只在验证器运行时覆盖并记录 SHA。

`model_1199.pt` 在官方 S52 MuJoCo 中完成 60 秒：

| 指标 | 结果 |
| --- | ---: |
| 失败 | 0 |
| 平均/最低高度 | 0.95249 / 0.94676 m |
| roll 绝对值 p95 | 0.344 deg |
| pitch 绝对值 p95 | 0.558 deg |
| roll/pitch 最大值 | 1.038 deg |
| 平面终点/最大漂移 | 0.00573 / 0.01486 m |
| 速度 RMSE | 0.00730 m/s |
| 左/右脚接触占比 | 约 1.0 / 1.0 |
| 接触滑动 p95 | 0.000469 m/s |
| 峰值足端接触力 | 1936.6 N |
| 力矩限幅比 p95 / 最大值 | 0.1343 / 0.3929 |

数据：`analysis/s52_transfer/mujoco_base_validation/stand/model1199_full60s_contact4ms.npz`

结论：同一 S52 站立策略在 Lab 与 MuJoCo 双端均通过，S52 的基础模型、关节映射和静态接触闭环已验证正确。

## 8. MuJoCo 行走构建门

Lab 通过的 `model_998.pt` 在 MuJoCo 中不会立即跌倒，但会累积明显横移和偏航。为缩小 sim2sim 差异，新增独立任务：

- `Velocity-Flat-KuavoS52-Walk-Sim2Sim`
- `Velocity-Flat-KuavoS52-Walk-Sim2Sim-Play`
- experiment：`kuavoS52_walk_sim2sim`

任务使用窄范围质量、质心、摩擦、PD/armature 和小推力随机化，并加强横向速度、偏航、姿态、滑移和冲击约束。`128x300` 正常完成到 `model_1297.pt`，无 Traceback、NaN 或 OOM；全部 `1000/1050/1100/1150/1200/1250/1297` 候选均已导出 ONNX 并完成官方 S52 MuJoCo 快速联评。

无镜像校正的 60 秒结果：

| 模型 | 跌倒/失败 | 前进 x | 横移 y | 累积偏航 | 速度 RMSE | 滑动 p95 | 峰值冲击 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1050 | 0 | 1.512 m | -9.469 m | -168.8 deg | 0.206 m/s | 0.197 m/s | 2361 N |
| 1250 | 0 | -3.074 m | -6.156 m | -239.5 deg | 0.260 m/s | 0.167 m/s | 2329 N |
| 1297 | 0 | -2.742 m | -6.640 m | -228.1 deg | 0.243 m/s | 0.142 m/s | 2134 N |

观测和命令编码复核确认 Lab/MuJoCo 都使用 `[vx, sin(phase), cos(phase)]`，因此漂移属于两侧接触误差的长期积分，不是命令槽错误。对 `model_1297` 仅在部署侧融合左右镜像动作，扫描结果显示 `symmetry_blend=0.25` 最适合长测：

| 指标 | 60 秒结果 |
| --- | ---: |
| 跌倒/失败 | 0 |
| 前进距离 | 17.57 m |
| 终点横向偏差 | -0.29 m |
| 累积偏航 | +4.4 deg |
| roll / pitch p95 | 4.48 / 3.69 deg |
| 速度 RMSE | 0.220 m/s |
| 接触滑动 p95 | 0.134 m/s |
| 峰值足端接触力 | 3496 N |

结论：S52 Lab 资产、27 关节映射、基础动力学和官方 MuJoCo sim2sim 链路已经通过“60 秒持续直行且零跌倒”的模型构建门。该策略只作为楼梯迁移桥接控制器，冲击和速度振荡仍偏高，不是最终部署控制器。

## 9. 域随机化与跨型号迁移结论

一手资料采用顺序如下：

- [Dynamics Randomization](https://arxiv.org/abs/1710.06537)：在动力学参数分布上训练期望回报，并用历史状态适应不同动力学；说明质量、摩擦、时间步和观测噪声应按 episode 随机。
- [Sim-to-Real Agile Locomotion](https://arxiv.org/abs/1804.10332)：先做 SysID、执行器和延迟建模，再做物理随机化；DR 不能替代错误的名义模型。
- [Humanoid-Gym](https://arxiv.org/abs/2404.05695)：Isaac Gym 训练、MuJoCo sim2sim、XBot-S/XBot-L 真机；采用历史观测、特权 critic、相位接触奖励和质量/摩擦/PD/延迟/噪声随机化。
- [Blind Bipedal Stair Traversal](https://roboticsproceedings.org/rss17/p061.html)：Cassie 在不改平地奖励时通过台阶/指令随机化学习盲楼梯；适合固定路线通过后的鲁棒化，不适合现在直接替代精确突缘净空。
- [LearningHumanoidWalking](https://arxiv.org/abs/2207.12644)：未来两个有向落脚点、相位/接触和地形课程可跨 HRP5P/JVRC-1；适合把长参考动作拆成接触完成驱动的逐步目标。
- [Mind Your Steps](https://montenegroalessandro.github.io/mind-your-steps/)：在支撑脚坐标系表达 3D foothold，并在摆腿期间保持目标；适合 S52 的脚底轨迹、骨盆高度和接触时序优先迁移。
- [Footstep-Constrained Cassie](https://arxiv.org/abs/2203.07589)：显式 touchdown 约束和可达域模型；用于限制短踏面的落点采样。
- [Contact-conditioned locomotion](https://arxiv.org/abs/2408.00776)：未来接触模式、足端位置和切换时刻比仅速度/步态标签更利于分布外泛化。
- [McARL](https://arxiv.org/abs/2505.18418)：随机形态向量条件化 actor/critic；支持将质量、惯量、质心、PD、力矩和足底几何作为 S52/S53 morphology context。
- [ASAP](https://agile.human2humanoid.com/)：目标系统实际轨迹上的 delta action 与后续策略微调，适合 Lab S52 楼梯通过后的 MuJoCo residual/DAgger。

实施边界：先在固定 S52 名义模型和固定台阶完成完整路线；随后只随机摩擦、质量/质心、motor strength、armature、0-10 ms 延迟、传感噪声和小外力；连续多 seed 通过后才加入楼梯尺寸、落点和材质随机化。域随机化是 sim2real 的关键工具，但不是第一摆腿粘地、接触时序错误或错误 MJCF 的修复器。

## 10. 下一阶段：S52 Lab 台阶任务

1. 将已重定向的 S52 楼梯 CSV 用 S52 Lab 资产转换为运动学 NPZ，并逐帧复核 27 关节、双脚轨迹、骨盆高度和接触时序。
2. 新建独立 S52 task/Play/experiment，保持 148 维 actor 与 27 维动作契约；`model_92099.pt` 只作为冻结 teacher/warm-start。
3. 先在 Lab 做 SFT/DAgger 和低学习率 RL：支撑脚不滑、摆腿离地、未来 foothold、足底位置/俯仰、骨盆、接触转换优先；碰撞窗口弱化 teacher KL。
4. 固定楼梯完整通过后导出 ONNX，再到官方 S52 MuJoCo 做固定参数 sim2sim、窄 DR 与 residual adapter。
5. 所有过程继续无界面运行，不操作 Docker、ROS、`ROS_MASTER_URI` 或真机链路。

## 11. S52 台阶 v1--v4 结果与接触相位修正方向（2026-09-01）

S52 楼梯 motion 已转换为 1341 帧、50 Hz、27 关节/30 body 的独立 NPZ，楼梯任务保持
148 维 actor 和 27 维动作，`model_92099.pt` 始终是只读 teacher/warm-start。v1 直接迁移、
v2 严格相位终止、v3 平台软约束、v4 非饱和骨盆下穿与双脚约束均已完成训练曲线和固定
seed 真实 PhysX 联评，但全部否决。

v4 最终 `model_92178.pt` 可跑满 frame 1340 且零重置，实际平台骨盆 z RMSE 仍为
`0.333 m`。frame 725 时，右脚在支撑到摆腿转换中比参考提前前冲约 `9.1 cm`；参考脚仍在
运动，策略脚却已产生约 `132 N` 的提前接触并落低。接触顺序失配后，frame 900 骨盆高度
落后参考约 `0.416 m`。因此这不是 S52 本体基础动力学失败，而是 S53 参考到 S52 后的
摆腿/支撑与落脚时序没有获得直接训练信用。

下一版本采用 [Mind Your Steps](https://montenegroalessandro.github.io/mind-your-steps/) 的
支撑脚相对 foothold 和 [Contact-conditioned locomotion](https://arxiv.org/abs/2408.00776)
的未来接触思想做工程实现：用左右参考足端相对速度区分摆腿和支撑；在平台末端与下降窗口
稠密惩罚摆腿脚的前向超调、高度下穿与过早接触，并约束支撑脚速度和接触丢失。先保持 148 维接口，
验证 reward/critic 侧接触条件有效后，再考虑兼容 adapter 扩展未来接触输入。

域随机化仍关闭。只有 S52 在固定物理参数下多 seed 完成上楼、平台、四级正向下楼和落地，
才按摩擦、质量/质心、motor strength/PD、0--1 控制步延迟、观测噪声的窄范围顺序启用，
随后进入官方 S52 MuJoCo 固定参数及窄 DR 验收。

v4 失败报告和本地曲线：
`F:\桌面\20260521\S52_TRANSFER_20260830\STAIRS_SOFT_PLATEAU_TRUST_V4_FAILURE_REPORT_ZH.md`
及 `v4_artifacts`。真实双视图 MP4 仅保存于服务器训练记录目录。

## 2026-09-01 补充：楼梯策略接口探针与 v8 连续验收

- S52/S53 的 27 个受控关节和 148 维楼梯 actor 观测已逐组核对；S52 的两个头部关节只存在于
  资产状态，不进入 27 维 action 和策略 joint observation。
- 修复 S53 资产误设 `merge_fixed_joints=False` 后，S53 `model_92099` 可完成 1351 步、零重置
  物理 rollout；其 148 维观测与 S52 理想重定向参考首步最大组差约 `0.00404`。
- S53 成功物理 rollout 只作为 SFT/蒸馏监督集。将闭环物理轨迹直接替换 command 会造成关节
  速度分布偏移，不能作为 S52 `MotionCommand`。
- v8 `128x60` 的训练回报虽升至约 `93`，五个固定 seed S52 真实 PhysX 候选仍全部在 frame
  `263--265` 重置。frame 240 的右支撑脚前冲约 `31.7 cm` 是当前首个确定动力学失效。
- 因此“资产、27/148 接口、Lab 站立/平地、Lab-MuJoCo 基础一致性”已经通过；“S52 Lab
  完整楼梯”尚未通过。MuJoCo 楼梯和域随机化继续保持关闭，下一阶段先做首段上楼的完整前缀
  与接触相位适配。

## 12. v9/v10 与 actual-state replay 探针（2026-09-01）

v9 AscentPrefix 和 v10 ContactRelease 均完成无界面短预检，但固定 `seed=42` 的所有候选仍在
首段上楼 frame `260--266` 重置。v9 放松 teacher 后 KL 一度升到 `3.954`，属于全 actor 漂移；
v10 恢复强 trust 后 KL 降低，但从理想 frame 160--220 reset 得到的 episode length 改善没有
迁移到 frame-0 rollout。当前瓶颈是 S52 实际访问状态与理想 reference reset 之间的 covariate
shift，不是 27/148 映射、基础动力学或奖励数值饱和。

用原始 `model_92099.pt` 在 v10 相同任务重采目标域轨迹后，actual-state replay 探针可把
frame `160/180/207/220/240` 的根状态、29 关节和 148 维观测写回；写回观测最大误差不超过
`7.2e-7`，刚体位置不超过 `9.6e-7 m`。承重瞬间单步接触力仍有最高约 `204 N` 差异，说明
PhysX 接触求解器历史不能从 NPZ 直接复原。因此下一版从失败前 7--62 步的真实状态启动并保留
前置接触走廊，而不是直接从碰撞帧启动。

v11 `Tracking-Stairs-ActualReplay-KuavoS52` 已独立建立并通过 py_compile、任务注册和 265 行
teacher replay 数据校验；它保持 v10 奖励，只采用 55% frame 0 + 45% teacher 实际状态的
训练分布，并以更低学习率和更强 teacher trust 近似 residual adaptation。当前仍未宣称 S52
楼梯通过，MuJoCo 楼梯和域随机化继续关闭。
