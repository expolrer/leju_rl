# leju_rl

面向 Kuavo S53/S52 双足人形机器人的强化学习训练、Isaac Sim/PhysX 物理验收和
LejuLab-Deploy 仿真/实机部署仓库。当前任务为：机器人正向上四级楼梯，穿过
`1.0 m` 平台，再正向下四级楼梯并稳定落地。

楼梯固定几何：每级长 `0.28 m`、宽 `1.5 m`、高 `0.13 m`；中间平台长
`1.0 m`、宽 `1.5 m`、高 `0.52 m`。

> 本仓库由 LejuLab-Train 的长期实验工作区整理而来。发布仓库不包含原始日志、
> TensorBoard event、失败 rollout、视频和未采用模型，只保留可运行代码、每轮奖励
> 配置与最终曲线，以及经过真实物理联评后采用的 checkpoint。

## 当前模型

当前采用模型为 `checkpoints/kuavo_s53_stairs/model_92150.pt`：

| 指标 | 固定 seed=131 物理验收结果 |
| --- | ---: |
| 完整上楼、平台、四级正向下楼、落地 | 通过 |
| 中途重置 | 0 |
| 最坏脚尖-突缘净空 | 2.410 mm |
| 同源刚体净空 cost | 0.0000416 |
| 足底滑移 p95 | 0.4329 m/s |
| 峰值接触力 | 968.7 N |
| 足底不完整支撑比例 | 19.12% |

SHA-256：

```text
1a4902952951da68a80d23522a5d3b7004bb0cc46f844f54f90313c6bc2f7b0f
```

最终训练 checkpoint 不等于最优模型。`model_92739.pt` 虽然总回报更高，但最坏
净空退化到 `0.862 mm`，因此没有保留。

## S52 迁移状态

S53 `model_92099.pt` 只作为 S52 的只读 warm-start/teacher，不直接视为 S52 最终
策略。受保护副本位于 `checkpoints/kuavo_s52_stairs/model_92099.pt`：

```text
6483ff66456f1e218713f228114a89b6bd5688d94d4c2612ebc247375812f0f4
```

截至 v11 开始前，验收状态如下：

| 项目 | 状态 |
| --- | --- |
| 策略关节 / 仿真关节 | 27 / 29，逐项校验通过 |
| policy observation / action | 148 / 27，校验通过 |
| S52 Isaac Lab 站立与行走 | 通过 |
| S52 MuJoCo 站立与行走 | 通过 |
| S52 Isaac Lab 完整楼梯 | 尚未通过；v9/v10 在首次上楼阶段失败 |
| S52 MuJoCo 完整楼梯 | 未开始，等待 Lab 固定物理验收 |
| 域随机化 | 暂停，等待 Lab 与 MuJoCo 名义参数均通过 |

当前 v11 `ActualReplay` 使用 55% 帧零起点和 45% teacher 实际状态回放起点，目标是
修复 v9/v10 暴露的首次上楼状态分布偏移。详细差异、曲线和失败原因见 `docs/` 与
`experiments/rounds/s52_*`。只有 Lab 中完整通过上楼、平台、四级正向下楼和落地后，
才进入 MuJoCo 楼梯验证；两侧名义物理均通过后，再分阶段加入摩擦、质量、惯量、
执行器延迟和观测噪声随机化。

## 软件栈

### 训练与 Isaac Sim

- Ubuntu 22.04 LTS；
- NVIDIA GPU，驱动需满足 Isaac Sim 4.5.0 要求；
- CUDA 由 Isaac Sim/Conda 环境提供；
- Python 3.10；
- Isaac Sim 4.5.0；
- Isaac Lab 2.1；
- RSL-RL（Isaac Lab 2.1 自带版本）；
- PyTorch、Gymnasium、TensorBoard、PyYAML。

### Mujoco 仿真与实机

- [LejuLab-Deploy](https://github.com/LejuRobotics/LejuLab-Deploy)；
- ROS Noetic；
- Catkin；
- CycloneDDS/iceoryx；
- ONNX Runtime 或 OpenVINO；
- Kuavo S52 部署侧使用 `ROBOT_VERSION=52`；S53 checkpoint 只作 teacher。

训练仓库本身不会启动或修改主机 ROS 链路。实机部署脚本默认 `dry_run=true`，避免
误连接真实机器人。

## 目录结构

```text
leju_rl/
├── checkpoints/kuavo_s53_stairs/  # 仅保留采用的 model_92150.pt
├── checkpoints/kuavo_s52_stairs/  # 仅保留受保护的 model_92099 teacher
├── config/
│   ├── train_stairs.yaml           # 训练、奖励、CMDP 和课程参数
│   ├── sim.yaml                    # Isaac/PhysX 固定种子验收参数
│   └── lab.yaml                    # Mujoco/实机部署参数与安全限制
├── docs/                           # S52/S53 差异、验收状态与迁移研究
├── experiments/rounds/             # 每轮只有配置、曲线和采用/失败结论
├── scripts/
│   ├── setup_env.sh                # 安装训练环境
│   ├── activate_env.sh             # 激活可移植运行环境
│   ├── train.sh                    # YAML 驱动的一键训练
│   ├── eval_sim.sh                 # 无界面真实物理 rollout + 双视图
│   ├── export_policy.sh            # 导出 ONNX/JIT
│   └── deploy_lab.sh               # 部署到 LejuLab-Deploy，默认 dry-run
└── source/leju_robot/              # Isaac Lab 扩展、机器人资产和任务代码
```

## 安装

### 1. 基础环境

先按 Isaac Lab 官方文档安装 Isaac Sim 4.5.0 与 Isaac Lab 2.1，并创建 Python 3.10
环境。模型、机器人网格和动作数据使用 Git LFS 管理，首次克隆必须拉取 LFS 对象：

```bash
git lfs install
git clone git@github.com:expolrer/leju_rl.git
cd leju_rl
git lfs pull
```

服务器已存在环境时可直接指定环境名：

```bash
cd /home/$USER/hhw/leju_rl
CONDA_ENV=lejulab_isaac bash scripts/setup_env.sh
```

脚本执行：

1. 激活 `${HOME}/miniconda3` 下的 Conda 环境；
2. 以 editable 模式安装 `source/leju_robot`；
3. 安装 YAML、曲线分析和 TensorBoard 工具依赖。

Conda 不在默认路径时：

```bash
CONDA_SH=/opt/miniconda3/etc/profile.d/conda.sh \
CONDA_ENV=lejulab_isaac \
bash scripts/setup_env.sh
```

### 2. 激活环境

```bash
source scripts/activate_env.sh
```

该脚本不写死用户名和仓库路径，会根据当前仓库与 Conda site-packages 自动构造
`PYTHONPATH`。Omniverse 缓存默认位于仓库 `.omni_home/`，不会污染用户主目录。

### 3. 自检

```bash
python scripts/list_envs.py | grep ContactGatedMarginCMDP
python scripts/list_envs.py | grep ActualReplay
python -m py_compile scripts/run_config.py
python scripts/run_config.py train config/train_stairs.yaml --dry-run
python scripts/run_config.py sim config/sim.yaml --dry-run
```

## 一键训练

正式训练：

```bash
./scripts/train.sh config/train_stairs.yaml
```

S52 v11 实际状态回放短预检：

```bash
./scripts/train.sh config/train_s52_actual_replay_v11.yaml
```

该配置固定使用 `model_92099.pt` 作为只读 warm-start/teacher，并重置优化器；不会把
teacher 直接标记为 S52 最终策略。

32 环境、2 iteration 冒烟测试：

```bash
cp config/train_stairs.yaml config/local_smoke.yaml
# 将 local_smoke.yaml 中 num_envs 改为 32、max_iterations 改为 2、run_name 改为唯一名称
./scripts/train.sh config/local_smoke.yaml
```

128 环境短预检：

```bash
cp config/train_stairs.yaml config/local_preflight.yaml
# 将 num_envs 改为 128，max_iterations 设置为 40-120
./scripts/train.sh config/local_preflight.yaml
```

`config/local*.yaml` 已被 `.gitignore` 排除，不会把机器专用路径或临时实验参数提交到
仓库。

## YAML 奖励配置

所有常用训练参数集中在 `config/train_stairs.yaml`，每个字段旁有中文注释。

### 普通奖励

| YAML 字段 | 作用 |
| --- | --- |
| `motion_feet_position` | 双脚参考落点跟踪，维持已有完整下楼能力 |
| `motion_feet_velocity` | 双脚摆动与落地速度跟踪 |
| `spatial_riser_corridor` | 惩罚摆脚进入梯级突缘危险空间 |
| `pre_touchdown_soft_landing` | 降低触地前垂向速度和落地冲击 |
| `feet_slide_velocity` | 限制接触阶段足底水平滑移 |
| `feet_contact_force` | 限制过大接触力 |
| `post_touchdown_full_sole` | 预留的落地后短窗口完整足底支撑约束 |

### 突缘净空 CMDP

`contact_gated_clearance` 不作为普通奖励叠加，而由约束优化器单独处理：

- `require_low_contact_for_swing=true`：只有低接触力摆脚才激活，避免错误惩罚承重脚；
- `safety_distance=0.006`：脚掌刚体进入 6 mm 走廊后连续增加风险；
- `hard_distance=0.002`：2 mm 内为硬危险区域；
- `cost_budget`：允许的平均 CMDP 预算；
- PID-Lagrangian 的 P/I/D 增益动态调节约束强度。

不要只提高稀疏最小净空惩罚。此前实验表明，这会让总回报上升但最坏净空反而下降。

### Teacher trust region

`teacher.action_coef` 与 `teacher.kl_coef` 将微调策略限制在安全教师附近。当前教师为
`model_92150.pt`。`reset_optimizer=true` 表示保留模型与归一化状态、重新初始化优化器，
适合修改奖励后的保守迭代。

### 固定场景课程

在固定楼梯尚未跨多个 seed 连续安全通过前，以下选项必须保持 `false`：

- `randomize_stair_geometry`；
- `randomize_friction`；
- `randomize_dynamics`。

先解决固定几何的最坏突缘净空、滑移、冲击和足底支撑，再进入 sim2real 随机化。

## Isaac Sim 真实物理验收

```bash
./scripts/eval_sim.sh config/sim.yaml
```

脚本会：

1. 在无界面 Isaac/PhysX 环境加载 checkpoint；
2. 从运动帧 0 开始执行完整上楼、平台、四级下楼和落地；
3. 输出包含机器人刚体、关节、接触、环境网格和净空指标的 NPZ；
4. 从真实 rollout 生成固定 `model_29999` 双视图 MP4。

验收阈值写在 `config/sim.yaml`。模型必须完整通关、零重置，且最坏净空、同源 cost、
滑移、冲击和足底支撑均不得相对当前安全基线严重退化。不能用脱离物理的关节插值动画
代替验收。

## 导出策略

```bash
./scripts/export_policy.sh
```

输出：

```text
exported/model_92150/policy.onnx
exported/model_92150/policy.pt
```

导出后应在相同观测顺序、动作顺序、归一化参数和 50 Hz 控制周期下比较 Isaac Sim 与
部署侧输出。ONNX 文件是构建产物，不提交到 Git。

## Mujoco 仿真部署

先安装 LejuLab-Deploy：

```bash
cd /home/$USER/hhw
git clone https://github.com/LejuRobotics/LejuLab-Deploy.git
cd LejuLab-Deploy
source installed/setup.bash
catkin build
```

编辑 `config/lab.yaml`，保持 `dry_run: true`，然后：

```bash
cd /home/$USER/hhw/leju_rl
./scripts/export_policy.sh
./scripts/deploy_lab.sh config/lab.yaml
```

脚本只复制并校验模型，不启动 ROS。随后按 LejuLab-Deploy 文档启动 Mujoco：

```bash
cd /home/$USER/hhw/LejuLab-Deploy
source devel/setup.bash
export ROBOT_VERSION=52
roslaunch leju_launch load_mujoco_sim.launch
```

确认观测、动作、关节顺序、控制频率和 action scale 与 Isaac 侧一致后，再接入 RL
controller。Mujoco 必须先完成平台切换、四级下楼和跌倒恢复测试。

## 实机部署

实机测试前必须具备：吊架或防跌保护、机械/软件急停、关节限位、低增益首测、空旷区域、
硬件负责人现场确认。默认配置不会连接实机。

1. 在 `config/lab.yaml` 中复核模型路径、50 Hz、action/Kp/Kd scale 和关节增量限制；
2. 先保持 `dry_run: true` 执行部署脚本；
3. 在 Mujoco 完整通过后，将 `dry_run` 改为 `false`；
4. 明确确认急停后运行：

```bash
./scripts/deploy_lab.sh config/lab.yaml --confirm-real-robot
```

实机 ROS/DDS 域必须由部署仓库管理。本仓库不会修改 `ROS_MASTER_URI`、
`CYCLONEDDS_URI` 或服务器与其他真机的现有链路。

## 每轮实验归档

每轮结束后只保留：奖励配置快照、`training_overview.png`、
`all_reward_terms.png`、可选 `safety_diagnostics.png` 和 `training_summary.json`。

不保留：

- 未采用的 `model_*.pt`；
- TensorBoard event；
- 外部训练日志；
- rollout NPZ；
- MP4；
- smoke/preflight 临时目录；
- 源文件备份。

当前采用模型是唯一例外，位于 `checkpoints/`。详细规则见 `experiments/README.md`。

## 训练结论

1. 总回报不是安全指标，最终 checkpoint 不能默认最优；
2. 净空 cost 必须与物理验收使用同一刚体点集和接触相位定义；
3. 支撑脚与摆动脚必须先按真实接触模式分离；
4. 0.28 m 短踏面对脚掌俯仰和摆脚突缘净空非常敏感；
5. 下一阶段只在确认落地后的短窗口改善完整足底支撑，不能侵入空中摆腿窗口；
6. 固定几何稳定后，才加入台阶、摩擦和动力学随机化构建 sim2real 鲁棒性。

## 许可证与上游

代码沿用原仓库许可证，见 `LICENCE`。基础训练框架来源于 LejuLab-Train；部署接口参照
[LejuLab-Deploy](https://github.com/LejuRobotics/LejuLab-Deploy)。Isaac Sim、Isaac Lab、
RSL-RL、ROS、MuJoCo 和机器人资产分别受其各自许可证约束。
