# S52 楼梯训练归档

截至 2026-10-04，S52 的 Lab/MuJoCo 站立与行走已验证，完整楼梯与 MuJoCo 楼梯部署未验收。公开记录仅保留实际任务/训练配置、完整奖励曲线数据、物理验收与失败分析；**不上传失败候选权重或视频**。

| 轮次 | 配置与源码 | 曲线 | 物理验收与结论 |
| --- | --- | --- | --- |
| v67 FutureContactResidual | [任务配置](v67_future_contact_residual/tracking_env_cfg.py)、[PPO 配置](v67_future_contact_residual/ppo_cfg.py)、[观测](v67_future_contact_residual/observations.py)、[策略](v67_future_contact_residual/residual_contact_actor_critic.py) | [CSV](v67_future_contact_residual/all_tensorboard_scalars.csv)、[SVG](v67_future_contact_residual/reward_and_safety_curves.svg)、[标量汇总](v67_future_contact_residual/all_tensorboard_scalar_summary.json) | [五候选路线验收](v67_future_contact_residual/absolute_route_rollout_comparison.json)、[失败报告](v67_future_contact_residual/FAILURE_REPORT_ZH.md) |
| v68 FutureContactPhaseResidual | [任务配置](v68_future_contact_phase_residual/tracking_env_cfg.py)、[PPO 配置](v68_future_contact_phase_residual/ppo_cfg.py)、[观测](v68_future_contact_phase_residual/observations.py)、[策略](v68_future_contact_phase_residual/residual_contact_actor_critic.py) | [CSV](v68_future_contact_phase_residual/all_tensorboard_scalars.csv)、[SVG](v68_future_contact_phase_residual/reward_and_safety_curves.svg)、[标量汇总](v68_future_contact_phase_residual/all_tensorboard_scalar_summary.json) | [最终模型路线验收](v68_future_contact_phase_residual/absolute_route_rollout_comparison.json)、[失败报告](v68_future_contact_phase_residual/FAILURE_REPORT_ZH.md) |

完整状态、受保护模型用途和 SHA-256 见 [S52 楼梯状态](../../docs/S52_STAIRS_STATUS_20261004_ZH.md)。原始 TensorBoard PNG 与 Isaac rollout NPZ/JSON 保留在 6 服务器各轮 `analysis/s52_transfer/training_records`；公开 CSV/SVG 可以直接复现曲线，避免将未通过的权重误认为可部署模型。
