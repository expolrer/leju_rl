# 训练轮次归档规则

`experiments/rounds/<版本>/` 每轮只允许包含：

- `reward_config.yaml`：该轮奖励、约束和优化器配置快照；
- `training_overview.png`：总回报、episode length 与优化器曲线；
- `all_reward_terms.png`：该轮全部奖励项曲线；
- `safety_diagnostics.png`：存在安全指标时的分阶段曲线；
- `training_summary.json`：曲线统计与结论。

不得提交 TensorBoard event、训练日志、回放视频、临时 rollout 或未采用 checkpoint。当前采用模型单独保存在 `checkpoints/`。

