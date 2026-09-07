# 训练轮次归档规则

`experiments/rounds/<版本>/` 每轮只允许包含：

- `reward_config.yaml` 或 `reward_config.py`：该轮实际生效的奖励、约束配置；
- `reward_functions.py`：仅在该轮使用自定义奖励实现时保留；
- `training_objective.yaml` 或 `training_objective.py`：SFT/adapter 轮的实际损失定义；
- `reward_curve.png` 或 `all_reward_terms.png`：训练结束后生成的奖励/目标函数曲线。

失败轮不得提交 checkpoint、TensorBoard event、训练日志、回放、rollout、抽帧、统计 JSON
或失败报告；这些完整证据只留在训练服务器。仓库中的训练轮次只承担“奖励函数/训练目标 +
对应曲线”的版本追溯。经过真实物理联评后采用的模型才允许单独保存在 `checkpoints/`。
