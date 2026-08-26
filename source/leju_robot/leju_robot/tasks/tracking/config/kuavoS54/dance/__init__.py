import gymnasium as gym

gym.register(
    id="Tracking-Dance-Flat-KuavoS54",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS54FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.tracking.agents.rsl_rl_ppo_cfg:KuavoS54DancePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Dance-Flat-KuavoS54-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS54FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.tracking.agents.rsl_rl_ppo_cfg:KuavoS54DancePPORunnerCfg",
    },
)
