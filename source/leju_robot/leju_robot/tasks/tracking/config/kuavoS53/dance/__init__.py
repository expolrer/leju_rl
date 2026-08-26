import gymnasium as gym

gym.register(
    id="Tracking-Dance-Flat-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.tracking.agents.rsl_rl_ppo_cfg:KuavoS53DancePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Dance-Flat-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "leju_robot.tasks.tracking.agents.rsl_rl_ppo_cfg:KuavoS53DancePPORunnerCfg",
    },
)
