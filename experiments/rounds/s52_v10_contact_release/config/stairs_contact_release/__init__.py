import gymnasium as gym


gym.register(
    id="Tracking-Stairs-ContactRelease-KuavoS52",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS52StairsContactReleaseEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS52StairsContactReleasePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-ContactRelease-KuavoS52-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS52StairsContactReleaseEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS52StairsContactReleasePPORunnerCfg"
        ),
    },
)
