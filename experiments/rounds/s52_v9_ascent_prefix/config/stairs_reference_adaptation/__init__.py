import gymnasium as gym


gym.register(
    id="Tracking-Stairs-ReferenceAdaptation-KuavoS52",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS52StairsReferenceAdaptationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS52StairsReferenceAdaptationPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-ReferenceAdaptation-KuavoS52-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS52StairsReferenceAdaptationEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS52StairsReferenceAdaptationPPORunnerCfg"
        ),
    },
)
