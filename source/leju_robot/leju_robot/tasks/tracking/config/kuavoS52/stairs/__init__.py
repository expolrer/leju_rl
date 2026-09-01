import gymnasium as gym


gym.register(
    id="Tracking-Stairs-Transfer-KuavoS52",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS52StairsTransferEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS52StairsTransferPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-Transfer-KuavoS52-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS52StairsTransferEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS52StairsTransferPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-PhaseAligned-KuavoS52",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS52StairsPhaseAlignedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS52StairsPhaseAlignedPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-PhaseAligned-KuavoS52-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS52StairsPhaseAlignedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS52StairsPhaseAlignedPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-SoftPlateau-KuavoS52",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS52StairsSoftPlateauEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS52StairsSoftPlateauPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-SoftPlateau-KuavoS52-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS52StairsSoftPlateauEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS52StairsSoftPlateauPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-SoftPlateauTrust-KuavoS52",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS52StairsSoftPlateauTrustEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS52StairsSoftPlateauTrustPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-SoftPlateauTrust-KuavoS52-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS52StairsSoftPlateauTrustEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS52StairsSoftPlateauTrustPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-ContactPhase-KuavoS52",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS52StairsContactPhaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS52StairsContactPhasePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-ContactPhase-KuavoS52-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS52StairsContactPhaseEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS52StairsContactPhasePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-ScheduledFoothold-KuavoS52",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS52StairsScheduledFootholdEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS52StairsScheduledFootholdPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-ScheduledFoothold-KuavoS52-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS52StairsScheduledFootholdEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS52StairsScheduledFootholdPPORunnerCfg",
    },
)
