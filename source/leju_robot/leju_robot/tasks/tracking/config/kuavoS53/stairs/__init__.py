import gymnasium as gym


gym.register(
    id="Tracking-Stairs-Step1-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53StairsStep1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS53StairsStep1PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-Step1-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53StairsStep1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS53StairsStep1PPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-Full-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53StairsFullEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS53StairsFullPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-Full-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53StairsFullEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS53StairsFullPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-UpDown-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53StairsUpDownEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS53StairsUpDownPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-UpDown-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53StairsUpDownEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS53StairsUpDownPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-ForwardDown-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53StairsForwardDownEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS53StairsForwardDownPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-ForwardDown-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53StairsForwardDownEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS53StairsForwardDownPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-ForwardUpDown-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53StairsForwardUpDownEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS53StairsForwardUpDownPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-ForwardUpDown-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53StairsForwardUpDownEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS53StairsForwardUpDownPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-ForwardUpDownStable-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53StairsForwardUpDownStableEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS53StairsForwardUpDownStablePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-ForwardUpDownStable-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53StairsForwardUpDownStableEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS53StairsForwardUpDownStablePPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-StepToDown-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53StairsStepToDownEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS53StairsStepToDownPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-StepToDown-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:KuavoS53StairsStepToDownEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:KuavoS53StairsStepToDownPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Stairs-StepToDownGateFixed-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsStepToDownGateFixedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsStepToDownGateFixedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-StepToDownGateFixed-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsStepToDownGateFixedEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsStepToDownGateFixedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-StepToDownNosingSafe-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsStepToDownNosingSafeEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsStepToDownNosingSafePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-StepToDownNosingSafe-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsStepToDownNosingSafeEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsStepToDownNosingSafePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-StepToDownPreserve-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsStepToDownPreserveEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsStepToDownPreservePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-StepToDownPreserve-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsStepToDownPreserveEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsStepToDownPreservePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-StepToDownTgmpFoothold-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsStepToDownTgmpFootholdEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsStepToDownTgmpFootholdPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-StepToDownTgmpFoothold-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsStepToDownTgmpFootholdEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsStepToDownTgmpFootholdPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Tracking-Stairs-TgmpRewardBaseline-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsTgmpRewardBaselineEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsTgmpRewardBaselinePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-TgmpRewardBaseline-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsTgmpRewardBaselineEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsTgmpRewardBaselinePPORunnerCfg"
        ),
    },
)


gym.register(
    id="Tracking-Stairs-TgmpTerrainConditioned-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsTgmpTerrainConditionedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsTgmpTerrainConditionedPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-TgmpTerrainConditioned-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsTgmpTerrainConditionedEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsTgmpTerrainConditionedPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Tracking-Stairs-TgmpRiserSafe-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsTgmpRiserSafeEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsTgmpRiserSafePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-TgmpRiserSafe-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsTgmpRiserSafeEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsTgmpRiserSafePPORunnerCfg"
        ),
    },
)


gym.register(
    id="Tracking-Stairs-MindStepsTact-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsMindStepsTactEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsMindStepsTactPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-MindStepsTact-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsMindStepsTactEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsMindStepsTactPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Tracking-Stairs-PredictiveSweep-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsPredictiveSweepEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsPredictiveSweepPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-PredictiveSweep-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsPredictiveSweepEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsPredictiveSweepPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Tracking-Stairs-SwingAwareSweep-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsSwingAwareSweepEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsSwingAwareSweepPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-SwingAwareSweep-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsSwingAwareSweepEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsSwingAwareSweepPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Tracking-Stairs-TailRiskSweep-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsTailRiskSweepEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsTailRiskSweepPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-TailRiskSweep-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsTailRiskSweepEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsTailRiskSweepPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Tracking-Stairs-ToeBarrier-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsToeBarrierEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsToeBarrierPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-ToeBarrier-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsToeBarrierEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsToeBarrierPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Tracking-Stairs-RunningMinBarrier-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsRunningMinBarrierEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsRunningMinBarrierPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-RunningMinBarrier-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsRunningMinBarrierEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsRunningMinBarrierPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Tracking-Stairs-TimeToRiserCone-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsTimeToRiserConeEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsTimeToRiserConePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-TimeToRiserCone-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsTimeToRiserConeEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsTimeToRiserConePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-SpatialRiserCorridor-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsSpatialRiserCorridorEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsSpatialRiserCorridorPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-SpatialRiserCorridor-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsSpatialRiserCorridorEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsSpatialRiserCorridorPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-ClearanceSoftLanding-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsClearanceSoftLandingEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsClearanceSoftLandingPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-ClearanceSoftLanding-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsClearanceSoftLandingEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsClearanceSoftLandingPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-ConservativeTailReplay-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsConservativeTailReplayEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsConservativeTailReplayPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-ConservativeTailReplay-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsConservativeTailReplayEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsConservativeTailReplayPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-ConstrainedTeacherProjection-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsConstrainedTeacherProjectionEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsConstrainedTeacherProjectionPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-ConstrainedTeacherProjection-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsConstrainedTeacherProjectionEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsConstrainedTeacherProjectionPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Tracking-Stairs-RiserConstraintCaT-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsRiserConstraintCaTEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsRiserConstraintCaTPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-RiserConstraintCaT-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsRiserConstraintCaTEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsRiserConstraintCaTPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-RiserClearanceLagrangian-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsRiserClearanceLagrangianEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsRiserClearanceLagrangianPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-RiserClearanceLagrangian-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsRiserClearanceLagrangianEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsRiserClearanceLagrangianPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-SharedClearanceCMDP-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsSharedClearanceCmdpEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsSharedClearanceCmdpPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-SharedClearanceCMDP-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsSharedClearanceCmdpEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsSharedClearanceCmdpPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-PIDWorstSegmentCMDP-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsPidWorstSegmentCmdpEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsPidWorstSegmentCmdpPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-PIDWorstSegmentCMDP-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsPidWorstSegmentCmdpEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsPidWorstSegmentCmdpPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-CalibratedTailCMDP-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsCalibratedTailCmdpEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsCalibratedTailCmdpPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-CalibratedTailCMDP-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsCalibratedTailCmdpEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsCalibratedTailCmdpPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-ContactGatedCMDP-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsContactGatedCmdpEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsContactGatedCmdpPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-ContactGatedCMDP-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsContactGatedCmdpEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsContactGatedCmdpPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-ContactGatedMarginCMDP-KuavoS53",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsContactGatedMarginCmdpEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsContactGatedMarginCmdpPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Tracking-Stairs-ContactGatedMarginCMDP-KuavoS53-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tracking_env_cfg:KuavoS53StairsContactGatedMarginCmdpEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.ppo_cfg:KuavoS53StairsContactGatedMarginCmdpPPORunnerCfg"
        ),
    },
)
