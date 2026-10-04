from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..stairs_finite_memory_composite.tracking_env_cfg import (
    KuavoS52FiniteMemoryCompositeRewardsCfg,
    KuavoS52StairsFiniteMemoryCompositeEnvCfg,
)
from . import rewards as local_rewards


@configclass
class KuavoS52TerminalFutureContactRewardsCfg(
    KuavoS52FiniteMemoryCompositeRewardsCfg
):
    terminal_future_contact_residual = RewTerm(
        func=local_rewards.terminal_future_contact_residual_reward,
        weight=0.20,
        params={
            "command_name": "motion",
            "body_names": ["leg_l6_link", "leg_r6_link"],
            "frame_start": 1190,
            "target_frame": 1340,
            "forward_std": 0.10,
            "lateral_std": 0.07,
            "height_std": 0.04,
            "speed_std": 0.12,
        },
    )


@configclass
class KuavoS52StairsTerminalFutureContactEnvCfg(
    KuavoS52StairsFiniteMemoryCompositeEnvCfg
):
    rewards: KuavoS52TerminalFutureContactRewardsCfg = (
        KuavoS52TerminalFutureContactRewardsCfg()
    )


@configclass
class KuavoS52StairsTerminalFutureContactEnvCfg_PLAY(
    KuavoS52StairsTerminalFutureContactEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 5.0
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.commands.motion.zero_start_fraction = 1.0
        self.commands.motion.terrain_focus_fraction = 0.0
        self.commands.motion.adaptive_uniform_ratio = 0.0
