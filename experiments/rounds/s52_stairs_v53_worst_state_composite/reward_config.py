from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from ..stairs_route_safety_composite.tracking_env_cfg import (
    KuavoS52RouteSafetyCompositeRewardsCfg,
    KuavoS52StairsRouteSafetyCompositeEnvCfg,
)
from . import rewards as local_rewards


@configclass
class KuavoS52WorstStateCompositeRewardsCfg(
    KuavoS52RouteSafetyCompositeRewardsCfg
):
    worst_state_composite_probe = RewTerm(
        func=local_rewards.worst_state_composite_constraint_probe,
        weight=-1.0e-8,
        params={
            "command_name": "motion",
            "window_start_frame": 800,
            "window_end_frame": 1340,
        },
    )


@configclass
class KuavoS52StairsWorstStateCompositeEnvCfg(
    KuavoS52StairsRouteSafetyCompositeEnvCfg
):
    rewards: KuavoS52WorstStateCompositeRewardsCfg = (
        KuavoS52WorstStateCompositeRewardsCfg()
    )


@configclass
class KuavoS52StairsWorstStateCompositeEnvCfg_PLAY(
    KuavoS52StairsWorstStateCompositeEnvCfg
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
