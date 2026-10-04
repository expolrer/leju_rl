from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ..stairs_finite_memory_composite.tracking_env_cfg import (
    KuavoS52FiniteMemoryCompositeRewardsCfg,
    KuavoS52StairsFiniteMemoryCompositeEnvCfg,
)
from . import rewards as local_rewards


FEET = ["leg_l6_link", "leg_r6_link"]


@configclass
class KuavoS52TerminalContactTimeGateRewardsCfg(
    KuavoS52FiniteMemoryCompositeRewardsCfg
):
    terminal_contact_time_gate = RewTerm(
        func=local_rewards.terminal_contact_time_gate_reward,
        weight=0.14,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "body_names": FEET,
            "frame_start": 1160,
            "target_frame": 1280,
            "frame_end": 1340,
            "forward_std": 0.12,
            "lateral_std": 0.08,
            "height_std": 0.05,
            "velocity_std": 0.22,
            "stable_speed_std": 0.10,
            "max_desired_speed": 0.45,
            "contact_force_threshold": 40.0,
            "touchdown_window_s": 0.12,
        },
    )


@configclass
class KuavoS52StairsTerminalContactTimeGateEnvCfg(
    KuavoS52StairsFiniteMemoryCompositeEnvCfg
):
    rewards: KuavoS52TerminalContactTimeGateRewardsCfg = (
        KuavoS52TerminalContactTimeGateRewardsCfg()
    )


@configclass
class KuavoS52StairsTerminalContactTimeGateEnvCfg_PLAY(
    KuavoS52StairsTerminalContactTimeGateEnvCfg
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
