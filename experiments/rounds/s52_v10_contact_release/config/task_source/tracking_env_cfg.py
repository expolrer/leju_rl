from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from ..stairs_ascent_prefix.tracking_env_cfg import (
    S52_ASCENT_SWING_RANGES,
    KuavoS52AscentPrefixRewardsCfg,
    KuavoS52StairsAscentPrefixEnvCfg,
)
from . import rewards as local_rewards


@configclass
class KuavoS52ContactReleaseRewardsCfg(KuavoS52AscentPrefixRewardsCfg):
    """Non-saturating credit around S52 ascent liftoff and foothold errors."""

    ascent_swing_path_l1 = RewTerm(
        func=local_rewards.scheduled_swing_path_l1_penalty,
        weight=-1.0,
        params={
            "command_name": "motion",
            "horizontal_scale": 0.10,
            "vertical_scale": 0.08,
            "foot_frame_ranges": S52_ASCENT_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    ascent_future_foothold_l1 = RewTerm(
        func=local_rewards.scheduled_future_foothold_l1_penalty,
        weight=-0.75,
        params={
            "command_name": "motion",
            "horizontal_scale": 0.10,
            "foot_frame_ranges": S52_ASCENT_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    ascent_swing_log_contact_force = RewTerm(
        func=local_rewards.scheduled_swing_log_contact_force_penalty,
        weight=-2.5,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "contact_force_threshold": 35.0,
            "contact_force_scale": 120.0,
            "release_grace_frames": 3,
            "touchdown_grace_frames": 5,
            "foot_frame_ranges": S52_ASCENT_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class KuavoS52StairsContactReleaseEnvCfg(KuavoS52StairsAscentPrefixEnvCfg):
    rewards: KuavoS52ContactReleaseRewardsCfg = KuavoS52ContactReleaseRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Preserve full-prefix learning while exposing the failed second swing
        # often enough to learn liftoff and load transfer under S52 dynamics.
        self.commands.motion.zero_start_fraction = 0.55
        self.commands.motion.terrain_focus_fraction = 0.35
        self.commands.motion.adaptive_uniform_ratio = 0.10
        self.commands.motion.terrain_focus_frames = (180, 195, 210, 225, 240)
        self.commands.motion.terrain_focus_approach_steps = 20

        # Exponential terms remain as near-target shaping. The new L1/log terms
        # carry credit when the foot is still far from the collision-free path.
        self.rewards.ascent_swing_path.weight = 1.5
        self.rewards.ascent_future_foothold_xy.weight = 0.75
        self.rewards.ascent_swing_early_contact.weight = -0.30
        self.rewards.ascent_swing_height_shortage.weight = -3.0
        self.rewards.ascent_touchdown_position.weight = 3.0
        self.rewards.ascent_pelvis_under_height.weight = -4.0


@configclass
class KuavoS52StairsContactReleaseEnvCfg_PLAY(
    KuavoS52StairsContactReleaseEnvCfg
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
        self.commands.motion.adaptive_uniform_ratio = 0.10
