from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from leju_robot.tasks.tracking import mdp

from ..stairs_reference_adaptation.tracking_env_cfg import (
    KuavoS52ReferenceAdaptationRewardsCfg,
    KuavoS52StairsReferenceAdaptationEnvCfg,
)


# Four ascent contacts plus the final step onto the platform. These windows are
# extracted from the retargeted S52 reference and shared by Train and Play.
S52_ASCENT_SWING_RANGES = (
    ((84, 148), (207, 267), (326, 377)),
    ((145, 207), (266, 329)),
)


@configclass
class KuavoS52AscentPrefixRewardsCfg(KuavoS52ReferenceAdaptationRewardsCfg):
    """Dense swing, touchdown, and stance credit for the initial ascent."""

    ascent_swing_path = RewTerm(
        func=mdp.motion_scheduled_swing_path_tracking_exp,
        weight=5.0,
        params={
            "command_name": "motion",
            "horizontal_std": 0.055,
            "vertical_std": 0.045,
            "foot_frame_ranges": S52_ASCENT_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    ascent_future_foothold_xy = RewTerm(
        func=mdp.motion_scheduled_future_foothold_xy_tracking_exp,
        weight=2.0,
        params={
            "command_name": "motion",
            "std": 0.070,
            "foot_frame_ranges": S52_ASCENT_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    ascent_swing_height_shortage = RewTerm(
        func=mdp.motion_scheduled_swing_height_shortage_penalty,
        weight=-2.5,
        params={
            "command_name": "motion",
            "height_scale": 0.050,
            "allowed_below": 0.012,
            "foot_frame_ranges": S52_ASCENT_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    ascent_swing_early_contact = RewTerm(
        func=mdp.motion_scheduled_swing_early_contact_penalty,
        weight=-1.5,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "contact_force_threshold": 40.0,
            "contact_force_scale": 100.0,
            "release_grace_frames": 4,
            "touchdown_grace_frames": 5,
            "foot_frame_ranges": S52_ASCENT_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    ascent_touchdown_position = RewTerm(
        func=mdp.motion_scheduled_touchdown_position_tracking_exp,
        weight=4.0,
        params={
            "command_name": "motion",
            "horizontal_std": 0.055,
            "vertical_std": 0.040,
            "touchdown_window": 12,
            "foot_frame_ranges": S52_ASCENT_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    ascent_touchdown_contact_loss = RewTerm(
        func=mdp.motion_scheduled_touchdown_contact_loss_penalty,
        weight=-0.75,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "support_force_threshold": 60.0,
            "touchdown_window": 12,
            "foot_frame_ranges": S52_ASCENT_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    ascent_opposite_stance_velocity = RewTerm(
        func=mdp.motion_scheduled_opposite_stance_velocity_penalty,
        weight=-1.0,
        params={
            "command_name": "motion",
            "speed_scale": 0.16,
            "foot_frame_ranges": S52_ASCENT_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    ascent_opposite_stance_contact_loss = RewTerm(
        func=mdp.motion_scheduled_opposite_stance_contact_loss_penalty,
        weight=-0.75,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "support_force_threshold": 60.0,
            "foot_frame_ranges": S52_ASCENT_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    ascent_pelvis_under_height = RewTerm(
        func=mdp.motion_frame_ranges_body_under_reference_height_penalty,
        weight=-3.0,
        params={
            "command_name": "motion",
            "height_scale": 0.080,
            "allowed_below": 0.020,
            "frame_ranges": [(80, 390)],
            "body_names": ["base_link"],
        },
    )


@configclass
class KuavoS52StairsAscentPrefixEnvCfg(KuavoS52StairsReferenceAdaptationEnvCfg):
    rewards: KuavoS52AscentPrefixRewardsCfg = KuavoS52AscentPrefixRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Every episode starts from frame zero. This makes mean episode length a
        # real continuous-prefix metric instead of a random-local-start metric.
        self.commands.motion.zero_start_fraction = 1.0
        self.commands.motion.terrain_focus_fraction = 0.0
        # The base sampler runs before the all-zero-start override. Keep a
        # nonzero uniform floor so its initial all-zero failure histogram is
        # still a valid probability distribution.
        self.commands.motion.adaptive_uniform_ratio = 0.10

        # Allow dynamics adaptation while retaining model_92099 as a prior.
        self.rewards.reference_anchor_position_l1.weight = -1.0
        self.rewards.motion_feet_pos.weight = 2.5
        self.rewards.motion_feet_vel.weight = 0.5
        self.rewards.feet_slide_vel.weight = -0.50
        self.rewards.tgmp_contact_force.weight = -0.0025
        self.rewards.tgmp_contact_force.params["threshold"] = 450.0

        self.terminations.anchor_pos.params["threshold"] = 0.42
        self.terminations.ee_body_pos.params["threshold"] = 0.50
        self.terminations.anchor_ori.params["threshold"] = 0.85


@configclass
class KuavoS52StairsAscentPrefixEnvCfg_PLAY(KuavoS52StairsAscentPrefixEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 5.0
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
