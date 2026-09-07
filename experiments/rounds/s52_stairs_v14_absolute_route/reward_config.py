from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from ..stairs.tracking_env_cfg import (
    KuavoS52ScheduledFootholdRewardsCfg,
    KuavoS52StairsScheduledFootholdEnvCfg,
)
from . import rewards as local_rewards


@configclass
class KuavoS52AbsoluteRouteRewardsCfg(KuavoS52ScheduledFootholdRewardsCfg):
    """v14 dense world-frame route credit for the native S52 interface."""

    absolute_anchor_route = RewTerm(
        func=local_rewards.absolute_anchor_route_huber_penalty,
        weight=-1.25,
        params={
            "command_name": "motion",
            "forward_scale": 0.16,
            "lateral_scale": 0.10,
            "height_scale": 0.10,
            "cap": 6.0,
        },
    )
    absolute_feet_route = RewTerm(
        func=local_rewards.absolute_feet_route_huber_penalty,
        weight=-0.60,
        params={
            "command_name": "motion",
            "horizontal_scale": 0.14,
            "vertical_scale": 0.10,
            "body_names": ["leg_l6_link", "leg_r6_link"],
            "frame_start": 0,
            "frame_end": 1340,
            "cap": 6.0,
        },
    )
    late_absolute_feet_route = RewTerm(
        func=local_rewards.absolute_feet_route_huber_penalty,
        weight=-0.75,
        params={
            "command_name": "motion",
            "horizontal_scale": 0.12,
            "vertical_scale": 0.09,
            "body_names": ["leg_l6_link", "leg_r6_link"],
            "frame_start": 900,
            "frame_end": 1340,
            "cap": 6.0,
        },
    )
    forward_route_overshoot = RewTerm(
        func=local_rewards.forward_route_overshoot_penalty,
        weight=-1.50,
        params={
            "command_name": "motion",
            "allowed_ahead": 0.06,
            "scale": 0.12,
            "frame_start": 650,
            "cap": 6.0,
        },
    )
    terminal_double_support_route = RewTerm(
        func=local_rewards.terminal_double_support_route_reward,
        weight=3.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "support_force_threshold": 80.0,
            "position_std": 0.18,
            "speed_std": 0.20,
            "frame_start": 1260,
        },
    )


@configclass
class KuavoS52StairsAbsoluteRouteEnvCfg(KuavoS52StairsScheduledFootholdEnvCfg):
    """Native-S52 fixed-geometry task that rejects frame-only route completion."""

    rewards: KuavoS52AbsoluteRouteRewardsCfg = KuavoS52AbsoluteRouteRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Preserve the 27-D action and 148-D observation interfaces. The parent
        # task already uses S52-native action scale, PD, effort, and sole geometry.
        self.commands.motion.zero_start_fraction = 0.65
        self.commands.motion.terrain_focus_fraction = 0.30
        self.commands.motion.adaptive_uniform_ratio = 0.05
        self.commands.motion.terrain_focus_frames = (
            700,
            820,
            940,
            1040,
            1140,
            1240,
        )
        self.commands.motion.terrain_focus_approach_steps = 35

        self.rewards.motion_global_anchor_pos.weight = 1.50
        self.rewards.motion_global_anchor_pos.params["std"] = 0.28
        self.rewards.motion_forward_pos.weight = 2.50
        self.rewards.motion_forward_pos.params["std"] = 0.20
        self.rewards.motion_anchor_lateral_pos.weight = 1.25
        self.rewards.motion_height.weight = 3.50
        self.rewards.motion_forward_velocity.weight = 1.0
        self.rewards.motion_body_pos.weight = 0.65
        self.rewards.motion_body_ori.weight = 0.65
        self.rewards.scheduled_swing_path.weight = 4.0
        self.rewards.scheduled_future_foothold_xy.weight = 1.5
        self.rewards.scheduled_touchdown_position.weight = 3.0
        self.rewards.feet_slide_vel.weight = -0.40
        self.rewards.tgmp_contact_force.weight = -0.002
        self.rewards.tgmp_contact_force.params["threshold"] = 450.0
        self.rewards.action_rate_l2.weight = -0.010
        self.rewards.action_smoothness_l2.weight = -0.020

        # Keep late-route failures observable, but do not permit multi-meter v13
        # overshoot to count as completion.
        self.terminations.anchor_pos.params["threshold"] = 0.75
        self.terminations.ee_body_pos.params["threshold"] = 0.70
        self.terminations.anchor_ori.params["threshold"] = 0.95


@configclass
class KuavoS52StairsAbsoluteRouteEnvCfg_PLAY(KuavoS52StairsAbsoluteRouteEnvCfg):
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
