from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from ..stairs.tracking_env_cfg import (
    KuavoS52ScheduledFootholdRewardsCfg,
    KuavoS52StairsScheduledFootholdEnvCfg,
)
from . import observations as local_observations
from . import rewards as local_rewards
from . import terminations as local_terminations


@configclass
class KuavoS52RouteObservedRewardsCfg(KuavoS52ScheduledFootholdRewardsCfg):
    """v15 route-aware, non-saturating rewards for the native S52 interface."""

    absolute_anchor_route = RewTerm(
        func=local_rewards.absolute_anchor_route_pseudo_huber_penalty,
        weight=-0.80,
        params={
            "command_name": "motion",
            "forward_scale": 0.20,
            "lateral_scale": 0.12,
            "height_scale": 0.12,
        },
    )
    absolute_feet_route = RewTerm(
        func=local_rewards.absolute_feet_route_pseudo_huber_penalty,
        weight=-0.35,
        params={
            "command_name": "motion",
            "horizontal_scale": 0.16,
            "vertical_scale": 0.12,
            "body_names": ["leg_l6_link", "leg_r6_link"],
            "frame_start": 0,
            "frame_end": 1340,
        },
    )
    late_absolute_feet_route = RewTerm(
        func=local_rewards.absolute_feet_route_pseudo_huber_penalty,
        weight=-0.45,
        params={
            "command_name": "motion",
            "horizontal_scale": 0.14,
            "vertical_scale": 0.10,
            "body_names": ["leg_l6_link", "leg_r6_link"],
            "frame_start": 850,
            "frame_end": 1340,
        },
    )
    forward_route_overshoot = RewTerm(
        func=local_rewards.forward_route_overshoot_pseudo_huber_penalty,
        weight=-0.65,
        params={
            "command_name": "motion",
            "allowed_ahead": 0.08,
            "scale": 0.18,
            "frame_start": 550,
        },
    )
    absolute_route_corridor = RewTerm(
        func=local_rewards.absolute_route_corridor_reward,
        weight=1.0,
        params={
            "command_name": "motion",
            "horizontal_std": 0.24,
            "vertical_std": 0.14,
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
class KuavoS52StairsRouteObservedEnvCfg(KuavoS52StairsScheduledFootholdEnvCfg):
    """Native-S52 route task with a shape-compatible horizontal error signal."""

    rewards: KuavoS52RouteObservedRewardsCfg = KuavoS52RouteObservedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        route_observation_params = {
            "command_name": "motion",
            "position_scales": (0.45, 0.25, 0.25),
            "route_gain": 0.30,
            "route_clip": 1.0,
        }
        self.observations.policy.motion_anchor_ori_b.func = (
            local_observations.motion_anchor_ori_route_error_b
        )
        self.observations.policy.motion_anchor_ori_b.params = route_observation_params
        self.observations.critic.motion_anchor_ori_b.func = (
            local_observations.motion_anchor_ori_route_error_b
        )
        self.observations.critic.motion_anchor_ori_b.params = route_observation_params

        self.commands.motion.zero_start_fraction = 0.50
        self.commands.motion.terrain_focus_fraction = 0.45
        self.commands.motion.adaptive_uniform_ratio = 0.05
        self.commands.motion.terrain_focus_frames = (
            500,
            650,
            780,
            900,
            980,
            1080,
            1180,
            1280,
        )
        self.commands.motion.terrain_focus_approach_steps = 30

        # Local pose terms remain useful but can no longer dominate world route.
        self.rewards.motion_global_anchor_pos.weight = 0.75
        self.rewards.motion_global_anchor_pos.params["std"] = 0.30
        self.rewards.motion_forward_pos.weight = 1.25
        self.rewards.motion_forward_pos.params["std"] = 0.24
        self.rewards.motion_anchor_lateral_pos.weight = 0.75
        self.rewards.motion_height.weight = 3.0
        self.rewards.motion_forward_velocity.weight = 0.75
        self.rewards.motion_body_pos.weight = 0.45
        self.rewards.motion_body_ori.weight = 0.55
        self.rewards.scheduled_swing_path.weight = 4.0
        self.rewards.scheduled_future_foothold_xy.weight = 1.5
        self.rewards.scheduled_touchdown_position.weight = 3.0
        self.rewards.feet_slide_vel.weight = -0.40
        self.rewards.tgmp_contact_force.weight = -0.002
        self.rewards.tgmp_contact_force.params["threshold"] = 450.0
        self.rewards.action_rate_l2.weight = -0.010
        self.rewards.action_smoothness_l2.weight = -0.020

        self.terminations.anchor_pos.func = local_terminations.bad_absolute_anchor_route
        self.terminations.anchor_pos.params = {
            "command_name": "motion",
            "horizontal_threshold": 0.55,
            "vertical_threshold": 0.35,
        }
        self.terminations.ee_body_pos.params["threshold"] = 0.70
        self.terminations.anchor_ori.params["threshold"] = 0.95


@configclass
class KuavoS52StairsRouteObservedEnvCfg_PLAY(KuavoS52StairsRouteObservedEnvCfg):
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
