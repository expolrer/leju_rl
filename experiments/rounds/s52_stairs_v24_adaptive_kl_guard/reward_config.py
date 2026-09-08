from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from ..stairs.tracking_env_cfg import (
    KuavoS52ScheduledFootholdRewardsCfg,
    KuavoS52StairsScheduledFootholdEnvCfg,
)
from . import commands as local_commands
from . import observations as local_observations
from . import rewards as local_rewards
from . import terminations as local_terminations


FEET = ["leg_l6_link", "leg_r6_link"]


@configclass
class KuavoS52TerminalBrakeRewardsCfg(KuavoS52ScheduledFootholdRewardsCfg):
    absolute_anchor_route = RewTerm(
        func=local_rewards.absolute_anchor_route_pseudo_huber_penalty,
        weight=-0.75,
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
            "body_names": FEET,
            "frame_start": 0,
            "frame_end": 1340,
        },
    )
    platform_absolute_feet = RewTerm(
        func=local_rewards.absolute_feet_route_pseudo_huber_penalty,
        weight=-0.90,
        params={
            "command_name": "motion",
            "horizontal_scale": 0.12,
            "vertical_scale": 0.09,
            "body_names": FEET,
            "frame_start": 700,
            "frame_end": 920,
        },
    )
    late_absolute_feet_route = RewTerm(
        func=local_rewards.absolute_feet_route_pseudo_huber_penalty,
        weight=-0.40,
        params={
            "command_name": "motion",
            "horizontal_scale": 0.14,
            "vertical_scale": 0.10,
            "body_names": FEET,
            "frame_start": 900,
            "frame_end": 1340,
        },
    )
    forward_route_overshoot = RewTerm(
        func=local_rewards.forward_route_overshoot_pseudo_huber_penalty,
        weight=-0.60,
        params={
            "command_name": "motion",
            "allowed_ahead": 0.08,
            "scale": 0.18,
            "frame_start": 550,
        },
    )
    route_velocity_feedback = RewTerm(
        func=local_rewards.route_velocity_feedback_penalty,
        weight=-1.25,
        params={
            "command_name": "motion",
            "position_gain": 0.60,
            "max_position_correction": 0.25,
            "velocity_scale": 0.18,
        },
    )
    platform_stop_speed = RewTerm(
        func=local_rewards.platform_stop_speed_penalty,
        weight=-1.75,
        params={
            "command_name": "motion",
            "linear_scale": 0.12,
            "yaw_scale": 0.30,
            "frame_start": 720,
            "frame_end": 900,
        },
    )
    platform_stationary_feet = RewTerm(
        func=local_rewards.platform_stationary_feet_penalty,
        weight=-0.50,
        params={
            "command_name": "motion",
            "body_names": FEET,
            "speed_scale": 0.10,
            "frame_start": 720,
            "frame_end": 900,
        },
    )
    platform_double_support_hold = RewTerm(
        func=local_rewards.platform_double_support_hold_reward,
        weight=3.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "support_force_threshold": 80.0,
            "position_std": 0.28,
            "speed_std": 0.16,
            "frame_start": 720,
            "frame_end": 900,
        },
    )
    platform_anchor_orientation = RewTerm(
        func=local_rewards.platform_anchor_orientation_reward,
        weight=2.50,
        params={
            "command_name": "motion",
            "orientation_std": 0.30,
            "frame_start": 700,
            "frame_end": 920,
        },
    )
    terminal_double_support_route = RewTerm(
        func=local_rewards.terminal_double_support_route_reward,
        weight=3.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET),
            "support_force_threshold": 80.0,
            "position_std": 0.18,
            "speed_std": 0.20,
            "frame_start": 1260,
        },
    )
    terminal_route_brake = RewTerm(
        func=local_rewards.terminal_route_brake_pseudo_huber_penalty,
        weight=-0.75,
        params={
            "command_name": "motion",
            "allowed_ahead": 0.04,
            "position_scale": 0.10,
            "position_gain": 0.80,
            "max_velocity_correction": 0.25,
            "velocity_scale": 0.12,
            "yaw_scale": 0.25,
            "frame_start": 1190,
            "full_weight_frame": 1240,
            "frame_end": 1340,
        },
    )


@configclass
class KuavoS52StairsAdaptiveKLGuardEnvCfg(KuavoS52StairsScheduledFootholdEnvCfg):
    rewards: KuavoS52TerminalBrakeRewardsCfg = KuavoS52TerminalBrakeRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        motion_cfg = self.commands.motion
        self.commands.motion = local_commands.FastPlateauBridgeMotionCommandCfg(
            resampling_time_range=motion_cfg.resampling_time_range,
            motion_file=motion_cfg.motion_file,
            asset_name=motion_cfg.asset_name,
            anchor_body=motion_cfg.anchor_body,
            body_names=motion_cfg.body_names,
            pose_range=motion_cfg.pose_range,
            velocity_range=motion_cfg.velocity_range,
            joint_position_range=motion_cfg.joint_position_range,
            adaptive_kernel_size=motion_cfg.adaptive_kernel_size,
            adaptive_lambda=motion_cfg.adaptive_lambda,
            adaptive_uniform_ratio=motion_cfg.adaptive_uniform_ratio,
            adaptive_alpha=motion_cfg.adaptive_alpha,
            start_hold_steps=motion_cfg.start_hold_steps,
            end_hold_steps=motion_cfg.end_hold_steps,
            anchor_pos_threshold=motion_cfg.anchor_pos_threshold,
            anchor_ori_threshold=motion_cfg.anchor_ori_threshold,
            debug_vis=False,
            zero_start_fraction=motion_cfg.zero_start_fraction,
            terrain_focus_fraction=motion_cfg.terrain_focus_fraction,
            terrain_focus_frames=motion_cfg.terrain_focus_frames,
            terrain_focus_approach_steps=motion_cfg.terrain_focus_approach_steps,
            phase_sync_start_frame=680,
            phase_sync_end_frame=946,
            phase_sync_trigger_ahead_m=0.08,
            phase_reference_lead_m=0.03,
            phase_sync_max_advance_per_step=15,
        )

        observation_params = {
            "command_name": "motion",
            "position_scales": (0.45, 0.25, 0.25),
            "route_gain": 0.35,
            "route_clip": 1.0,
        }
        self.observations.policy.motion_anchor_ori_b.func = (
            local_observations.motion_anchor_ori_route_error_b
        )
        self.observations.policy.motion_anchor_ori_b.params = observation_params
        self.observations.critic.motion_anchor_ori_b.func = (
            local_observations.motion_anchor_ori_route_error_b
        )
        self.observations.critic.motion_anchor_ori_b.params = observation_params

        self.commands.motion.zero_start_fraction = 0.45
        self.commands.motion.terrain_focus_fraction = 0.50
        self.commands.motion.adaptive_uniform_ratio = 0.05
        self.commands.motion.terrain_focus_frames = (
            650,
            690,
            720,
            740,
            780,
            820,
            860,
            900,
            930,
            980,
            1080,
            1180,
        )
        self.commands.motion.terrain_focus_approach_steps = 30

        self.rewards.motion_global_anchor_pos.weight = 0.65
        self.rewards.motion_global_anchor_pos.params["std"] = 0.30
        self.rewards.motion_forward_pos.weight = 1.0
        self.rewards.motion_forward_pos.params["std"] = 0.24
        self.rewards.motion_anchor_lateral_pos.weight = 0.70
        self.rewards.motion_height.weight = 3.0
        self.rewards.motion_forward_velocity.weight = 0.50
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

        self.terminations.anchor_pos.func = (
            local_terminations.catastrophic_absolute_anchor_route
        )
        self.terminations.anchor_pos.params = {
            "command_name": "motion",
            "horizontal_threshold": 1.80,
            "vertical_threshold": 0.35,
        }
        self.terminations.ee_body_pos.params["threshold"] = 0.70
        self.terminations.anchor_ori.params["threshold"] = 0.95


@configclass
class KuavoS52StairsAdaptiveKLGuardEnvCfg_PLAY(
    KuavoS52StairsAdaptiveKLGuardEnvCfg
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
