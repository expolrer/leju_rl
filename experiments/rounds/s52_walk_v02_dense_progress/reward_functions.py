import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from leju_robot.tasks.locomotion.velocity.config.kuavoS54.rough_env_cfg import (
    RewardsCfg as KuavoVelocityRewardsCfg,
)

from .kuavoS52 import KuavoS52_CFG
from .rough_env_cfg import KuavoS52RoughEnvCfg


def forward_velocity_progress(
    env,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward commanded forward progress without rewarding stationary policies."""

    command = env.command_manager.get_command(command_name)[:, 0]
    velocity = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0]
    moving = torch.abs(command) > command_threshold
    direction = torch.sign(command)
    ratio = direction * velocity / torch.clamp(torch.abs(command), min=command_threshold)
    return torch.where(moving, torch.clamp(ratio, min=0.0, max=1.0), 0.0)


def forward_velocity_error_l1(
    env,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Apply a non-saturating velocity error on commanded walking environments."""

    command = env.command_manager.get_command(command_name)[:, 0]
    velocity = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0]
    moving = torch.abs(command) > command_threshold
    return torch.where(moving, torch.abs(command - velocity), 0.0)


def backward_velocity_l1(
    env,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize motion opposite to a non-zero forward command."""

    command = env.command_manager.get_command(command_name)[:, 0]
    velocity = env.scene[asset_cfg.name].data.root_lin_vel_b[:, 0]
    moving = torch.abs(command) > command_threshold
    opposite_speed = torch.relu(-torch.sign(command) * velocity)
    return torch.where(moving, opposite_speed, 0.0)


@configclass
class KuavoS52WalkRewardsCfg(KuavoVelocityRewardsCfg):
    forward_velocity_progress = RewTerm(
        func=forward_velocity_progress,
        weight=6.0,
        params={"command_name": "base_velocity", "command_threshold": 0.1},
    )
    forward_velocity_error_l1 = RewTerm(
        func=forward_velocity_error_l1,
        weight=-6.0,
        params={"command_name": "base_velocity", "command_threshold": 0.1},
    )
    backward_velocity_l1 = RewTerm(
        func=backward_velocity_l1,
        weight=-8.0,
        params={"command_name": "base_velocity", "command_threshold": 0.1},
    )


def _disable_morphology_randomization(cfg) -> None:
    """Use nominal S52 physics for the first cross-simulator validation."""

    cfg.events.physics_material = None
    cfg.events.add_base_mass = None
    cfg.events.scale_link_mass = None
    cfg.events.randomize_rigid_body_com = None
    cfg.events.scale_actuator_gains = None
    cfg.events.scale_joint_parameters = None
    cfg.events.push_robot = None
    cfg.events.add_joint_default_pos.params = {
        "asset_cfg": KuavoS52_CFG.preserve_joint_order,
        "pos_distribution_params": (0.0, 0.0),
        "operation": "add",
    }
    cfg.events.reset_base.params = {
        "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }
    cfg.events.reset_robot_joints.params = {
        "position_range": (-0.02, 0.02),
        "velocity_range": (0.0, 0.0),
    }


@configclass
class KuavoS52FlatEnvCfg(KuavoS52RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.Feet_L_scanner = None
        self.scene.Feet_R_scanner = None
        self.observations.critic.height_scan = None
        self.observations.critic.feet_heights = None
        self.curriculum.terrain_levels = None


@configclass
class KuavoS52StandEnvCfg(KuavoS52FlatEnvCfg):
    """Zero-command task used to prove stable S52 contact dynamics."""

    def __post_init__(self):
        super().__post_init__()
        _disable_morphology_randomization(self)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False

        # A stand task must not inherit rewards that ask the robot to step.
        self.rewards.feet_air_time = None
        self.rewards.alternating_contacts = None
        self.rewards.arm_swing = None
        self.rewards.feet_clearance = None
        self.rewards.contact_force = None

        self.rewards.alive.weight = 0.5
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.flat_orientation_l1.weight = -5.0
        self.rewards.base_height_l1.weight = -10.0
        self.rewards.base_height_l1.params["target_height"] = 0.955
        self.rewards.stand_still_without_cmd.weight = -1.5
        self.rewards.feet_on_ground_at_rest.weight = 5.0
        self.rewards.stand_still_base_vel.weight = -5.0
        self.rewards.feet_slide.weight = -0.5
        self.terminations.bad_orientation.params["limit_angle"] = 0.60


@configclass
class KuavoS52StandEnvCfg_PLAY(KuavoS52StandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }


@configclass
class KuavoS52WalkEnvCfg(KuavoS52FlatEnvCfg):
    """Conservative forward-only task used before stairs or randomization."""

    rewards: KuavoS52WalkRewardsCfg = KuavoS52WalkRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _disable_morphology_randomization(self)
        self.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.40)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.05
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False

        self.rewards.track_lin_vel_xy_exp.weight = 6.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.20
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.20
        self.rewards.alive.weight = 0.10
        self.rewards.arm_swing = None
        self.rewards.base_height_l1.weight = -8.0
        self.rewards.base_height_l1.params["target_height"] = 0.945
        self.rewards.flat_orientation_l2.weight = -3.0
        self.rewards.flat_orientation_l1.weight = -3.0
        self.rewards.feet_air_time.weight = 4.0
        self.rewards.alternating_contacts.weight = 3.0
        self.rewards.feet_clearance.weight = 0.5
        self.rewards.feet_slide.weight = -0.5
        self.rewards.contact_force.weight = -0.3
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness_l2.weight = -0.01
        self.rewards.joint_deviation_arms_other.weight = -2.0
        self.terminations.bad_orientation.params["limit_angle"] = 0.60


@configclass
class KuavoS52WalkEnvCfg_PLAY(KuavoS52WalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 60.0
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }
        self.commands.base_velocity.ranges.lin_vel_x = (0.30, 0.30)
