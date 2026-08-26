from __future__ import annotations

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from leju_robot.assets.motion_data import MOTION_DIR
import leju_robot.tasks.tracking.mdp as mdp
from leju_robot.tasks.tracking.config.kuavoS53.dance.tracking_env_cfg import KuavoS53FlatEnvCfg
from leju_robot.tasks.tracking.config.kuavoS54.dance.tracking_env_cfg import (
    RewardsCfg,
    RobotSceneCfg,
    TerminationsCfg,
)


STAIR_HEIGHT = 0.13
STAIR_TREAD = 0.28
STAIR_WIDTH = 1.5
FIRST_RISER_X = 0.14
PLATFORM_LENGTH = 1.0

STEP1_MOTION = f"{MOTION_DIR}/mimic/npz_data/kuavoS53_stairs_step1_medoid_50fps.npz"
FULL_MOTION = f"{MOTION_DIR}/mimic/npz_data/kuavoS53_stairs_full_medoid_50fps.npz"
UPDOWN_MOTION = f"{MOTION_DIR}/mimic/npz_data/kuavoS53_stairs_updown_50fps.npz"
FORWARD_UPDOWN_MOTION = f"{MOTION_DIR}/mimic/npz_data/kuavoS53_stairs_forward_updown_50fps.npz"
FORWARD_UPDOWN_STABLE_MOTION = (
    f"{MOTION_DIR}/mimic/npz_data/kuavoS53_stairs_forward_updown_stable_50fps.npz"
)
STEP_TO_DOWN_MOTION = f"{MOTION_DIR}/mimic/npz_data/kuavoS53_stairs_step_to_down_50fps.npz"
FORWARD_DOWN_MOTION = f"{MOTION_DIR}/mimic/npz_data/kuavoS53_stairs_forward_down_50fps.npz"

# The converter samples [0, duration) and produces 1342 frames from the
# 1343-row source CSV, so the final valid NPZ index is 1341.
MOTION_LAST_FRAME = 1341.0
PRE_DESCENT_PHASE_START = 708.0 / MOTION_LAST_FRAME
PRE_DESCENT_PHASE_END = 742.0 / MOTION_LAST_FRAME
DESCENT_PHASE_START = 743.0 / MOTION_LAST_FRAME
DESCENT_PHASE_END = 1292.0 / MOTION_LAST_FRAME

# Exact frame ranges for the fixed-coordinate, left-first step-to descent.
STEP_TO_MOTION_LAST_FRAME = 1361.0
STEP_TO_PRE_DESCENT_PHASE_START = 728.0 / STEP_TO_MOTION_LAST_FRAME
STEP_TO_PRE_DESCENT_PHASE_END = 762.0 / STEP_TO_MOTION_LAST_FRAME
STEP_TO_DESCENT_PHASE_START = 763.0 / STEP_TO_MOTION_LAST_FRAME
STEP_TO_DESCENT_PHASE_END = 1282.0 / STEP_TO_MOTION_LAST_FRAME
STEP_TO_GATE_FRAMES = (762, 892, 1022, 1152, 1282)
STEP_TO_DOUBLE_FOOT_GATE_RANGES = (
    (728, 762),
    (868, 892),
    (998, 1022),
    (1128, 1152),
    (1258, 1282),
)
STEP_TO_LEFT_SWING_RANGES = ((763, 807), (893, 937), (1023, 1067), (1153, 1197))
STEP_TO_RIGHT_SWING_RANGES = ((823, 867), (953, 997), (1083, 1127), (1213, 1257))
STEP_TO_DESCENT_RANGES = ((763, 1282),)
# The model_113994 physical rollout first became airborne at these frames.
# Restrict nosing shaping to this early-flight window so normal stance and
# intended lower-tread landing contact are not penalized.
STEP_TO_LEFT_NOSING_APPROACH_RANGES = (
    (776, 797),
    (906, 927),
    (1036, 1057),
    (1165, 1187),
)
# Geometry-aware windows extend until the old support edge has cleared the sole.
STEP_TO_LEFT_FOOTHOLD_APPROACH_RANGES = (
    (776, 807),
    (906, 937),
    (1036, 1067),
    (1165, 1197),
)
STEP_TO_DESCENT_RISER_X = (2.26, 2.54, 2.82, 3.10)
STEP_TO_DESCENT_UPPER_HEIGHTS = (0.52, 0.39, 0.26, 0.13)
STEP_TO_DESCENT_LOWER_HEIGHTS = (0.39, 0.26, 0.13, 0.0)
STEP_TO_DESCENT_TREAD_HEIGHTS = (0.52, 0.39, 0.26, 0.13, 0.0)
STEP_TO_GATE_FOOTHOLD_EDGES = (-1.0, 2.26, 2.54, 2.82, 3.10)
STEP_TO_GATE_FOOTHOLD_HEIGHTS = (0.52, 0.39, 0.26, 0.13, 0.0)

# Samples follow the actual S53 URDF collision footprint rather than a centered ankle point.
S53_FOREFOOT_SAMPLES = (
    (0.042, -0.040, -0.062),
    (0.042, 0.040, -0.062),
    (0.100, -0.040, -0.062),
    (0.100, 0.040, -0.062),
    (0.130, -0.040, -0.062),
    (0.130, 0.040, -0.062),
    (0.165, -0.032, -0.062),
    (0.165, 0.032, -0.062),
)
S53_SOLE_SAMPLES = (
    (-0.063, -0.040, -0.062),
    (-0.063, 0.040, -0.062),
    (-0.036, -0.040, -0.062),
    (-0.036, 0.040, -0.062),
    (0.042, -0.045, -0.062),
    (0.042, 0.045, -0.062),
    (0.130, -0.040, -0.062),
    (0.130, 0.040, -0.062),
    (0.165, -0.032, -0.062),
    (0.165, 0.032, -0.062),
)
S53_HEEL_SAMPLES = S53_SOLE_SAMPLES[:4]

# Full fixed task geometry in coordinates relative to each environment origin.
T_GMP_RISER_X = (0.14, 0.42, 0.70, 0.98, 2.26, 2.54, 2.82, 3.10)
T_GMP_RISER_UPPER_HEIGHTS = (0.13, 0.26, 0.39, 0.52, 0.52, 0.39, 0.26, 0.13)
T_GMP_RISER_LOWER_HEIGHTS = (0.00, 0.13, 0.26, 0.39, 0.39, 0.26, 0.13, 0.00)
T_GMP_TREAD_HEIGHTS = (0.00, 0.13, 0.26, 0.39, 0.52, 0.39, 0.26, 0.13, 0.00)
MINDSTEPS_DESCENT_TARGET_RANGES = (
    (950, 1015),
    (1016, 1060),
    (1061, 1120),
    (1121, 1210),
)
MINDSTEPS_DESCENT_TREAD_EDGES = (2.26, 2.54, 2.82, 3.10)
MINDSTEPS_DESCENT_TREAD_HEIGHTS = (0.39, 0.26, 0.13, 0.00)


def kuavo_stairs_terrain(
    difficulty: float, cfg: terrain_gen.SubTerrainBaseCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate symmetric four-step stairs joined by a 1 m top platform."""
    del difficulty
    origin_x = 0.50
    origin_y = 0.50 * cfg.size[1]
    terrain_depth = 1.0
    meshes = [
        trimesh.creation.box(
            (cfg.size[0], cfg.size[1], terrain_depth),
            trimesh.transformations.translation_matrix(
                (0.5 * cfg.size[0], 0.5 * cfg.size[1], -0.5 * terrain_depth)
            ),
        )
    ]
    for level in range(1, 5):
        start_x = origin_x + FIRST_RISER_X + (level - 1) * STAIR_TREAD
        length = STAIR_TREAD if level < 4 else STAIR_TREAD + PLATFORM_LENGTH
        height = level * STAIR_HEIGHT
        center = (start_x + 0.5 * length, origin_y, 0.5 * height)
        meshes.append(
            trimesh.creation.box(
                (length, STAIR_WIDTH, height),
                trimesh.transformations.translation_matrix(center),
            )
        )
    descent_start_x = origin_x + FIRST_RISER_X + 4 * STAIR_TREAD + PLATFORM_LENGTH
    for index, level in enumerate((3, 2, 1)):
        start_x = descent_start_x + index * STAIR_TREAD
        height = level * STAIR_HEIGHT
        center = (start_x + 0.5 * STAIR_TREAD, origin_y, 0.5 * height)
        meshes.append(
            trimesh.creation.box(
                (STAIR_TREAD, STAIR_WIDTH, height),
                trimesh.transformations.translation_matrix(center),
            )
        )
    return meshes, np.array((origin_x, origin_y, 0.0), dtype=np.float64)


@configclass
class KuavoStairsTerrainCfg(terrain_gen.SubTerrainBaseCfg):
    function = kuavo_stairs_terrain


KUAVO_STAIRS_TERRAIN_GENERATOR_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=42,
    size=(4.2, 2.5),
    border_width=0.5,
    border_height=1.0,
    num_rows=32,
    num_cols=32,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=False,
    color_scheme="none",
    use_cache=False,
    sub_terrains={"kuavo_stairs": KuavoStairsTerrainCfg(proportion=1.0)},
)


@configclass
class KuavoS53StairsSceneCfg(RobotSceneCfg):
    """Four steps up, a 1 m platform, and four steps down."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=KUAVO_STAIRS_TERRAIN_GENERATOR_CFG,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.9,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        visual_material=None,
        debug_vis=False,
    )


@configclass
class StairsTrackingRewardsCfg(RewardsCfg):
    motion_feet_pos = RewTerm(
        func=mdp.motion_feet_position_error_exp,
        weight=3.0,
        params={
            "command_name": "motion",
            "std": 0.06,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class FullStairsTrackingRewardsCfg(StairsTrackingRewardsCfg):
    """Phase-two rewards aimed at the third/fourth stair failure segment."""

    motion_anchor_lateral_pos = RewTerm(
        func=mdp.motion_global_anchor_lateral_position_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 0.08},
    )
    motion_feet_xy = RewTerm(
        func=mdp.motion_feet_horizontal_position_error_exp,
        weight=2.0,
        params={
            "command_name": "motion",
            "std": 0.05,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    motion_late_body_pos = RewTerm(
        func=mdp.motion_phase_weighted_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.16, "phase_start": 0.25},
    )
    motion_late_feet_pos = RewTerm(
        func=mdp.motion_phase_weighted_body_position_error_exp,
        weight=2.5,
        params={
            "command_name": "motion",
            "std": 0.065,
            "phase_start": 0.25,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    motion_phase_progress = RewTerm(
        func=mdp.motion_phase_progress_reward,
        weight=1.0,
        params={"command_name": "motion", "exponent": 2.0},
    )
    motion_feet_under_clearance = RewTerm(
        func=mdp.motion_feet_under_clearance_penalty,
        weight=-0.75,
        params={
            "command_name": "motion",
            "std": 0.06,
            "margin": 0.01,
            "swing_velocity_threshold": 0.05,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class UpDownStairsTrackingRewardsCfg(FullStairsTrackingRewardsCfg):
    """Stronger contact tracking for platform transitions and descent."""

    motion_feet_pos = RewTerm(
        func=mdp.motion_feet_position_error_exp,
        weight=4.0,
        params={
            "command_name": "motion",
            "std": 0.065,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    motion_late_feet_pos = RewTerm(
        func=mdp.motion_phase_weighted_body_position_error_exp,
        weight=3.5,
        params={
            "command_name": "motion",
            "std": 0.07,
            "phase_start": 0.38,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    motion_phase_progress = RewTerm(
        func=mdp.motion_phase_progress_reward,
        weight=1.25,
        params={"command_name": "motion", "exponent": 2.0},
    )


@configclass
class ForwardDownStairsTrackingRewardsCfg(FullStairsTrackingRewardsCfg):
    """Task-first rewards for refining a nominal forward descent gait."""

    motion_forward_pos = RewTerm(
        func=mdp.motion_global_anchor_forward_position_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.28},
    )
    motion_height = RewTerm(
        func=mdp.motion_global_anchor_height_error_exp,
        weight=2.5,
        params={"command_name": "motion", "std": 0.10},
    )
    motion_forward_velocity = RewTerm(
        func=mdp.motion_anchor_forward_velocity_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.18},
    )
    backward_velocity = RewTerm(
        func=mdp.motion_backward_velocity_penalty,
        weight=-1.0,
        params={"command_name": "motion"},
    )
    motion_feet_pos = RewTerm(
        func=mdp.motion_feet_position_error_exp,
        weight=0.75,
        params={
            "command_name": "motion",
            "std": 0.15,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    motion_feet_xy = RewTerm(
        func=mdp.motion_feet_horizontal_position_error_exp,
        weight=0.75,
        params={
            "command_name": "motion",
            "std": 0.12,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    motion_late_body_pos = RewTerm(
        func=mdp.motion_phase_weighted_body_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.25, "phase_start": 0.15},
    )
    motion_late_feet_pos = RewTerm(
        func=mdp.motion_phase_weighted_body_position_error_exp,
        weight=0.75,
        params={
            "command_name": "motion",
            "std": 0.16,
            "phase_start": 0.15,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    motion_phase_progress = RewTerm(
        func=mdp.motion_phase_progress_reward,
        weight=1.5,
        params={"command_name": "motion", "exponent": 1.5},
    )
    motion_feet_under_clearance = RewTerm(
        func=mdp.motion_feet_under_clearance_penalty,
        weight=-0.25,
        params={
            "command_name": "motion",
            "std": 0.10,
            "margin": 0.02,
            "swing_velocity_threshold": 0.04,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class ForwardUpDownStairsTrackingRewardsCfg(ForwardDownStairsTrackingRewardsCfg):
    """Balanced rewards for ascent, platform traversal, and forward descent."""

    motion_anchor_lateral_pos = RewTerm(
        func=mdp.motion_global_anchor_lateral_position_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.10},
    )
    motion_forward_pos = RewTerm(
        func=mdp.motion_global_anchor_forward_position_error_exp,
        weight=1.25,
        params={"command_name": "motion", "std": 0.24},
    )
    motion_height = RewTerm(
        func=mdp.motion_global_anchor_height_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 0.09},
    )
    motion_forward_velocity = RewTerm(
        func=mdp.motion_anchor_forward_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.20},
    )
    backward_velocity = RewTerm(
        func=mdp.motion_backward_velocity_penalty,
        weight=-0.75,
        params={"command_name": "motion"},
    )
    motion_feet_pos = RewTerm(
        func=mdp.motion_feet_position_error_exp,
        weight=2.0,
        params={
            "command_name": "motion",
            "std": 0.10,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    motion_feet_xy = RewTerm(
        func=mdp.motion_feet_horizontal_position_error_exp,
        weight=1.5,
        params={
            "command_name": "motion",
            "std": 0.08,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    motion_late_body_pos = RewTerm(
        func=mdp.motion_phase_weighted_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.18, "phase_start": 0.45},
    )
    motion_late_feet_pos = RewTerm(
        func=mdp.motion_phase_weighted_body_position_error_exp,
        weight=1.75,
        params={
            "command_name": "motion",
            "std": 0.10,
            "phase_start": 0.45,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    motion_phase_progress = RewTerm(
        func=mdp.motion_phase_progress_reward,
        weight=1.25,
        params={"command_name": "motion", "exponent": 1.5},
    )
    motion_feet_under_clearance = RewTerm(
        func=mdp.motion_feet_under_clearance_penalty,
        weight=-0.5,
        params={
            "command_name": "motion",
            "std": 0.08,
            "margin": 0.015,
            "swing_velocity_threshold": 0.04,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class TgmpRewardBaselineStairsTrackingRewardsCfg(StairsTrackingRewardsCfg):
    """Published T-GMP task, regularization, and foothold reward structure."""

    # A small direct-state prior substitutes for the unpublished CVAE/AMP model.
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.75,
        params={"command_name": "motion", "std": 0.30},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.50,
        params={"command_name": "motion", "std": 0.40},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=0.50,
        params={"command_name": "motion", "std": 0.30},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=0.50,
        params={"command_name": "motion", "std": 0.40},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=5.0,
        params={
            "command_name": "motion",
            "std": 0.45,
            "body_names": ["waist_yaw_link"],
        },
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=3.0,
        params={
            "command_name": "motion",
            "std": 0.75,
            "body_names": ["waist_yaw_link"],
        },
    )
    motion_feet_pos = RewTerm(
        func=mdp.motion_feet_position_error_exp,
        weight=0.75,
        params={
            "command_name": "motion",
            "std": 0.12,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    motion_feet_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.50,
        params={
            "command_name": "motion",
            "std": 1.0,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    motion_feet_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.25,
        params={
            "command_name": "motion",
            "std": 3.14,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )

    joint_velocity_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.0e-3)
    joint_acceleration_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torque_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    joint_power_l1 = RewTerm(func=mdp.joint_power_l1, weight=-2.0e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    feet_slide_vel = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["leg_l6_link", "leg_r6_link"]
            ),
        },
    )
    feet_contact_forces = RewTerm(
        func=mdp.feet_contact_forces,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["leg_l6_link", "leg_r6_link"]
            ),
        },
    )
    tgmp_contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-0.001,
        params={
            "threshold": 500.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
        },
    )
    tgmp_toe_clearance = RewTerm(
        func=mdp.motion_fixed_stairs_toe_clearance_penalty,
        weight=-1.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "riser_x_offsets": T_GMP_RISER_X,
            "upper_heights": T_GMP_RISER_UPPER_HEIGHTS,
            "lower_heights": T_GMP_RISER_LOWER_HEIGHTS,
            "safety_distance": 0.035,
            "swing_speed_threshold": 0.04,
            "contact_force_threshold": 40.0,
            "local_points": S53_FOREFOOT_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    tgmp_sole_support = RewTerm(
        func=mdp.motion_fixed_stairs_sole_support_distance_penalty,
        weight=-0.5,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": ((0, int(MOTION_LAST_FRAME)),),
            "riser_x_offsets": T_GMP_RISER_X,
            "tread_heights": T_GMP_TREAD_HEIGHTS,
            "max_gap_threshold": 0.012,
            "gap_scale": 0.030,
            "contact_force_threshold": 60.0,
            "local_points": S53_SOLE_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TgmpTerrainConditionedStairsTrackingRewardsCfg(
    TgmpRewardBaselineStairsTrackingRewardsCfg
):
    """Terrain-focused T-GMP proxy with stronger contact quality shaping."""


@configclass
class TgmpRiserSafeStairsTrackingRewardsCfg(
    TgmpTerrainConditionedStairsTrackingRewardsCfg
):
    """Persistent descent-riser clearance with corrected contact accounting."""

    continuous_riser_safety = RewTerm(
        func=mdp.motion_fixed_stairs_continuous_riser_safety_penalty,
        weight=-4.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": T_GMP_RISER_X,
            "upper_heights": T_GMP_RISER_UPPER_HEIGHTS,
            "lower_heights": T_GMP_RISER_LOWER_HEIGHTS,
            "tread_heights": T_GMP_TREAD_HEIGHTS,
            "safety_distance": 0.035,
            "near_contact_force_threshold": 10.0,
            "settled_contact_force_threshold": 60.0,
            "settled_speed_threshold": 0.05,
            "settled_max_gap": 0.012,
            "contact_force_scale": 100.0,
            "contact_penalty_scale": 2.0,
            "forefoot_points": S53_FOREFOOT_SAMPLES,
            "sole_points": S53_SOLE_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class MindStepsTactStairsTrackingRewardsCfg(
    TgmpTerrainConditionedStairsTrackingRewardsCfg
):
    """Short-tread footholds with TACT-style swing geometry and contact timing."""

    bezier_riser_corridor = RewTerm(
        func=mdp.motion_fixed_stairs_bezier_riser_corridor_penalty,
        weight=-0.75,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "approach_distance": 0.080,
            "landing_distance": 0.120,
            "arc_height": 0.100,
            "clearance_scale": 0.030,
            "min_toe_up": 0.012,
            "max_toe_up": 0.055,
            "swing_speed_threshold": 0.035,
            "near_contact_force_threshold": 10.0,
            "settled_contact_force_threshold": 60.0,
            "contact_force_scale": 120.0,
            "forefoot_points": S53_FOREFOOT_SAMPLES,
            "heel_points": S53_HEEL_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    foothold_sequence = RewTerm(
        func=mdp.motion_fixed_stairs_foothold_sequence_reward,
        weight=2.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": MINDSTEPS_DESCENT_TARGET_RANGES,
            "tread_edges": MINDSTEPS_DESCENT_TREAD_EDGES,
            "tread_heights": MINDSTEPS_DESCENT_TREAD_HEIGHTS,
            "foot_center_offset": 0.089,
            "ankle_height": 0.062,
            "std_x": 0.060,
            "std_y": 0.080,
            "std_z": 0.045,
            "contact_force_threshold": 60.0,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    support_margin = RewTerm(
        func=mdp.motion_fixed_stairs_support_margin_penalty,
        weight=-0.25,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": ((900, 1250),),
            "tread_edges": MINDSTEPS_DESCENT_TREAD_EDGES,
            "tread_heights": MINDSTEPS_DESCENT_TREAD_HEIGHTS,
            "tread_length": STAIR_TREAD,
            "edge_margin": 0.010,
            "ankle_height": 0.062,
            "height_tolerance": 0.055,
            "margin_scale": 0.040,
            "contact_force_threshold": 60.0,
            "sole_points": S53_SOLE_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    descent_swing_contact = RewTerm(
        func=mdp.motion_phase_swing_foot_contact_penalty,
        weight=-0.35,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "swing_velocity_threshold": 0.045,
            "contact_force_threshold": 120.0,
            "phase_start": 800.0 / MOTION_LAST_FRAME,
            "phase_end": 1250.0 / MOTION_LAST_FRAME,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class PredictiveSweepStairsTrackingRewardsCfg(
    TgmpTerrainConditionedStairsTrackingRewardsCfg
):
    """Rigid-foot predictive clearance with local imitation release."""

    motion_feet_pos = RewTerm(
        func=mdp.motion_riser_risk_gated_feet_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.12,
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "safety_distance": 0.035,
            "lookahead_time": 0.080,
            "approach_distance": 0.120,
            "release_distance": 0.015,
            "min_forward_speed": 0.020,
            "minimum_tracking_weight": 0.20,
            "local_points": S53_SOLE_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    predictive_riser_sweep = RewTerm(
        func=mdp.motion_predictive_riser_sweep_penalty,
        weight=-1.25,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "safety_distance": 0.035,
            "hard_distance": 0.012,
            "lookahead_time": 0.080,
            "approach_distance": 0.120,
            "release_distance": 0.015,
            "min_forward_speed": 0.020,
            "speed_scale": 0.300,
            "near_contact_force_threshold": 10.0,
            "contact_force_scale": 100.0,
            "hard_penalty_scale": 1.5,
            "contact_penalty_scale": 1.0,
            "local_points": S53_SOLE_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class SwingAwarePredictiveSweepStairsTrackingRewardsCfg(
    TgmpTerrainConditionedStairsTrackingRewardsCfg
):
    """Support-aware predictive clearance that preserves the v7 descent prior."""

    motion_feet_pos = RewTerm(
        func=mdp.motion_riser_risk_gated_feet_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.12,
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "safety_distance": 0.020,
            "lookahead_time": 0.060,
            "approach_distance": 0.100,
            "release_distance": 0.010,
            "min_forward_speed": 0.015,
            "minimum_tracking_weight": 0.65,
            "local_points": S53_SOLE_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    predictive_riser_sweep = RewTerm(
        func=mdp.motion_predictive_riser_sweep_penalty,
        weight=-0.75,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "tread_heights": STEP_TO_DESCENT_TREAD_HEIGHTS,
            "safety_distance": 0.020,
            "hard_distance": 0.006,
            "lookahead_time": 0.060,
            "approach_distance": 0.100,
            "release_distance": 0.010,
            "min_forward_speed": 0.015,
            "speed_scale": 0.250,
            "near_contact_force_threshold": 10.0,
            "contact_force_scale": 100.0,
            "hard_penalty_scale": 2.0,
            "contact_penalty_scale": 0.75,
            "reference_swing_speed_threshold": 0.060,
            "reference_swing_contrast_threshold": 0.060,
            "settled_contact_force_threshold": 60.0,
            "settled_speed_threshold": 0.080,
            "settled_max_gap": 0.012,
            "local_points": S53_SOLE_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class TailRiskPredictiveSweepStairsTrackingRewardsCfg(
    SwingAwarePredictiveSweepStairsTrackingRewardsCfg
):
    """Target the lowest-clearance swing samples while preserving the v7 gait."""

    motion_feet_pos = RewTerm(
        func=mdp.motion_riser_risk_gated_feet_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.12,
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "safety_distance": 0.020,
            "lookahead_time": 0.060,
            "approach_distance": 0.100,
            "release_distance": 0.010,
            "min_forward_speed": 0.015,
            "minimum_tracking_weight": 0.85,
            "local_points": S53_SOLE_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    predictive_riser_sweep = RewTerm(
        func=mdp.motion_predictive_riser_sweep_penalty,
        weight=-0.50,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "tread_heights": STEP_TO_DESCENT_TREAD_HEIGHTS,
            "safety_distance": 0.020,
            "hard_distance": 0.006,
            "lookahead_time": 0.060,
            "approach_distance": 0.100,
            "release_distance": 0.010,
            "min_forward_speed": 0.015,
            "speed_scale": 0.250,
            "near_contact_force_threshold": 10.0,
            "contact_force_scale": 100.0,
            "hard_penalty_scale": 2.5,
            "contact_penalty_scale": 0.75,
            "reference_swing_speed_threshold": 0.060,
            "reference_swing_contrast_threshold": 0.060,
            "settled_contact_force_threshold": 60.0,
            "settled_speed_threshold": 0.080,
            "settled_max_gap": 0.012,
            "tail_fraction": 0.10,
            "tail_extra_weight": 2.5,
            "tail_min_active_envs": 16,
            "local_points": S53_SOLE_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class ReferenceSwingToeBarrierStairsTrackingRewardsCfg(
    TailRiskPredictiveSweepStairsTrackingRewardsCfg
):
    """Align training with the rollout's worst toe-to-riser clearance metric."""

    reference_swing_toe_barrier = RewTerm(
        func=mdp.motion_reference_swing_toe_riser_barrier_penalty,
        weight=-0.80,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "tread_heights": STEP_TO_DESCENT_TREAD_HEIGHTS,
            "safety_distance": 0.008,
            "hard_distance": 0.003,
            "near_contact_force_threshold": 10.0,
            "contact_force_scale": 100.0,
            "hard_penalty_scale": 3.0,
            "contact_penalty_scale": 0.50,
            "reference_swing_speed_threshold": 0.035,
            "reference_swing_contrast_threshold": 0.020,
            "settled_contact_force_threshold": 60.0,
            "settled_speed_threshold": 0.080,
            "settled_max_gap": 0.012,
            "local_points": S53_FOREFOOT_SAMPLES,
            "sole_points": S53_SOLE_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class RunningMinToeBarrierStairsTrackingRewardsCfg(
    ReferenceSwingToeBarrierStairsTrackingRewardsCfg
):
    """Use each swing's worst observed toe-riser clearance as the barrier state."""

    reference_swing_toe_barrier = None

    swing_running_min_toe_barrier = RewTerm(
        func=mdp.motion_swing_running_min_toe_riser_barrier_penalty,
        weight=-0.55,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "tread_heights": STEP_TO_DESCENT_TREAD_HEIGHTS,
            "safety_distance": 0.008,
            "hard_distance": 0.003,
            "near_contact_force_threshold": 10.0,
            "contact_force_scale": 100.0,
            "hard_penalty_scale": 2.0,
            "contact_penalty_scale": 0.50,
            "persistence_scale": 0.35,
            "new_minimum_scale": 2.0,
            "terminal_scale": 1.0,
            "reference_swing_speed_threshold": 0.035,
            "reference_swing_contrast_threshold": 0.020,
            "settled_contact_force_threshold": 60.0,
            "settled_speed_threshold": 0.080,
            "settled_max_gap": 0.012,
            "local_points": S53_FOREFOOT_SAMPLES,
            "sole_points": S53_SOLE_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class TimeToRiserConeStairsTrackingRewardsCfg(
    RunningMinToeBarrierStairsTrackingRewardsCfg
):
    """Dense pre-contact riser risk based on each forefoot point's TTC cone."""

    swing_running_min_toe_barrier = None

    time_to_riser_cone = RewTerm(
        func=mdp.motion_time_to_riser_cone_penalty,
        weight=-0.45,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "approach_distance": 0.140,
            "prediction_horizon": 0.35,
            "time_scale": 0.12,
            "safety_height": 0.015,
            "hard_height": 0.005,
            "minimum_forward_speed": 0.025,
            "reference_swing_speed_threshold": 0.035,
            "reference_swing_contrast_threshold": 0.020,
            "minimum_toe_up": 0.008,
            "toe_pitch_scale": 0.20,
            "near_contact_force_threshold": 10.0,
            "contact_force_scale": 100.0,
            "contact_penalty_scale": 0.40,
            "forefoot_points": S53_FOREFOOT_SAMPLES,
            "heel_points": S53_HEEL_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class SpatialRiserCorridorStairsTrackingRewardsCfg(
    TimeToRiserConeStairsTrackingRewardsCfg
):
    """Velocity-independent local obstacle potential for v16."""

    time_to_riser_cone = None

    spatial_riser_corridor = RewTerm(
        func=mdp.motion_spatial_riser_corridor_penalty,
        weight=-0.35,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "window_before": 0.140,
            "window_after": 0.060,
            "corridor_sigma": 0.035,
            "base_clearance": 0.003,
            "peak_clearance": 0.018,
            "hard_clearance": 0.003,
            "reference_swing_speed_threshold": 0.035,
            "reference_swing_contrast_threshold": 0.020,
            "minimum_toe_up": 0.008,
            "toe_pitch_scale": 0.15,
            "near_contact_force_threshold": 10.0,
            "contact_force_scale": 100.0,
            "contact_penalty_scale": 0.35,
            "forefoot_points": S53_FOREFOOT_SAMPLES,
            "heel_points": S53_HEEL_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class ClearanceSoftLandingStairsTrackingRewardsCfg(
    SpatialRiserCorridorStairsTrackingRewardsCfg
):
    """v17: retain riser clearance while distributing touchdown risk densely."""

    pre_touchdown_soft_landing = RewTerm(
        func=mdp.motion_pre_touchdown_soft_landing_penalty,
        weight=-0.50,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "tread_heights": STEP_TO_DESCENT_TREAD_HEIGHTS,
            "approach_height": 0.070,
            "below_tolerance": 0.005,
            "safe_downward_speed": 0.18,
            "downward_speed_scale": 0.25,
            "reference_swing_speed_threshold": 0.035,
            "reference_swing_contrast_threshold": 0.020,
            "contact_force_threshold": 1100.0,
            "contact_force_scale": 350.0,
            "impact_scale": 0.35,
            "sole_points": S53_SOLE_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class RiserClearanceLagrangianStairsTrackingRewardsCfg(
    ClearanceSoftLandingStairsTrackingRewardsCfg
):
    """v22: episode-preserving adaptive cost for rigid-foot riser clearance."""

    riser_clearance_lagrangian = RewTerm(
        func=mdp.motion_riser_clearance_lagrangian_cost,
        weight=-1.0,
        params={
            "command_name": "motion",
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "window_before": 0.030,
            "window_after": 0.020,
            "safety_distance": 0.0030,
            "hard_distance": 0.0005,
            "hard_cost_scale": 3.0,
            "cost_budget": 0.020,
            "dual_learning_rate": 0.020,
            "dual_ema_decay": 0.98,
            "min_dual_multiplier": 0.05,
            "max_dual_multiplier": 2.0,
            "update_period_steps": 24,
            "initial_dual_multiplier": 0.10,
            "reference_swing_speed_threshold": 0.035,
            "reference_swing_contrast_threshold": 0.020,
            "local_points": S53_FOREFOOT_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class SharedClearanceCmdpStairsTrackingRewardsCfg(
    ClearanceSoftLandingStairsTrackingRewardsCfg
):
    """v23: expose shared actual/planned-swing geometry only through CMDP cost."""

    shared_riser_clearance_cost = RewTerm(
        func=mdp.motion_shared_riser_clearance_cost,
        # RewardManager skips evaluating exact zero-weight terms. Keep this
        # numerically inert weight so the raw CMDP cost is produced every step.
        weight=-1.0e-8,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "window_before": 0.050,
            "window_after": 0.030,
            "safety_distance": 0.0030,
            "hard_distance": 0.0005,
            "hard_cost_scale": 3.0,
            "actual_swing_speed_threshold": 0.025,
            "support_force_threshold": 40.0,
            "reference_swing_speed_threshold": 0.035,
            "reference_swing_contrast_threshold": 0.020,
            "local_points": S53_FOREFOOT_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class ConservativeTailReplayStairsTrackingRewardsCfg(
    ClearanceSoftLandingStairsTrackingRewardsCfg
):
    """v18: replay each swing's worst riser distance while retaining soft landing."""

    swing_running_min_toe_barrier = RewTerm(
        func=mdp.motion_swing_running_min_toe_riser_barrier_penalty,
        weight=-0.25,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "tread_heights": STEP_TO_DESCENT_TREAD_HEIGHTS,
            "safety_distance": 0.008,
            "hard_distance": 0.003,
            "near_contact_force_threshold": 10.0,
            "contact_force_scale": 100.0,
            "hard_penalty_scale": 2.0,
            "contact_penalty_scale": 0.35,
            "persistence_scale": 1.0,
            "new_minimum_scale": 0.50,
            "terminal_scale": 1.20,
            "reference_swing_speed_threshold": 0.035,
            "reference_swing_contrast_threshold": 0.020,
            "settled_contact_force_threshold": 60.0,
            "settled_speed_threshold": 0.080,
            "settled_max_gap": 0.012,
            "local_points": S53_FOREFOOT_SAMPLES,
            "sole_points": S53_SOLE_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class ForwardUpDownStableStairsTrackingRewardsCfg(ForwardUpDownStairsTrackingRewardsCfg):
    """Fixed platform gate and contact-clean forward descent rewards."""

    pre_descent_gate_position = RewTerm(
        func=mdp.motion_phase_anchor_position_error_exp,
        weight=3.0,
        params={
            "command_name": "motion",
            "std": 0.05,
            "phase_start": PRE_DESCENT_PHASE_START,
            "phase_end": PRE_DESCENT_PHASE_END,
        },
    )
    pre_descent_gate_orientation = RewTerm(
        func=mdp.motion_phase_anchor_orientation_error_exp,
        weight=2.0,
        params={
            "command_name": "motion",
            "std": 0.15,
            "phase_start": PRE_DESCENT_PHASE_START,
            "phase_end": PRE_DESCENT_PHASE_END,
        },
    )
    pre_descent_stability = RewTerm(
        func=mdp.motion_phase_anchor_stability_penalty,
        weight=-2.0,
        params={
            "command_name": "motion",
            "linear_velocity_scale": 0.12,
            "angular_velocity_scale": 0.35,
            "phase_start": PRE_DESCENT_PHASE_START,
            "phase_end": PRE_DESCENT_PHASE_END,
        },
    )
    descent_feet_position = RewTerm(
        func=mdp.motion_phase_body_position_error_exp,
        weight=3.0,
        params={
            "command_name": "motion",
            "std": 0.07,
            "phase_start": DESCENT_PHASE_START,
            "phase_end": DESCENT_PHASE_END,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    descent_feet_slide = RewTerm(
        func=mdp.motion_phase_body_slide_penalty,
        weight=-1.5,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "contact_threshold": 80.0,
            "phase_start": DESCENT_PHASE_START,
            "phase_end": DESCENT_PHASE_END,
        },
    )
    descent_swing_foot_contact = RewTerm(
        func=mdp.motion_phase_swing_foot_contact_penalty,
        weight=-1.75,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "swing_velocity_threshold": 0.06,
            "contact_force_threshold": 80.0,
            "phase_start": DESCENT_PHASE_START,
            "phase_end": DESCENT_PHASE_END,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    descent_feet_under_clearance = RewTerm(
        func=mdp.motion_feet_under_clearance_penalty,
        weight=-1.0,
        params={
            "command_name": "motion",
            "std": 0.06,
            "margin": 0.01,
            "swing_velocity_threshold": 0.04,
            "body_names": ["leg_l6_link", "leg_r6_link"],
            "phase_start": DESCENT_PHASE_START,
            "phase_end": DESCENT_PHASE_END,
        },
    )


@configclass
class StepToDownStairsTrackingRewardsCfg(ForwardUpDownStableStairsTrackingRewardsCfg):
    """Left-first step-to descent with per-level double-foot localization gates."""

    pre_descent_gate_position = RewTerm(
        func=mdp.motion_phase_anchor_position_error_exp,
        weight=2.5,
        params={
            "command_name": "motion",
            "std": 0.05,
            "phase_start": STEP_TO_PRE_DESCENT_PHASE_START,
            "phase_end": STEP_TO_PRE_DESCENT_PHASE_END,
        },
    )
    pre_descent_gate_orientation = RewTerm(
        func=mdp.motion_phase_anchor_orientation_error_exp,
        weight=2.0,
        params={
            "command_name": "motion",
            "std": 0.15,
            "phase_start": STEP_TO_PRE_DESCENT_PHASE_START,
            "phase_end": STEP_TO_PRE_DESCENT_PHASE_END,
        },
    )
    pre_descent_stability = RewTerm(
        func=mdp.motion_phase_anchor_stability_penalty,
        weight=-1.5,
        params={
            "command_name": "motion",
            "linear_velocity_scale": 0.10,
            "angular_velocity_scale": 0.30,
            "phase_start": STEP_TO_PRE_DESCENT_PHASE_START,
            "phase_end": STEP_TO_PRE_DESCENT_PHASE_END,
        },
    )
    descent_feet_position = RewTerm(
        func=mdp.motion_frame_ranges_body_position_error_exp,
        weight=3.5,
        params={
            "command_name": "motion",
            "std": 0.075,
            "frame_ranges": list(STEP_TO_DESCENT_RANGES),
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    descent_feet_slide = RewTerm(
        func=mdp.motion_phase_body_slide_penalty,
        weight=-1.75,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "contact_threshold": 70.0,
            "phase_start": STEP_TO_DESCENT_PHASE_START,
            "phase_end": STEP_TO_DESCENT_PHASE_END,
        },
    )
    descent_swing_foot_contact = RewTerm(
        func=mdp.motion_phase_swing_foot_contact_penalty,
        weight=-2.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "swing_velocity_threshold": 0.05,
            "contact_force_threshold": 70.0,
            "phase_start": STEP_TO_DESCENT_PHASE_START,
            "phase_end": STEP_TO_DESCENT_PHASE_END,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    descent_feet_under_clearance = RewTerm(
        func=mdp.motion_feet_under_clearance_penalty,
        weight=-1.0,
        params={
            "command_name": "motion",
            "std": 0.055,
            "margin": 0.008,
            "swing_velocity_threshold": 0.04,
            "body_names": ["leg_l6_link", "leg_r6_link"],
            "phase_start": STEP_TO_DESCENT_PHASE_START,
            "phase_end": STEP_TO_DESCENT_PHASE_END,
        },
    )
    double_foot_gate_position = RewTerm(
        func=mdp.motion_frame_ranges_body_position_error_exp,
        weight=4.0,
        params={
            "command_name": "motion",
            "std": 0.035,
            "frame_ranges": list(STEP_TO_DOUBLE_FOOT_GATE_RANGES),
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    double_foot_same_level = RewTerm(
        func=mdp.motion_frame_ranges_feet_same_level_error_exp,
        weight=2.0,
        params={
            "command_name": "motion",
            "std": 0.025,
            "frame_ranges": list(STEP_TO_DOUBLE_FOOT_GATE_RANGES),
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    double_foot_gate_velocity = RewTerm(
        func=mdp.motion_frame_ranges_body_velocity_penalty,
        weight=-1.5,
        params={
            "command_name": "motion",
            "speed_scale": 0.10,
            "frame_ranges": list(STEP_TO_DOUBLE_FOOT_GATE_RANGES),
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    double_foot_gate_stability = RewTerm(
        func=mdp.motion_frame_ranges_anchor_stability_penalty,
        weight=-1.5,
        params={
            "command_name": "motion",
            "linear_velocity_scale": 0.10,
            "angular_velocity_scale": 0.30,
            "frame_ranges": list(STEP_TO_DOUBLE_FOOT_GATE_RANGES),
        },
    )
    right_support_position = RewTerm(
        func=mdp.motion_frame_ranges_body_position_error_exp,
        weight=2.25,
        params={
            "command_name": "motion",
            "std": 0.035,
            "frame_ranges": list(STEP_TO_LEFT_SWING_RANGES),
            "body_names": ["leg_r6_link"],
        },
    )
    right_support_velocity = RewTerm(
        func=mdp.motion_frame_ranges_body_velocity_penalty,
        weight=-1.25,
        params={
            "command_name": "motion",
            "speed_scale": 0.08,
            "frame_ranges": list(STEP_TO_LEFT_SWING_RANGES),
            "body_names": ["leg_r6_link"],
        },
    )
    left_support_position = RewTerm(
        func=mdp.motion_frame_ranges_body_position_error_exp,
        weight=2.25,
        params={
            "command_name": "motion",
            "std": 0.035,
            "frame_ranges": list(STEP_TO_RIGHT_SWING_RANGES),
            "body_names": ["leg_l6_link"],
        },
    )
    left_support_velocity = RewTerm(
        func=mdp.motion_frame_ranges_body_velocity_penalty,
        weight=-1.25,
        params={
            "command_name": "motion",
            "speed_scale": 0.08,
            "frame_ranges": list(STEP_TO_RIGHT_SWING_RANGES),
            "body_names": ["leg_l6_link"],
        },
    )
    left_swing_backward_kick = RewTerm(
        func=mdp.motion_frame_ranges_backward_body_velocity_penalty,
        weight=-2.0,
        params={
            "command_name": "motion",
            "velocity_scale": 0.12,
            "frame_ranges": list(STEP_TO_LEFT_SWING_RANGES),
            "body_names": ["leg_l6_link"],
        },
    )
    right_swing_backward_kick = RewTerm(
        func=mdp.motion_frame_ranges_backward_body_velocity_penalty,
        weight=-1.5,
        params={
            "command_name": "motion",
            "velocity_scale": 0.12,
            "frame_ranges": list(STEP_TO_RIGHT_SWING_RANGES),
            "body_names": ["leg_r6_link"],
        },
    )
    descent_landing_impact = RewTerm(
        func=mdp.motion_frame_ranges_contact_impact_penalty,
        weight=-1.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "force_threshold": 850.0,
            "frame_ranges": list(STEP_TO_DESCENT_RANGES),
        },
    )


@configclass
class StepToDownNosingSafeStairsTrackingRewardsCfg(
    StepToDownStairsTrackingRewardsCfg
):
    """Protect left-foot stair-edge clearance and preserve stable gate passage."""

    left_nosing_clearance = RewTerm(
        func=mdp.motion_frame_ranges_body_under_reference_height_penalty,
        weight=-2.5,
        params={
            "command_name": "motion",
            "height_scale": 0.025,
            "allowed_below": 0.005,
            "frame_ranges": list(STEP_TO_LEFT_NOSING_APPROACH_RANGES),
            "body_names": ["leg_l6_link"],
        },
    )
    left_nosing_early_contact = RewTerm(
        func=mdp.motion_frame_ranges_contact_force_penalty,
        weight=-1.5,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link"]
            ),
            "force_threshold": 20.0,
            "force_scale": 200.0,
            "frame_ranges": list(STEP_TO_LEFT_NOSING_APPROACH_RANGES),
        },
    )
    gate_stable_progress = RewTerm(
        func=mdp.motion_gate_consecutive_stability_reward,
        weight=1.5,
        params={"command_name": "motion", "exponent": 2.0},
    )


@configclass
class StepToDownPreserveStairsTrackingRewardsCfg(
    StepToDownNosingSafeStairsTrackingRewardsCfg
):
    """Preserve the verified policy while making gate passage the profitable outcome."""

    # Partial 1-6 step cycles caused the previous policy to hover below the hard gate.
    gate_stable_progress = None
    gate_anchor_angular_excess = RewTerm(
        func=mdp.motion_gate_anchor_angular_excess_penalty,
        weight=-6.0,
        params={
            "command_name": "motion",
            "target_speed": 0.18,
            "speed_scale": 0.08,
        },
    )
    gate_wait = RewTerm(
        func=mdp.motion_gate_wait_penalty,
        weight=-8.0,
        params={
            "command_name": "motion",
            "grace_steps": 10,
            "max_wait_steps": 100,
            "exponent": 2.0,
        },
    )
    gate_stable_reset = RewTerm(
        func=mdp.motion_gate_stable_reset_penalty,
        weight=-10.0,
        params={"command_name": "motion"},
    )
    gate_pass = RewTerm(
        func=mdp.motion_gate_pass_bonus,
        weight=50.0,
        params={"command_name": "motion"},
    )
    gate_task_complete = RewTerm(
        func=mdp.motion_gate_task_complete_bonus,
        weight=250.0,
        params={"command_name": "motion"},
    )
    gate_timeout = RewTerm(
        func=mdp.motion_gate_timeout_penalty,
        weight=-250.0,
        params={"command_name": "motion", "max_wait_steps": 100},
    )


@configclass
class StepToDownTgmpFootholdStairsTrackingRewardsCfg(
    StepToDownPreserveStairsTrackingRewardsCfg
):
    """Terrain-conditioned toe clearance and whole-sole support quality."""

    left_forefoot_nosing_distance = RewTerm(
        func=mdp.motion_frame_ranges_foot_nosing_distance_penalty,
        weight=-4.0,
        params={
            "command_name": "motion",
            "frame_ranges": list(STEP_TO_LEFT_FOOTHOLD_APPROACH_RANGES),
            "riser_x_offsets": list(STEP_TO_DESCENT_RISER_X),
            "upper_heights": list(STEP_TO_DESCENT_UPPER_HEIGHTS),
            "lower_heights": list(STEP_TO_DESCENT_LOWER_HEIGHTS),
            "safety_distance": 0.040,
            "local_points": list(S53_FOREFOOT_SAMPLES),
            "body_names": ["leg_l6_link"],
        },
    )
    descent_sole_support_distance = RewTerm(
        func=mdp.motion_fixed_stairs_sole_support_distance_penalty,
        weight=-5.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "frame_ranges": list(STEP_TO_DESCENT_RANGES),
            "riser_x_offsets": list(STEP_TO_DESCENT_RISER_X),
            "tread_heights": list(STEP_TO_DESCENT_TREAD_HEIGHTS),
            "max_gap_threshold": 0.025,
            "gap_scale": 0.080,
            "contact_force_threshold": 40.0,
            "local_points": list(S53_SOLE_SAMPLES),
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class StepToDownPreserveTerminationsCfg(TerminationsCfg):
    """Treat gate stalls as task failures instead of successful episode timeouts."""

    gate_wait_timeout = DoneTerm(
        func=mdp.motion_gate_wait_timeout,
        params={"command_name": "motion", "max_wait_steps": 100},
        time_out=False,
    )


@configclass
class RiserConstraintCaTTerminationsCfg(TerminationsCfg):
    """Terminate locally sampled unsafe swing states without gating progress."""

    riser_clearance_cat = DoneTerm(
        func=mdp.motion_riser_clearance_cat,
        params={
            "command_name": "motion",
            "frame_ranges": ((800, 1250),),
            "riser_x_offsets": STEP_TO_DESCENT_RISER_X,
            "upper_heights": STEP_TO_DESCENT_UPPER_HEIGHTS,
            "lower_heights": STEP_TO_DESCENT_LOWER_HEIGHTS,
            "window_before": 0.030,
            "window_after": 0.020,
            "safety_distance": 0.0025,
            "hard_distance": 0.00025,
            "max_termination_probability": 0.05,
            "risk_exponent": 4.0,
            "reference_swing_speed_threshold": 0.035,
            "reference_swing_contrast_threshold": 0.020,
            "local_points": S53_FOREFOOT_SAMPLES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
        time_out=False,
    )


@configclass
class KuavoS53StairsStep1EnvCfg(KuavoS53FlatEnvCfg):
    """Warm-start tracking task for left-first/right-second stair contacts."""

    scene: KuavoS53StairsSceneCfg = KuavoS53StairsSceneCfg(num_envs=1024, env_spacing=3.0)
    rewards: StairsTrackingRewardsCfg = StairsTrackingRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = STEP1_MOTION
        self.commands.motion.debug_vis = False
        self.commands.motion.start_hold_steps = 25
        self.commands.motion.end_hold_steps = 50
        self.commands.motion.pose_range = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (-0.005, 0.005),
            "roll": (-0.02, 0.02),
            "pitch": (-0.02, 0.02),
            "yaw": (-0.03, 0.03),
        }
        self.commands.motion.velocity_range = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.03, 0.03),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        }

        self.episode_length_s = 6.0
        self.scene.contact_forces.debug_vis = False
        self.viewer.eye = (2.2, 1.8, 1.4)

        self.rewards.motion_global_anchor_pos.weight = 1.5
        self.rewards.motion_global_anchor_pos.params["std"] = 0.18
        self.rewards.motion_global_anchor_ori.weight = 0.8
        self.rewards.motion_body_pos.weight = 1.5
        self.rewards.motion_body_pos.params["std"] = 0.18
        self.rewards.motion_knee_pos.weight = 1.2
        self.rewards.motion_feet_vel.weight = 1.0
        self.rewards.feet_contact_forces.weight = -0.1
        self.rewards.feet_slide_vel.weight = -0.5

        # Phase one learns the demonstrated contact sequence before robustness.
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (-0.02, 0.02)
        self.events.torso_com = None
        self.events.base_com = None
        self.events.add_torso_mass = None
        self.events.add_base_mass = None
        self.events.link_com = None
        self.events.add_link_mass = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None


@configclass
class KuavoS53StairsFullEnvCfg(KuavoS53StairsStep1EnvCfg):
    """Four-level reference-tracking task used after the step-one warm start."""

    rewards: FullStairsTrackingRewardsCfg = FullStairsTrackingRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = FULL_MOTION
        self.episode_length_s = 14.0
        self.commands.motion.adaptive_uniform_ratio = 0.25

        # Let phase-two PPO experience and recover from a one-step vertical miss.
        self.terminations.anchor_pos.params["threshold"] = 0.32
        self.terminations.ee_body_pos.params["threshold"] = 0.32
        self.terminations.ee_body_pos.params["body_names"] = ["leg_l6_link", "leg_r6_link"]
        self.terminations.anchor_ori.params["threshold"] = 0.9


@configclass
class KuavoS53StairsUpDownEnvCfg(KuavoS53StairsFullEnvCfg):
    """Complete four-step ascent, platform walk, and four-step descent."""

    rewards: UpDownStairsTrackingRewardsCfg = UpDownStairsTrackingRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = UPDOWN_MOTION
        self.commands.motion.adaptive_uniform_ratio = 0.20
        self.episode_length_s = 32.0
        self.scene.env_spacing = 4.5
        self.viewer.eye = (4.0, 2.8, 1.8)

        self.rewards.motion_global_anchor_pos.weight = 2.0
        self.rewards.motion_global_anchor_pos.params["std"] = 0.20
        self.rewards.motion_body_pos.weight = 2.0
        self.rewards.motion_body_pos.params["std"] = 0.20
        self.rewards.motion_feet_vel.weight = 1.25
        self.rewards.feet_slide_vel.weight = -0.7

        # Downward contacts have larger transient errors than ascent contacts.
        self.terminations.anchor_pos.params["threshold"] = 0.38
        self.terminations.ee_body_pos.params["threshold"] = 0.38
        self.terminations.anchor_ori.params["threshold"] = 1.0


@configclass
class KuavoS53StairsForwardDownEnvCfg(KuavoS53StairsFullEnvCfg):
    """Platform approach and forward-facing descent specialization."""

    rewards: ForwardDownStairsTrackingRewardsCfg = ForwardDownStairsTrackingRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = FORWARD_DOWN_MOTION
        self.commands.motion.adaptive_uniform_ratio = 0.35
        self.episode_length_s = 18.0
        self.scene.env_spacing = 4.5
        self.viewer.eye = (4.0, 2.8, 1.8)

        self.rewards.motion_global_anchor_pos.weight = 0.75
        self.rewards.motion_global_anchor_pos.params["std"] = 0.35
        self.rewards.motion_global_anchor_ori.weight = 1.5
        self.rewards.motion_body_pos.weight = 0.35
        self.rewards.motion_body_pos.params["std"] = 0.28
        self.rewards.motion_body_ori.weight = 0.5
        self.rewards.motion_body_lin_vel.weight = 0.5
        self.rewards.motion_body_ang_vel.weight = 0.5
        self.rewards.motion_feet_vel.weight = 0.5
        self.rewards.motion_feet_ang_vel.weight = 0.5
        self.rewards.feet_slide_vel.weight = -0.6

        self.terminations.anchor_pos.params["threshold"] = 0.55
        self.terminations.ee_body_pos.params["threshold"] = 0.65
        self.terminations.anchor_ori.params["threshold"] = 1.0


@configclass
class KuavoS53StairsForwardUpDownEnvCfg(KuavoS53StairsForwardDownEnvCfg):
    """Final unified ascent, platform, and forward-descent fine-tuning task."""

    rewards: ForwardUpDownStairsTrackingRewardsCfg = ForwardUpDownStairsTrackingRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = FORWARD_UPDOWN_MOTION
        self.episode_length_s = 32.0
        self.commands.motion.adaptive_uniform_ratio = 0.30

        # Restore imitation pressure lost in the descent specialist while keeping
        # enough task-space freedom to learn the platform and forward descent.
        self.rewards.motion_global_anchor_pos.weight = 1.5
        self.rewards.motion_global_anchor_pos.params["std"] = 0.24
        self.rewards.motion_global_anchor_ori.weight = 2.0
        self.rewards.motion_body_pos.weight = 1.25
        self.rewards.motion_body_pos.params["std"] = 0.22
        self.rewards.motion_body_ori.weight = 0.75
        self.rewards.motion_body_lin_vel.weight = 0.75
        self.rewards.motion_body_ang_vel.weight = 0.75
        self.rewards.motion_feet_vel.weight = 0.8
        self.rewards.motion_feet_ang_vel.weight = 0.6
        self.rewards.feet_slide_vel.weight = -0.7

        # Warm-start with recovery room; tighten these after full-cycle success.
        self.terminations.anchor_pos.params["threshold"] = 0.45
        self.terminations.ee_body_pos.params["threshold"] = 0.50
        self.terminations.anchor_ori.params["threshold"] = 1.0


@configclass
class KuavoS53StairsTgmpRewardBaselineEnvCfg(KuavoS53StairsForwardUpDownEnvCfg):
    """Continuous full-course task with T-GMP's published reward structure."""

    rewards: TgmpRewardBaselineStairsTrackingRewardsCfg = (
        TgmpRewardBaselineStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = FORWARD_UPDOWN_MOTION
        self.commands.motion.adaptive_uniform_ratio = 0.30

        # The task has no platform gate, fixed foothold, or prescribed leg order.
        self.rewards.motion_global_anchor_pos.weight = 0.75
        self.rewards.motion_global_anchor_pos.params["std"] = 0.30
        self.rewards.motion_global_anchor_ori.weight = 0.50
        self.rewards.motion_body_pos.weight = 0.50
        self.rewards.motion_body_pos.params["std"] = 0.30
        self.rewards.motion_body_ori.weight = 0.50
        self.rewards.motion_body_lin_vel.weight = 5.0
        self.rewards.motion_body_lin_vel.params["std"] = 0.45
        self.rewards.motion_body_lin_vel.params["body_names"] = ["waist_yaw_link"]
        self.rewards.motion_body_ang_vel.weight = 3.0
        self.rewards.motion_body_ang_vel.params["std"] = 0.75
        self.rewards.motion_body_ang_vel.params["body_names"] = ["waist_yaw_link"]
        self.rewards.motion_feet_pos.weight = 0.75
        self.rewards.motion_feet_pos.params["std"] = 0.12
        self.rewards.motion_feet_vel.weight = 0.50
        self.rewards.motion_feet_ang_vel.weight = 0.25
        self.rewards.motion_hand_pos.weight = 0.0
        self.rewards.motion_hand_vel.weight = 0.0
        self.rewards.motion_knee_pos.weight = 0.0
        self.rewards.motion_knee_vel.weight = 0.0
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.feet_slide_vel.weight = -0.10
        self.rewards.feet_contact_forces.weight = 0.0

        # Keep the first reward-port stage deterministic enough to preserve the
        # successful model_89996 traversal. Robustness randomization comes later.
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None

        self.terminations.anchor_pos.params["threshold"] = 0.48
        self.terminations.ee_body_pos.params["threshold"] = 0.52
        self.terminations.anchor_ori.params["threshold"] = 1.0


@configclass
class KuavoS53StairsTgmpTerrainConditionedEnvCfg(
    KuavoS53StairsTgmpRewardBaselineEnvCfg
):
    """Terrain-conditioned sampling and contact-aware T-GMP fine-tuning."""

    rewards: TgmpTerrainConditionedStairsTrackingRewardsCfg = (
        TgmpTerrainConditionedStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        motion_cfg = self.commands.motion
        self.commands.motion = mdp.TerrainBiasedMotionCommandCfg(
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
            adaptive_uniform_ratio=0.25,
            adaptive_alpha=motion_cfg.adaptive_alpha,
            start_hold_steps=motion_cfg.start_hold_steps,
            end_hold_steps=motion_cfg.end_hold_steps,
            anchor_pos_threshold=motion_cfg.anchor_pos_threshold,
            anchor_ori_threshold=motion_cfg.anchor_ori_threshold,
            debug_vis=False,
            zero_start_fraction=0.30,
            terrain_focus_fraction=0.45,
            terrain_focus_frames=(708, 760, 930, 990, 1050, 1110, 1170),
            terrain_focus_approach_steps=40,
        )

        # Preserve full-course motion while allowing terrain-local leg corrections.
        self.rewards.motion_global_anchor_pos.weight = 1.0
        self.rewards.motion_global_anchor_ori.weight = 0.75
        self.rewards.motion_body_pos.weight = 0.75
        self.rewards.motion_body_ori.weight = 0.75
        self.rewards.motion_feet_pos.weight = 1.0

        # Curves showed high-frequency action drift after model_90100.
        self.rewards.action_rate_l2.weight = -0.010
        self.rewards.action_smoothness_l2.weight = -0.020

        # T-GMP foothold shaping is strengthened only through terrain-focused sampling.
        self.rewards.feet_slide_vel.weight = -0.25
        self.rewards.tgmp_contact_force.weight = -0.002
        self.rewards.tgmp_contact_force.params["threshold"] = 400.0
        self.rewards.tgmp_toe_clearance.weight = -3.0
        self.rewards.tgmp_sole_support.weight = -1.5


@configclass
class KuavoS53StairsTgmpRiserSafeEnvCfg(
    KuavoS53StairsTgmpTerrainConditionedEnvCfg
):
    """v8: remove the contact blind spot while preserving full-course motion."""

    rewards: TgmpRiserSafeStairsTrackingRewardsCfg = (
        TgmpRiserSafeStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.zero_start_fraction = 0.30
        self.commands.motion.terrain_focus_fraction = 0.50
        self.commands.motion.adaptive_uniform_ratio = 0.20
        self.commands.motion.terrain_focus_frames = (
            900, 938, 975, 1019, 1043, 1080, 1134, 1170
        )
        self.commands.motion.terrain_focus_approach_steps = 30

        # The old reference and teacher both pass within millimeters of risers.
        self.rewards.motion_feet_pos.weight = 0.25
        self.rewards.motion_feet_vel.weight = 0.15
        self.rewards.motion_feet_ang_vel.weight = 0.05
        self.rewards.tgmp_toe_clearance.weight = 0.0
        self.rewards.tgmp_sole_support.weight = -2.0
        self.rewards.feet_slide_vel.weight = -0.30


@configclass
class KuavoS53StairsMindStepsTactEnvCfg(
    KuavoS53StairsTgmpTerrainConditionedEnvCfg
):
    """v9: retain full descent while refining short-tread swing and landing."""

    rewards: MindStepsTactStairsTrackingRewardsCfg = (
        MindStepsTactStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.zero_start_fraction = 0.35
        self.commands.motion.terrain_focus_fraction = 0.45
        self.commands.motion.adaptive_uniform_ratio = 0.20
        self.commands.motion.terrain_focus_frames = (
            900, 950, 990, 1025, 1050, 1080, 1110, 1145, 1185
        )
        self.commands.motion.terrain_focus_approach_steps = 35

        # Keep global progress and natural upper-body coordination as hard priors.
        self.rewards.motion_global_anchor_pos.weight = 1.25
        self.rewards.motion_global_anchor_ori.weight = 0.85
        self.rewards.motion_body_pos.weight = 0.85
        self.rewards.motion_body_ori.weight = 0.85

        # Let the feet depart from the unsafe reference only inside dense geometry rewards.
        self.rewards.motion_feet_pos.weight = 0.35
        self.rewards.motion_feet_vel.weight = 0.10
        self.rewards.motion_feet_ang_vel.weight = 0.05
        self.rewards.tgmp_toe_clearance.weight = 0.0
        self.rewards.tgmp_sole_support.weight = -1.0
        self.rewards.feet_slide_vel.weight = -0.30
        self.rewards.tgmp_contact_force.weight = -0.002
        self.rewards.tgmp_contact_force.params["threshold"] = 400.0

        self.rewards.action_rate_l2.weight = -0.012
        self.rewards.action_smoothness_l2.weight = -0.025

        # Fixed geometry first; robustness randomization comes after clean rollout.
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsPredictiveSweepEnvCfg(
    KuavoS53StairsTgmpTerrainConditionedEnvCfg
):
    """v10: preserve model_92099 while correcting imminent riser sweeps."""

    rewards: PredictiveSweepStairsTrackingRewardsCfg = (
        PredictiveSweepStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.zero_start_fraction = 0.35
        self.commands.motion.terrain_focus_fraction = 0.45
        self.commands.motion.adaptive_uniform_ratio = 0.20
        self.commands.motion.terrain_focus_frames = (
            900, 938, 975, 1019, 1043, 1080, 1134, 1170
        )
        self.commands.motion.terrain_focus_approach_steps = 40

        # Keep the v7 full-course prior. Only position imitation is relaxed by risk.
        self.rewards.motion_global_anchor_pos.weight = 1.0
        self.rewards.motion_global_anchor_ori.weight = 0.75
        self.rewards.motion_body_pos.weight = 0.75
        self.rewards.motion_body_ori.weight = 0.75
        self.rewards.motion_feet_pos.weight = 1.0
        self.rewards.motion_feet_vel.weight = 0.35
        self.rewards.motion_feet_ang_vel.weight = 0.15

        # Preserve model_92099's low-slip contact quality and remove the old blind term.
        self.rewards.tgmp_toe_clearance.weight = 0.0
        self.rewards.tgmp_sole_support.weight = -1.5
        self.rewards.feet_slide_vel.weight = -0.25
        self.rewards.tgmp_contact_force.weight = -0.002
        self.rewards.tgmp_contact_force.params["threshold"] = 400.0
        self.rewards.action_rate_l2.weight = -0.010
        self.rewards.action_smoothness_l2.weight = -0.020

        # Fixed geometry first. Randomization starts only after multi-seed clearance.
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsSwingAwareSweepEnvCfg(
    KuavoS53StairsTgmpTerrainConditionedEnvCfg
):
    """v11: refine swing clearance without penalizing settled tread support."""

    rewards: SwingAwarePredictiveSweepStairsTrackingRewardsCfg = (
        SwingAwarePredictiveSweepStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.zero_start_fraction = 0.45
        self.commands.motion.terrain_focus_fraction = 0.40
        self.commands.motion.adaptive_uniform_ratio = 0.15
        self.commands.motion.terrain_focus_frames = (
            900, 938, 975, 1019, 1043, 1080, 1134, 1170
        )
        self.commands.motion.terrain_focus_approach_steps = 40

        self.rewards.motion_global_anchor_pos.weight = 1.1
        self.rewards.motion_global_anchor_ori.weight = 0.85
        self.rewards.motion_body_pos.weight = 0.85
        self.rewards.motion_body_ori.weight = 0.85
        self.rewards.motion_feet_pos.weight = 1.0
        self.rewards.motion_feet_vel.weight = 0.45
        self.rewards.motion_feet_ang_vel.weight = 0.20

        self.rewards.tgmp_toe_clearance.weight = 0.0
        self.rewards.tgmp_sole_support.weight = -1.5
        self.rewards.feet_slide_vel.weight = -0.30
        self.rewards.tgmp_contact_force.weight = -0.002
        self.rewards.tgmp_contact_force.params["threshold"] = 400.0
        self.rewards.action_rate_l2.weight = -0.012
        self.rewards.action_smoothness_l2.weight = -0.025

        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsTailRiskSweepEnvCfg(
    KuavoS53StairsSwingAwareSweepEnvCfg
):
    """v12: optimize the lowest-clearance tail without changing fixed geometry."""

    rewards: TailRiskPredictiveSweepStairsTrackingRewardsCfg = (
        TailRiskPredictiveSweepStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.motion_feet_vel.weight = 0.50
        self.rewards.feet_slide_vel.weight = -0.30
        self.rewards.tgmp_contact_force.weight = -0.002
        self.rewards.tgmp_contact_force.params["threshold"] = 400.0
        self.rewards.action_rate_l2.weight = -0.012
        self.rewards.action_smoothness_l2.weight = -0.025


@configclass
class KuavoS53StairsToeBarrierEnvCfg(KuavoS53StairsTailRiskSweepEnvCfg):
    """v13: direct reference-swing toe barrier with a tightly preserved gait."""

    rewards: ReferenceSwingToeBarrierStairsTrackingRewardsCfg = (
        ReferenceSwingToeBarrierStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.predictive_riser_sweep.weight = -0.15
        self.rewards.motion_feet_pos.params["minimum_tracking_weight"] = 0.92
        self.rewards.motion_feet_vel.weight = 0.55
        self.rewards.feet_slide_vel.weight = -0.30
        self.rewards.tgmp_contact_force.weight = -0.002
        self.rewards.action_rate_l2.weight = -0.012
        self.rewards.action_smoothness_l2.weight = -0.025


@configclass
class KuavoS53StairsRunningMinBarrierEnvCfg(KuavoS53StairsToeBarrierEnvCfg):
    """v14: align training with full-swing worst toe-riser clearance."""

    rewards: RunningMinToeBarrierStairsTrackingRewardsCfg = (
        RunningMinToeBarrierStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.predictive_riser_sweep.weight = -0.10
        self.rewards.motion_feet_pos.params["minimum_tracking_weight"] = 0.94
        self.rewards.motion_feet_vel.weight = 0.55
        self.rewards.feet_slide_vel.weight = -0.32
        self.rewards.tgmp_contact_force.weight = -0.002
        self.rewards.action_rate_l2.weight = -0.012
        self.rewards.action_smoothness_l2.weight = -0.025


@configclass
class KuavoS53StairsTimeToRiserConeEnvCfg(
    KuavoS53StairsRunningMinBarrierEnvCfg
):
    """v15: dense velocity-cone shaping before a forefoot reaches a riser."""

    rewards: TimeToRiserConeStairsTrackingRewardsCfg = (
        TimeToRiserConeStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.predictive_riser_sweep.weight = -0.05
        self.rewards.motion_feet_pos.params["minimum_tracking_weight"] = 0.96
        self.rewards.motion_feet_vel.weight = 0.55
        self.rewards.feet_slide_vel.weight = -0.32
        self.rewards.tgmp_contact_force.weight = -0.002
        self.rewards.action_rate_l2.weight = -0.012
        self.rewards.action_smoothness_l2.weight = -0.025


@configclass
class KuavoS53StairsSpatialRiserCorridorEnvCfg(
    KuavoS53StairsTimeToRiserConeEnvCfg
):
    """v16: spatial riser corridor that cannot be avoided by changing velocity."""

    rewards: SpatialRiserCorridorStairsTrackingRewardsCfg = (
        SpatialRiserCorridorStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.predictive_riser_sweep.weight = -0.03
        self.rewards.motion_feet_pos.params["minimum_tracking_weight"] = 0.98
        self.rewards.motion_feet_vel.weight = 0.60


@configclass
class KuavoS53StairsClearanceSoftLandingEnvCfg(
    KuavoS53StairsSpatialRiserCorridorEnvCfg
):
    """v17: clearance-preserving refinement with local touchdown damping."""

    rewards: ClearanceSoftLandingStairsTrackingRewardsCfg = (
        ClearanceSoftLandingStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.spatial_riser_corridor.weight = -0.30
        self.rewards.motion_feet_pos.params["minimum_tracking_weight"] = 0.98
        self.rewards.motion_feet_vel.weight = 0.60
        self.rewards.feet_slide_vel.weight = -0.32
        self.rewards.tgmp_sole_support.weight = -1.8
        self.rewards.tgmp_contact_force.weight = -0.0025
        self.rewards.action_rate_l2.weight = -0.012
        self.rewards.action_smoothness_l2.weight = -0.025


@configclass
class KuavoS53StairsConservativeTailReplayEnvCfg(
    KuavoS53StairsClearanceSoftLandingEnvCfg
):
    """v18: short-horizon tail-risk refinement around model_92138."""

    rewards: ConservativeTailReplayStairsTrackingRewardsCfg = (
        ConservativeTailReplayStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.spatial_riser_corridor.weight = -0.30
        self.rewards.pre_touchdown_soft_landing.weight = -0.55
        self.rewards.tgmp_sole_support.weight = -2.0
        self.rewards.tgmp_contact_force.weight = -0.0025
        self.rewards.motion_feet_pos.params["minimum_tracking_weight"] = 0.99
        self.rewards.motion_feet_vel.weight = 0.62


@configclass
class KuavoS53StairsConstrainedTeacherProjectionEnvCfg(
    KuavoS53StairsClearanceSoftLandingEnvCfg
):
    """v20: v17 physical objectives with hard actor projection in PPO."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pre_touchdown_soft_landing.weight = -0.50
        self.rewards.spatial_riser_corridor.weight = -0.30
        self.rewards.motion_feet_pos.params["minimum_tracking_weight"] = 0.99


@configclass
class KuavoS53StairsRiserConstraintCaTEnvCfg(
    KuavoS53StairsClearanceSoftLandingEnvCfg
):
    """v21: direct physical riser-clearance cost with local CaT credit."""

    terminations: RiserConstraintCaTTerminationsCfg = (
        RiserConstraintCaTTerminationsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pre_touchdown_soft_landing.weight = -0.50
        self.rewards.spatial_riser_corridor.weight = -0.24
        self.rewards.motion_feet_pos.params["minimum_tracking_weight"] = 0.99
        self.rewards.motion_feet_vel.weight = 0.62
        self.rewards.feet_slide_vel.weight = -0.34


@configclass
class KuavoS53StairsRiserClearanceLagrangianEnvCfg(
    KuavoS53StairsClearanceSoftLandingEnvCfg
):
    """v22: preserve complete episodes while adapting rigid-foot safety cost."""

    rewards: RiserClearanceLagrangianStairsTrackingRewardsCfg = (
        RiserClearanceLagrangianStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pre_touchdown_soft_landing.weight = -0.50
        self.rewards.spatial_riser_corridor.weight = -0.24
        self.rewards.motion_feet_pos.params["minimum_tracking_weight"] = 0.99
        self.rewards.motion_feet_vel.weight = 0.62
        self.rewards.feet_slide_vel.weight = -0.34


@configclass
class KuavoS53StairsSharedClearanceCmdpEnvCfg(
    KuavoS53StairsClearanceSoftLandingEnvCfg
):
    """v23: physical task reward plus a separate shared-geometry CMDP cost."""

    rewards: SharedClearanceCmdpStairsTrackingRewardsCfg = (
        SharedClearanceCmdpStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pre_touchdown_soft_landing.weight = -0.50
        self.rewards.spatial_riser_corridor.weight = -0.24
        self.rewards.motion_feet_pos.params["minimum_tracking_weight"] = 0.99
        self.rewards.motion_feet_vel.weight = 0.62
        self.rewards.feet_slide_vel.weight = -0.34


@configclass
class KuavoS53StairsPidWorstSegmentCmdpEnvCfg(
    KuavoS53StairsSharedClearanceCmdpEnvCfg
):
    """v24 keeps the v23 shared physical cost; aggregation lives in storage."""

    pass


@configclass
class KuavoS53StairsCalibratedTailCmdpEnvCfg(
    KuavoS53StairsSharedClearanceCmdpEnvCfg
):
    """v25 keeps the shared physical cost and calibrates tail risk in PPO."""

    pass


@configclass
class KuavoS53StairsContactGatedCmdpEnvCfg(
    KuavoS53StairsSharedClearanceCmdpEnvCfg
):
    """v26: evaluate rigid-foot clearance only during low-contact swing."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.shared_riser_clearance_cost.params[
            "require_low_contact_for_swing"
        ] = True


@configclass
class KuavoS53StairsContactGatedMarginCmdpEnvCfg(
    KuavoS53StairsContactGatedCmdpEnvCfg
):
    """v27: densify contact-gated credit inside a 6 mm safety corridor."""

    def __post_init__(self):
        super().__post_init__()
        params = self.rewards.shared_riser_clearance_cost.params
        params["safety_distance"] = 0.006
        params["hard_distance"] = 0.002


@configclass
class KuavoS53StairsForwardUpDownStableEnvCfg(KuavoS53StairsForwardUpDownEnvCfg):
    """First stable-descent stage with a deterministic platform switch gate."""

    rewards: ForwardUpDownStableStairsTrackingRewardsCfg = ForwardUpDownStableStairsTrackingRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = FORWARD_UPDOWN_STABLE_MOTION
        self.commands.motion.adaptive_uniform_ratio = 0.35
        self.commands.motion.pose_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.commands.motion.velocity_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

        self.rewards.motion_forward_velocity.weight = 0.75
        self.rewards.motion_feet_pos.weight = 2.5
        self.rewards.motion_feet_pos.params["std"] = 0.085
        self.rewards.motion_feet_xy.weight = 2.0
        self.rewards.motion_feet_xy.params["std"] = 0.07
        self.rewards.motion_height.weight = 2.25
        self.rewards.motion_phase_progress.weight = 1.0
        self.rewards.feet_slide_vel.weight = -1.0
        self.rewards.feet_contact_forces.weight = -0.2

        # Stage one is deterministic. Coordinate and dynamics randomization are
        # introduced only after the fixed gate and clean contacts are validated.
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)

        self.terminations.anchor_pos.params["threshold"] = 0.42
        self.terminations.ee_body_pos.params["threshold"] = 0.45
        self.terminations.anchor_ori.params["threshold"] = 0.95


@configclass
class KuavoS53StairsStepToDownEnvCfg(KuavoS53StairsForwardUpDownStableEnvCfg):
    """Fixed-coordinate four-level step-to descent before robustness randomization."""

    rewards: StepToDownStairsTrackingRewardsCfg = StepToDownStairsTrackingRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        motion_cfg = self.commands.motion
        self.commands.motion = mdp.StepToGatedMotionCommandCfg(
            motion_file=STEP_TO_DOWN_MOTION,
            asset_name=motion_cfg.asset_name,
            anchor_body=motion_cfg.anchor_body,
            resampling_time_range=motion_cfg.resampling_time_range,
            debug_vis=False,
            pose_range=motion_cfg.pose_range,
            velocity_range=motion_cfg.velocity_range,
            joint_position_range=motion_cfg.joint_position_range,
            adaptive_kernel_size=motion_cfg.adaptive_kernel_size,
            adaptive_lambda=motion_cfg.adaptive_lambda,
            adaptive_uniform_ratio=0.40,
            adaptive_alpha=motion_cfg.adaptive_alpha,
            start_hold_steps=motion_cfg.start_hold_steps,
            end_hold_steps=motion_cfg.end_hold_steps,
            anchor_pos_threshold=motion_cfg.anchor_pos_threshold,
            anchor_ori_threshold=motion_cfg.anchor_ori_threshold,
            body_names=motion_cfg.body_names,
            gate_frames=STEP_TO_GATE_FRAMES,
            gate_body_names=("leg_l6_link", "leg_r6_link"),
            gate_position_tolerance=0.035,
            gate_foot_speed_tolerance=0.08,
            gate_anchor_speed_tolerance=0.08,
            gate_anchor_angular_speed_tolerance=0.25,
            gate_stable_steps=10,
        )
        self.episode_length_s = 34.0

        self.rewards.motion_feet_pos.weight = 3.0
        self.rewards.motion_feet_pos.params["std"] = 0.075
        self.rewards.motion_feet_xy.weight = 2.5
        self.rewards.motion_feet_xy.params["std"] = 0.06
        self.rewards.motion_phase_progress.weight = 0.75
        self.rewards.feet_slide_vel.weight = -1.25
        self.rewards.feet_contact_forces.weight = -0.30

        # This stage intentionally keeps every switch coordinate deterministic.
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)

        self.terminations.anchor_pos.params["threshold"] = 0.48
        self.terminations.ee_body_pos.params["threshold"] = 0.52
        self.terminations.anchor_ori.params["threshold"] = 0.95


@configclass
class KuavoS53StairsStepToDownGateFixedEnvCfg(KuavoS53StairsStepToDownEnvCfg):
    """Gate-frame corrected fine-tuning task with conservative contact shaping."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.descent_feet_slide.weight = -2.25
        self.rewards.descent_landing_impact.weight = -1.5
        self.rewards.descent_landing_impact.params["force_threshold"] = 750.0
        self.rewards.right_swing_backward_kick.func = (
            mdp.motion_frame_ranges_backward_body_velocity_log_penalty
        )
        self.rewards.right_swing_backward_kick.weight = -0.75
        self.rewards.right_swing_backward_kick.params["velocity_scale"] = 0.12
        self.rewards.feet_contact_forces.weight = -0.35


@configclass
class KuavoS53StairsStepToDownNosingSafeEnvCfg(
    KuavoS53StairsStepToDownGateFixedEnvCfg
):
    """Fine-tune model_113994 without nosing strikes or gate-stall regression."""

    rewards: StepToDownNosingSafeStairsTrackingRewardsCfg = (
        StepToDownNosingSafeStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.rewards.descent_feet_under_clearance.weight = -1.5
        self.rewards.pre_descent_stability.weight = -2.25
        self.rewards.pre_descent_stability.params["linear_velocity_scale"] = 0.08
        self.rewards.pre_descent_stability.params["angular_velocity_scale"] = 0.25
        self.rewards.double_foot_gate_velocity.weight = -1.75
        self.rewards.double_foot_gate_velocity.params["speed_scale"] = 0.08
        self.rewards.double_foot_gate_stability.weight = -2.5
        self.rewards.double_foot_gate_stability.params["linear_velocity_scale"] = 0.08
        self.rewards.double_foot_gate_stability.params["angular_velocity_scale"] = 0.25


@configclass
class KuavoS53StairsStepToDownPreserveEnvCfg(
    KuavoS53StairsStepToDownNosingSafeEnvCfg
):
    """Teacher-preserved fine-tuning with anti-stall gate objectives."""

    rewards: StepToDownPreserveStairsTrackingRewardsCfg = (
        StepToDownPreserveStairsTrackingRewardsCfg()
    )
    terminations: StepToDownPreserveTerminationsCfg = StepToDownPreserveTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Full-sequence and gate-approach episodes remain visible despite adaptive sampling.
        self.commands.motion.zero_start_fraction = 0.30
        self.commands.motion.gate_approach_fraction = 0.35
        self.commands.motion.gate_approach_steps = 45
        self.commands.motion.gate_max_wait_steps = 100

        # Keep the edge terms local and weaker than the verified teacher behavior.
        self.rewards.left_nosing_clearance.weight = -1.0
        self.rewards.left_nosing_early_contact.weight = -1.0
        self.rewards.descent_feet_under_clearance.weight = -1.0
        self.rewards.pre_descent_stability.weight = -1.0
        self.rewards.double_foot_gate_stability.weight = -1.0


@configclass
class KuavoS53StairsStepToDownTgmpFootholdEnvCfg(
    KuavoS53StairsStepToDownPreserveEnvCfg
):
    """T-GMP-inspired fixed-terrain foothold refinement without gate regression."""

    rewards: StepToDownTgmpFootholdStairsTrackingRewardsCfg = (
        StepToDownTgmpFootholdStairsTrackingRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        motion_cfg = self.commands.motion

        # Preserve full tasks and gates while explicitly sampling each risky nosing approach.
        motion_cfg.zero_start_fraction = 0.25
        motion_cfg.gate_approach_fraction = 0.25
        motion_cfg.foothold_approach_fraction = 0.35
        motion_cfg.foothold_approach_frames = tuple(
            frame_range[0] for frame_range in STEP_TO_LEFT_FOOTHOLD_APPROACH_RANGES
        )
        motion_cfg.foothold_approach_steps = 30

        # The S53 sole spans x=[-0.063, 0.165] m about the ankle. The old targets
        # leave the toe about 3 cm beyond the next edge, so move both descent
        # footholds backward while smoothly preserving the platform approach.
        motion_cfg.foothold_target_x_offset = -0.055
        motion_cfg.left_target_offset_ramp = (763, 807)
        motion_cfg.right_target_offset_ramp = (823, 867)
        motion_cfg.gate_foothold_edge_x_offsets = STEP_TO_GATE_FOOTHOLD_EDGES
        motion_cfg.gate_foothold_ground_heights = STEP_TO_GATE_FOOTHOLD_HEIGHTS
        motion_cfg.gate_foothold_tread_length = STAIR_TREAD
        motion_cfg.gate_foothold_margin = 0.012
        motion_cfg.gate_sole_max_gap = 0.025
        motion_cfg.gate_sole_local_points = (
            (-0.063, -0.040, -0.062),
            (-0.063, 0.040, -0.062),
            (0.165, -0.040, -0.062),
            (0.165, 0.040, -0.062),
        )

        # Replace ankle-center and net-force proxies with explicit geometry terms.
        self.rewards.left_nosing_clearance = None
        self.rewards.left_nosing_early_contact = None


@configclass
class KuavoS53StairsStep1EnvCfg_PLAY(KuavoS53StairsStep1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsFullEnvCfg_PLAY(KuavoS53StairsFullEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsUpDownEnvCfg_PLAY(KuavoS53StairsUpDownEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.5
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsForwardDownEnvCfg_PLAY(KuavoS53StairsForwardDownEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsForwardUpDownEnvCfg_PLAY(KuavoS53StairsForwardUpDownEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsTgmpRewardBaselineEnvCfg_PLAY(
    KuavoS53StairsTgmpRewardBaselineEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsTgmpTerrainConditionedEnvCfg_PLAY(
    KuavoS53StairsTgmpTerrainConditionedEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsTgmpRiserSafeEnvCfg_PLAY(
    KuavoS53StairsTgmpRiserSafeEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsMindStepsTactEnvCfg_PLAY(
    KuavoS53StairsMindStepsTactEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsPredictiveSweepEnvCfg_PLAY(
    KuavoS53StairsPredictiveSweepEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsSwingAwareSweepEnvCfg_PLAY(
    KuavoS53StairsSwingAwareSweepEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsTailRiskSweepEnvCfg_PLAY(
    KuavoS53StairsTailRiskSweepEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsToeBarrierEnvCfg_PLAY(KuavoS53StairsToeBarrierEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsRunningMinBarrierEnvCfg_PLAY(
    KuavoS53StairsRunningMinBarrierEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsTimeToRiserConeEnvCfg_PLAY(
    KuavoS53StairsTimeToRiserConeEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsSpatialRiserCorridorEnvCfg_PLAY(
    KuavoS53StairsSpatialRiserCorridorEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsClearanceSoftLandingEnvCfg_PLAY(
    KuavoS53StairsClearanceSoftLandingEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsConservativeTailReplayEnvCfg_PLAY(
    KuavoS53StairsConservativeTailReplayEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsConstrainedTeacherProjectionEnvCfg_PLAY(
    KuavoS53StairsConstrainedTeacherProjectionEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsRiserConstraintCaTEnvCfg_PLAY(
    KuavoS53StairsRiserConstraintCaTEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)
        self.terminations.riser_clearance_cat = None


@configclass
class KuavoS53StairsRiserClearanceLagrangianEnvCfg_PLAY(
    KuavoS53StairsRiserClearanceLagrangianEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsSharedClearanceCmdpEnvCfg_PLAY(
    KuavoS53StairsSharedClearanceCmdpEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsPidWorstSegmentCmdpEnvCfg_PLAY(
    KuavoS53StairsPidWorstSegmentCmdpEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsCalibratedTailCmdpEnvCfg_PLAY(
    KuavoS53StairsCalibratedTailCmdpEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.commands.motion.rel_standing_envs = 0.0
        self.events.base_external_force_torque = None
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsContactGatedCmdpEnvCfg_PLAY(
    KuavoS53StairsContactGatedCmdpEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.commands.motion.rel_standing_envs = 0.0
        self.events.base_external_force_torque = None
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsContactGatedMarginCmdpEnvCfg_PLAY(
    KuavoS53StairsContactGatedMarginCmdpEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.commands.motion.rel_standing_envs = 0.0
        self.events.base_external_force_torque = None
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsForwardUpDownStableEnvCfg_PLAY(KuavoS53StairsForwardUpDownStableEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsStepToDownEnvCfg_PLAY(KuavoS53StairsStepToDownEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsStepToDownGateFixedEnvCfg_PLAY(
    KuavoS53StairsStepToDownGateFixedEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsStepToDownNosingSafeEnvCfg_PLAY(
    KuavoS53StairsStepToDownNosingSafeEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsStepToDownPreserveEnvCfg_PLAY(
    KuavoS53StairsStepToDownPreserveEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)


@configclass
class KuavoS53StairsStepToDownTgmpFootholdEnvCfg_PLAY(
    KuavoS53StairsStepToDownTgmpFootholdEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)
