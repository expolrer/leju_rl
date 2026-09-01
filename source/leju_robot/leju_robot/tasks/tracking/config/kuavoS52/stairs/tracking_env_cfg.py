from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from leju_robot.assets.motion_data import MOTION_DIR
from leju_robot.tasks.tracking import mdp
from leju_robot.tasks.tracking.config.kuavoS53.stairs.tracking_env_cfg import (
    KuavoS53StairsTgmpTerrainConditionedEnvCfg,
    TgmpTerrainConditionedStairsTrackingRewardsCfg,
)

from .kuavoS52 import KuavoS52_ACTION_SCALE, KuavoS52_TRACKING_CFG


S52_MODEL92099_MOTION = (
    f"{MOTION_DIR}/mimic/npz_data/kuavoS52_model92099_stairs_retargeted_50fps.npz"
)

# Official S52 MuJoCo uses six 5 mm spheres under each sole. These samples are
# expressed in the leg_*6_link frame and replace the S53 box-foot samples.
S52_FOREFOOT_SAMPLES = (
    (0.040, -0.050, -0.0595),
    (0.040, 0.050, -0.0595),
    (0.100, -0.050, -0.0595),
    (0.100, 0.050, -0.0595),
    (0.150, -0.050, -0.0595),
    (0.150, 0.050, -0.0595),
    (0.17084, -0.050, -0.0595),
    (0.17084, 0.050, -0.0595),
)
S52_SOLE_SAMPLES = (
    (-0.073164, -0.050, -0.0595),
    (-0.073164, 0.000, -0.0595),
    (-0.073164, 0.050, -0.0595),
    (0.17084, -0.050, -0.0595),
    (0.17084, 0.000, -0.0595),
    (0.17084, 0.050, -0.0595),
)

# Explicit swing/contact schedule extracted once from the retargeted S52 motion.
# Training and Play share this table so a touchdown cannot change meaning at
# evaluation time. Each outer entry follows [left foot, right foot].
S52_STAIR_SWING_RANGES = (
    (
        (584, 604),
        (633, 653),
        (685, 704),
        (776, 842),
        (979, 1050),
        (1188, 1258),
    ),
    (
        (547, 570),
        (612, 630),
        (661, 681),
        (718, 738),
        (877, 950),
        (1088, 1162),
    ),
)


def _use_s52_policy_interface(cfg) -> None:
    cfg.scene.robot = KuavoS52_TRACKING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions.joint_pos.scale = KuavoS52_ACTION_SCALE
    cfg.actions.joint_pos.joint_names = KuavoS52_TRACKING_CFG.preserve_joint_order.joint_names

    ordered = {"asset_cfg": KuavoS52_TRACKING_CFG.preserve_joint_order}
    cfg.observations.policy.joint_pos.params = ordered
    cfg.observations.policy.joint_vel.params = ordered
    cfg.observations.critic.joint_pos.params = ordered
    cfg.observations.critic.joint_vel.params = ordered
    cfg.events.add_joint_default_pos.params = {
        "asset_cfg": KuavoS52_TRACKING_CFG.preserve_joint_order,
        "pos_distribution_params": (0.0, 0.0),
        "operation": "add",
    }


def _disable_domain_randomization(cfg) -> None:
    # First prove the fixed S52 route. Narrow DR is enabled only after nominal
    # ascent, platform traversal, descent, and landing pass physical rollout.
    for name in (
        "physics_material",
        "torso_com",
        "base_com",
        "add_torso_mass",
        "add_base_mass",
        "link_com",
        "add_link_mass",
        "scale_actuator_gains",
        "scale_joint_parameters",
        "push_robot",
        "base_external_force_torque",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)


@configclass
class KuavoS52PhaseAlignedRewardsCfg(TgmpTerrainConditionedStairsTrackingRewardsCfg):
    """Dense route-timing rewards for the first S52 stair adaptation stage."""

    motion_anchor_lateral_pos = RewTerm(
        func=mdp.motion_global_anchor_lateral_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.10},
    )
    motion_forward_pos = RewTerm(
        func=mdp.motion_global_anchor_forward_position_error_exp,
        weight=2.0,
        params={"command_name": "motion", "std": 0.16},
    )
    motion_height = RewTerm(
        func=mdp.motion_global_anchor_height_error_exp,
        weight=4.0,
        params={"command_name": "motion", "std": 0.07},
    )
    motion_forward_velocity = RewTerm(
        func=mdp.motion_anchor_forward_velocity_error_exp,
        weight=1.5,
        params={"command_name": "motion", "std": 0.16},
    )
    backward_velocity = RewTerm(
        func=mdp.motion_backward_velocity_penalty,
        weight=-0.5,
        params={"command_name": "motion"},
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


@configclass
class KuavoS52SoftPlateauRewardsCfg(KuavoS52PhaseAlignedRewardsCfg):
    """Soft platform hold that preserves credit assignment into descent."""

    platform_anchor_pose = RewTerm(
        func=mdp.motion_phase_anchor_position_error_exp,
        weight=6.0,
        params={
            "command_name": "motion",
            "std": 0.09,
            "phase_start": 700.0 / 1340.0,
            "phase_end": 960.0 / 1340.0,
        },
    )
    platform_anchor_stability = RewTerm(
        func=mdp.motion_phase_anchor_stability_penalty,
        weight=-0.5,
        params={
            "command_name": "motion",
            "linear_velocity_scale": 0.18,
            "angular_velocity_scale": 0.35,
            "phase_start": 700.0 / 1340.0,
            "phase_end": 960.0 / 1340.0,
        },
    )


@configclass
class KuavoS52SoftPlateauTrustRewardsCfg(KuavoS52SoftPlateauRewardsCfg):
    """v4 platform support terms with dense credit for pelvis-height failure."""

    platform_anchor_pose = RewTerm(
        func=mdp.motion_phase_anchor_position_error_exp,
        weight=4.0,
        params={
            "command_name": "motion",
            "std": 0.10,
            "phase_start": 700.0 / 1340.0,
            "phase_end": 960.0 / 1340.0,
        },
    )
    platform_anchor_orientation = RewTerm(
        func=mdp.motion_phase_anchor_orientation_error_exp,
        weight=2.0,
        params={
            "command_name": "motion",
            "std": 0.20,
            "phase_start": 700.0 / 1340.0,
            "phase_end": 960.0 / 1340.0,
        },
    )
    platform_pelvis_under_height = RewTerm(
        func=mdp.motion_frame_ranges_body_under_reference_height_penalty,
        weight=-2.5,
        params={
            "command_name": "motion",
            "height_scale": 0.10,
            "allowed_below": 0.025,
            "frame_ranges": [(700, 960)],
            "body_names": ["base_link"],
        },
    )
    platform_feet_position = RewTerm(
        func=mdp.motion_frame_ranges_body_position_error_exp,
        weight=2.0,
        params={
            "command_name": "motion",
            "std": 0.10,
            "frame_ranges": [(700, 960)],
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    platform_feet_velocity = RewTerm(
        func=mdp.motion_frame_ranges_body_velocity_penalty,
        weight=-0.35,
        params={
            "command_name": "motion",
            "speed_scale": 0.20,
            "frame_ranges": [(700, 960)],
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class KuavoS52ContactPhaseRewardsCfg(KuavoS52SoftPlateauTrustRewardsCfg):
    """v5 contact-conditioned credit for the S52 platform-to-descent transition."""

    # v4 slowed both feet and therefore opposed the intended swing. Settle only
    # the foot identified as stance by the retargeted reference.
    platform_feet_velocity = None
    contact_phase_swing_forward_overshoot = RewTerm(
        func=mdp.motion_frame_ranges_reference_swing_forward_overshoot_penalty,
        weight=-2.0,
        params={
            "command_name": "motion",
            "forward_scale": 0.05,
            "allowed_forward": 0.020,
            "swing_speed_threshold": 0.14,
            "relative_speed_margin": 0.04,
            "frame_ranges": [(670, 960)],
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    contact_phase_swing_height_shortage = RewTerm(
        func=mdp.motion_frame_ranges_reference_swing_height_shortage_penalty,
        weight=-2.5,
        params={
            "command_name": "motion",
            "height_scale": 0.06,
            "allowed_below": 0.015,
            "swing_speed_threshold": 0.14,
            "relative_speed_margin": 0.04,
            "frame_ranges": [(670, 960)],
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    contact_phase_swing_contact = RewTerm(
        func=mdp.motion_frame_ranges_reference_swing_contact_penalty,
        weight=-0.8,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "contact_force_threshold": 50.0,
            "contact_force_scale": 100.0,
            "swing_speed_threshold": 0.14,
            "relative_speed_margin": 0.04,
            "frame_ranges": [(670, 960)],
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    contact_phase_stance_velocity = RewTerm(
        func=mdp.motion_frame_ranges_reference_stance_velocity_penalty,
        weight=-0.4,
        params={
            "command_name": "motion",
            "speed_scale": 0.20,
            "swing_speed_threshold": 0.14,
            "relative_speed_margin": 0.04,
            "frame_ranges": [(670, 960)],
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    contact_phase_stance_contact_loss = RewTerm(
        func=mdp.motion_frame_ranges_reference_stance_contact_loss_penalty,
        weight=-0.35,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "support_force_threshold": 60.0,
            "swing_speed_threshold": 0.14,
            "relative_speed_margin": 0.04,
            "frame_ranges": [(670, 960)],
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class KuavoS52ScheduledFootholdRewardsCfg(KuavoS52ContactPhaseRewardsCfg):
    """v6 explicit foothold and touchdown schedule for the full stair route."""

    # Instantaneous speed classification was fragmented around liftoff and
    # touchdown. Replace every v5 proxy with the shared offline phase table.
    contact_phase_swing_forward_overshoot = None
    contact_phase_swing_height_shortage = None
    contact_phase_swing_contact = None
    contact_phase_stance_velocity = None
    contact_phase_stance_contact_loss = None

    scheduled_swing_path = RewTerm(
        func=mdp.motion_scheduled_swing_path_tracking_exp,
        weight=5.0,
        params={
            "command_name": "motion",
            "horizontal_std": 0.050,
            "vertical_std": 0.040,
            "foot_frame_ranges": S52_STAIR_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    scheduled_future_foothold_xy = RewTerm(
        func=mdp.motion_scheduled_future_foothold_xy_tracking_exp,
        weight=1.5,
        params={
            "command_name": "motion",
            "std": 0.080,
            "foot_frame_ranges": S52_STAIR_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    scheduled_swing_height_shortage = RewTerm(
        func=mdp.motion_scheduled_swing_height_shortage_penalty,
        weight=-3.5,
        params={
            "command_name": "motion",
            "height_scale": 0.040,
            "allowed_below": 0.008,
            "foot_frame_ranges": S52_STAIR_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    scheduled_swing_early_contact = RewTerm(
        func=mdp.motion_scheduled_swing_early_contact_penalty,
        weight=-2.5,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "contact_force_threshold": 40.0,
            "contact_force_scale": 100.0,
            "release_grace_frames": 3,
            "touchdown_grace_frames": 4,
            "foot_frame_ranges": S52_STAIR_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    scheduled_touchdown_position = RewTerm(
        func=mdp.motion_scheduled_touchdown_position_tracking_exp,
        weight=3.0,
        params={
            "command_name": "motion",
            "horizontal_std": 0.050,
            "vertical_std": 0.035,
            "touchdown_window": 10,
            "foot_frame_ranges": S52_STAIR_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    scheduled_touchdown_contact_loss = RewTerm(
        func=mdp.motion_scheduled_touchdown_contact_loss_penalty,
        weight=-0.75,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "support_force_threshold": 60.0,
            "touchdown_window": 10,
            "foot_frame_ranges": S52_STAIR_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    scheduled_opposite_stance_velocity = RewTerm(
        func=mdp.motion_scheduled_opposite_stance_velocity_penalty,
        weight=-0.5,
        params={
            "command_name": "motion",
            "speed_scale": 0.18,
            "foot_frame_ranges": S52_STAIR_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )
    scheduled_opposite_stance_contact_loss = RewTerm(
        func=mdp.motion_scheduled_opposite_stance_contact_loss_penalty,
        weight=-0.5,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_l6_link", "leg_r6_link"]
            ),
            "support_force_threshold": 60.0,
            "foot_frame_ranges": S52_STAIR_SWING_RANGES,
            "body_names": ["leg_l6_link", "leg_r6_link"],
        },
    )


@configclass
class KuavoS52StairsTransferEnvCfg(KuavoS53StairsTgmpTerrainConditionedEnvCfg):
    """Nominal S52 transfer task with model_92099 kept as a frozen teacher."""

    def __post_init__(self):
        super().__post_init__()
        _use_s52_policy_interface(self)
        _disable_domain_randomization(self)

        self.commands.motion.motion_file = S52_MODEL92099_MOTION
        self.commands.motion.zero_start_fraction = 0.40
        self.commands.motion.terrain_focus_fraction = 0.40
        self.commands.motion.adaptive_uniform_ratio = 0.20

        self.rewards.tgmp_toe_clearance.params["local_points"] = S52_FOREFOOT_SAMPLES
        self.rewards.tgmp_sole_support.params["local_points"] = S52_SOLE_SAMPLES

        # Preserve route imitation first. S52-specific contact rewards are local
        # and deliberately conservative so the teacher is not treated as final.
        self.rewards.motion_global_anchor_pos.weight = 1.0
        self.rewards.motion_global_anchor_ori.weight = 0.75
        self.rewards.motion_body_pos.weight = 0.75
        self.rewards.motion_body_ori.weight = 0.75
        self.rewards.motion_feet_pos.weight = 2.0
        self.rewards.motion_feet_pos.params["std"] = 0.10
        self.rewards.motion_feet_vel.weight = 0.75
        self.rewards.feet_slide_vel.weight = -0.35
        self.rewards.tgmp_contact_force.weight = -0.002
        self.rewards.tgmp_contact_force.params["threshold"] = 500.0
        self.rewards.tgmp_toe_clearance.weight = -2.0
        self.rewards.tgmp_sole_support.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness_l2.weight = -0.02

        self.terminations.anchor_pos.params["threshold"] = 0.55
        self.terminations.ee_body_pos.params["threshold"] = 0.60
        self.terminations.anchor_ori.params["threshold"] = 1.0


@configclass
class KuavoS52StairsTransferEnvCfg_PLAY(KuavoS52StairsTransferEnvCfg):
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


@configclass
class KuavoS52StairsPhaseAlignedEnvCfg(KuavoS52StairsTransferEnvCfg):
    """v2: reject early platform drop and late-route phase drift explicitly."""

    rewards: KuavoS52PhaseAlignedRewardsCfg = KuavoS52PhaseAlignedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.commands.motion.zero_start_fraction = 0.35
        self.commands.motion.terrain_focus_fraction = 0.50
        self.commands.motion.adaptive_uniform_ratio = 0.15
        self.commands.motion.terrain_focus_frames = (
            680,
            740,
            820,
            900,
            980,
            1060,
            1140,
            1220,
        )
        self.commands.motion.terrain_focus_approach_steps = 35

        # v1 increased mean reward while the pelvis dropped about 0.44 m early.
        # Decomposed route terms make that failure dense and directly observable.
        self.rewards.motion_global_anchor_pos.weight = 0.5
        self.rewards.motion_body_lin_vel.weight = 2.5
        self.rewards.motion_body_ang_vel.weight = 2.0
        self.rewards.motion_feet_pos.weight = 2.5
        self.rewards.motion_feet_pos.params["std"] = 0.09
        self.rewards.motion_feet_vel.weight = 0.5

        # Keep contact shaping local and conservative during nominal transfer.
        self.rewards.feet_slide_vel.weight = -0.45
        self.rewards.tgmp_contact_force.weight = -0.003
        self.rewards.tgmp_contact_force.params["threshold"] = 350.0
        self.rewards.tgmp_toe_clearance.weight = -2.0
        self.rewards.tgmp_sole_support.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.012
        self.rewards.action_smoothness_l2.weight = -0.025

        # A 0.44 m early drop must not be counted as a successful full route.
        self.terminations.anchor_pos.params["threshold"] = 0.32
        self.terminations.ee_body_pos.params["threshold"] = 0.45


@configclass
class KuavoS52StairsPhaseAlignedEnvCfg_PLAY(KuavoS52StairsPhaseAlignedEnvCfg):
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


@configclass
class KuavoS52StairsSoftPlateauEnvCfg(KuavoS52StairsPhaseAlignedEnvCfg):
    """v3: learn the platform hold softly before optimizing stair descent."""

    rewards: KuavoS52SoftPlateauRewardsCfg = KuavoS52SoftPlateauRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.zero_start_fraction = 0.30
        self.commands.motion.terrain_focus_fraction = 0.55
        self.commands.motion.adaptive_uniform_ratio = 0.15
        self.commands.motion.terrain_focus_frames = (
            660,
            700,
            740,
            780,
            820,
            860,
            900,
            940,
            1000,
            1100,
        )
        self.commands.motion.terrain_focus_approach_steps = 30

        # v2's 0.32 m hard gate cut every rollout near frame 824. Keep the
        # failure observable while allowing gradients to reach the descent.
        self.terminations.anchor_pos.params["threshold"] = 0.52
        self.terminations.ee_body_pos.params["threshold"] = 0.60


@configclass
class KuavoS52StairsSoftPlateauEnvCfg_PLAY(KuavoS52StairsSoftPlateauEnvCfg):
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


@configclass
class KuavoS52StairsSoftPlateauTrustEnvCfg(KuavoS52StairsSoftPlateauEnvCfg):
    """v4: preserve the S53 prior while fixing S52 platform support collapse."""

    rewards: KuavoS52SoftPlateauTrustRewardsCfg = KuavoS52SoftPlateauTrustRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.zero_start_fraction = 0.30
        self.commands.motion.terrain_focus_fraction = 0.60
        self.commands.motion.adaptive_uniform_ratio = 0.10
        self.commands.motion.terrain_focus_frames = (
            660,
            700,
            740,
            780,
            820,
            860,
            900,
            940,
            1000,
            1100,
        )
        self.commands.motion.terrain_focus_approach_steps = 30

        # Keep the failure visible through the platform/descent transition.
        self.terminations.anchor_pos.params["threshold"] = 0.52
        self.terminations.ee_body_pos.params["threshold"] = 0.60
        self.terminations.anchor_ori.params["threshold"] = 0.85


@configclass
class KuavoS52StairsSoftPlateauTrustEnvCfg_PLAY(KuavoS52StairsSoftPlateauTrustEnvCfg):
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


@configclass
class KuavoS52StairsContactPhaseEnvCfg(KuavoS52StairsSoftPlateauTrustEnvCfg):
    """v5: preserve the route while correcting swing/stance contact timing."""

    rewards: KuavoS52ContactPhaseRewardsCfg = KuavoS52ContactPhaseRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.zero_start_fraction = 0.25
        self.commands.motion.terrain_focus_fraction = 0.65
        self.commands.motion.adaptive_uniform_ratio = 0.10
        self.commands.motion.terrain_focus_frames = (
            650,
            680,
            710,
            740,
            770,
            800,
            840,
            880,
            920,
            960,
        )
        self.commands.motion.terrain_focus_approach_steps = 25


@configclass
class KuavoS52StairsContactPhaseEnvCfg_PLAY(KuavoS52StairsContactPhaseEnvCfg):
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


@configclass
class KuavoS52StairsScheduledFootholdEnvCfg(KuavoS52StairsContactPhaseEnvCfg):
    """v6: explicit swing trajectory, future foothold, and touchdown timing."""

    rewards: KuavoS52ScheduledFootholdRewardsCfg = KuavoS52ScheduledFootholdRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.zero_start_fraction = 0.20
        self.commands.motion.terrain_focus_fraction = 0.72
        self.commands.motion.adaptive_uniform_ratio = 0.08
        self.commands.motion.terrain_focus_frames = (
            540,
            580,
            610,
            630,
            660,
            685,
            715,
            775,
            875,
            975,
            1085,
            1185,
        )
        self.commands.motion.terrain_focus_approach_steps = 20


@configclass
class KuavoS52StairsScheduledFootholdEnvCfg_PLAY(KuavoS52StairsScheduledFootholdEnvCfg):
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
