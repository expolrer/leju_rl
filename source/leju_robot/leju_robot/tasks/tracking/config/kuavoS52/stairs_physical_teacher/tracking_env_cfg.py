from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from leju_robot.assets.motion_data import MOTION_DIR
from leju_robot.tasks.tracking import mdp
from leju_robot.tasks.tracking.config.kuavoS53.dance.kuavoS53 import KuavoS53_ACTION_SCALE

from ..stairs.tracking_env_cfg import (
    KuavoS52PhaseAlignedRewardsCfg,
    KuavoS52StairsTransferEnvCfg,
)


S52_PHYSICAL_TEACHER_MOTION = (
    f"{MOTION_DIR}/mimic/npz_data/kuavoS52_model92099_physical_teacher_canonical_50fps.npz"
)


@configclass
class KuavoS52PhysicalTeacherRewardsCfg(KuavoS52PhaseAlignedRewardsCfg):
    """Dense nominal-physics tracking around the successful S53 rollout."""

    # The former exponential forward term saturated after large platform drift.
    # Keep it for local shaping and add a non-saturating frame-aligned penalty.
    physical_teacher_anchor_position = RewTerm(
        func=mdp.motion_frame_ranges_anchor_position_l1_penalty,
        weight=-1.0,
        params={
            "command_name": "motion",
            "position_scale": 0.10,
            "frame_ranges": [(0, 1340)],
        },
    )


@configclass
class KuavoS52StairsPhysicalTeacherEnvCfg(KuavoS52StairsTransferEnvCfg):
    """S52 nominal task driven by the successful model_92099 physics trajectory."""

    rewards: KuavoS52PhysicalTeacherRewardsCfg = KuavoS52PhysicalTeacherRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Preserve the actor's normalized action and joint-relative observation
        # semantics. The S52 PD, effort limits, mass, inertia, and contacts remain
        # physical S52 values and are the dynamics to be adapted.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.925)
        self.scene.robot.init_state.joint_pos = {
            "leg_[l,r]1_joint": 0.0,
            "leg_[l,r]2_joint": 0.0,
            "leg_[l,r]3_joint": -0.4,
            "leg_[l,r]4_joint": 0.69,
            "leg_[l,r]5_joint": -0.33,
            "leg_[l,r]6_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "zarm_.*_joint": 0.0,
            "zhead_.*_joint": 0.0,
        }
        self.actions.joint_pos.scale = KuavoS53_ACTION_SCALE
        self.commands.motion.motion_file = S52_PHYSICAL_TEACHER_MOTION
        self.commands.motion.zero_start_fraction = 0.30
        self.commands.motion.terrain_focus_fraction = 0.60
        self.commands.motion.adaptive_uniform_ratio = 0.10
        self.commands.motion.terrain_focus_frames = (
            500,
            580,
            660,
            720,
            780,
            860,
            940,
            1020,
            1100,
            1180,
            1260,
        )
        self.commands.motion.terrain_focus_approach_steps = 25

        # Keep rollout failures observable without accepting a collapsed pelvis.
        self.terminations.anchor_pos.params["threshold"] = 0.42
        self.terminations.ee_body_pos.params["threshold"] = 0.50
        self.terminations.anchor_ori.params["threshold"] = 0.85


@configclass
class KuavoS52StairsPhysicalTeacherEnvCfg_PLAY(KuavoS52StairsPhysicalTeacherEnvCfg):
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
