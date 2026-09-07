from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from leju_robot.assets.motion_data import MOTION_DIR
from leju_robot.tasks.tracking import mdp
from leju_robot.tasks.tracking.config.kuavoS53.dance.kuavoS53 import (
    KuavoS53_ACTION_SCALE,
)

from ..stairs.tracking_env_cfg import (
    KuavoS52PhaseAlignedRewardsCfg,
    KuavoS52StairsTransferEnvCfg,
)


S52_IDEAL_REFERENCE_MOTION = (
    f"{MOTION_DIR}/mimic/npz_data/kuavoS52_model92099_stairs_retargeted_50fps.npz"
)


@configclass
class KuavoS52ReferenceAdaptationRewardsCfg(KuavoS52PhaseAlignedRewardsCfg):
    """Dense route credit without replacing the policy's command distribution."""

    reference_anchor_position_l1 = RewTerm(
        func=mdp.motion_frame_ranges_anchor_position_l1_penalty,
        weight=-0.75,
        params={
            "command_name": "motion",
            "position_scale": 0.10,
            "frame_ranges": [(0, 1340)],
        },
    )


@configclass
class KuavoS52StairsReferenceAdaptationEnvCfg(KuavoS52StairsTransferEnvCfg):
    """Adapt S53 stair behavior to fixed S52 dynamics and contact geometry."""

    rewards: KuavoS52ReferenceAdaptationRewardsCfg = (
        KuavoS52ReferenceAdaptationRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        # Keep the exact actor-side semantics used by model_92099. S52 mass,
        # inertia, torque limits, PD gains, and collision geometry stay native.
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

        # The ideal retarget is the command because it matches the 148-D
        # normalizer distribution. The successful S53 physical rollout remains
        # a separate SFT/distillation dataset, never a command-motion substitute.
        self.commands.motion.motion_file = S52_IDEAL_REFERENCE_MOTION
        self.commands.motion.zero_start_fraction = 0.45
        self.commands.motion.terrain_focus_fraction = 0.45
        self.commands.motion.adaptive_uniform_ratio = 0.10
        self.commands.motion.terrain_focus_frames = (
            120,
            150,
            180,
            210,
            240,
            270,
            300,
            340,
        )
        self.commands.motion.terrain_focus_approach_steps = 25

        # The direct S52 rollout first diverged at the initial ascent around
        # frame 220. Keep the failure dense while avoiding a platform hard gate.
        self.rewards.motion_global_anchor_pos.weight = 1.0
        self.rewards.motion_global_anchor_ori.weight = 0.75
        self.rewards.motion_body_pos.weight = 1.0
        self.rewards.motion_body_ori.weight = 0.75
        self.rewards.motion_feet_pos.weight = 2.0
        self.rewards.motion_feet_pos.params["std"] = 0.10
        self.rewards.motion_feet_vel.weight = 0.75
        self.rewards.feet_slide_vel.weight = -0.35
        self.rewards.tgmp_contact_force.weight = -0.002
        self.rewards.tgmp_contact_force.params["threshold"] = 500.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness_l2.weight = -0.02

        self.terminations.anchor_pos.params["threshold"] = 0.42
        self.terminations.ee_body_pos.params["threshold"] = 0.50
        self.terminations.anchor_ori.params["threshold"] = 0.85


@configclass
class KuavoS52StairsReferenceAdaptationEnvCfg_PLAY(
    KuavoS52StairsReferenceAdaptationEnvCfg
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
