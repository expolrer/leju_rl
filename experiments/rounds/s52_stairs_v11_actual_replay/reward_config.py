from isaaclab.utils import configclass

from leju_robot.assets.motion_data import MOTION_DIR

from ..stairs_contact_release.tracking_env_cfg import (
    KuavoS52ContactReleaseRewardsCfg,
    KuavoS52StairsContactReleaseEnvCfg,
)
from .actual_state_command import ActualStateReplayMotionCommandCfg


S52_ACTUAL_PREFIX_REPLAY = (
    f"{MOTION_DIR}/mimic/npz_data/"
    "kuavoS52_model92099_actual_prefix_replay_seed42.npz"
)


@configclass
class KuavoS52StairsActualReplayEnvCfg(KuavoS52StairsContactReleaseEnvCfg):
    """Train on frame-zero and actual S52 states before the failed second swing."""

    rewards: KuavoS52ContactReleaseRewardsCfg = KuavoS52ContactReleaseRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        motion_cfg = self.commands.motion
        self.commands.motion = ActualStateReplayMotionCommandCfg(
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
            adaptive_uniform_ratio=0.10,
            adaptive_alpha=motion_cfg.adaptive_alpha,
            start_hold_steps=motion_cfg.start_hold_steps,
            end_hold_steps=motion_cfg.end_hold_steps,
            anchor_pos_threshold=motion_cfg.anchor_pos_threshold,
            anchor_ori_threshold=motion_cfg.anchor_ori_threshold,
            debug_vis=False,
            actual_replay_file=S52_ACTUAL_PREFIX_REPLAY,
            actual_replay_rows=(145, 155, 165, 175, 185, 195, 200),
            actual_replay_fraction=0.45,
            zero_start_fraction=0.55,
        )


@configclass
class KuavoS52StairsActualReplayEnvCfg_PLAY(KuavoS52StairsActualReplayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 5.0
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.commands.motion.actual_replay_fraction = 0.0
        self.commands.motion.zero_start_fraction = 1.0
