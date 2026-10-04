from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from leju_robot.tasks.tracking.config.kuavoS54.dance.tracking_env_cfg import (
    ObservationsCfg as BaseObservationsCfg,
)

from ..stairs.tracking_env_cfg import S52_STAIR_SWING_RANGES
from ..stairs_support_foothold_planar_persistence.tracking_env_cfg import (
    KuavoS52StairsSupportFootholdPlanarPersistenceEnvCfg,
)
from . import observations as local_observations


FEET = ["leg_l6_link", "leg_r6_link"]


@configclass
class FutureContactPhaseResidualObservationsCfg(BaseObservationsCfg):
    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        future_foot_contacts = ObsTerm(
            func=local_observations.future_foot_contacts_with_phase_b,
            params={
                "command_name": "motion",
                "body_names": FEET,
                "foot_frame_ranges": S52_STAIR_SWING_RANGES,
                "time_horizon_frames": 180,
                "total_motion_frames": 1340,
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class KuavoS52StairsFutureContactPhaseResidualEnvCfg(
    KuavoS52StairsSupportFootholdPlanarPersistenceEnvCfg
):
    observations: FutureContactPhaseResidualObservationsCfg = (
        FutureContactPhaseResidualObservationsCfg()
    )


@configclass
class KuavoS52StairsFutureContactPhaseResidualEnvCfg_PLAY(
    KuavoS52StairsFutureContactPhaseResidualEnvCfg
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

