from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from ..stairs_support_foothold_planar_persistence.ppo_cfg import (
    KuavoS52StairsSupportFootholdPlanarPersistencePPORunnerCfg,
)


@configclass
class ResidualContactActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ResidualContactActorCritic"
    base_obs_dim: int = 148
    adapter_hidden_dims: list[int] = [64, 32]
    adapter_scale: float = 0.02


@configclass
class KuavoS52StairsFutureContactResidualRunnerCfg(
    KuavoS52StairsSupportFootholdPlanarPersistencePPORunnerCfg
):
    policy = ResidualContactActorCriticCfg(
        init_noise_std=0.02,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        base_obs_dim=148,
        adapter_hidden_dims=[64, 32],
        adapter_scale=0.02,
    )

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS52_stairs_future_contact_residual"
        self.max_iterations = 8
        self.save_interval = 1
        self.algorithm.learning_rate = 1.0e-6
        self.algorithm.adaptive_min_learning_rate = 1.0e-6
        self.algorithm.adaptive_max_learning_rate = 1.0e-6
        self.algorithm.teacher_action_coef = 2.0
        self.algorithm.teacher_kl_coef = 0.25
        self.algorithm.teacher_action_tail_fraction = 0.02
        self.algorithm.teacher_action_tail_coef = 2.0
        self.algorithm.teacher_hard_action_rmse_limit = 0.002
        self.algorithm.teacher_projection_iterations = 6
        self.algorithm.teacher_checkpoint = (
            "/home/zzx23457/hhw/LejuLab-Train/logs/rsl_rl/"
            "kuavoS52_stairs_future_contact_residual/"
            "s52_future_contact_residual_seed_v67_20260914/model_92150.pt"
        )

