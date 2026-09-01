from isaaclab.utils import configclass

from ..stairs_contact_release.ppo_cfg import (
    KuavoS52StairsContactReleasePPORunnerCfg,
)


@configclass
class KuavoS52StairsActualReplayPPORunnerCfg(
    KuavoS52StairsContactReleasePPORunnerCfg
):
    """Conservative target-state adaptation around the failed S52 ascent swing."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS52_stairs_actual_replay"
        self.max_iterations = 60
        self.save_interval = 10
        self.algorithm.learning_rate = 2.0e-6
        self.algorithm.clip_param = 0.03
        self.algorithm.desired_kl = 7.0e-4
        self.algorithm.teacher_action_coef = 6.0
        self.algorithm.teacher_kl_coef = 0.30
        self.algorithm.max_grad_norm = 0.25
