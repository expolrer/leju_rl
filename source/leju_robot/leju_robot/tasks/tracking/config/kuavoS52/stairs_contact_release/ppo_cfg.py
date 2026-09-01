from isaaclab.utils import configclass

from ..stairs_reference_adaptation.ppo_cfg import (
    KuavoS52StairsReferenceAdaptationPPORunnerCfg,
)


@configclass
class KuavoS52StairsContactReleasePPORunnerCfg(
    KuavoS52StairsReferenceAdaptationPPORunnerCfg
):
    """Adapt only enough to release and place the S52 ascent swing feet."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS52_stairs_contact_release"
        self.max_iterations = 80
        self.save_interval = 10
        self.algorithm.learning_rate = 3.0e-6
        self.algorithm.clip_param = 0.04
        self.algorithm.desired_kl = 0.001
        self.algorithm.teacher_action_coef = 4.0
        self.algorithm.teacher_kl_coef = 0.20
        self.algorithm.max_grad_norm = 0.30
