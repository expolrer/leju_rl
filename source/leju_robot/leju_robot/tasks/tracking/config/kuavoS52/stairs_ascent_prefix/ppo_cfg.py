from isaaclab.utils import configclass

from ..stairs_reference_adaptation.ppo_cfg import (
    KuavoS52StairsReferenceAdaptationPPORunnerCfg,
)


@configclass
class KuavoS52StairsAscentPrefixPPORunnerCfg(
    KuavoS52StairsReferenceAdaptationPPORunnerCfg
):
    """Adapt model_92099 on continuous S52 ascent prefixes."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS52_stairs_ascent_prefix"
        self.max_iterations = 120
        self.save_interval = 10
        self.algorithm.learning_rate = 8.0e-6
        self.algorithm.clip_param = 0.06
        self.algorithm.desired_kl = 0.002
        self.algorithm.teacher_action_coef = 1.0
        self.algorithm.teacher_kl_coef = 0.04
        self.algorithm.max_grad_norm = 0.40
