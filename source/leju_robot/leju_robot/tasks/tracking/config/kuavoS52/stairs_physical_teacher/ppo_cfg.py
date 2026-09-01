from isaaclab.utils import configclass

from ..stairs.ppo_cfg import KuavoS52StairsTransferPPORunnerCfg


@configclass
class KuavoS52StairsPhysicalTeacherPPORunnerCfg(KuavoS52StairsTransferPPORunnerCfg):
    """Conservative dynamics adaptation around the physical model_92099 teacher."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS52_stairs_physical_teacher_transfer"
        self.max_iterations = 60
        self.save_interval = 10
        self.algorithm.learning_rate = 3.0e-6
        self.algorithm.clip_param = 0.04
        self.algorithm.desired_kl = 0.001
        self.algorithm.teacher_action_coef = 4.0
        self.algorithm.teacher_kl_coef = 0.20
        self.algorithm.max_grad_norm = 0.30
