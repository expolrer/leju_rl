from isaaclab.utils import configclass

from leju_robot.tasks.tracking.agents.rsl_rl_ppo_cfg import KuavoS53DancePPORunnerCfg
from leju_robot.tasks.tracking.config.kuavoS53.stairs.ppo_cfg import (
    MODEL_92099,
    TeacherRegularizedPpoAlgorithmCfg,
)


@configclass
class KuavoS52StairsTransferPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    """Low-learning-rate S52 adaptation around the frozen S53 stair teacher."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS52_stairs_model92099_transfer"
        self.max_iterations = 120
        self.save_interval = 10
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.08,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=1.0e-5,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.002,
            max_grad_norm=0.5,
            teacher_checkpoint=MODEL_92099,
            teacher_action_coef=2.0,
            teacher_kl_coef=0.10,
            student_init_std=0.04,
            student_std_coef=2.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                1.0,
                0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
                0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
            ),
        )


@configclass
class KuavoS52StairsPhaseAlignedPPORunnerCfg(KuavoS52StairsTransferPPORunnerCfg):
    """Tighter teacher trust region for phase-aligned S52 adaptation."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS52_stairs_phase_aligned_transfer"
        self.max_iterations = 80
        self.algorithm.learning_rate = 5.0e-6
        self.algorithm.clip_param = 0.06
        self.algorithm.desired_kl = 0.0015
        self.algorithm.teacher_action_coef = 4.0
        self.algorithm.teacher_kl_coef = 0.25
        self.algorithm.max_grad_norm = 0.4


@configclass
class KuavoS52StairsSoftPlateauPPORunnerCfg(KuavoS52StairsPhaseAlignedPPORunnerCfg):
    """Allow morphology adaptation while retaining model_92099 as a prior."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS52_stairs_soft_plateau_transfer"
        self.max_iterations = 120
        self.algorithm.learning_rate = 1.5e-5
        self.algorithm.clip_param = 0.08
        self.algorithm.desired_kl = 0.003
        self.algorithm.teacher_action_coef = 0.75
        self.algorithm.teacher_kl_coef = 0.03
        self.algorithm.max_grad_norm = 0.5


@configclass
class KuavoS52StairsSoftPlateauTrustPPORunnerCfg(KuavoS52StairsSoftPlateauPPORunnerCfg):
    """v4 moderate trust region around model_92099 with dense platform credit."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS52_stairs_soft_plateau_trust_transfer"
        self.max_iterations = 80
        self.algorithm.learning_rate = 7.5e-6
        self.algorithm.clip_param = 0.06
        self.algorithm.desired_kl = 0.002
        self.algorithm.teacher_action_coef = 2.0
        self.algorithm.teacher_kl_coef = 0.10
        self.algorithm.max_grad_norm = 0.4


@configclass
class KuavoS52StairsContactPhasePPORunnerCfg(KuavoS52StairsSoftPlateauTrustPPORunnerCfg):
    """v5 narrow trust region for contact-phase correction around model_92099."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS52_stairs_contact_phase_transfer"
        self.max_iterations = 80
        self.algorithm.learning_rate = 5.0e-6
        self.algorithm.clip_param = 0.05
        self.algorithm.desired_kl = 0.0015
        self.algorithm.teacher_action_coef = 3.0
        self.algorithm.teacher_kl_coef = 0.15
        self.algorithm.max_grad_norm = 0.35


@configclass
class KuavoS52StairsScheduledFootholdPPORunnerCfg(KuavoS52StairsContactPhasePPORunnerCfg):
    """v6 local dynamics adaptation under an explicit contact schedule."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS52_stairs_scheduled_foothold_transfer"
        self.max_iterations = 80
        self.algorithm.learning_rate = 5.0e-6
        self.algorithm.clip_param = 0.04
        self.algorithm.desired_kl = 0.0012
        self.algorithm.teacher_action_coef = 2.0
        self.algorithm.teacher_kl_coef = 0.10
        self.algorithm.max_grad_norm = 0.30
