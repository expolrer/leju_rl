from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from leju_robot.tasks.tracking.agents.rsl_rl_ppo_cfg import KuavoS53DancePPORunnerCfg


MODEL_113994 = (
    "/home/zzx23457/hhw/LejuLab-Train/logs/rsl_rl/"
    "kuavoS53_stairs_step_to_down_tracking/"
    "2026-08-20_00-19-04_s53_step_to_down_gated_warm101995_v1/model_113994.pt"
)
MODEL_115993 = (
    "/home/zzx23457/hhw/LejuLab-Train/logs/rsl_rl/"
    "kuavoS53_stairs_step_to_down_tracking/"
    "2026-08-22_02-27-25_s53_step_to_down_preserve_warm113994_v4_std012/"
    "model_115993.pt"
)
MODEL_89996 = (
    "/home/zzx23457/hhw/LejuLab-Train/logs/rsl_rl/"
    "kuavoS53_stairs_tgmp_reward_baseline_tracking/"
    "model_89996_bootstrap/model_89996.pt"
)
MODEL_90100 = (
    "/home/zzx23457/hhw/LejuLab-Train/logs/rsl_rl/"
    "kuavoS53_stairs_tgmp_reward_baseline_tracking/"
    "2026-08-22_14-18-57_s53_tgmp_reward_warm89996_v6/model_90100.pt"
)
MODEL_92099 = (
    "/home/zzx23457/hhw/LejuLab-Train/logs/rsl_rl/"
    "kuavoS53_stairs_tgmp_terrain_conditioned_tracking/"
    "2026-08-22_19-01-28_s53_tgmp_terrain_warm90100_v7/model_92099.pt"
)
MODEL_92138 = (
    "/home/zzx23457/hhw/LejuLab-Train/logs/rsl_rl/"
    "kuavoS53_stairs_clearance_soft_landing_tracking/"
    "2026-08-26_03-23-26_s53_clearance_soft_landing_warm92099_v17_preflight128x40/"
    "model_92138.pt"
)


@configclass
class TeacherRegularizedPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO with a fixed model_113994 action teacher."""

    teacher_checkpoint: str = MODEL_113994
    teacher_action_coef: float = 2.0
    teacher_kl_coef: float = 0.1
    student_init_std: float = 0.12
    student_std_coef: float = 0.1
    freeze_observation_normalizer: bool = True
    teacher_action_weights: tuple[float, ...] = ()
    teacher_hard_action_rmse_limit: float = 0.0
    teacher_projection_iterations: int = 4


@configclass
class TeacherRegularizedConstrainedPpoAlgorithmCfg(
    TeacherRegularizedPpoAlgorithmCfg
):
    """Teacher PPO with a separate cost critic and Lagrangian actor advantage."""

    constraint_cost_metric: str = "shared_clearance_cost"
    constraint_active_metric: str = "shared_clearance_active_fraction"
    constraint_command_name: str = "motion"
    cost_gamma: float = 0.99
    cost_lam: float = 0.95
    cost_budget: float = 0.002
    segment_replay_scale: float = 1.0
    dual_proportional_gain: float = 25.0
    dual_integral_gain: float = 0.50
    dual_derivative_gain: float = 5.0
    initial_dual_multiplier: float = 0.50
    min_dual_multiplier: float = 0.0
    max_dual_multiplier: float = 5.0
    cost_value_loss_coef: float = 1.0
    cost_critic_learning_rate: float = 3.0e-4


@configclass
class KuavoS53StairsStep1PPORunnerCfg(KuavoS53DancePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_step1_tracking"
        self.max_iterations = 30000
        self.save_interval = 250


@configclass
class KuavoS53StairsFullPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_full_tracking"
        self.max_iterations = 30000
        self.save_interval = 250


@configclass
class KuavoS53StairsUpDownPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_updown_tracking"
        self.max_iterations = 30000
        self.save_interval = 250


@configclass
class KuavoS53StairsForwardDownPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_forward_down_tracking"
        self.max_iterations = 20000
        self.save_interval = 250


@configclass
class KuavoS53StairsForwardUpDownPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_forward_updown_tracking"
        self.max_iterations = 20000
        self.save_interval = 250
        # Conservative fine-tuning protects both specialist skills after the
        # optimizer is reset for the longer combined reference.
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.entropy_coef = 0.003
        self.algorithm.desired_kl = 0.008


@configclass
class KuavoS53StairsTgmpRewardBaselinePPORunnerCfg(KuavoS53DancePPORunnerCfg):
    """PPO settings matching the public T-GMP table for the reward-port stage."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_tgmp_reward_baseline_tracking"
        self.max_iterations = 5000
        self.save_interval = 100
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            teacher_checkpoint=MODEL_89996,
            teacher_action_coef=0.25,
            teacher_kl_coef=0.02,
            student_init_std=0.12,
            student_std_coef=0.05,
            freeze_observation_normalizer=True,
        )


@configclass
class KuavoS53StairsTgmpTerrainConditionedPPORunnerCfg(
    KuavoS53DancePPORunnerCfg
):
    """Conservative terrain-focused continuation from the best v6 candidate."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_tgmp_terrain_conditioned_tracking"
        self.max_iterations = 2000
        self.save_interval = 100
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.10,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=5.0e-5,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.003,
            max_grad_norm=0.5,
            teacher_checkpoint=MODEL_90100,
            teacher_action_coef=1.0,
            teacher_kl_coef=0.03,
            student_init_std=0.08,
            student_std_coef=2.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(
                0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
                0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        )


@configclass
class KuavoS53StairsTgmpRiserSafePPORunnerCfg(KuavoS53DancePPORunnerCfg):
    """Fine-tune model_92099 against persistent descent-riser contacts."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_tgmp_riser_safe_tracking"
        self.max_iterations = 2500
        self.save_interval = 100
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.08,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=2.0e-5,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.002,
            max_grad_norm=0.5,
            teacher_checkpoint=MODEL_92099,
            teacher_action_coef=0.5,
            teacher_kl_coef=0.002,
            student_init_std=0.06,
            student_std_coef=5.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(
                0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        )


@configclass
class KuavoS53StairsMindStepsTactPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    """Conservative foothold-and-swing refinement from full-course model_92099."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_mindsteps_tact_tracking"
        self.max_iterations = 2000
        self.save_interval = 100
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.06,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=1.5e-5,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.0015,
            max_grad_norm=0.5,
            teacher_checkpoint=MODEL_92099,
            teacher_action_coef=0.75,
            teacher_kl_coef=0.004,
            student_init_std=0.055,
            student_std_coef=5.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(
                0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
                0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        )


@configclass
class KuavoS53StairsPredictiveSweepPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    """Low-drift refinement of model_92099 with predictive rigid-foot clearance."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_predictive_sweep_tracking"
        self.max_iterations = 1500
        self.save_interval = 100
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.05,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=1.0e-5,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.0012,
            max_grad_norm=0.5,
            teacher_checkpoint=MODEL_92099,
            teacher_action_coef=1.0,
            teacher_kl_coef=0.004,
            student_init_std=0.045,
            student_std_coef=5.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(
                0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        )


@configclass
class KuavoS53StairsSwingAwareSweepPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    """Conservative model_92099 refinement with support-aware sweep shaping."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_swing_aware_sweep_tracking"
        self.max_iterations = 1000
        self.save_interval = 100
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.035,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=5.0e-6,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=8.0e-4,
            max_grad_norm=0.5,
            teacher_checkpoint=MODEL_92099,
            teacher_action_coef=1.5,
            teacher_kl_coef=0.02,
            student_init_std=0.035,
            student_std_coef=8.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(
                0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
                0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        )


@configclass
class KuavoS53StairsTailRiskSweepPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    """Short low-drift refinement focused on the worst clearance decile."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_tail_risk_sweep_tracking"
        self.max_iterations = 600
        self.save_interval = 50
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.030,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=3.0e-6,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=6.0e-4,
            max_grad_norm=0.5,
            teacher_checkpoint=MODEL_92099,
            teacher_action_coef=1.75,
            teacher_kl_coef=0.03,
            student_init_std=0.028,
            student_std_coef=10.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(
                0.45, 0.45, 0.45, 0.45, 0.45, 0.45,
                0.45, 0.45, 0.45, 0.45, 0.45, 0.45,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        )


@configclass
class KuavoS53StairsToeBarrierPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    """Very low-drift refinement around the physical toe-clearance metric."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_toe_barrier_tracking"
        self.max_iterations = 400
        self.save_interval = 50
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.025,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=2.0e-6,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=5.0e-4,
            max_grad_norm=0.5,
            teacher_checkpoint=MODEL_92099,
            teacher_action_coef=2.0,
            teacher_kl_coef=0.05,
            student_init_std=0.024,
            student_std_coef=12.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(
                0.60, 0.60, 0.60, 0.60, 0.60, 0.60,
                0.60, 0.60, 0.60, 0.60, 0.60, 0.60,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        )


@configclass
class KuavoS53StairsRunningMinBarrierPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    """Conservative low-drift refinement against each swing's worst clearance."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_running_min_barrier_tracking"
        self.max_iterations = 120
        self.save_interval = 20
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.020,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=1.5e-6,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=3.0e-4,
            max_grad_norm=0.5,
            teacher_checkpoint=MODEL_92099,
            teacher_action_coef=2.5,
            teacher_kl_coef=0.06,
            student_init_std=0.020,
            student_std_coef=15.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(
                0.70, 0.70, 0.70, 0.70, 0.70, 0.70,
                0.70, 0.70, 0.70, 0.70, 0.70, 0.70,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        )


@configclass
class KuavoS53StairsTimeToRiserConePPORunnerCfg(KuavoS53DancePPORunnerCfg):
    """Low-drift v15 refinement with dense pre-contact riser credit."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_time_to_riser_cone_tracking"
        self.max_iterations = 80
        self.save_interval = 20
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.015,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=1.0e-6,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=2.0e-4,
            max_grad_norm=0.5,
            teacher_checkpoint=MODEL_92099,
            teacher_action_coef=3.0,
            teacher_kl_coef=0.10,
            student_init_std=0.016,
            student_std_coef=20.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(
                0.80, 0.80, 0.80, 0.80, 0.80, 0.80,
                0.80, 0.80, 0.80, 0.80, 0.80, 0.80,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        )


@configclass
class KuavoS53StairsSpatialRiserCorridorPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    """Very low-drift v16 refinement with a velocity-independent corridor."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_spatial_riser_corridor_tracking"
        self.max_iterations = 60
        self.save_interval = 20
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.010,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=5.0e-7,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=1.0e-4,
            max_grad_norm=0.4,
            teacher_checkpoint=MODEL_92099,
            teacher_action_coef=4.0,
            teacher_kl_coef=0.15,
            student_init_std=0.012,
            student_std_coef=25.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(
                0.90, 0.90, 0.90, 0.90, 0.90, 0.90,
                0.90, 0.90, 0.90, 0.90, 0.90, 0.90,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        )


@configclass
class KuavoS53StairsClearanceSoftLandingPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    """Very low-drift v17 refinement around the model_92099 safety baseline."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_clearance_soft_landing_tracking"
        self.max_iterations = 40
        self.save_interval = 10
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.008,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=5.0e-7,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=1.0e-4,
            max_grad_norm=0.4,
            teacher_checkpoint=MODEL_92099,
            teacher_action_coef=4.5,
            teacher_kl_coef=0.18,
            student_init_std=0.010,
            student_std_coef=30.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(
                0.92, 0.92, 0.92, 0.92, 0.92, 0.92,
                0.92, 0.92, 0.92, 0.92, 0.92, 0.92,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        )


@configclass
class KuavoS53StairsConservativeTailReplayPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    """Short, strongly regularized v18 refinement around model_92138."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_conservative_tail_replay_tracking"
        self.max_iterations = 80
        self.save_interval = 20
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.006,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=3.0e-7,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=1.0e-4,
            max_grad_norm=0.35,
            teacher_checkpoint=MODEL_92138,
            teacher_action_coef=6.0,
            teacher_kl_coef=0.25,
            student_init_std=0.008,
            student_std_coef=35.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(
                0.96, 0.96, 0.96, 0.96, 0.96, 0.96,
                0.96, 0.96, 0.96, 0.96, 0.96, 0.96,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        )


@configclass
class KuavoS53StairsConstrainedTeacherProjectionPPORunnerCfg(
    KuavoS53DancePPORunnerCfg
):
    """v20: hard actor projection around the physically verified model_92138."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_constrained_teacher_projection_tracking"
        self.max_iterations = 40
        self.save_interval = 10
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.006,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=3.0e-7,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=1.0e-4,
            max_grad_norm=0.30,
            teacher_checkpoint=MODEL_92138,
            teacher_action_coef=6.0,
            teacher_kl_coef=0.25,
            student_init_std=0.006,
            student_std_coef=40.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(1.0,) * 27,
            teacher_hard_action_rmse_limit=0.003,
            teacher_projection_iterations=6,
        )


@configclass
class KuavoS53StairsRiserConstraintCaTPPORunnerCfg(
    KuavoS53DancePPORunnerCfg
):
    """v21: low-drift PPO with direct local riser Constraints-as-Terminations."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_riser_constraint_cat_tracking"
        self.max_iterations = 60
        self.save_interval = 10
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.006,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=3.0e-7,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=1.0e-4,
            max_grad_norm=0.30,
            teacher_checkpoint=MODEL_92138,
            teacher_action_coef=6.0,
            teacher_kl_coef=0.25,
            student_init_std=0.006,
            student_std_coef=40.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(1.0,) * 27,
        )


@configclass
class KuavoS53StairsRiserClearanceLagrangianPPORunnerCfg(
    KuavoS53DancePPORunnerCfg
):
    """v22: low-drift PPO with an episode-preserving adaptive clearance cost."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_riser_clearance_lagrangian_tracking"
        self.max_iterations = 60
        self.save_interval = 10
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.006,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=3.0e-7,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=1.0e-4,
            max_grad_norm=0.30,
            teacher_checkpoint=MODEL_92138,
            teacher_action_coef=6.0,
            teacher_kl_coef=0.25,
            student_init_std=0.006,
            student_std_coef=40.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(1.0,) * 27,
        )


@configclass
class KuavoS53StairsSharedClearanceCmdpPPORunnerCfg(
    KuavoS53DancePPORunnerCfg
):
    """v23: low-drift constrained PPO around the verified model_92138."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_shared_clearance_cmdp_tracking"
        self.max_iterations = 60
        self.save_interval = 10
        self.algorithm = TeacherRegularizedConstrainedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.006,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=3.0e-7,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=1.0e-4,
            max_grad_norm=0.30,
            teacher_checkpoint=MODEL_92138,
            teacher_action_coef=6.0,
            teacher_kl_coef=0.25,
            student_init_std=0.006,
            student_std_coef=40.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(1.0,) * 27,
            constraint_cost_metric="shared_clearance_cost",
            constraint_command_name="motion",
            cost_gamma=0.99,
            cost_lam=0.95,
            cost_budget=0.002,
            dual_learning_rate=0.05,
            initial_dual_multiplier=0.10,
            min_dual_multiplier=0.0,
            max_dual_multiplier=5.0,
            cost_value_loss_coef=1.0,
            cost_critic_learning_rate=3.0e-4,
        )


@configclass
class KuavoS53StairsPidWorstSegmentCmdpPPORunnerCfg(
    KuavoS53DancePPORunnerCfg
):
    """v24: PID Lagrangian over per-swing worst-step rigid-foot cost."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_pid_worst_segment_cmdp_tracking"
        self.max_iterations = 80
        self.save_interval = 10
        self.algorithm = TeacherRegularizedConstrainedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.004,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=1.5e-7,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=7.5e-5,
            max_grad_norm=0.25,
            teacher_checkpoint=MODEL_92138,
            teacher_action_coef=8.0,
            teacher_kl_coef=0.35,
            student_init_std=0.004,
            student_std_coef=60.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(1.0,) * 27,
            constraint_cost_metric="shared_clearance_cost",
            constraint_active_metric="shared_clearance_active_fraction",
            constraint_command_name="motion",
            cost_gamma=0.99,
            cost_lam=0.95,
            cost_budget=0.001,
            dual_proportional_gain=25.0,
            dual_integral_gain=0.50,
            dual_derivative_gain=5.0,
            initial_dual_multiplier=0.50,
            min_dual_multiplier=0.0,
            max_dual_multiplier=20.0,
            cost_value_loss_coef=1.0,
            cost_critic_learning_rate=2.0e-4,
        )


@configclass
class KuavoS53StairsCalibratedTailCmdpPPORunnerCfg(
    KuavoS53DancePPORunnerCfg
):
    """v25: baseline-calibrated, softly replayed per-swing tail cost."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_calibrated_tail_cmdp_tracking"
        self.max_iterations = 80
        self.save_interval = 10
        self.algorithm = TeacherRegularizedConstrainedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.003,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=1.0e-7,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=5.0e-5,
            max_grad_norm=0.20,
            teacher_checkpoint=MODEL_92138,
            teacher_action_coef=12.0,
            teacher_kl_coef=0.50,
            student_init_std=0.003,
            student_std_coef=80.0,
            freeze_observation_normalizer=True,
            teacher_action_weights=(1.0,) * 27,
            constraint_cost_metric="shared_clearance_cost",
            constraint_active_metric="shared_clearance_active_fraction",
            constraint_command_name="motion",
            cost_gamma=0.99,
            cost_lam=0.95,
            cost_budget=0.0203,
            segment_replay_scale=0.10,
            dual_proportional_gain=10.0,
            dual_integral_gain=0.10,
            dual_derivative_gain=2.0,
            initial_dual_multiplier=0.10,
            min_dual_multiplier=0.0,
            max_dual_multiplier=5.0,
            cost_value_loss_coef=0.50,
            cost_critic_learning_rate=1.0e-4,
        )


@configclass
class KuavoS53StairsContactGatedCmdpPPORunnerCfg(
    KuavoS53StairsCalibratedTailCmdpPPORunnerCfg
):
    """v26: calibrated tail CMDP with contact-consistent swing activation."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_contact_gated_cmdp_tracking"
        self.algorithm.cost_budget = 0.0010


@configclass
class KuavoS53StairsContactGatedMarginCmdpPPORunnerCfg(
    KuavoS53StairsContactGatedCmdpPPORunnerCfg
):
    """v27: denser margin cost with a tighter teacher trust region."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_contact_gated_margin_cmdp_tracking"
        self.algorithm.learning_rate = 5.0e-8
        self.algorithm.teacher_action_coef = 20.0
        self.algorithm.teacher_kl_coef = 1.0
        self.algorithm.student_init_std = 0.002
        self.algorithm.student_std_coef = 120.0
        self.algorithm.cost_budget = 0.0040
        self.algorithm.dual_proportional_gain = 15.0
        self.algorithm.dual_integral_gain = 0.20
        self.algorithm.dual_derivative_gain = 3.0
        self.algorithm.initial_dual_multiplier = 0.50


@configclass
class KuavoS53StairsForwardUpDownStablePPORunnerCfg(KuavoS53DancePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_forward_updown_stable_tracking"
        self.max_iterations = 12000
        self.save_interval = 250
        self.algorithm.learning_rate = 1.5e-4
        self.algorithm.entropy_coef = 0.0015
        self.algorithm.desired_kl = 0.006


@configclass
class KuavoS53StairsStepToDownPPORunnerCfg(KuavoS53DancePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS53_stairs_step_to_down_tracking"
        self.max_iterations = 12000
        self.save_interval = 250
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.005


@configclass
class KuavoS53StairsStepToDownGateFixedPPORunnerCfg(
    KuavoS53StairsStepToDownPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 6000
        self.save_interval = 250
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.entropy_coef = 5.0e-4
        self.algorithm.desired_kl = 0.003


@configclass
class KuavoS53StairsStepToDownNosingSafePPORunnerCfg(
    KuavoS53StairsStepToDownGateFixedPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 4000
        self.save_interval = 250
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.entropy_coef = 2.0e-4
        self.algorithm.desired_kl = 0.002


@configclass
class KuavoS53StairsStepToDownPreservePPORunnerCfg(
    KuavoS53StairsStepToDownNosingSafePPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2000
        self.save_interval = 100
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.10,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=5.0e-6,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.001,
            max_grad_norm=0.5,
            teacher_checkpoint=MODEL_113994,
            teacher_action_coef=2.0,
            teacher_kl_coef=0.1,
            student_init_std=0.12,
            student_std_coef=0.1,
            freeze_observation_normalizer=True,
        )


@configclass
class KuavoS53StairsStepToDownTgmpFootholdPPORunnerCfg(
    KuavoS53StairsStepToDownPreservePPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.save_interval = 100
        self.algorithm = TeacherRegularizedPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.08,
            entropy_coef=0.0,
            num_learning_epochs=3,
            num_mini_batches=4,
            learning_rate=3.0e-6,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.001,
            max_grad_norm=0.5,
            teacher_checkpoint=MODEL_115993,
            teacher_action_coef=2.0,
            teacher_kl_coef=0.05,
            student_init_std=0.10,
            student_std_coef=0.1,
            freeze_observation_normalizer=True,
            # Action order is left leg, right leg, waist, then both arms.
            # Release the legs from the colliding teacher trajectory while
            # preserving torso and arm coordination as a motion-style prior.
            teacher_action_weights=(
                0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
                0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
                1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ),
        )
