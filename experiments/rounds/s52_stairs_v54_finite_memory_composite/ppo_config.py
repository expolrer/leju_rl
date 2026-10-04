from isaaclab.utils import configclass

from ..stairs_route_safety_composite.ppo_cfg import (
    KuavoS52StairsRouteSafetyCompositePPORunnerCfg,
)


@configclass
class KuavoS52StairsFiniteMemoryCompositePPORunnerCfg(
    KuavoS52StairsRouteSafetyCompositePPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS52_stairs_finite_memory_composite"
        self.max_iterations = 6
        self.save_interval = 1
        self.algorithm.constraint_cost_metric = (
            "s52_finite_memory_composite_constraint_cost"
        )
        self.algorithm.constraint_active_metric = (
            "s52_finite_memory_composite_constraint_active"
        )
        self.algorithm.cost_budget = 0.02
        self.algorithm.segment_replay_scale = 0.25
        self.algorithm.dual_proportional_gain = 1.0
        self.algorithm.dual_integral_gain = 0.02
        self.algorithm.dual_derivative_gain = 0.10
