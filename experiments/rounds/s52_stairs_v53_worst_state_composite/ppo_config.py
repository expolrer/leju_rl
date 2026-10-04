from isaaclab.utils import configclass

from ..stairs_route_safety_composite.ppo_cfg import (
    KuavoS52StairsRouteSafetyCompositePPORunnerCfg,
)


@configclass
class KuavoS52StairsWorstStateCompositePPORunnerCfg(
    KuavoS52StairsRouteSafetyCompositePPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS52_stairs_worst_state_composite"
        self.max_iterations = 6
        self.save_interval = 1
        self.algorithm.constraint_cost_metric = (
            "s52_worst_state_composite_constraint_cost"
        )
        self.algorithm.constraint_active_metric = (
            "s52_worst_state_composite_constraint_active"
        )
        self.algorithm.cost_budget = 0.003
        self.algorithm.segment_replay_scale = 0.75
        self.algorithm.dual_proportional_gain = 2.5
        self.algorithm.dual_integral_gain = 0.06
