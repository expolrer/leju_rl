from isaaclab.utils import configclass

from ..stairs_finite_memory_composite.ppo_cfg import (
    KuavoS52StairsFiniteMemoryCompositePPORunnerCfg,
)


@configclass
class KuavoS52StairsTerminalContactTimeGatePPORunnerCfg(
    KuavoS52StairsFiniteMemoryCompositePPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "kuavoS52_stairs_terminal_contact_time_gate"
        self.max_iterations = 6
        self.save_interval = 1
