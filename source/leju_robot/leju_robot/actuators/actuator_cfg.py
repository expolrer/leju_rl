# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.actuators import DelayedPDActuatorCfg

from isaaclab.utils import configclass

# from .actuator_pd import DelayedImplicitActuator
from .actuator_pd import LejuDelayedPDActuator, LejuDelayedPDActuator_S17
from dataclasses import MISSING
import torch


@configclass
class LejuDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = LejuDelayedPDActuator
    effort_weaken_velocity_limit: float | dict[str, float] = 0.0
    friction_static: float | dict[str, float] = 0
    friction_activation_vel: float = torch.inf
    friction_dynamic: float | dict[str, float] = 0


@configclass
class LejuDelayedPDActuatorCfg_S17(LejuDelayedPDActuatorCfg):
    """Configuration for RobanS17 delayed PD actuator with rated effort limits."""

    class_type: type = LejuDelayedPDActuator_S17
    effort_limit_rated: float | dict[str, float] = MISSING


