import re

from leju_robot.assets.leju import KuavoS52ArticulationCfg


KuavoS52_TRACKING_CFG = KuavoS52ArticulationCfg()

# Match the tracking controller's effort/stiffness-derived action scale while
# keeping the exact 27-joint policy order used by model_92099.
KuavoS52_ACTION_SCALE: dict[str, float] = {}
for actuator in KuavoS52_TRACKING_CFG.actuators.values():
    effort_cfg = actuator.effort_limit_sim
    stiffness_cfg = actuator.stiffness
    patterns = actuator.joint_names_expr
    if not patterns:
        continue

    for joint_name in KuavoS52_TRACKING_CFG.preserve_joint_order.joint_names or []:
        if not any(re.fullmatch(pattern, joint_name) for pattern in patterns):
            continue

        def resolve(value):
            if not isinstance(value, dict):
                return value
            for pattern, item in value.items():
                if re.fullmatch(pattern, joint_name):
                    return item
            return None

        effort = resolve(effort_cfg)
        stiffness = resolve(stiffness_cfg)
        if effort is not None and stiffness not in (None, 0):
            KuavoS52_ACTION_SCALE[joint_name] = 0.25 * float(0.5 * effort) / float(stiffness)
