import re
from leju_robot.assets import leju

KuavoS54_CYLINDER_CFG = leju.KuavoS54ArticulationCfg()

KuavoS54_ACTION_SCALE = {}
for a in KuavoS54_CYLINDER_CFG.actuators.values():
    e_cfg = a.effort_limit_sim
    s_cfg = a.stiffness
    name_patterns = a.joint_names_expr

    if not name_patterns:
        continue

    candidate_joint_names = []
    joint_names_list = KuavoS54_CYLINDER_CFG.preserve_joint_order.joint_names or []
    for jn in joint_names_list:
        for pat in name_patterns:
            if re.fullmatch(pat, jn):
                candidate_joint_names.append(jn)
                break

    def _resolve_value(joint_name, cfg_value):
        if isinstance(cfg_value, dict):
            for pat, val in cfg_value.items():
                if re.fullmatch(pat, joint_name):
                    return val
            return None
        else:
            return cfg_value

    for joint_name in candidate_joint_names:
        e_val = _resolve_value(joint_name, e_cfg)
        s_val = _resolve_value(joint_name, s_cfg)
        if e_val is None or s_val in (None, 0):
            continue
        KuavoS54_ACTION_SCALE[joint_name] = 0.25 * float(0.5 * e_val) / float(s_val)

