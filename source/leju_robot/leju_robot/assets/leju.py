# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from leju_robot.assets.robots import ASSET_DIR
from leju_robot.actuators.actuator_cfg import LejuDelayedPDActuatorCfg, LejuDelayedPDActuatorCfg_S17

ROBANS14_MDP_JOINT_ORDER_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "waist_yaw_joint",
        "leg_l1_joint",
        "leg_l2_joint",
        "leg_l3_joint",
        "leg_l4_joint",
        "leg_l5_joint",
        "leg_l6_joint",
        "leg_r1_joint",
        "leg_r2_joint",
        "leg_r3_joint",
        "leg_r4_joint",
        "leg_r5_joint",
        "leg_r6_joint",
        "zarm_l1_joint",
        "zarm_l2_joint",
        "zarm_l3_joint",
        "zarm_l4_joint",
        "zarm_r1_joint",
        "zarm_r2_joint",
        "zarm_r3_joint",
        "zarm_r4_joint",
    ],
    body_names=[
        "waist_yaw_link",
        "leg_l1_link",
        "leg_l2_link",
        "leg_l3_link",
        "leg_l4_link",
        "leg_l5_link",
        "leg_l6_link",
        "leg_r1_link",
        "leg_r2_link",
        "leg_r3_link",
        "leg_r4_link",
        "leg_r5_link",
        "leg_r6_link",
        "zarm_l1_link",
        "zarm_l2_link",
        "zarm_l3_link",
        "zarm_l4_link",
        "zarm_r1_link",
        "zarm_r2_link",
        "zarm_r3_link",
        "zarm_r4_link",
    ],
    preserve_order=True
)



@configclass
class RobanS14ArticulationCfg(ArticulationCfg):
    """Configuration for Roban S14 articulation."""
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ASSET_DIR}/robanS14/urdf/robanS14_mini_col.urdf",
        # spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ASSET_DIR}/robanS14/usd/robanS14_mini_col.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        # joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    )
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.685),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "waist_yaw_joint": 0.0,
            "leg_l1_joint": -0.412,
            "leg_l2_joint": -0.0437,
            "leg_l3_joint": -0.287,
            "leg_l4_joint": 0.5,
            "leg_l5_joint": -0.2,
            "leg_l6_joint": 0.0,
            "leg_r1_joint": 0.412,
            "leg_r2_joint": 0.0437,
            "leg_r3_joint": 0.287,
            "leg_r4_joint": 0.5,
            "leg_r5_joint": -0.2,
            "leg_r6_joint": 0.0,

            "zarm_l1_joint": 0.2,
            "zarm_l2_joint": 0.16,
            "zarm_l3_joint": -0.4,
            "zarm_l4_joint": -0.5,
            "zarm_r1_joint": 0.2,
            "zarm_r2_joint": -0.16,
            "zarm_r3_joint": 0.4,
            "zarm_r4_joint": -0.5,
        },
        joint_vel={".*": 0.0},
    )
    soft_joint_pos_limit_factor=1.0
    actuators={
        "motor": LejuDelayedPDActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "leg_.*",
                "zarm_.*"
            ],
            effort_limit_sim={
                "waist_yaw_joint": 80.0,
                "leg_[lr]1_joint": 80.0,
                "leg_[lr]2_joint": 63.0,
                "leg_[lr]3_joint": 80.0,
                "leg_[lr]4_joint": 63.0,
                "leg_[lr]5_joint": 25.0,
                "leg_[lr]6_joint": 25.0,
                
                "zarm_[lr]1_joint": 14.1,
                "zarm_[lr]2_joint": 14.1,
                "zarm_[lr]3_joint": 14.1,
                "zarm_[lr]4_joint": 14.1,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 7.0,
                "leg_[lr]1_joint": 7.0,
                "leg_[lr]2_joint": 10.8,
                "leg_[lr]3_joint": 7.0,
                "leg_[lr]4_joint": 10.8,
                "leg_[lr]5_joint": 10.2,
                "leg_[lr]6_joint": 10.2,

                "zarm_[lr]1_joint":10.5,
                "zarm_[lr]2_joint":10.5,
                "zarm_[lr]3_joint":10.5,
                "zarm_[lr]4_joint":10.5,
            },
            effort_weaken_velocity_limit={
                "waist_yaw_joint": 7.0 * 0.2,
                "leg_[lr]1_joint": 7.0 * 0.2,
                "leg_[lr]2_joint": 10.8 * 0.2,
                "leg_[lr]3_joint": 7.0 * 0.2,
                "leg_[lr]4_joint": 10.8 * 0.2,
                "leg_[lr]5_joint": 10.2 * 0.2,
                "leg_[lr]6_joint": 10.2 * 0.2,

                "zarm_[lr]1_joint":10.5 * 0.2,
                "zarm_[lr]2_joint":10.5 * 0.2,
                "zarm_[lr]3_joint":10.5 * 0.2,
                "zarm_[lr]4_joint":10.5 * 0.2,
            },
            stiffness={
                "waist_yaw_joint": 100.0,
                "leg_[lr]1_joint": 100.0,
                "leg_[lr]2_joint": 100.0,
                "leg_[lr]3_joint": 100.0,
                "leg_[lr]4_joint": 150.0,
                "leg_[lr]5_joint": 20.0,
                "leg_[lr]6_joint": 20.0,

                "zarm_[lr]1_joint": 20.0,
                "zarm_[lr]2_joint": 20.0,
                "zarm_[lr]3_joint": 20.0,
                "zarm_[lr]4_joint": 20.0,
            },
            damping={
                "waist_yaw_joint": 2.0,
                "leg_[lr]1_joint": 2.0,
                "leg_[lr]2_joint": 2.0,
                "leg_[lr]3_joint": 2.0,
                "leg_[lr]4_joint": 2.0,
                "leg_[lr]5_joint": 1.0,
                "leg_[lr]6_joint": 1.0,

                "zarm_[lr]1_joint": 1.0,
                "zarm_[lr]2_joint": 1.0,
                "zarm_[lr]3_joint": 1.0,
                "zarm_[lr]4_joint": 1.0,
            },
            armature={
                "waist_yaw_joint": 0.05,
                "leg_[lr]1_joint": 0.05,
                "leg_[lr]2_joint": 0.05,
                "leg_[lr]3_joint": 0.05,
                "leg_[lr]4_joint": 0.05,
                "leg_[lr]5_joint": 0.02,
                "leg_[lr]6_joint": 0.02,

                "zarm_[lr]1_joint": 0.02,
                "zarm_[lr]2_joint": 0.02,
                "zarm_[lr]3_joint": 0.02,
                "zarm_[lr]4_joint": 0.02,
            },
            friction=0,
            min_delay=0,
            max_delay=4,
            friction_static={
                "waist_yaw_joint": 2.0,
                "leg_[lr]1_joint": 2.0,
                "leg_[lr]2_joint": 2.0,
                "leg_[lr]3_joint": 2.0,
                "leg_[lr]4_joint": 2.0,
                "leg_[lr]5_joint": 2.0,
                "leg_[lr]6_joint": 2.0,

                "zarm_[lr]1_joint": 1.0,
                "zarm_[lr]2_joint": 1.0,
                "zarm_[lr]3_joint": 1.0,
                "zarm_[lr]4_joint": 1.0,
            },
            friction_activation_vel=0.01,
            friction_dynamic={
                "waist_yaw_joint": 0.2,
                "leg_[lr]1_joint": 0.2,
                "leg_[lr]2_joint": 0.2,
                "leg_[lr]3_joint": 0.2,
                "leg_[lr]4_joint": 0.2,
                "leg_[lr]5_joint": 0.2,
                "leg_[lr]6_joint": 0.2,

                "zarm_[lr]1_joint": 0.1,
                "zarm_[lr]2_joint": 0.1,
                "zarm_[lr]3_joint": 0.1,
                "zarm_[lr]4_joint": 0.1,
            },
        ),
    }
    preserve_joint_order = ROBANS14_MDP_JOINT_ORDER_CFG
    end_effector_configs = [
        ("leg_l6_link", None),
        ("leg_r6_link", None),
        ("zarm_l4_link", [0.0, 0.0, -0.2]),
        ("zarm_r4_link", [0.0, 0.0, -0.2]),
    ]


@configclass
class RobanS14ArticulationCfg_New_Year_Dance(ArticulationCfg):
    """Configuration for Roban S14 articulation."""
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ASSET_DIR}/robanS14/urdf/robanS14_mini_col.urdf",
        # spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ASSET_DIR}/robanS14/usd/robanS14_mini_col.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        # joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    )
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.685),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "waist_yaw_joint": 0.0,
            "leg_l1_joint": -0.412,
            "leg_l2_joint": -0.0437,
            "leg_l3_joint": -0.287,
            "leg_l4_joint": 0.5,
            "leg_l5_joint": -0.2,
            "leg_l6_joint": 0.0,
            "leg_r1_joint": 0.412,
            "leg_r2_joint": 0.0437,
            "leg_r3_joint": 0.287,
            "leg_r4_joint": 0.5,
            "leg_r5_joint": -0.2,
            "leg_r6_joint": 0.0,

            "zarm_l1_joint": 0.2,
            "zarm_l2_joint": 0.16,
            "zarm_l3_joint": -0.4,
            "zarm_l4_joint": -0.5,
            "zarm_r1_joint": 0.2,
            "zarm_r2_joint": -0.16,
            "zarm_r3_joint": 0.4,
            "zarm_r4_joint": -0.5,
        },
        joint_vel={".*": 0.0},
    )
    soft_joint_pos_limit_factor=0.95
    actuators={
        "motor": LejuDelayedPDActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "leg_.*",
                "zarm_.*"
            ],
            effort_limit_sim={
                "waist_yaw_joint": 80.0,
                "leg_[lr]1_joint": 80.0,
                "leg_[lr]2_joint": 63.0,
                "leg_[lr]3_joint": 80.0,
                "leg_[lr]4_joint": 63.0,
                "leg_[lr]5_joint": 25.0,
                "leg_[lr]6_joint": 25.0,
                
                "zarm_[lr]1_joint": 14.1,
                "zarm_[lr]2_joint": 14.1,
                "zarm_[lr]3_joint": 14.1,
                "zarm_[lr]4_joint": 14.1,
            },
            velocity_limit={
                "waist_yaw_joint": 10.8,
                "leg_[l,r]1_joint": 10.8,
                "leg_[l,r]2_joint": 10.8,
                "leg_[l,r]3_joint": 10.8,
                "leg_[l,r]4_joint": 10.8,
                "leg_[l,r]5_joint": 10.2,
                "leg_[l,r]6_joint": 10.2,
                "zarm_[l,r]1_joint": 9.0,
                "zarm_[l,r]2_joint": 9.0,
                "zarm_[l,r]3_joint": 9.0,
                "zarm_[l,r]4_joint": 9.0,
            },
            effort_weaken_velocity_limit={
                "waist_yaw_joint": 7.0 * 0.2,
                "leg_[lr]1_joint": 7.0 * 0.2,
                "leg_[lr]2_joint": 10.8 * 0.2,
                "leg_[lr]3_joint": 7.0 * 0.2,
                "leg_[lr]4_joint": 10.8 * 0.2,
                "leg_[lr]5_joint": 10.2 * 0.2,
                "leg_[lr]6_joint": 10.2 * 0.2,

                "zarm_[lr]1_joint":10.5 * 0.2,
                "zarm_[lr]2_joint":10.5 * 0.2,
                "zarm_[lr]3_joint":10.5 * 0.2,
                "zarm_[lr]4_joint":10.5 * 0.2,
            },
            stiffness={
                "waist_yaw_joint": 40.1792,
                "leg_[l,r]1_joint": 40.1792,
                "leg_[l,r]2_joint": 99.0984,
                "leg_[l,r]3_joint": 40.1792,
                "leg_[l,r]4_joint": 99.0984,
                "leg_[l,r]5_joint": 14.2506,
                "leg_[l,r]6_joint": 14.2506,
                "zarm_[l,r]1_joint": 14.2506,
                "zarm_[l,r]2_joint": 14.2506,
                "zarm_[l,r]3_joint": 14.2506,
                "zarm_[l,r]4_joint": 14.2506,
            },
            damping={
                "waist_yaw_joint": 2.5579,
                "leg_[l,r]1_joint": 2.5579,
                "leg_[l,r]2_joint": 6.3088,
                "leg_[l,r]3_joint": 2.5579,
                "leg_[l,r]4_joint": 6.3088,
                "leg_[l,r]5_joint": 0.9072,
                "leg_[l,r]6_joint": 0.9072,
                "zarm_[l,r]1_joint": 0.9072,
                "zarm_[l,r]2_joint": 0.9072,
                "zarm_[l,r]3_joint": 0.9072,
                "zarm_[l,r]4_joint": 0.9072,
            },
            armature={
                "waist_yaw_joint": 0.01017752,
                "leg_[l,r]1_joint": 0.01017752,
                "leg_[l,r]2_joint": 0.025101925,
                "leg_[l,r]3_joint": 0.01017752,
                "leg_[l,r]4_joint": 0.025101925,
                "leg_[l,r]5_joint": 0.003609725,
                "leg_[l,r]6_joint": 0.003609725,
                "zarm_[l,r]1_joint": 0.003609725,
                "zarm_[l,r]2_joint": 0.003609725,
                "zarm_[l,r]3_joint": 0.003609725,
                "zarm_[l,r]4_joint": 0.003609725,
            },
            friction=0,
            min_delay=0,
            max_delay=4,
            friction_static={
                "waist_yaw_joint": 0.2,
                "leg_[l,r]1_joint": 0.2,
                "leg_[l,r]2_joint": 0.2,
                "leg_[l,r]3_joint": 0.2,
                "leg_[l,r]4_joint": 0.2,
                "leg_[l,r]5_joint": 0.2,
                "leg_[l,r]6_joint": 0.2,
                "zarm_[l,r]1_joint": 0.2,
                "zarm_[l,r]2_joint": 0.2,
                "zarm_[l,r]3_joint": 0.2,
                "zarm_[l,r]4_joint": 0.2,
            },
            friction_activation_vel=0.1,
            friction_dynamic=0,
        ),
    }
    preserve_joint_order = ROBANS14_MDP_JOINT_ORDER_CFG
    # end_effector_configs = [
    #     ("leg_l6_link", None),
    #     ("leg_r6_link", None),
    #     ("zarm_l4_link", [0.0, 0.0, -0.2]),
    #     ("zarm_r4_link", [0.0, 0.0, -0.2]),
    # ]




KUAVOS54_MDP_JOINT_ORDER_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "leg_l1_joint",
        "leg_l2_joint",
        "leg_l3_joint",
        "leg_l4_joint",
        "leg_l5_joint",
        "leg_l6_joint",
        "leg_r1_joint",
        "leg_r2_joint",
        "leg_r3_joint",
        "leg_r4_joint",
        "leg_r5_joint",
        "leg_r6_joint",
        "waist_yaw_joint",
        "zarm_l1_joint",
        "zarm_l2_joint",
        "zarm_l3_joint",
        "zarm_l4_joint",
        "zarm_l5_joint",
        "zarm_l6_joint",
        "zarm_l7_joint",
        "zarm_r1_joint",
        "zarm_r2_joint",
        "zarm_r3_joint",
        "zarm_r4_joint",
        "zarm_r5_joint",
        "zarm_r6_joint",
        "zarm_r7_joint",
    ],
    body_names=[
        "leg_l1_link",
        "leg_l2_link",
        "leg_l3_link",
        "leg_l4_link",
        "leg_l5_link",
        "leg_l6_link",
        "leg_r1_link",
        "leg_r2_link",
        "leg_r3_link",
        "leg_r4_link",
        "leg_r5_link",
        "leg_r6_link",
        "waist_yaw_link",
        "zarm_l1_link",
        "zarm_l2_link",
        "zarm_l3_link",
        "zarm_l4_link",
        "zarm_l5_link",
        "zarm_l6_link",
        "zarm_l7_link",
        "zarm_r1_link",
        "zarm_r2_link",
        "zarm_r3_link",
        "zarm_r4_link",
        "zarm_r5_link",
        "zarm_r6_link",
        "zarm_r7_link",
    ],
    preserve_order=True
)

@configclass
class KuavoS54ArticulationCfg(ArticulationCfg):
    """Configuration for Kuavo S54 articulation."""
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/kuavos54/urdf/kuavoS54_mini_col.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    )
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.925),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "leg_[l,r]1_joint": 0.0,
            "leg_[l,r]2_joint": 0.0,
            "leg_[l,r]3_joint": -0.4,
            "leg_[l,r]4_joint": 0.69,
            "leg_[l,r]5_joint": -0.33,
            "leg_[l,r]6_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "zarm_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    )
    soft_joint_pos_limit_factor=0.95
    actuators={
        "motor": LejuDelayedPDActuatorCfg(
            joint_names_expr=[
                "waist_.*",
                "leg_.*",
                "zarm_.*"
            ],
            effort_limit_sim={
                "leg_[l,r]1_joint": 101.6,
                "leg_[l,r]2_joint": 56.8,
                "leg_[l,r]3_joint": 105.6,
                "leg_[l,r]4_joint": 224.0,
                "leg_[l,r]5_joint": 91.6,
                "leg_[l,r]6_joint": 57.0,
                "waist_yaw_joint": 81.6,
                "zarm_[l,r]1_joint": 52.8,
                "zarm_[l,r]2_joint": 60.0,
                "zarm_[l,r]3_joint": 45.6,
                "zarm_[l,r]4_joint": 60.0,
                "zarm_[l,r]5_joint": 11.0,
                "zarm_[l,r]6_joint": 11.0,
                "zarm_[l,r]7_joint": 11.0,
            },
            velocity_limit_sim={
                "leg_[l,r]1_joint": 10.4,
                "leg_[l,r]2_joint": 8.7,
                "leg_[l,r]3_joint": 12.7,
                "leg_[l,r]4_joint": 10.4,
                "leg_[l,r]5_joint": 17.8,
                "leg_[l,r]6_joint": 17.8,

                "waist_yaw_joint": 8.7,
                "zarm_[l,r]1_joint": 18.8,
                "zarm_[l,r]2_joint": 8.0,
                "zarm_[l,r]3_joint": 7.5,
                "zarm_[l,r]4_joint": 8.0,
                "zarm_[l,r]5_joint": 17.5,
                "zarm_[l,r]6_joint": 17.5,
                "zarm_[l,r]7_joint": 17.5,
            },
            effort_weaken_velocity_limit={
                "leg_[lr]1_joint": 10.4 * 0.2,
                "leg_[lr]2_joint": 8.7 * 0.2,
                "leg_[lr]3_joint": 12.7 * 0.2,
                "leg_[lr]4_joint": 10.4 * 0.2,
                "leg_[lr]5_joint": 17.8 * 0.2,
                "leg_[lr]6_joint": 17.8 * 0.2,
                "waist_yaw_joint": 8.7 * 0.2,

                "zarm_[lr]1_joint": 18.8 * 0.2,
                "zarm_[lr]2_joint": 8.0 * 0.2,
                "zarm_[lr]3_joint": 7.5 * 0.2,
                "zarm_[lr]4_joint": 8.0 * 0.2,
                "zarm_[lr]5_joint": 17.5 * 0.2,
                "zarm_[lr]6_joint": 17.5 * 0.2,
                "zarm_[lr]7_joint": 17.5 * 0.2,
            },
            stiffness={
                "leg_[l,r]1_joint": 60.0,
                "leg_[l,r]2_joint": 60.0,
                "leg_[l,r]3_joint": 80.0,
                "leg_[l,r]4_joint": 95.0,
                "leg_[l,r]5_joint": 55.0,
                "leg_[l,r]6_joint": 55.0,
                "waist_yaw_joint":  40.0,
                "zarm_[l,r]1_joint": 20.0,
                "zarm_[l,r]2_joint": 20.0,
                "zarm_[l,r]3_joint": 20.0,
                "zarm_[l,r]4_joint": 20.0,
                "zarm_[l,r]5_joint": 15.0,
                "zarm_[l,r]6_joint": 15.0,
                "zarm_[l,r]7_joint": 15.0,
            },
            damping={
                "leg_[l,r]1_joint": 6.0,
                "leg_[l,r]2_joint": 6.0,
                "leg_[l,r]3_joint": 6.0,
                "leg_[l,r]4_joint": 6.0,
                "leg_[l,r]5_joint": 7.5,
                "leg_[l,r]6_joint": 7.5,
                "waist_yaw_joint": 4.0,
                "zarm_[l,r]1_joint": 3.0,
                "zarm_[l,r]2_joint": 3.0,
                "zarm_[l,r]3_joint": 3.0,
                "zarm_[l,r]4_joint": 3.0,
                "zarm_[l,r]5_joint": 3.0,
                "zarm_[l,r]6_joint": 3.0,
                "zarm_[l,r]7_joint": 3.0,
            },
            armature={
                "leg_[l,r]1_joint": 0.05,
                "leg_[l,r]2_joint": 0.025,
                "leg_[l,r]3_joint": 0.025,
                "leg_[l,r]4_joint": 0.05,
                "leg_[l,r]5_joint": 0.05,
                "leg_[l,r]6_joint": 0.05,
                "waist_yaw_joint": 0.025,

                "zarm_[l,r]1_joint": 0.025,
                "zarm_[l,r]2_joint": 0.02,
                "zarm_[l,r]3_joint": 0.02,
                "zarm_[l,r]4_joint": 0.02,
                "zarm_[l,r]5_joint": 0.01,
                "zarm_[l,r]6_joint": 0.01,
                "zarm_[l,r]7_joint": 0.01,
            },
            friction=0,
            min_delay=0,
            max_delay=4,
            friction_static={
                "leg_[l,r]1_joint": 1.0,
                "leg_[l,r]2_joint": 0.5,
                "leg_[l,r]3_joint": 0.5,
                "leg_[l,r]4_joint": 1.0,
                "leg_[l,r]5_joint": 0.2,
                "leg_[l,r]6_joint": 0.2,
                "waist_yaw_joint": 0.2,
                "zarm_[l,r]1_joint": 0.5,
                "zarm_[l,r]2_joint": 0.3,
                "zarm_[l,r]3_joint": 0.2,
                "zarm_[l,r]4_joint": 0.3,
                "zarm_[l,r]5_joint": 0.1,
                "zarm_[l,r]6_joint": 0.1,
                "zarm_[l,r]7_joint": 0.1
            },
            friction_activation_vel=0.1,
            friction_dynamic={
                "leg_[lr]1_joint": 0.2,
                "leg_[lr]2_joint": 0.2,
                "leg_[lr]3_joint": 0.2,
                "leg_[lr]4_joint": 0.2,
                "leg_[lr]5_joint": 0.2,
                "leg_[lr]6_joint": 0.2,
                "waist_yaw_joint": 0.2,

                "zarm_[lr]1_joint": 0.1,
                "zarm_[lr]2_joint": 0.1,
                "zarm_[lr]3_joint": 0.1,
                "zarm_[lr]4_joint": 0.1,
                "zarm_[lr]5_joint": 0.1,
                "zarm_[lr]6_joint": 0.1,
                "zarm_[lr]7_joint": 0.1,
            },
        ),
    }
    preserve_joint_order = KUAVOS54_MDP_JOINT_ORDER_CFG

    end_effector_configs = [
        ("leg_l6_link", None),
        ("leg_r6_link", None),
        ("zarm_l7_link", [0.0, 0.0, 0.0]),
        ("zarm_r7_link", [0.0, 0.0, 0.0]),
    ]

RobanS14_CFG = RobanS14ArticulationCfg()
KuavoS54_CFG = KuavoS54ArticulationCfg()


ROBANS17_MDP_JOINT_ORDER_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "waist_yaw_joint",
        "leg_l1_joint", "leg_l2_joint", "leg_l3_joint",
        "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
        "leg_r1_joint", "leg_r2_joint", "leg_r3_joint",
        "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
        "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint",
        "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint",
    ],
    body_names=[
        "waist_yaw_link",
        "leg_l1_link", "leg_l2_link", "leg_l3_link",
        "leg_l4_link", "leg_l5_link", "leg_l6_link",
        "leg_r1_link", "leg_r2_link", "leg_r3_link",
        "leg_r4_link", "leg_r5_link", "leg_r6_link",
        "zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link",
        "zarm_r1_link", "zarm_r2_link", "zarm_r3_link", "zarm_r4_link",
    ],
    preserve_order=True,
)


@configclass
class RobanS17ArticulationCfg(ArticulationCfg):
    """Configuration for Roban S17 (S2.2 / biped_s17) articulation."""

    spawn = sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ASSET_DIR}/robanS17/urdf/biped_s17.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    )
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "waist_yaw_joint": 0.0,
            "leg_l1_joint": 0.0, "leg_l2_joint": 0.0, "leg_l3_joint": 0.0,
            "leg_l4_joint": 0.0, "leg_l5_joint": 0.0, "leg_l6_joint": 0.0,
            "leg_r1_joint": 0.0, "leg_r2_joint": 0.0, "leg_r3_joint": 0.0,
            "leg_r4_joint": 0.0, "leg_r5_joint": 0.0, "leg_r6_joint": 0.0,
            "zarm_l1_joint": 0.0, "zarm_l2_joint": 0.0,
            "zarm_l3_joint": 0.0, "zarm_l4_joint": 0.0,
            "zarm_r1_joint": 0.0, "zarm_r2_joint": 0.0,
            "zarm_r3_joint": 0.0, "zarm_r4_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    )
    soft_joint_pos_limit_factor = 0.95
    actuators = {
        "motor": LejuDelayedPDActuatorCfg_S17(
            joint_names_expr=["waist_yaw_joint", "leg_.*", "zarm_.*"],
            effort_limit_sim={
                "waist_yaw_joint": 80.0,
                "leg_[lr]1_joint": 150.0,
                "leg_[lr]2_joint": 150.0,
                "leg_[lr]3_joint": 70.0,
                "leg_[lr]4_joint": 150.0,
                "leg_[lr]5_joint": 74.0,
                "leg_[lr]6_joint": 74.0,
                "zarm_[lr]1_joint": 14.1,
                "zarm_[lr]2_joint": 37.0,
                "zarm_[lr]3_joint": 37.0,
                "zarm_[lr]4_joint": 37.0,
            },
            effort_limit_rated={
                "waist_yaw_joint": 35.0,
                "leg_[lr]1_joint": 51.0,
                "leg_[lr]2_joint": 51.0,
                "leg_[lr]3_joint": 34.0,
                "leg_[lr]4_joint": 51.0,
                "leg_[lr]5_joint": 37.0,
                "leg_[lr]6_joint": 37.0,
                "zarm_[lr]1_joint": 6.1,
                "zarm_[lr]2_joint": 11.0,
                "zarm_[lr]3_joint": 11.0,
                "zarm_[lr]4_joint": 11.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 12.0,
                "leg_[lr]1_joint": 14.6,
                "leg_[lr]2_joint": 14.6,
                "leg_[lr]3_joint": 12.0,
                "leg_[lr]4_joint": 14.6,
                "leg_[lr]5_joint": 17.0,
                "leg_[lr]6_joint": 17.0,
                "zarm_[lr]1_joint": 10.5,
                "zarm_[lr]2_joint": 15.0,
                "zarm_[lr]3_joint": 15.0,
                "zarm_[lr]4_joint": 15.0,
            },
            stiffness={
                "waist_yaw_joint": 40.1792,
                "leg_[lr]1_joint": 90.1792,
                "leg_[lr]2_joint": 150.0984,
                "leg_[lr]3_joint": 40.1792,
                "leg_[lr]4_joint": 150.0984,
                "leg_[lr]5_joint": 34.2506,
                "leg_[lr]6_joint": 34.2506,
                "zarm_[lr]1_joint": 14.2506,
                "zarm_[lr]2_joint": 14.2506,
                "zarm_[lr]3_joint": 14.2506,
                "zarm_[lr]4_joint": 14.2506,
            },
            damping={
                "waist_yaw_joint": 3.5579,
                "leg_[lr]1_joint": 3.5579,
                "leg_[lr]2_joint": 8.3088,
                "leg_[lr]3_joint": 3.5579,
                "leg_[lr]4_joint": 8.3088,
                "leg_[lr]5_joint": 2.9072,
                "leg_[lr]6_joint": 2.9072,
                "zarm_[lr]1_joint": 1.9072,
                "zarm_[lr]2_joint": 1.9072,
                "zarm_[lr]3_joint": 1.9072,
                "zarm_[lr]4_joint": 1.9072,
            },
            armature={
                "waist_yaw_joint": 0.03178707,
                "leg_[lr]1_joint": 0.0493,
                "leg_[lr]2_joint": 0.0493,
                "leg_[lr]3_joint": 0.03178707,
                "leg_[lr]4_joint": 0.0493,
                "leg_[lr]5_joint": 0.036,
                "leg_[lr]6_joint": 0.036,
                "zarm_[lr]1_joint": 0.015,
                "zarm_[lr]2_joint": 0.018,
                "zarm_[lr]3_joint": 0.018,
                "zarm_[lr]4_joint": 0.018,
            },
            friction=0,
            min_delay=0,
            max_delay=4,
            friction_static=0.2,
            friction_activation_vel=0.1,
            friction_dynamic=0,
        ),
    }
    preserve_joint_order = ROBANS17_MDP_JOINT_ORDER_CFG
    end_effector_configs = [
        ("leg_l6_link", None),
        ("leg_r6_link", None),
        ("zarm_l4_link", [0.0, 0.0, -0.2]),
        ("zarm_r4_link", [0.0, 0.0, -0.2]),
    ]


RobanS17_CFG = RobanS17ArticulationCfg()

RobanS17_AMP_CFG = RobanS17ArticulationCfg(
    spawn=RobanS17_CFG.spawn.replace(asset_path=f"{ASSET_DIR}/robanS17/urdf/robanS17_0204.urdf"),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.77),
        rot=(1.0, 0.0, 0.0, 0.0),
        # 0309默认姿态
        joint_pos={
            "waist_yaw_joint": 0.0,
            "leg_l1_joint": -0.05033,
            "leg_l2_joint": -0.0164,
            "leg_l3_joint": -0.0233,
            "leg_l4_joint": 0.155,
            "leg_l5_joint": -0.1,
            "leg_l6_joint": 0,
            
            "leg_r1_joint": 0.05033,
            "leg_r2_joint": 0.0164,
            "leg_r3_joint": -0.0233,
            "leg_r4_joint": 0.155,
            "leg_r5_joint": -0.1,
            "leg_r6_joint": 0,

            # "zarm_l1_joint": 0.2,
            # "zarm_l2_joint": 0.210,
            # "zarm_l3_joint": -0.4,
            # "zarm_l4_joint": -0.6,
            # "zarm_r1_joint": 0.2,
            # "zarm_r2_joint": -0.210,
            # "zarm_r3_joint": 0.4,
            # "zarm_r4_joint": -0.6,

            "zarm_l1_joint": 0.2,
            "zarm_l2_joint": 0.1,
            "zarm_l3_joint": -0.2,
            "zarm_l4_joint": -0.6,

            "zarm_r1_joint": 0.2,
            "zarm_r2_joint": -0.1,
            "zarm_r3_joint": 0.2,
            "zarm_r4_joint": -0.6,
        },   
        
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "motor": RobanS17_CFG.actuators["motor"].replace(
            damping={
                "waist_yaw_joint": 2.557,
                "leg_[l,r]1_joint": 3.557,
                "leg_[l,r]2_joint": 8.308,
                "leg_[l,r]3_joint": 3.557,
                "leg_[l,r]4_joint": 8.308,
                "leg_[l,r]5_joint": 1.907,
                "leg_[l,r]6_joint": 1.907,
                "zarm_[l,r]1_joint": 0.907,
                "zarm_[l,r]2_joint": 0.907,
                "zarm_[l,r]3_joint": 0.907,
                "zarm_[l,r]4_joint": 0.907,
            },
        ),
    },
)


KUAVOS46_MDP_JOINT_ORDER_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "leg_l1_joint", "leg_l2_joint", "leg_l3_joint",
        "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
        "leg_r1_joint", "leg_r2_joint", "leg_r3_joint",
        "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
        "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint", "zarm_l4_joint",
        "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
        "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint", "zarm_r4_joint",
        "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
    ],
    body_names=[
        "leg_l1_link", "leg_l2_link", "leg_l3_link",
        "leg_l4_link", "leg_l5_link", "leg_l6_link",
        "leg_r1_link", "leg_r2_link", "leg_r3_link",
        "leg_r4_link", "leg_r5_link", "leg_r6_link",
        "zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link",
        "zarm_l5_link", "zarm_l6_link", "zarm_l7_link",
        "zarm_r1_link", "zarm_r2_link", "zarm_r3_link", "zarm_r4_link",
        "zarm_r5_link", "zarm_r6_link", "zarm_r7_link",
    ],
    preserve_order=True,
)


@configclass
class KuavoS46ArticulationCfg(ArticulationCfg):
    """Configuration for Kuavo S46 articulation (USD-based, 26 DOF, no waist)."""

    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/kuavoS46/usd/biped_s46.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    )
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "leg_[l,r]1_joint": 0.0,
            "leg_[l,r]2_joint": 0.0,
            "leg_[l,r]3_joint": -0.27,
            "leg_[l,r]4_joint": 0.52,
            "leg_[l,r]5_joint": -0.3,
            "leg_[l,r]6_joint": 0.0,
            "zarm_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    )
    soft_joint_pos_limit_factor = 0.9
    actuators = {
        "motor": LejuDelayedPDActuatorCfg(
            joint_names_expr=["leg_.*", "zarm_.*_joint"],
            effort_limit_sim={
                "leg_[lr]1_joint": 180.0,
                "leg_[lr]2_joint": 100.0,
                "leg_[lr]3_joint": 100.0,
                "leg_[lr]4_joint": 180.0,
                "leg_[lr]5_joint": 72.0,
                "leg_[lr]6_joint": 36.0,
                "zarm_[lr]1_joint": 100.0,
                "zarm_[lr]2_joint": 50.0,
                "zarm_[lr]3_joint": 36.0,
                "zarm_[lr]4_joint": 50.0,
                "zarm_[lr]5_joint": 12.0,
                "zarm_[lr]6_joint": 12.0,
                "zarm_[lr]7_joint": 12.0,
            },
            velocity_limit_sim={
                "leg_[lr]1_joint": 14.0,
                "leg_[lr]2_joint": 23.0,
                "leg_[lr]3_joint": 23.0,
                "leg_[lr]4_joint": 14.0,
                "leg_[lr]5_joint": 10.0,
                "leg_[lr]6_joint": 10.0,
                "zarm_[lr]1_joint": 23.0,
                "zarm_[lr]2_joint": 23.0,
                "zarm_[lr]3_joint": 23.0,
                "zarm_[lr]4_joint": 23.0,
                "zarm_[lr]5_joint": 23.0,
                "zarm_[lr]6_joint": 23.0,
                "zarm_[lr]7_joint": 23.0,
            },
            effort_weaken_velocity_limit={
                "leg_[lr]1_joint": 14.0 * 0.2,
                "leg_[lr]2_joint": 23.0 * 0.2,
                "leg_[lr]3_joint": 23.0 * 0.2,
                "leg_[lr]4_joint": 14.0 * 0.2,
                "leg_[lr]5_joint": 10.0 * 0.2,
                "leg_[lr]6_joint": 10.0 * 0.2,
                "zarm_[lr]1_joint": 23.0 * 0.2,
                "zarm_[lr]2_joint": 23.0 * 0.2,
                "zarm_[lr]3_joint": 23.0 * 0.2,
                "zarm_[lr]4_joint": 23.0 * 0.2,
                "zarm_[lr]5_joint": 23.0 * 0.2,
                "zarm_[lr]6_joint": 23.0 * 0.2,
                "zarm_[lr]7_joint": 23.0 * 0.2,
            },
            stiffness={
                "leg_[lr]1_joint": 100.0,
                "leg_[lr]2_joint": 100.0,
                "leg_[lr]3_joint": 100.0,
                "leg_[lr]4_joint": 150.0,
                "leg_[lr]5_joint": 40.0,
                "leg_[lr]6_joint": 40.0,
                "zarm_[lr]1_joint": 30.0,
                "zarm_[lr]2_joint": 30.0,
                "zarm_[lr]3_joint": 30.0,
                "zarm_[lr]4_joint": 20.0,
                "zarm_[lr]5_joint": 10.0,
                "zarm_[lr]6_joint": 10.0,
                "zarm_[lr]7_joint": 10.0,
            },
            damping={
                "leg_[lr]1_joint": 4.0,
                "leg_[lr]2_joint": 4.0,
                "leg_[lr]3_joint": 4.0,
                "leg_[lr]4_joint": 8.0,
                "leg_[lr]5_joint": 4.0,
                "leg_[lr]6_joint": 4.0,
                "zarm_[lr]1_joint": 3.0,
                "zarm_[lr]2_joint": 3.0,
                "zarm_[lr]3_joint": 3.0,
                "zarm_[lr]4_joint": 3.0,
                "zarm_[lr]5_joint": 3.0,
                "zarm_[lr]6_joint": 3.0,
                "zarm_[lr]7_joint": 3.0,
            },
            armature={
                "leg_[lr]1_joint": 0.05,
                "leg_[lr]2_joint": 0.025,
                "leg_[lr]3_joint": 0.025,
                "leg_[lr]4_joint": 0.05,
                "leg_[lr]5_joint": 0.05,
                "leg_[lr]6_joint": 0.05,
                "zarm_[lr]1_joint": 0.025,
                "zarm_[lr]2_joint": 0.02,
                "zarm_[lr]3_joint": 0.02,
                "zarm_[lr]4_joint": 0.02,
                "zarm_[lr]5_joint": 0.01,
                "zarm_[lr]6_joint": 0.01,
                "zarm_[lr]7_joint": 0.01,
            },
            friction=0,
            min_delay=0,
            max_delay=4,
            friction_static={
                "leg_[lr]1_joint": 1.0,
                "leg_[lr]2_joint": 0.5,
                "leg_[lr]3_joint": 0.5,
                "leg_[lr]4_joint": 1.0,
                "leg_[lr]5_joint": 0.2,
                "leg_[lr]6_joint": 0.2,
                "zarm_[lr]1_joint": 0.5,
                "zarm_[lr]2_joint": 0.3,
                "zarm_[lr]3_joint": 0.2,
                "zarm_[lr]4_joint": 0.3,
                "zarm_[lr]5_joint": 0.1,
                "zarm_[lr]6_joint": 0.1,
                "zarm_[lr]7_joint": 0.1,
            },
            friction_activation_vel=0.1,
            friction_dynamic=0,
        ),
    }
    preserve_joint_order = KUAVOS46_MDP_JOINT_ORDER_CFG


KuavoS46_CFG = KuavoS46ArticulationCfg()


KUAVOS53_MDP_JOINT_ORDER_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "leg_l1_joint", "leg_l2_joint", "leg_l3_joint",
        "leg_l4_joint", "leg_l5_joint", "leg_l6_joint",
        "leg_r1_joint", "leg_r2_joint", "leg_r3_joint",
        "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
        "waist_yaw_joint",
        "zarm_l1_joint", "zarm_l2_joint", "zarm_l3_joint",
        "zarm_l4_joint", "zarm_l5_joint", "zarm_l6_joint", "zarm_l7_joint",
        "zarm_r1_joint", "zarm_r2_joint", "zarm_r3_joint",
        "zarm_r4_joint", "zarm_r5_joint", "zarm_r6_joint", "zarm_r7_joint",
    ],
    body_names=[
        "leg_l1_link", "leg_l2_link", "leg_l3_link",
        "leg_l4_link", "leg_l5_link", "leg_l6_link",
        "leg_r1_link", "leg_r2_link", "leg_r3_link",
        "leg_r4_link", "leg_r5_link", "leg_r6_link",
        "waist_yaw_link",
        "zarm_l1_link", "zarm_l2_link", "zarm_l3_link",
        "zarm_l4_link", "zarm_l5_link", "zarm_l6_link", "zarm_l7_link",
        "zarm_r1_link", "zarm_r2_link", "zarm_r3_link",
        "zarm_r4_link", "zarm_r5_link", "zarm_r6_link", "zarm_r7_link",
    ],
    preserve_order=True,
)


@configclass
class KuavoS53ArticulationCfg(ArticulationCfg):
    """Configuration for Kuavo S53 articulation. Actuator params identical to S54."""

    spawn = sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/kuavos53/urdf/biped_s53.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    )
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.925),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "leg_[l,r]1_joint": 0.0,
            "leg_[l,r]2_joint": 0.0,
            "leg_[l,r]3_joint": -0.4,
            "leg_[l,r]4_joint": 0.69,
            "leg_[l,r]5_joint": -0.33,
            "leg_[l,r]6_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "zarm_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    )
    soft_joint_pos_limit_factor = 0.95
    actuators = {
        "motor": LejuDelayedPDActuatorCfg(
            joint_names_expr=["waist_.*", "leg_.*", "zarm_.*"],
            effort_limit_sim={
                "leg_[l,r]1_joint": 101.6,
                "leg_[l,r]2_joint": 56.8,
                "leg_[l,r]3_joint": 105.6,
                "leg_[l,r]4_joint": 224.0,
                "leg_[l,r]5_joint": 91.6,
                "leg_[l,r]6_joint": 57.0,
                "waist_yaw_joint": 81.6,
                "zarm_[l,r]1_joint": 52.8,
                "zarm_[l,r]2_joint": 60.0,
                "zarm_[l,r]3_joint": 45.6,
                "zarm_[l,r]4_joint": 60.0,
                "zarm_[l,r]5_joint": 11.0,
                "zarm_[l,r]6_joint": 11.0,
                "zarm_[l,r]7_joint": 11.0,
            },
            velocity_limit_sim={
                "leg_[l,r]1_joint": 10.4,
                "leg_[l,r]2_joint": 8.7,
                "leg_[l,r]3_joint": 12.7,
                "leg_[l,r]4_joint": 10.4,
                "leg_[l,r]5_joint": 17.8,
                "leg_[l,r]6_joint": 17.8,
                "waist_yaw_joint": 8.7,
                "zarm_[l,r]1_joint": 18.8,
                "zarm_[l,r]2_joint": 8.0,
                "zarm_[l,r]3_joint": 7.5,
                "zarm_[l,r]4_joint": 8.0,
                "zarm_[l,r]5_joint": 17.5,
                "zarm_[l,r]6_joint": 17.5,
                "zarm_[l,r]7_joint": 17.5,
            },
            effort_weaken_velocity_limit={
                "leg_[lr]1_joint": 10.4 * 0.2,
                "leg_[lr]2_joint": 8.7 * 0.2,
                "leg_[lr]3_joint": 12.7 * 0.2,
                "leg_[lr]4_joint": 10.4 * 0.2,
                "leg_[lr]5_joint": 17.8 * 0.2,
                "leg_[lr]6_joint": 17.8 * 0.2,
                "waist_yaw_joint": 8.7 * 0.2,
                "zarm_[lr]1_joint": 18.8 * 0.2,
                "zarm_[lr]2_joint": 8.0 * 0.2,
                "zarm_[lr]3_joint": 7.5 * 0.2,
                "zarm_[lr]4_joint": 8.0 * 0.2,
                "zarm_[lr]5_joint": 17.5 * 0.2,
                "zarm_[lr]6_joint": 17.5 * 0.2,
                "zarm_[lr]7_joint": 17.5 * 0.2,
            },
            stiffness={
                "leg_[l,r]1_joint": 60.0,
                "leg_[l,r]2_joint": 60.0,
                "leg_[l,r]3_joint": 80.0,
                "leg_[l,r]4_joint": 95.0,
                "leg_[l,r]5_joint": 55.0,
                "leg_[l,r]6_joint": 55.0,
                "waist_yaw_joint": 40.0,
                "zarm_[l,r]1_joint": 20.0,
                "zarm_[l,r]2_joint": 20.0,
                "zarm_[l,r]3_joint": 20.0,
                "zarm_[l,r]4_joint": 20.0,
                "zarm_[l,r]5_joint": 15.0,
                "zarm_[l,r]6_joint": 15.0,
                "zarm_[l,r]7_joint": 15.0,
            },
            damping={
                "leg_[l,r]1_joint": 6.0,
                "leg_[l,r]2_joint": 6.0,
                "leg_[l,r]3_joint": 6.0,
                "leg_[l,r]4_joint": 6.0,
                "leg_[l,r]5_joint": 7.5,
                "leg_[l,r]6_joint": 7.5,
                "waist_yaw_joint": 4.0,
                "zarm_[l,r]1_joint": 3.0,
                "zarm_[l,r]2_joint": 3.0,
                "zarm_[l,r]3_joint": 3.0,
                "zarm_[l,r]4_joint": 3.0,
                "zarm_[l,r]5_joint": 3.0,
                "zarm_[l,r]6_joint": 3.0,
                "zarm_[l,r]7_joint": 3.0,
            },
            armature={
                "leg_[l,r]1_joint": 0.05,
                "leg_[l,r]2_joint": 0.025,
                "leg_[l,r]3_joint": 0.025,
                "leg_[l,r]4_joint": 0.05,
                "leg_[l,r]5_joint": 0.05,
                "leg_[l,r]6_joint": 0.05,
                "waist_yaw_joint": 0.025,
                "zarm_[l,r]1_joint": 0.025,
                "zarm_[l,r]2_joint": 0.02,
                "zarm_[l,r]3_joint": 0.02,
                "zarm_[l,r]4_joint": 0.02,
                "zarm_[l,r]5_joint": 0.01,
                "zarm_[l,r]6_joint": 0.01,
                "zarm_[l,r]7_joint": 0.01,
            },
            friction=0,
            min_delay=0,
            max_delay=4,
            friction_static={
                "leg_[l,r]1_joint": 1.0,
                "leg_[l,r]2_joint": 0.5,
                "leg_[l,r]3_joint": 0.5,
                "leg_[l,r]4_joint": 1.0,
                "leg_[l,r]5_joint": 0.2,
                "leg_[l,r]6_joint": 0.2,
                "waist_yaw_joint": 0.2,
                "zarm_[l,r]1_joint": 0.5,
                "zarm_[l,r]2_joint": 0.3,
                "zarm_[l,r]3_joint": 0.2,
                "zarm_[l,r]4_joint": 0.3,
                "zarm_[l,r]5_joint": 0.1,
                "zarm_[l,r]6_joint": 0.1,
                "zarm_[l,r]7_joint": 0.1,
            },
            friction_activation_vel=0.1,
            friction_dynamic={
                "leg_[lr]1_joint": 0.2,
                "leg_[lr]2_joint": 0.2,
                "leg_[lr]3_joint": 0.2,
                "leg_[lr]4_joint": 0.2,
                "leg_[lr]5_joint": 0.2,
                "leg_[lr]6_joint": 0.2,
                "waist_yaw_joint": 0.2,
                "zarm_[lr]1_joint": 0.1,
                "zarm_[lr]2_joint": 0.1,
                "zarm_[lr]3_joint": 0.1,
                "zarm_[lr]4_joint": 0.1,
                "zarm_[lr]5_joint": 0.1,
                "zarm_[lr]6_joint": 0.1,
                "zarm_[lr]7_joint": 0.1,
            },
        ),
    }
    preserve_joint_order = KUAVOS53_MDP_JOINT_ORDER_CFG

    end_effector_configs = [
        ("leg_l6_link", None),
        ("leg_r6_link", None),
        ("zarm_l7_link", [0.0, 0.0, 0.0]),
        ("zarm_r7_link", [0.0, 0.0, 0.0]),
    ]


KuavoS53_CFG = KuavoS53ArticulationCfg()