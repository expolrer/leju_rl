from isaaclab.utils import configclass

from leju_robot.assets.motion_data import MOTION_DIR
from leju_robot.tasks.tracking.config.kuavoS54.dance.tracking_env_cfg import (
    KuavoS54FlatEnvCfg,
    KuavoS54FlatEnvCfg_PLAY,
)
from leju_robot.tasks.tracking.config.kuavoS53.dance.kuavoS53 import KuavoS53_ACTION_SCALE, KuavoS53_CYLINDER_CFG


@configclass
class KuavoS53FlatEnvCfg(KuavoS54FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Replace robot asset
        self.scene.robot = KuavoS53_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Motion file (S53 and S54 share the same NPZ data)
        self.commands.motion.motion_file = f"{MOTION_DIR}/mimic/npz_data/kuavos54_dance_50fps.npz"
        # Replace action scale and joint order
        self.actions.joint_pos.scale = KuavoS53_ACTION_SCALE
        self.actions.joint_pos.joint_names = KuavoS53_CYLINDER_CFG.preserve_joint_order.joint_names
        self.observations.policy.joint_pos.params = {"asset_cfg": KuavoS53_CYLINDER_CFG.preserve_joint_order}
        self.observations.policy.joint_vel.params = {"asset_cfg": KuavoS53_CYLINDER_CFG.preserve_joint_order}
        self.observations.critic.joint_pos.params = {"asset_cfg": KuavoS53_CYLINDER_CFG.preserve_joint_order}
        self.observations.critic.joint_vel.params = {"asset_cfg": KuavoS53_CYLINDER_CFG.preserve_joint_order}
        self.events.add_joint_default_pos.params = {
            "asset_cfg": KuavoS53_CYLINDER_CFG.preserve_joint_order,
            "pos_distribution_params": (-0.1, 0.1),
            "operation": "add",
        }


@configclass
class KuavoS53FlatEnvCfg_PLAY(KuavoS53FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.episode_length_s = 1e9

        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        self.events.physics_material = None
        self.events.add_joint_default_pos.params = {
            "asset_cfg": KuavoS53_CYLINDER_CFG.preserve_joint_order,
            "pos_distribution_params": (-0.0, 0.0),
            "operation": "add",
        }
        self.events.torso_com = None
        self.events.base_com = None
        self.events.add_torso_mass = None
        self.events.add_base_mass = None
        self.events.link_com = None
        self.events.add_link_mass = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None

        self.events.push_robot = None
        self.events.base_external_force_torque = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
