from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class PushManager:
    """State manager for external force and torque disturbances."""
    
    def __init__(
        self,
        env: ManagerBasedEnv,
        disturbance_categories: list[dict],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        """Initialize PushManager."""
        first_category = disturbance_categories[0]
        asset_name = first_category["asset_cfg"].name
        self.asset: RigidObject | Articulation = env.scene[asset_name]
        self.device = self.asset.device
        self.disturbance_categories = disturbance_categories

        num_envs = env.scene.num_envs

        self.active = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.time_remaining = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.category_idx = torch.randint(0, len(disturbance_categories), (num_envs,), device=self.device)
        self.body_ids_list = [None] * num_envs
        max_bodies = 0
        for category in disturbance_categories:
            cat_asset_cfg = category["asset_cfg"]
            if cat_asset_cfg.body_ids == slice(None):
                max_bodies = max(max_bodies, self.asset.num_bodies)
            else:
                max_bodies = max(max_bodies, len(cat_asset_cfg.body_ids))
        
        self.forces = torch.zeros(num_envs, max_bodies, 3, dtype=torch.float32, device=self.device)
        self.torques = torch.zeros(num_envs, max_bodies, 3, dtype=torch.float32, device=self.device)
    
    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor | None):
        """Step state update and apply disturbances."""
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.device)

        dt = env.step_dt

        self.time_remaining -= dt

        need_clear = self.active & (self.time_remaining <= 0)
        if need_clear.any():
            clear_env_ids = torch.where(need_clear)[0]
            for env_id in clear_env_ids:
                cat_idx = self.category_idx[env_id].item()
                category = self.disturbance_categories[cat_idx]
                duration_off = math_utils.sample_uniform(
                    category["interval_range_off"][0],
                    category["interval_range_off"][1],
                    (1,),
                    device=self.device
                )[0]
                self.time_remaining[env_id] = duration_off
            self.active[clear_env_ids] = False
            self.forces[clear_env_ids] = 0.0
            self.torques[clear_env_ids] = 0.0

        need_start = ~self.active & (self.time_remaining <= 0)
        if need_start.any():
            start_env_ids = torch.where(need_start)[0]

            for env_id in start_env_ids:
                cat_idx = self.category_idx[env_id].item()
                category = self.disturbance_categories[cat_idx]
                cat_asset_cfg = category["asset_cfg"]

                if cat_asset_cfg.body_ids == slice(None):
                    body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device=self.device)
                else:
                    body_ids = torch.tensor(cat_asset_cfg.body_ids, dtype=torch.int, device=self.device)

                self.body_ids_list[env_id] = body_ids
                duration_on = math_utils.sample_uniform(
                    category["duration_range_on"][0],
                    category["duration_range_on"][1],
                    (1,),
                    device=self.device
                )[0]

                force_x = math_utils.sample_uniform(
                    category["force_range_xy"][0], category["force_range_xy"][1], (1,), device=self.device
                )[0]
                force_y = math_utils.sample_uniform(
                    category["force_range_xy"][0], category["force_range_xy"][1], (1,), device=self.device
                )[0]
                force_z = math_utils.sample_uniform(
                    category["force_range_z"][0], category["force_range_z"][1], (1,), device=self.device
                )[0]

                torque_x = math_utils.sample_uniform(
                    category["torque_range_xy"][0], category["torque_range_xy"][1], (1,), device=self.device
                )[0]
                torque_y = math_utils.sample_uniform(
                    category["torque_range_xy"][0], category["torque_range_xy"][1], (1,), device=self.device
                )[0]
                torque_z = math_utils.sample_uniform(
                    category["torque_range_z"][0], category["torque_range_z"][1], (1,), device=self.device
                )[0]
                
                num_bodies = len(body_ids)
                for i, body_id in enumerate(body_ids):
                    self.forces[env_id, i, 0] = force_x
                    self.forces[env_id, i, 1] = force_y
                    self.forces[env_id, i, 2] = force_z
                    self.torques[env_id, i, 0] = torque_x
                    self.torques[env_id, i, 1] = torque_y
                    self.torques[env_id, i, 2] = torque_z

                self.active[env_id] = True
                self.time_remaining[env_id] = duration_on

        active_env_ids = torch.where(self.active)[0]
        if len(active_env_ids) > 0:
            body_ids_groups = {}
            for env_id in active_env_ids.cpu().tolist():
                body_ids = self.body_ids_list[env_id]
                if body_ids is None:
                    continue
                body_ids_key = tuple(body_ids.cpu().tolist())
                if body_ids_key not in body_ids_groups:
                    body_ids_groups[body_ids_key] = []
                body_ids_groups[body_ids_key].append(env_id)

            for body_ids_key, group_env_ids in body_ids_groups.items():
                body_ids = torch.tensor(body_ids_key, dtype=torch.int, device=self.device)
                group_env_ids_tensor = torch.tensor(group_env_ids, dtype=torch.long, device=self.device)

                num_bodies = len(body_ids)
                group_forces = self.forces[group_env_ids_tensor, :num_bodies, :]
                group_torques = self.torques[group_env_ids_tensor, :num_bodies, :]

                self.asset.set_external_force_and_torque(
                    group_forces, group_torques, env_ids=group_env_ids_tensor, body_ids=body_ids
                )

        inactive_env_ids = torch.where(~self.active)[0]
        if len(inactive_env_ids) > 0:
            all_body_ids = set()
            for env_id in inactive_env_ids.cpu().tolist():
                body_ids = self.body_ids_list[env_id]
                if body_ids is not None:
                    all_body_ids.add(tuple(body_ids.cpu().tolist()))

            for body_ids_key in all_body_ids:
                body_ids = torch.tensor(body_ids_key, dtype=torch.int, device=self.device)
                relevant_inactive = [
                    env_id for env_id in inactive_env_ids.cpu().tolist()
                    if self.body_ids_list[env_id] is not None
                    and tuple(self.body_ids_list[env_id].cpu().tolist()) == body_ids_key
                ]
                if len(relevant_inactive) > 0:
                    relevant_inactive_tensor = torch.tensor(relevant_inactive, dtype=torch.long, device=self.device)
                    num_bodies = len(body_ids)
                    zero_forces = torch.zeros(len(relevant_inactive), num_bodies, 3, dtype=torch.float32, device=self.device)
                    zero_torques = torch.zeros(len(relevant_inactive), num_bodies, 3, dtype=torch.float32, device=self.device)
                    self.asset.set_external_force_and_torque(
                        zero_forces, zero_torques, env_ids=relevant_inactive_tensor, body_ids=body_ids
                    )


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        # env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos
        env.action_manager.get_term("joint_pos")._offset[env_ids, :] = pos.unsqueeze(1)


def set_joint_soft_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    margin: float | dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set joint soft limits using absolute margin values instead of percentages.

    Called during environment initialization. Soft limits are based on hard limits (defined in URDF)
    minus specified absolute margin values.

    Example:
        Usage in tracking_env_cfg.py:
        ```python
        @configclass
        class EventCfg:
            startup = EventTermCfg(
                func=mdp.set_joint_soft_limits,
                params={
                    "margin": {
                        "waist_yaw_joint": 0.1,
                        "leg_[l,r]1_joint": 0.1,
                        # ...
                    },
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                },
                mode="startup",
            )
        ```
    """
    import re
    
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    
    if isinstance(asset_cfg.joint_ids, slice):
        if asset_cfg.joint_ids == slice(None):
            joint_ids = list(range(asset.num_joints))
        else:
            joint_ids = list(range(*asset_cfg.joint_ids.indices(asset.num_joints)))
    elif isinstance(asset_cfg.joint_ids, list):
        joint_ids = asset_cfg.joint_ids
    else:
        joint_ids = list(asset_cfg.joint_ids)

    joint_names = [asset.joint_names[i] for i in joint_ids]

    if isinstance(margin, dict):
        margin_list = []
        for joint_name in joint_names:
            matched_margin = 0.0
            for pattern, margin_value in margin.items():
                regex_pattern = pattern.replace("[l,r]", "[lr]")
                if re.match(f"^{regex_pattern}$", joint_name):
                    matched_margin = margin_value
                    break
            margin_list.append(matched_margin)
        margin_tensor = torch.tensor(margin_list, device=asset.device, dtype=torch.float32)
    else:
        margin_tensor = torch.full((len(joint_ids),), margin, device=asset.device, dtype=torch.float32)

    joint_limits = asset.data.joint_pos_limits[:, joint_ids, :]

    for env_id in env_ids:
        asset.data.soft_joint_pos_limits[env_id, joint_ids, 0] = joint_limits[env_id, :, 0] + margin_tensor
        asset.data.soft_joint_pos_limits[env_id, joint_ids, 1] = joint_limits[env_id, :, 1] - margin_tensor

    print(f"[set_joint_soft_limits] Set custom soft limits for {len(env_ids)} environments")
    print(f"[set_joint_soft_limits] Affected joints: {joint_names}")


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def init_push_manager(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    disturbance_categories: list[dict],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Initialize PushManager, called during environment startup."""
    env._push_manager = PushManager(env, disturbance_categories, asset_cfg)


def push_by_impulse_from_force_duration(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    disturbance_categories: list[dict],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply external force and torque disturbances with multiple categories.

    Uses PushManager to manage state. PushManager should be initialized in startup event.
    """
    if not hasattr(env, "_push_manager"):
        env._push_manager = PushManager(env, disturbance_categories, asset_cfg)

    env._push_manager.step(env, env_ids)

def apply_disturbance_with_duration(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    disturbance_categories: list[dict],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply external force and torque disturbances with duration and interval.

    Note: This function should be called every step (using interval mode with small interval_range_s, e.g., 0.01s)
    to precisely control duration and interval.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    device = asset.device

    if not hasattr(env, "_disturbance_active"):
        num_envs = env.scene.num_envs
        env._disturbance_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        env._disturbance_time_remaining = torch.zeros(num_envs, dtype=torch.float32, device=device)
        env._disturbance_forces = torch.zeros(num_envs, asset.num_bodies, 3, dtype=torch.float32, device=device)
        env._disturbance_torques = torch.zeros(num_envs, asset.num_bodies, 3, dtype=torch.float32, device=device)
        env._disturbance_category_idx = torch.randint(0, len(disturbance_categories), (num_envs,), device=device)

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)

    dt = env.step_dt

    env._disturbance_time_remaining -= dt

    need_clear = env._disturbance_active & (env._disturbance_time_remaining <= 0)
    if need_clear.any():
        clear_env_ids = torch.where(need_clear)[0]
        for env_id in clear_env_ids:
            category_idx = env._disturbance_category_idx[env_id].item()
            category = disturbance_categories[category_idx]
            duration_off = math_utils.sample_uniform(
                category["duration_off_range"][0],
                category["duration_off_range"][1],
                (1,),
                device=device
            )[0]
            env._disturbance_time_remaining[env_id] = duration_off
        env._disturbance_active[clear_env_ids] = False
        env._disturbance_forces[clear_env_ids] = 0.0
        env._disturbance_torques[clear_env_ids] = 0.0

    need_start = ~env._disturbance_active & (env._disturbance_time_remaining <= 0)
    if need_start.any():
        start_env_ids = torch.where(need_start)[0]

        for env_id in start_env_ids:
            category_idx = env._disturbance_category_idx[env_id].item()
            category = disturbance_categories[category_idx]

            body_names = category["body_names"]
            if isinstance(body_names, str):
                body_names = [body_names]

            body_ids = []
            for body_name in body_names:
                try:
                    found_bodies = asset.find_bodies(body_name)
                    if len(found_bodies) > 0:
                        body_ids.extend(found_bodies)
                except (IndexError, ValueError):
                    continue

            if len(body_ids) == 0:
                continue

            duration_on = math_utils.sample_uniform(
                category["duration_on_range"][0],
                category["duration_on_range"][1],
                (1,),
                device=device
            )[0]

            num_bodies = len(body_ids)
            force_xy = math_utils.sample_uniform(
                category["force_range_xy"][0],
                category["force_range_xy"][1],
                (num_bodies, 2),
                device=device
            )
            force_z = math_utils.sample_uniform(
                category["force_range_z"][0],
                category["force_range_z"][1],
                (num_bodies, 1),
                device=device
            )
            forces = torch.cat([force_xy, force_z], dim=-1)

            torque_xy = math_utils.sample_uniform(
                category["torque_range_xy"][0],
                category["torque_range_xy"][1],
                (num_bodies, 2),
                device=device
            )
            torque_z = math_utils.sample_uniform(
                category["torque_range_z"][0],
                category["torque_range_z"][1],
                (num_bodies, 1),
                device=device
            )
            torques = torch.cat([torque_xy, torque_z], dim=-1)

            env._disturbance_forces[env_id, :] = 0.0
            env._disturbance_torques[env_id, :] = 0.0
            for i, body_id in enumerate(body_ids):
                env._disturbance_forces[env_id, body_id] = forces[i]
                env._disturbance_torques[env_id, body_id] = torques[i]

            env._disturbance_active[env_id] = True
            env._disturbance_time_remaining[env_id] = duration_on

    active_env_ids = torch.where(env._disturbance_active)[0]
    if len(active_env_ids) > 0:
        active_forces = env._disturbance_forces[active_env_ids]
        active_torques = env._disturbance_torques[active_env_ids]

        asset.set_external_force_and_torque(
            active_forces,
            active_torques,
            env_ids=active_env_ids,
            body_ids=slice(None)
        )

    inactive_env_ids = torch.where(~env._disturbance_active)[0]
    if len(inactive_env_ids) > 0:
        num_bodies = asset.num_bodies
        zero_forces = torch.zeros(len(inactive_env_ids), num_bodies, 3, dtype=torch.float32, device=device)
        zero_torques = torch.zeros(len(inactive_env_ids), num_bodies, 3, dtype=torch.float32, device=device)
        asset.set_external_force_and_torque(
            zero_forces,
            zero_torques,
            env_ids=inactive_env_ids,
            body_ids=slice(None)
        )

def apply_external_force_torque_stochastic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # clear the existing forces and torques
    asset._external_force_b *= 0
    asset._external_torque_b *= 0

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    random_values = torch.rand(env_ids.shape, device=env_ids.device)
    mask = random_values < probability
    masked_env_ids = env_ids[mask]

    if len(masked_env_ids) == 0:
        return

    # resolve number of bodies
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )

    # sample random forces and torques
    size = (len(masked_env_ids), num_bodies, 3)
    force_range_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    force_range = torch.tensor(force_range_list, device=asset.device)
    forces = math_utils.sample_uniform(
        force_range[:, 0], force_range[:, 1], size, asset.device
    )
    torque_range_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    torque_range = torch.tensor(torque_range_list, device=asset.device)
    torques = math_utils.sample_uniform(
        torque_range[:, 0], torque_range[:, 1], size, asset.device
    )
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(
        forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids
    )

