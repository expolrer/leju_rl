"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --motion_file source/xxxx/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch
import os
import time
from collections.abc import Sequence

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to local motion npz (overrides registry, if not provided, will use MOTION_FILES list)")
parser.add_argument("--robot", type=str, help="robot name")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_rotate
from dataclasses import MISSING

from leju_robot.assets.leju import RobanS14_CFG, KuavoS54_CFG, RobanS17_CFG, KuavoS53_CFG
from leju_robot.assets.motion_data import MOTION_DIR

ROBOT_CONFIGS = {
    "robanS14": RobanS14_CFG,
    "kuavoS54": KuavoS54_CFG,
    "robanS17": RobanS17_CFG,
    "kuavoS53": KuavoS53_CFG,
}
MOTION_FILES = {
    f"{MOTION_DIR}/mimic/npz_data/kuavos54_dance_50fps.npz",
}

def compute_npz_to_isaac_indices(robot: Articulation, npz_joint_order: list[str]) -> list[int]:
    """
    计算NPZ到IsaacLab的关节映射索引（在初始化时调用一次）
    
    Args:
        robot: IsaacLab机器人对象，用于查找关节索引
        npz_joint_order: NPZ文件中使用的关节顺序（PRESERVE_JOINT_ORDER_ASSET_CFG中的joint_names顺序）
    
    Returns:
        npz_to_isaac_indices: 映射索引列表，第i个元素表示IsaacLab中第i个关节（按索引排序）在NPZ中的索引
    """
    # 使用 find_joints 获取每个 NPZ 关节在 IsaacLab 中的索引
    # isaac_joint_indices[i] 表示 NPZ 中第 i 个关节在 IsaacLab 中的索引
    isaac_joint_indices, _ = robot.find_joints(npz_joint_order, preserve_order=True)
    isaac_joint_indices = list(isaac_joint_indices)
    
    # 构建映射：对于 IsaacLab 中的每个索引 j，找到它在 NPZ 中的索引
    # 创建一个字典：IsaacLab索引 -> NPZ索引
    isaac_to_npz_map = {isaac_idx: npz_idx for npz_idx, isaac_idx in enumerate(isaac_joint_indices)}
    
    # 获取 IsaacLab 中所有关节的索引（从小到大排序）
    all_isaac_indices = sorted(isaac_to_npz_map.keys())
    
    # 构建 NPZ 到 IsaacLab 的映射索引列表
    # npz_to_isaac_indices[i] 表示 IsaacLab 中第 i 个关节（按索引排序）在 NPZ 中的索引
    npz_to_isaac_indices = [isaac_to_npz_map[isaac_idx] for isaac_idx in all_isaac_indices]
    
    return npz_to_isaac_indices


def reorder_joint_data_for_isaac(joint_data: torch.Tensor, npz_to_isaac_indices: list[int]) -> torch.Tensor:
    """
    将NPZ文件中的关节数据重排序为IsaacLab期望的顺序
    
    Args:
        joint_data: NPZ文件中的关节数据 (按NPZ顺序排列)
        npz_to_isaac_indices: NPZ到IsaacLab的映射索引列表（由compute_npz_to_isaac_indices计算）
    
    Returns:
        重排序后的关节数据 (按IsaacLab顺序排列)
    """
    # 应用重排序
    if joint_data.dim() == 1:
        # 一维数据: (num_joints,)
        return joint_data[npz_to_isaac_indices]
    else:
        # 二维数据: (num_envs, num_joints)
        return joint_data[:, npz_to_isaac_indices]


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = MISSING
    
    def __post_init__(self):
        super().__post_init__()
        robot_name = getattr(args_cli, "robot", None) or "robanS14"
        if robot_name not in ROBOT_CONFIGS:
            available = ", ".join(ROBOT_CONFIGS.keys())
            raise ValueError(f"Unknown robot '{robot_name}'. Available options: {available}")
        self.robot = ROBOT_CONFIGS[robot_name].replace(prim_path="{ENV_REGEX_NS}/Robot")

class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Determine motion files: prefer --motion_file, otherwise use MOTION_FILES list, or WandB registry
    if args_cli.motion_file is not None:
        motion_files = [args_cli.motion_file]
    else:
        # 使用脚本中定义的MOTION_FILES集合
        motion_files = MOTION_FILES
    
    # 确保所有文件路径都是绝对路径（相对于workspace root）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.dirname(os.path.dirname(script_dir))  # scripts/mimic的父目录的父目录是workspace root
    motion_files = [
        os.path.abspath(os.path.join(workspace_root, f)) if not os.path.isabs(f) else f 
        for f in motion_files
    ]
    
    # 验证文件存在
    for f in motion_files:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Motion file not found: {f}")
    
    print(f"\n{'='*60}")
    print(f"总共找到 {len(motion_files)} 个NPZ文件，将循环播放")
    print(f"{'='*60}\n")
    
    # NPZ文件中使用的关节顺序
    npz_joint_order = scene.cfg.robot.preserve_joint_order.joint_names
    
    # 在初始化时计算一次映射索引（映射是固定的，不需要在循环中重复计算）
    npz_to_isaac_indices = compute_npz_to_isaac_indices(robot, npz_joint_order)
    print(f"[INFO] Computed NPZ->IsaacLab joint mapping: {len(npz_to_isaac_indices)} joints")

    # 相机自动跟随标志
    camera_follow = False
    initial_camera_set = False
    camera_distance = 3.0
    camera_height = 1.2
    camera_lookahead = 0.5

    def get_camera_pose(root_state: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        pos = root_state[:3]
        quat = root_state[3:7]
        forward_local = torch.tensor([1.0, 0.0, 0.0], device=root_state.device).unsqueeze(0)
        forward = quat_rotate(quat.unsqueeze(0), forward_local).squeeze(0)
        forward = forward / (torch.linalg.norm(forward) + 1e-8)
        pos_np = pos.cpu().numpy()
        forward_np = forward.cpu().numpy()
        camera_pos = pos_np - forward_np * camera_distance
        # camera_pos[2] += camera_height
        camera_pos[1] = 1.0
        look_at = pos_np + forward_np * camera_lookahead
        return camera_pos, look_at

    # 循环播放所有NPZ文件
    current_file_index = 0
    motion = None
    time_steps = None
    frame_dt = None  # 每帧的时间间隔（根据 NPZ 文件的 fps 计算）
    
    cam_pos = [2,2,2]
    cam_look = [0,0,1]
    sim.set_camera_view(cam_pos, cam_look)
    initial_camera_set = True
    while simulation_app.is_running():
        # 如果当前motion文件播放完毕，切换到下一个文件
        if motion is None or (time_steps is not None and time_steps[0].item() >= motion.time_step_total - 1):
            # 加载下一个motion文件
            motion_file = motion_files[current_file_index]
            
            print(f"\n{'='*60}")
            print(f"[{current_file_index + 1}/{len(motion_files)}] 正在加载NPZ文件:")
            print(f"  {os.path.basename(motion_file)}")
            print(f"  完整路径: {motion_file}")
            print(f"{'='*60}\n")
            
            motion = MotionLoader(
                motion_file,
                torch.tensor([0], dtype=torch.long, device=sim.device),
                sim.device,
            )
            time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
            initial_camera_set = False  # 重置相机设置，为新文件设置初始视角
            
            # 从 NPZ 文件中获取 fps，计算每帧的时间间隔
            if hasattr(motion, 'fps') and motion.fps is not None:
                motion_fps = float(motion.fps) if not isinstance(motion.fps, (int, float)) else float(motion.fps)
                frame_dt = 1.0 / motion_fps
                print(f"  Using fps from NPZ file: {motion_fps} Hz (frame_dt: {frame_dt:.4f}s)")
            else:
                # 使用默认 fps (50 Hz)
                motion_fps = 50.0
                frame_dt = 1.0 / motion_fps
                print(f"  Warning: No fps found in NPZ file, using default: {motion_fps} Hz (frame_dt: {frame_dt:.4f}s)")
            
            # 移动到下一个文件索引（循环）
            current_file_index = (current_file_index + 1) % len(motion_files)

        # Record start time for motion frame time matching
        loop_start_time = time.time()
        
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins[:, None, :]
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        
        # 重排序关节数据：从NPZ顺序转换为IsaacLab顺序（使用预计算的映射索引）
        npz_joint_pos = motion.joint_pos[time_steps]
        npz_joint_vel = motion.joint_vel[time_steps]
        
        isaac_joint_pos = reorder_joint_data_for_isaac(npz_joint_pos, npz_to_isaac_indices)
        isaac_joint_vel = reorder_joint_data_for_isaac(npz_joint_vel, npz_to_isaac_indices)
        
        robot.write_joint_state_to_sim(isaac_joint_pos, isaac_joint_vel)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)
        
        # Sleep to match motion frame time (based on NPZ fps)
        loop_elapsed_time = time.time() - loop_start_time
        sleep_time = frame_dt - loop_elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"Warning: simulation loop is slower ({loop_elapsed_time:.4f}s) than frame dt ({frame_dt:.4f}s)")

        # if not initial_camera_set or camera_follow:
        #     cam_pos, cam_look = get_camera_pose(root_states[0])
        #     sim.set_camera_view(cam_pos, cam_look)
        #     initial_camera_set = True


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
