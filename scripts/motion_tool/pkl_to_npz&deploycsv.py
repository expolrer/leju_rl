"""This script replay a motion from a pkl file and output it to a npz file

.. code-block:: bash

    # Usage for PKL
    python pkl_to_npz.py --input_file motion.pkl \
    --output_file ./motions/motion.npz --output_fps 50 --robot robanS14
    
    # If pkl file contains fps, it will be used automatically (unless --input_fps is provided)
    python pkl_to_npz.py --input_file motion.pkl --input_fps 30 \
    --output_file ./motions/motion.npz --output_fps 50 --robot robanS14
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import os
import sys
import time
import pickle
from pathlib import Path
from dataclasses import MISSING
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from pkl file and output to npz file.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion pkl file.")
parser.add_argument("--input_fps", type=int, default=None, help="The fps of the input motion. If not provided, will try to use fps from pkl file (if available), otherwise defaults to 30.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")
parser.add_argument("--npz_output", type=str, default="/tmp/motion.npz", help="Path to save the generated npz.")
parser.add_argument("--csv_output", type=str, default=None, help="Path to save the generated csv file. If not provided, will auto-generate based on npz path.")
parser.add_argument("--robot", type=str, help="robot name")
# Note: AppLauncher adds --device argument automatically
# For single environment, using CPU is faster: --device cpu
# Default is usually GPU (cuda), but for this script CPU is recommended

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# For single environment, CPU is faster (no GPU transfer overhead)
# Set default device to CPU if not explicitly provided
if not hasattr(args_cli, 'device') or args_cli.device is None:
    args_cli.device = "cpu"
    print("[INFO] Using CPU device by default (recommended for single environment)")
elif args_cli.device != "cpu":
    print(f"[INFO] Using device: {args_cli.device} (consider using --device cpu for single environment)")

if args_cli.input_fps is None:
    args_cli.input_fps = 30
    args_cli._input_fps_provided = False
else:
    args_cli._input_fps_provided = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_conjugate, quat_mul, quat_rotate, quat_slerp

##
# Pre-defined configs
##
from leju_robot.assets.leju import RobanS14_CFG, KuavoS54_CFG, RobanS17_CFG, KuavoS53_CFG

ROBOT_CONFIGS = {
    "robanS14": RobanS14_CFG,
    "kuavoS54": KuavoS54_CFG,
    "robanS17": RobanS17_CFG,
    "kuavoS53": KuavoS53_CFG,
}

@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
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
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
        num_robot_joints: int | None = None,
        use_pkl_fps: bool = False,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self.num_robot_joints = num_robot_joints
        self.use_pkl_fps = use_pkl_fps
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from pkl file."""
        file_ext = os.path.splitext(self.motion_file)[1].lower()
        
        if file_ext != ".pkl":
            raise ValueError(f"Unsupported file format: {file_ext}. Only .pkl files are supported.")
        
        with open(self.motion_file, "rb") as f:
            pkl_data = pickle.load(f)
        
        if "fps" in pkl_data and self.use_pkl_fps:
            pkl_fps = int(pkl_data["fps"])
            if pkl_fps != self.input_fps:
                self.input_fps = pkl_fps
                self.input_dt = 1.0 / self.input_fps
                print(f"Using fps from pkl file: {self.input_fps} (overriding provided fps)")
        
        root_pos = pkl_data.get("root_pos")
        root_rot = pkl_data.get("root_rot")
        dof_pos = pkl_data.get("dof_pos")
        
        if root_pos is None or root_rot is None or dof_pos is None:
            raise ValueError(
                f"PKL file must contain 'root_pos', 'root_rot', and 'dof_pos' keys. "
                f"Found keys: {list(pkl_data.keys())}"
            )
        
        root_pos = torch.from_numpy(np.array(root_pos)).to(torch.float32)
        root_rot = torch.from_numpy(np.array(root_rot)).to(torch.float32)
        dof_pos = torch.from_numpy(np.array(dof_pos)).to(torch.float32)
        
        start_idx = self.frame_range[0] - 1
        end_idx = self.frame_range[1]
        root_pos = root_pos[start_idx:end_idx]
        root_rot = root_rot[start_idx:end_idx]
        dof_pos = dof_pos[start_idx:end_idx]
    
        first_frame_xy_offset = root_pos[0, :2].clone()
        root_pos[:, :2] = root_pos[:, :2] - first_frame_xy_offset
        print(f"Centered root position: subtracted first frame xy offset ({first_frame_xy_offset[0]:.3f}, {first_frame_xy_offset[1]:.3f})")
        
        pkl_num_joints = dof_pos.shape[1]
        if self.num_robot_joints is not None:
            if pkl_num_joints != self.num_robot_joints:
                raise ValueError(
                    f"PKL file has {pkl_num_joints} joints, but robot model has {self.num_robot_joints} joints. "
                    f"Please ensure PKL file has the same number of joints as the robot."
                )
        print(
            f"Using all PKL joints (sequential mapping): {pkl_num_joints} joints "
            f"(PKL and robot have the same number of joints)."
        )
        
        self.motion_base_poss_input = root_pos.to(self.device)
        self.motion_base_rots_input = root_rot.to(self.device)
        self.motion_dof_poss_input = dof_pos.to(self.device)
        
        self.input_frames = root_pos.shape[0]

        self.duration = (self.input_frames - 1) * self.input_dt
        print(
            f"Motion loaded ({self.motion_file}), duration: {self.duration:.2f} sec, "
            f"frames: {self.input_frames}, joints: {self.motion_dof_poss_input.shape[1]}"
        )

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]
    preserve_joint_order = scene.cfg.robot.preserve_joint_order.joint_names
    npz_to_isaac_indices = robot.find_joints(preserve_joint_order, preserve_order=True)[0]
    num_robot_joints = len(preserve_joint_order)
    
    file_ext = os.path.splitext(args_cli.input_file)[1].lower()
    use_pkl_fps = (file_ext == ".pkl" and not getattr(args_cli, "_input_fps_provided", False))
    
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
        num_robot_joints=num_robot_joints,
        use_pkl_fps=use_pkl_fps,
    )
    
    end_effector_configs = getattr(scene.cfg.robot, 'end_effector_configs', None)
    
    if end_effector_configs is None:
        raise ValueError("end_effector_configs not found in robot configuration. Please ensure the robot config defines end_effector_configs.")
    
    # 将配置中的 local_offset 转换为 torch tensor（如果是列表/元组）
    end_effector_configs = [
        (body_name, torch.tensor(local_offset, device=robot.device, dtype=torch.float32) if local_offset is not None else None)
        for body_name, local_offset in end_effector_configs
    ]
    print("Using end-effector configs from robot configuration")
    
    end_effector_names = [cfg[0] for cfg in end_effector_configs]
    end_effector_body_ids, _ = robot.find_bodies(name_keys=end_effector_names, preserve_order=True)

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "end_effector_pos_b": [],
        "end_effector_names": end_effector_names,
    }
    file_saved = False

    physics_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        loop_start_time = time.time() 
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, npz_to_isaac_indices] = motion_dof_pos
        joint_vel[:, npz_to_isaac_indices] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(physics_dt)
        if args_cli.headless is False:
            loop_elapsed_time = time.time() - loop_start_time
            sleep_time = physics_dt - loop_elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Warning: simulation loop is slower ({loop_elapsed_time:.4f}s) than physics dt ({physics_dt:.4f}s)")


        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :][npz_to_isaac_indices].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :][npz_to_isaac_indices].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())
            
            root_pos_w = robot.data.root_pos_w[0, :]
            root_quat_w = robot.data.root_quat_w[0, :]
            
            end_effector_pos_relative_w_list = []
            
            for i, (body_name, local_offset) in enumerate(end_effector_configs):
                body_id = end_effector_body_ids[i]
                body_pos_w = robot.data.body_pos_w[0, body_id, :]
                
                if local_offset is not None:
                    body_quat_w = robot.data.body_quat_w[0, body_id, :]
                    local_offset_expanded = local_offset.unsqueeze(0)
                    body_quat_w_expanded = body_quat_w.unsqueeze(0)
                    offset_world = quat_rotate(body_quat_w_expanded, local_offset_expanded).squeeze(0)
                    end_pos_w = body_pos_w + offset_world
                else:
                    end_pos_w = body_pos_w
                
                end_pos_relative_w = end_pos_w - root_pos_w
                end_effector_pos_relative_w_list.append(end_pos_relative_w)
            
            end_effector_pos_relative_w = torch.stack(end_effector_pos_relative_w_list, dim=0)
            
            root_quat_conj = quat_conjugate(root_quat_w)
            root_quat_conj_expanded = root_quat_conj.unsqueeze(0).repeat(4, 1)
            end_effector_pos_b = quat_apply(root_quat_conj_expanded, end_effector_pos_relative_w)
            
            log["end_effector_pos_b"].append(end_effector_pos_b.cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "end_effector_pos_b",
            ):
                log[k] = np.stack(log[k], axis=0)
            
            log["end_effector_names"] = np.array(log["end_effector_names"], dtype=object)

            npz_output_dir = os.path.dirname(args_cli.npz_output)
            if npz_output_dir:
                os.makedirs(npz_output_dir, exist_ok=True)
            np.savez(args_cli.npz_output, **log)
            print(f"[INFO]: Motion saved to: {args_cli.npz_output}")
            
            body_pos = log["body_pos_w"][:, 0, :]
            body_quat = log["body_quat_w"][:, 0, :]
            joint_pos = log["joint_pos"]
            joint_vel = log["joint_vel"]
            
            num_frames = body_pos.shape[0]
            num_joints = joint_pos.shape[1]
            
            combined_data = np.concatenate([
                body_pos,
                body_quat,
                joint_pos,
                joint_vel
            ], axis=1)
            
            header = [
                'body_pos_x', 'body_pos_y', 'body_pos_z',
                'body_quat_w', 'body_quat_x', 'body_quat_y', 'body_quat_z'
            ]
            header += [f'joint_pos_{i:02d}' for i in range(num_joints)]
            header += [f'joint_vel_{i:02d}' for i in range(num_joints)]
            
            if args_cli.csv_output is not None:
                csv_output_path = args_cli.csv_output
            else:
                csv_output_path = os.path.splitext(args_cli.npz_output)[0] + '_deploy.csv'
            
            csv_output_dir = os.path.dirname(csv_output_path)
            if csv_output_dir:
                os.makedirs(csv_output_dir, exist_ok=True)
            
            np.savetxt(csv_output_path, combined_data, delimiter=',',
                      header=','.join(header), comments='', fmt='%.6f')
            
            print(f"[INFO]: CSV file saved to: {csv_output_path}")
            print(f"[INFO]: CSV contains {num_frames} frames, {len(header)} columns")
            sys.exit(0)

def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
