# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os 
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from leju_robot.tasks.tracking.mdp.commands import MotionCommand


def export_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    """Export trained motion tracking policy to ONNX format for deployment."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    """ONNX policy exporter for motion tracking tasks."""
    
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        
        if "motion" in env.command_manager.active_terms:
            cmd: MotionCommand = env.command_manager.get_term("motion")
            self.time_step_total = cmd.motion.time_step_total
        else:
            self.time_step_total = None

    def forward(self, x):
        """Forward pass returning only actions."""
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        """Export model to ONNX format."""
        self.to("cpu")
        
        obs = torch.zeros(1, self.actor[0].in_features)
        
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    """Convert array/list to CSV format string."""
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    """Attach training environment configuration to ONNX model metadata."""
    onnx_path = os.path.join(path, filename)
    
    action_scale = env.action_manager.get_term("joint_pos")._scale
    if isinstance(action_scale, torch.Tensor):
        action_scale_list = action_scale[0].cpu().tolist()
    else:
        action_scale_list = [float(action_scale)]
    
    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": env.observation_manager.active_terms["policy"],
        "action_scale": action_scale_list,
    }

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
