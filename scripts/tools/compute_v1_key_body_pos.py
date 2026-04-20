"""Compute key_body_pos for v1 motion data using FK in Isaac Sim.

For each frame in the pkl, sets root pose and joint positions on the articulation,
steps physics to resolve FK, and records the world positions of the key bodies.
Overwrites the pkl with the added key_body_pos and loop_mode fields.
"""

import argparse
import os
import pickle

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compute key_body_pos for v1 motion pkl files.")
parser.add_argument("--pkl", type=str, required=True, help="Path to the pkl file to process.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)

import joblib
import torch
import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from legged_lab.assets.v1 import V1_CFG

KEY_BODY_NAMES = [
    "left_toe",
    "right_toe",
]


def main():
    pkl_path = os.path.abspath(args.pkl)
    print(f"Loading motion data from: {pkl_path}")

    data = joblib.load(pkl_path)

    root_pos = data["root_pos"]  # (T, 3)
    root_rot = data["root_rot"]  # (T, 4) wxyz
    dof_pos = data["dof_pos"]  # (T, num_dofs)
    num_frames = root_pos.shape[0]
    print(f"Motion has {num_frames} frames, {dof_pos.shape[1]} DOFs")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    prim_utils.create_prim("/World/Origin", "Xform", translation=[0.0, 0.0, 0.0])
    robot = Articulation(V1_CFG.replace(prim_path="/World/Origin/Robot"))

    sim.reset()

    body_names = robot.body_names
    key_body_indices = []
    for kb_name in KEY_BODY_NAMES:
        if kb_name in body_names:
            key_body_indices.append(body_names.index(kb_name))
        else:
            raise ValueError(f"Key body '{kb_name}' not found in robot body names: {body_names}")

    print(f"Key body indices: {key_body_indices} for names: {KEY_BODY_NAMES}")

    key_body_pos_all = np.zeros((num_frames, len(KEY_BODY_NAMES), 3), dtype=np.float32)

    root_pos_t = torch.from_numpy(root_pos).float().to(robot.device)
    root_rot_t = torch.from_numpy(root_rot).float().to(robot.device)
    dof_pos_t = torch.from_numpy(dof_pos).float().to(robot.device)

    joint_names = robot.joint_names
    num_joints = len(joint_names)
    print(f"Robot joint names ({num_joints}): {joint_names}")

    for i in range(num_frames):
        root_state = torch.zeros(1, 13, device=robot.device)
        root_state[0, 0:3] = root_pos_t[i]
        root_state[0, 3:7] = root_rot_t[i]
        root_state[0, 7:13] = 0.0

        robot.write_root_state_to_sim(root_state)

        jp = dof_pos_t[i].unsqueeze(0)
        jv = torch.zeros_like(jp)
        robot.write_joint_state_to_sim(jp, jv)

        sim.step(render=False)
        robot.update(dt=sim.get_physics_dt())

        body_pos_w = robot.data.body_pos_w[0]  # (num_bodies, 3)
        for j, idx in enumerate(key_body_indices):
            key_body_pos_all[i, j, :] = body_pos_w[idx].cpu().numpy()

        if (i + 1) % 50 == 0 or i == num_frames - 1:
            print(f"  Processed frame {i + 1}/{num_frames}")

    data["key_body_pos"] = key_body_pos_all
    if "loop_mode" not in data:
        data["loop_mode"] = 0

    joblib.dump(data, pkl_path)
    print(f"Saved updated pkl with key_body_pos shape={key_body_pos_all.shape} to {pkl_path}")


if __name__ == "__main__":
    main()
    app_launcher.app.close()
