"""Functions to specify the symmetry in the observation and action space for V1 12dof (lower body only).

The observation layout below must stay in sync with
``legged_lab.tasks.locomotion.amp.amp_env_cfg.ObservationsCfg.PolicyCfg``
and ``legged_lab.tasks.locomotion.amp.config.v1.v1_amp_env_cfg.KEY_BODY_NAMES``.
Per-step policy observation = [ang_vel(3), root_local_rot_tan_norm(6),
velocity_commands(3), joint_pos(12), joint_vel(12), last_actions(12),
key_body_pos_b(4*3)] flattened over ``history_length=5``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


# ------------------------------------------------------------
# Observation / action layout constants
# ------------------------------------------------------------
_HISTORY_LEN = 5
_ANG_VEL_DIM = 3
_ROT_TAN_NORM_DIM = 6
_VEL_CMD_DIM = 3
_JOINT_DIM = 12  # V1: 6 joints/side, lower body only
_KEY_BODY_NUM = 4  # left/right ankle_roll + left/right toe

# Any joint matching this regex has its signed angle flipped under a left-right
# reflection, because its axis (roll/yaw) is reflected to the opposite direction
# across the sagittal plane. Pitch / knee joints keep their sign (axis parallel
# to +y on both sides by URDF convention).
_FLIP_SIGN_PATTERN = re.compile(r".*_(hip_roll|hip_yaw|ankle_roll)_joint$")


# ------------------------------------------------------------
# Lazy per-env cache of joint index mappings.
# ------------------------------------------------------------
_CACHED: dict = {
    "env_id": None,
    "left_indices": None,   # (N_pair,) long tensor
    "right_indices": None,  # (N_pair,) long tensor
    "flip_sign_mask": None, # (_JOINT_DIM,) float tensor of +/-1
}


def _ensure_indices(env: "ManagerBasedRLEnv", device: torch.device) -> None:
    """Build left/right joint index maps from the robot's actual joint_names order.

    Using runtime introspection avoids hard-coding IsaacLab's articulation
    ordering, which differs from the SDK / URDF ordering.
    """
    global _CACHED
    env_id = id(env)
    if _CACHED["env_id"] == env_id and _CACHED["left_indices"] is not None:
        return

    robot = env.scene["robot"]
    joint_names = list(robot.joint_names)
    if len(joint_names) != _JOINT_DIM:
        raise ValueError(
            f"v1 symmetry expects {_JOINT_DIM} joints, got {len(joint_names)}: {joint_names}"
        )

    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    left_indices: list[int] = []
    right_indices: list[int] = []
    for name in joint_names:
        if name.startswith("left_"):
            mirror = "right_" + name[len("left_") :]
            if mirror not in name_to_idx:
                raise ValueError(f"v1 symmetry: no mirror joint found for {name}")
            left_indices.append(name_to_idx[name])
            right_indices.append(name_to_idx[mirror])

    if len(left_indices) * 2 != _JOINT_DIM:
        raise ValueError(
            f"v1 symmetry: joints are not fully paired left/right: {joint_names}"
        )

    flip_sign = torch.ones(_JOINT_DIM, device=device)
    for i, n in enumerate(joint_names):
        if _FLIP_SIGN_PATTERN.match(n):
            flip_sign[i] = -1.0

    _CACHED.update(
        env_id=env_id,
        left_indices=torch.tensor(left_indices, dtype=torch.long, device=device),
        right_indices=torch.tensor(right_indices, dtype=torch.long, device=device),
        flip_sign_mask=flip_sign,
    )


# ------------------------------------------------------------
# Per-tensor helpers
# ------------------------------------------------------------
def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left<->right joint columns and flip sign on roll/yaw joints."""
    left = _CACHED["left_indices"]
    right = _CACHED["right_indices"]
    flip = _CACHED["flip_sign_mask"]

    out = joint_data.clone()
    out[..., left] = joint_data[..., right]
    out[..., right] = joint_data[..., left]
    out = out * flip  # broadcast along the last dim
    return out


def _switch_key_body_pos_left_right(key_body_pos: torch.Tensor) -> torch.Tensor:
    """Swap paired key bodies and negate the y-coordinate of every key body.

    Assumes KEY_BODY_NAMES is laid out as [left_0, right_0, left_1, right_1, ...],
    which is the V1 convention (left_ankle_roll, right_ankle_roll, left_toe, right_toe).
    """
    out = key_body_pos.clone()
    num_pairs = _KEY_BODY_NUM // 2
    for p in range(num_pairs):
        left_idx = p * 2
        right_idx = p * 2 + 1
        out[..., left_idx * 3 : left_idx * 3 + 3] = key_body_pos[
            ..., right_idx * 3 : right_idx * 3 + 3
        ]
        out[..., right_idx * 3 : right_idx * 3 + 3] = key_body_pos[
            ..., left_idx * 3 : left_idx * 3 + 3
        ]
        out[..., left_idx * 3 + 1] *= -1.0
        out[..., right_idx * 3 + 1] *= -1.0
    return out


# ------------------------------------------------------------
# Policy observation transform
# ------------------------------------------------------------
def _transform_policy_obs_left_right(
    env: "ManagerBasedRLEnv", obs: torch.Tensor
) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device
    _ensure_indices(env, device)

    end_idx = 0

    # base_ang_vel: (wx, wy, wz) -> (-wx, wy, -wz)
    ang_vel_sign = torch.tensor([-1.0, 1.0, -1.0], device=device)
    for _ in range(_HISTORY_LEN):
        start_idx, end_idx = end_idx, end_idx + _ANG_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * ang_vel_sign

    # root_local_rot_tan_norm: two 3D vectors, flip y-component on each
    rot_sign = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, 1.0], device=device)
    for _ in range(_HISTORY_LEN):
        start_idx, end_idx = end_idx, end_idx + _ROT_TAN_NORM_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * rot_sign

    # velocity_commands: (vx, vy, wz) -> (vx, -vy, -wz)
    vel_cmd_sign = torch.tensor([1.0, -1.0, -1.0], device=device)
    for _ in range(_HISTORY_LEN):
        start_idx, end_idx = end_idx, end_idx + _VEL_CMD_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * vel_cmd_sign

    # joint_pos / joint_vel / last_actions: all indexed the same way.
    for _ in range(3):
        for _ in range(_HISTORY_LEN):
            start_idx, end_idx = end_idx, end_idx + _JOINT_DIM
            obs[:, start_idx:end_idx] = _switch_joints_left_right(obs[:, start_idx:end_idx])

    # key_body_pos_b
    for _ in range(_HISTORY_LEN):
        start_idx, end_idx = end_idx, end_idx + _KEY_BODY_NUM * 3
        obs[:, start_idx:end_idx] = _switch_key_body_pos_left_right(obs[:, start_idx:end_idx])

    expected = _HISTORY_LEN * (
        _ANG_VEL_DIM + _ROT_TAN_NORM_DIM + _VEL_CMD_DIM + 3 * _JOINT_DIM + _KEY_BODY_NUM * 3
    )
    if end_idx != obs.shape[-1] or end_idx != expected:
        raise RuntimeError(
            f"v1 symmetry: policy obs layout mismatch (consumed={end_idx}, "
            f"expected={expected}, got_dim={obs.shape[-1]}). If you changed "
            f"PolicyCfg or KEY_BODY_NAMES, update v1.py accordingly."
        )
    return obs


# ------------------------------------------------------------
# Action transform
# ------------------------------------------------------------
def _transform_actions_left_right(
    env: "ManagerBasedRLEnv", actions: torch.Tensor
) -> torch.Tensor:
    _ensure_indices(env, actions.device)
    return _switch_joints_left_right(actions.clone())


# ------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------
@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv",
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augment a batch by appending its left-right reflected copy.

    Mirrors the policy observation group and the action vector. Other obs
    groups (critic / disc / disc_demo) are copied untouched, following the
    same convention as legged_lab's G1 symmetry implementation.
    """
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(
            env.unwrapped, obs["policy"][:]
        )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(
            env.unwrapped, actions
        )
    else:
        actions_aug = None

    return obs_aug, actions_aug
