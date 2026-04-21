import math
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.v1 import V1_IMPLICIT_CFG
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg

KEY_BODY_NAMES = [
    "left_toe",
    "right_toe",
]
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4


@configclass
class V1AmpRewards:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.15,
        params={"command_name": "base_velocity", "std": math.sqrt(0.36)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    # Match G1 AMP: -0.05 over-penalizes action changes on this lighter robot and hurts corrective balance.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.35,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
        },
    )
    # Penalize foot not parallel to ground while in contact. Directly discourages
    # heel-only / toe-only contact (the "tiptoe" pattern) on V1's ankle_roll body,
    # since the full foot collision plate is rigidly attached to ankle_roll.
    feet_orientation_l2 = RewTerm(
        func=mdp.feet_orientation_l2,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll"),
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class V1AmpEnvCfg(LocomotionAmpEnvCfg):
    """Configuration for the v1 AMP environment."""

    rewards: V1AmpRewards = V1AmpRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = V1_IMPLICIT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        # New Kimodo-generated + GMR-retargeted dataset (17 categories x 3 seeds = 51 clips).
        # Each clip already carries cmd_lin_vel_x/y, cmd_ang_vel_z, category and priority metadata
        # inside the .pkl (baked in by dataset_retarget.py --manifest).
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "v1", "amp_dataset"
        )
        # Weights follow manifest "priority" per category (shared across the 3 seeds s00/s01/s02).
        _category_priority = {
            "stand_idle": 1.5,
            "stand_shift_weight": 1.0,
            "walk_forward_very_slow": 1.5,
            "walk_forward_slow": 1.7,
            "walk_forward_mid": 1.7,
            "walk_forward_normal": 1.7,
            "walk_backward_slow": 1.2,
            "side_step_left_slow": 1.2,
            "side_step_right_slow": 1.2,
            "turn_left_in_place": 1.2,
            "turn_right_in_place": 1.2,
            "walk_arc_left_slow": 1.5,
            "walk_arc_right_slow": 1.5,
            "walk_arc_left_normal": 1.2,
            "walk_arc_right_normal": 1.2,
            "walk_diag_left_slow": 1.0,
            "walk_diag_right_slow": 1.0,
        }
        self.motion_data.motion_dataset.motion_data_weights = {
            f"{category}_s{seed:02d}": weight
            for category, weight in _category_priority.items()
            for seed in range(3)
        }

        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------
        self.terminal_obs_groups = ("disc",)

        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }

        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }

        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.disc.history_length = AMP_NUM_STEPS

        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME

        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso"]
        self.events.reset_from_ref.params = {"animation": ANIMATION_TERM_NAME, "height_offset": 0.1}

        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        # Aligned with manifest.command_ranges of the Kimodo dataset: lin_vel_x ∈ [-0.5, 1.0],
        # lin_vel_y ∈ [-0.5, 0.5], ang_vel_z ∈ [-1.0, 1.0] (reference side_step and in-place-turn
        # clips now exist in amp_dataset/ to cover the widened y / yaw ranges).
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        # Disable heading setpoint: with heading_command + full heading range the policy fights
        # "go straight" vs "face random yaw", which often looks like one-step wobble / no sustained vx.
        self.commands.base_velocity.heading_command = False
        # Slightly gentler joint deltas than default 0.25 for this 12-DoF humanoid.
        self.actions.joint_pos.scale = 0.2

        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None

        # ------------------------------------------------------
        # terminations
        # ------------------------------------------------------
        self.terminations.base_contact = None


@configclass
class V1AmpEnvCfg_PLAY(V1AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_from_ref = None
