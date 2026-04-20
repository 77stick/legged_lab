"""Configuration for v1 humanoid robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.v1_actuators import (
    V1ActuatorCfg_RI50,
    V1ActuatorCfg_RI60,
    V1ActuatorCfg_RI70,
    V1ActuatorCfg_RI100,
)


_V1_SPAWN = sim_utils.UsdFileCfg(
    usd_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/v1/usd/v1.usd",
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
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    ),
)

_V1_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.915),
    joint_pos={
        ".*_hip_yaw_joint": 0.0,
        ".*_hip_roll_joint": 0.0,
        "left_hip_pitch_joint": -0.145,
        "right_hip_pitch_joint": -0.145,
        ".*_knee_joint": 0.28,
        ".*_ankle_pitch_joint": -0.15,
        ".*_ankle_roll_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)

# fmt: off
_V1_JOINT_SDK_NAMES = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
# fmt: on


@configclass
class V1ArticulationCfg(ArticulationCfg):
    """Configuration for v1 articulation."""

    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.9


# ── Stribeck actuator version (DC motor T-N curve + Stribeck friction + delay) ──

V1_CFG = V1ArticulationCfg(
    spawn=_V1_SPAWN,
    init_state=_V1_INIT_STATE,
    actuators={
        "hip_yaw_roll": V1ActuatorCfg_RI70(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint"],
            effort_limit=76.5,
        ),
        "hip_pitch": V1ActuatorCfg_RI100(
            joint_names_expr=[".*_hip_pitch_joint"],
            effort_limit=308.25,
        ),
        "knee": V1ActuatorCfg_RI100(
            joint_names_expr=[".*_knee_joint"],
            effort_limit=308.25,
        ),
        "ankle_pitch": V1ActuatorCfg_RI60(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit=64.35,
        ),
        "ankle_roll": V1ActuatorCfg_RI50(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit=28.73,
        ),
    },
    joint_sdk_names=_V1_JOINT_SDK_NAMES,
)

# ── Implicit actuator version (simple PD, same as G1 style) ──

V1_IMPLICIT_CFG = V1ArticulationCfg(
    spawn=_V1_SPAWN,
    init_state=_V1_INIT_STATE,
    actuators={
        "hip_yaw_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint"],
            effort_limit=76.5,
            velocity_limit=8.378,
            stiffness=300.0,
            damping=9.0,
            armature=1.9127,
        ),
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint"],
            effort_limit=308.25,
            velocity_limit=6.06,
            stiffness=450.0,
            damping=12.0,
            armature=8.1263,
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit=308.25,
            velocity_limit=6.06,
            stiffness=450.0,
            damping=12.0,
            armature=8.1263,
        ),
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit=64.35,
            velocity_limit=8.586,
            stiffness=50.0,
            damping=7.0,
            armature=0.692086845,
        ),
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit=28.73,
            velocity_limit=10.252,
            stiffness=30.0,
            damping=5.0,
            armature=0.173886246,
        ),
    },
    joint_sdk_names=_V1_JOINT_SDK_NAMES,
)
