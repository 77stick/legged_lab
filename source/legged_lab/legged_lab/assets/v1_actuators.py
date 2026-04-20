"""V1 robot actuator models with DC motor T-N curve, Stribeck friction, and command delay."""

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class V1StribeckActuator(DelayedPDActuator):
    """Actuator with DC motor T-N curve, Stribeck friction, and optional command delay.

    Inherits from DelayedPDActuator to get delay buffer support.
    Manually implements the DCMotor linear four-quadrant torque-speed curve
    via ``_clip_effort``, and applies a Stribeck friction torque in ``compute``:

        T_friction = Fs * tanh(v / Vs) + Fd * sign(v) + Fv * v

    where Fs is static friction, Fd is dynamic (Coulomb) friction,
    Fv is viscous friction, and Vs is the Stribeck transition velocity.
    """

    cfg: V1StribeckActuatorCfg

    def __init__(self, cfg: V1StribeckActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # DC motor T-N curve parameters (ported from DCMotor.__init__)
        self._saturation_effort = cfg.saturation_effort
        if self._saturation_effort is None:
            raise ValueError("saturation_effort must be provided for V1StribeckActuator.")
        if cfg.velocity_limit is None:
            raise ValueError("velocity_limit must be provided for V1StribeckActuator.")
        self._vel_at_effort_lim = self.velocity_limit * (1 + self.effort_limit / self._saturation_effort)
        self._joint_vel = torch.zeros_like(self.computed_effort)

        # Stribeck friction parameters
        self._friction_static = self._parse_joint_parameter(cfg.static_friction, 0.0)
        self._friction_dynamic = self._parse_joint_parameter(cfg.dynamic_friction, 0.0)
        self._friction_viscous = self._parse_joint_parameter(cfg.viscous_friction, 0.0)
        self._stribeck_velocity = self._parse_joint_parameter(cfg.stribeck_velocity, 0.01)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save joint vel for _clip_effort (called inside super().compute via IdealPDActuator)
        self._joint_vel[:] = joint_vel
        # DelayedPDActuator.compute applies delay, then IdealPDActuator.compute does PD + _clip_effort
        control_action = super().compute(control_action, joint_pos, joint_vel)

        # apply Stribeck friction after PD + T-N clipping
        friction_torque = (
            self._friction_static * torch.tanh(joint_vel / self._stribeck_velocity)
            + self._friction_dynamic * torch.sign(joint_vel)
            + self._friction_viscous * joint_vel
        )
        self.applied_effort -= friction_torque
        control_action.joint_efforts = self.applied_effort
        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """Linear four-quadrant DC motor torque-speed curve (ported from DCMotor)."""
        self._joint_vel[:] = torch.clip(self._joint_vel, min=-self._vel_at_effort_lim, max=self._vel_at_effort_lim)
        torque_speed_top = self._saturation_effort * (1.0 - self._joint_vel / self.velocity_limit)
        torque_speed_bottom = self._saturation_effort * (-1.0 - self._joint_vel / self.velocity_limit)
        max_effort = torch.clip(torque_speed_top, max=self.effort_limit)
        min_effort = torch.clip(torque_speed_bottom, min=-self.effort_limit)
        return torch.clip(effort, min=min_effort, max=max_effort)


@configclass
class V1StribeckActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for V1 Stribeck DC motor actuator with optional delay."""

    class_type: type = V1StribeckActuator

    saturation_effort: float = MISSING
    """Peak motor stall torque in N*m (from DC motor T-N curve)."""

    static_friction: float = 0.0
    """Static (stiction) friction torque in N*m."""

    dynamic_friction: float = 0.0
    """Dynamic (Coulomb) friction torque in N*m."""

    viscous_friction: float = 0.0
    """Viscous friction coefficient in N*m*s/rad."""

    stribeck_velocity: float = 0.01
    """Stribeck transition velocity in rad/s."""


# ---------------------------------------------------------------------------
# Per-motor presets (parameters from mjlab v1 robot config)
# Delay defaults to 0 (disabled). Override min_delay/max_delay to enable.
# ---------------------------------------------------------------------------


@configclass
class V1ActuatorCfg_RI50(V1StribeckActuatorCfg):
    """RI50 motor -- ankle_roll joints."""

    saturation_effort = 44.2
    velocity_limit = 10.252
    armature = 0.173886246
    stiffness = 30.0
    damping = 5.0
    static_friction = 0.65
    dynamic_friction = 0.5
    viscous_friction = 0.03
    stribeck_velocity = 0.16


@configclass
class V1ActuatorCfg_RI60(V1StribeckActuatorCfg):
    """RI60 motor -- ankle_pitch joints."""

    saturation_effort = 99.0
    velocity_limit = 8.586
    armature = 0.692086845
    stiffness = 50.0
    damping = 7.0
    static_friction = 0.7
    dynamic_friction = 0.5
    viscous_friction = 0.04
    stribeck_velocity = 0.18


@configclass
class V1ActuatorCfg_RI70(V1StribeckActuatorCfg):
    """RI70 motor -- hip_yaw and hip_roll joints."""

    saturation_effort = 102.0
    velocity_limit = 8.378
    armature = 1.9127
    stiffness = 300.0
    damping = 9.0
    static_friction = 8.0183
    dynamic_friction = 6.9467
    viscous_friction = 11.0158
    stribeck_velocity = 0.02


@configclass
class V1ActuatorCfg_RI100(V1StribeckActuatorCfg):
    """RI100 motor -- hip_pitch and knee joints."""

    saturation_effort = 411.0
    velocity_limit = 6.06
    armature = 8.1263
    stiffness = 450.0
    damping = 12.0
    static_friction = 20.2984
    dynamic_friction = 17.6954
    viscous_friction = 18.4632
    stribeck_velocity = 0.02
