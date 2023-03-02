# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Physics-based models for actuators.

Currently the following models are supported:
* Ideal actuator
* DC motor
* Variable gear ratio DC motor
"""


import torch
from typing import Optional, Sequence, Union

from .actuator_cfg import DCMotorCfg, IdealActuatorCfg, VariableGearRatioDCMotorCfg


class IdealActuator:
    r"""Ideal torque-controlled actuator model with a simple saturation model.

    It employs the following model for computing torques for the actuated joint :math:`j`:

    .. math::

        \tau_{j, computed} = k_p * (q - q_{des}) + k_d * (\dot{q} - \dot{q}_{des}) + \tau_{ff}

    where, :math:`k_p` and :math:`k_d` are joint stiffness and damping gains, :math:`q` and :math:`\dot{q}`
    are the current joint positions and velocities, :math:`q_{des}`, :math:`\dot{q}_{des}` and :math:`\tau_{ff}`
    are the desired joint positions, velocities and torques commands.

    The clipping model is based on the maximum torque applied by the motor. It is implemented as:

    .. math::

        \tau_{j, max} & = \gamma \times \tau_{motor, max} \\
        \tau_{j, applied} & = clip(\tau_{computed}, -\tau_{j, max}, \tau_{j, max})

    where the clipping function is defined as :math:`clip(x, x_{min}, x_{max}) = min(max(x, x_{min}), x_{max})`.
    The parameters :math:`\gamma` is the gear ratio of the gear box connecting the motor and the actuated joint ends,
    and :math:`\tau_{motor, max}` is the maximum motor effort possible. These parameters are read from
    the configuration instance passed to the class.
    """

    cfg: IdealActuatorCfg
    """The configuration for the actuator model."""
    num_actuators: int
    """The number of actuators using the model."""
    num_envs: int
    """The number of instances of the articulation."""
    device: str
    """The computation device."""

    def __init__(self, cfg: IdealActuatorCfg, num_actuators: int, num_envs: int, device: str):
        """Initializes the ideal actuator model.

        Args:
            cfg (IdealActuatorCfg): The configuration for the actuator model.
            num_actuators (int): The number of actuators using the model.
            num_envs (int): The number of instances of the articulation.
            device (str): The computation device.
        """
        # store inputs to class
        self.cfg = cfg
        self.num_actuators = num_actuators
        self.num_envs = num_envs
        self.device = device

        # create buffers for allocation
        # -- joint commands
        self._des_dof_pos = torch.zeros(self.num_envs, self.num_actuators, device=self.device)
        self._des_dof_vel = torch.zeros_like(self._des_dof_pos)
        # -- PD gains
        self._p_gains = torch.zeros_like(self._des_dof_pos)
        self._d_gains = torch.zeros_like(self._des_dof_pos)
        # -- feed-forward torque
        self._torque_ff = torch.zeros_like(self._des_dof_pos)

    """
    Properties
    """

    @property
    def gear_ratio(self) -> float:
        """Gear-box conversion factor from motor axis to joint axis."""
        return self.cfg.gear_ratio

    """
    Operations- State.
    """

    def set_command(
        self,
        dof_pos: Optional[Union[torch.Tensor, float]] = None,
        dof_vel: Optional[Union[torch.Tensor, float]] = None,
        p_gains: Optional[Union[torch.Tensor, float]] = None,
        d_gains: Optional[Union[torch.Tensor, float]] = None,
        torque_ff: Optional[Union[torch.Tensor, float]] = None,
    ):
        """Sets the desired joint positions, velocities, gains and feed-forward torques.

        If the values are :obj:`None`, the previous values are retained.

        Args:
            dof_pos (Optional[Union[torch.Tensor, float]], optional): The desired joint positions. Defaults to None.
            dof_vel (Optional[Union[torch.Tensor, float]], optional): The desired joint velocities. Defaults to None.
            p_gains (Optional[Union[torch.Tensor, float]], optional): The stiffness gains of the drive. Defaults to None.
            d_gains (Optional[Union[torch.Tensor, float]], optional): The damping gains of the drive. Defaults to None.
            torque_ff (Optional[Union[torch.Tensor, float]], optional): The desired joint torque. Defaults to None.
        """
        if dof_pos is not None:
            self._des_dof_pos[:] = dof_pos
        if dof_vel is not None:
            self._des_dof_vel[:] = dof_vel
        if p_gains is not None:
            self._p_gains[:] = p_gains
        if d_gains is not None:
            self._d_gains[:] = d_gains
        if torque_ff is not None:
            self._torque_ff[:] = torque_ff

    """
    Operations- Main.
    """

    def reset(self, env_ids: Sequence[int]):
        """Resets the internal buffers or state of the actuator model.

        Args:
            env_ids (Sequence[int]): The ids to reset.
        """
        # reset desired joint positions and velocities
        self._des_dof_pos[env_ids] = 0.0
        self._des_dof_vel[env_ids] = 0.0
        # reset feed-forward torque
        self._torque_ff[env_ids] = 0.0

    def compute_torque(self, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> torch.Tensor:
        """Computes the desired joint torques using the input commands and current joint states.

        Args:
            dof_pos (torch.Tensor): The joint positions of the actuators.
            dof_vel (torch.Tensor): The joint velocities of the actuators.

        Returns:
            torch.Tensor: The desired joint torques to achieve the input commands.
        """
        # compute errors
        dof_pos_error = self._des_dof_pos - dof_pos
        dof_vel_error = self._des_dof_vel - dof_vel
        # compute torques
        desired_torques = self._p_gains * dof_pos_error + self._d_gains * dof_vel_error + self._torque_ff
        # return torques
        return desired_torques

    def clip_torques(self, desired_torques: torch.Tensor, **kwargs) -> torch.Tensor:
        """Clip the desired torques based on the motor limits.

        Args:
            desired_torques (torch.Tensor): The desired torques to clip.

        Returns:
            torch.Tensor: The clipped torques.
        """
        # evaluate parameters from motor axel to dof axel
        torque_limit = self.cfg.motor_torque_limit * self.gear_ratio
        # saturate torques
        return torch.clip(desired_torques, -torque_limit, torque_limit)


class DCMotor(IdealActuator):
    r"""
    Direct control (DC) motor actuator model with velocity-based saturation model.

    It uses the same model as the :class:`IdealActuator` for computing the torques from input commands.
    However, it implements a saturation model defined by DC motor characteristics.

    A DC motor is a type of electric motor that is powered by direct current electricity. In most cases,
    the motor is connected to a constant source of voltage supply, and the current is controlled by a rheostat.
    Depending on various design factors such as windings and materials, the motor can draw a limited maximum power
    from the electronic source, which limits the produced motor torque and speed.

    A DC motor characteristics are defined by the following parameters:

    * Continuous-rated speed (:math:`\dot{q}_{motor, max}`) : The maximum-rated speed of the motor.
    * Continuous-stall torque (:math:`\tau_{motor, max}`): The maximum-rated torque produced at 0 speed.
    * Peak torque (:math:`\tau_{motor, peak}`): The maximum torque that can be outputted for a short period.

    Based on these parameters, the instantaneous minimum and maximum torques are defined as follows:

    .. math::

        \tau_{j, max}(\dot{q}) & = clip \left (\tau_{j, peak} \times \left(1 -
            \frac{\dot{q}}{\dot{q}_{j, max}}\right), 0.0, \tau_{j, max} \right) \\
        \tau_{j, min}(\dot{q}) & = clip \left (\tau_{j, peak} \times \left( -1 -
            \frac{\dot{q}}{\dot{q}_{j, max}}\right), - \tau_{j, max}, 0.0 \right)

    where :math:`\gamma` is the gear ratio of the gear box connecting the motor and the actuated joint ends,
    :math:`\dot{q}_{j, max} = \gamma^{-1} \times  \dot{q}_{motor, max}`, :math:`\tau_{j, max} =
    \gamma \times \tau_{motor, max}` and :math:`\tau_{j, peak} = \gamma \times \tau_{motor, peak}`
    are the maximum joint velocity, maximum joint torque and peak torque, respectively. These parameters
    are read from the configuration instance passed to the class.

    Using these values, the computed torques are clipped to the minimum and maximum values based on the
    instantaneous joint velocity:

    .. math::

        \tau_{j, applied} = clip(\tau_{computed}, \tau_{j, min}(\dot{q}), \tau_{j, max}(\dot{q}))

    """

    cfg: DCMotorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: DCMotorCfg, num_actuators: int, num_envs: int, device: str):
        """Initializes the DC motor actuator model.

        Args:
            cfg (DCMotorCfg): The configuration for the actuator model.
            num_actuators (int): The number of actuators using the model.
            num_envs (int): The number of instances of the articulation.
            device (str): The computation device.
        """
        super().__init__(cfg, num_actuators, num_envs, device)

    def clip_torques(self, desired_torques: torch.Tensor, dof_vel: torch.Tensor, **kwargs) -> torch.Tensor:
        """Clip the desired torques based on the motor limits.

        Args:
            desired_torques (torch.Tensor): The desired torques to clip.
            dof_vel (torch.Tensor): The current joint velocities.

        Returns:
            torch.Tensor: The clipped torques.
        """
        # evaluate parameters from motor axel to dof axel
        peak_torque = self.cfg.peak_motor_torque * self.gear_ratio
        torque_limit = self.cfg.motor_torque_limit * self.gear_ratio
        velocity_limit = self.cfg.motor_velocity_limit / self.gear_ratio
        # compute torque limits
        # -- max limit
        max_torques = peak_torque * (1.0 - dof_vel / velocity_limit)
        max_torques = torch.clip(max_torques, min=0.0, max=torque_limit)
        # -- min limit
        min_torques = peak_torque * (-1.0 - dof_vel / velocity_limit)
        min_torques = torch.clip(min_torques, min=-torque_limit, max=0.0)
        # saturate torques
        return torch.clip(desired_torques, min_torques, max_torques)


class VariableGearRatioDCMotor(DCMotor):
    r"""Torque-controlled actuator with variable gear-ratio based saturation model.

    Instead of using a fixed gear box, some motors are equipped with variators that allow the gear-ratio
    to be changed continuously (instead of steps). This model implements a DC motor with a variable
    gear ratio function that computes the gear-ratio as a function of the joint position, i.e.:

    .. math::

        \gamma = \gamma(q)

    where :math:`\gamma(\cdot)` is read from the configuration instance passed to the class. The gear-ratio function is evaluated at
    every time step and the motor parameters are computed accordingly.

    """

    cfg: VariableGearRatioDCMotorCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: VariableGearRatioDCMotorCfg, num_actuators: int, num_envs: int, device: str):
        """Initializes the variable gear ratio DC actuator model.

        Args:
            cfg (VariableGearRatioDCMotorCfg): The configuration for the actuator model.
            num_actuators (int): The number of actuators using the model.
            num_envs (int): The number of instances of the articulation.
            device (str): The computation device.
        """
        super().__init__(cfg, num_actuators, num_envs, device)
        # parse the configuration
        if isinstance(self.cfg.gear_ratio, str):
            self._gear_ratio_fn = eval(self.cfg.gear_ratio)
        else:
            self._gear_ratio_fn = self.cfg.gear_ratio
        # check configuration
        if not callable(self._gear_ratio_fn):
            raise ValueError(f"Expected a callable gear ratio function. Received: {self.cfg.gear_ratio}.")
        # create buffers
        self._gear_ratio = torch.ones(self.num_envs, self.num_actuators, device=self.device)

    @property
    def gear_ratio(self) -> torch.Tensor:
        """Gear-box conversion factor from motor axis to joint axis."""
        return self._gear_ratio

    def clip_torques(
        self, desired_torques: torch.Tensor, dof_pos: torch.Tensor, dof_vel: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Clip the desired torques based on the motor limits.

        Args:
            desired_torques (torch.Tensor): The desired torques to clip.
            dof_pos (torch.Tensor): The current joint positions.
            dof_vel (torch.Tensor): The current joint velocities.

        Returns:
            torch.Tensor: The clipped torques.
        """
        # compute gear ratio
        self._gear_ratio = self._gear_ratio_fn(dof_pos)
        # clip torques using model from parent
        super().clip_torques(desired_torques, dof_vel=dof_vel)
