omni.isaac.orbit.actuators.model
================================

There are two main categories of actuator models that are supported:

- **Implicit**: Motor model with ideal PD from the physics engine.
- **Explicit**: Motor models based on physical drive models.

  - **Physics-based**: Derives the motor models based on first-principles.
  - **Neural Network-based**: Learned motor models from actuator data.

Currently, all the explicit motor models convert input joint commands into joint efforts to
apply into the simulation. This process comprise of three main steps:

1. :func:`set_command`: Set the desired command to the model. These include joint positions, velocities, feedforward torque, stiffness and damping gain.
2. :func:`compute_torque`: Compute the joint efforts using the actuator model.
3. :func:`clip_torques`: Clip the desired torques epxlicitly using an actuator saturation model.

It is up to the model how the input values from step (1) are processed and dealt with in step (2).
The steps (2) and (3) are segregrated explicitly, since many times in learning, we need both the
computed (desired) or clipped (applied) joint efforts. For instance, to penalize the difference
between the computed and clipped joint efforts, so that the learned policy does not keep outputting
arbitrarily large commands.

All explicit models inherit from the base actuator model, :class:`IdealActuator`, which implements a
PD controller with feed-forward effort, and simple clipping based on the configured maximum effort.

The following is a simple example of using the actuator model:

.. code-block:: python

   import torch
   from omni.isaac.orbit.actuators.model import IdealActuator, IdealActuatorCfg

   # joint information from the articulation
   num_actuators, num_envs = 5, 32
   device ="cpu"
   # create a configuration instance
   cfg = IdealActuatorCfg(motor_torque_limit=20, gear_ratio=1)
   # create the actuator model instance
   model = IdealActuator(cfg, num_actuators, num_envs, device)
   # clear history in the actuator model (if any)
   model.reset()

   # creat random commands
   dof_pos_des, dof_vel_des = torch.rand(32, 5), torch.rand(32, 5)
   # create random state
   dof_pos, dof_vel = torch.rand(32, 5), torch.rand(32, 5)

   # set desired joint state
   model.set_command(dof_pos_des, dof_vel_des)
   # compute the torques
   computed_torques = model.compute_torques(dof_pos, dof_vel)
   applied_torques = model.clip_torques(computed_torques)


Implicit Actuator
-----------------

.. autoclass:: omni.isaac.orbit.actuators.model.ImplicitActuatorCfg
   :members:
   :show-inheritance:

Ideal Actuator
---------------

.. autoclass:: omni.isaac.orbit.actuators.model.IdealActuator
   :members:
   :show-inheritance:

.. autoclass:: omni.isaac.orbit.actuators.model.IdealActuatorCfg
   :members:
   :show-inheritance:

Direct Control (DC) Actuator
----------------------------

Fixed Gear Ratio
~~~~~~~~~~~~~~~~

.. autoclass:: omni.isaac.orbit.actuators.model.DCMotor
   :members:
   :show-inheritance:

.. autoclass:: omni.isaac.orbit.actuators.model.DCMotorCfg
   :members:
   :show-inheritance:

Variable Gear Ratio
~~~~~~~~~~~~~~~~~~~

.. autoclass:: omni.isaac.orbit.actuators.model.VariableGearRatioDCMotor
   :members:
   :show-inheritance:

.. autoclass:: omni.isaac.orbit.actuators.model.VariableGearRatioDCMotorCfg
   :members:
   :show-inheritance:

Actuator Networks
-----------------

Multi-layer Perceptron (MLP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: omni.isaac.orbit.actuators.model.ActuatorNetMLP
   :members:
   :show-inheritance:

.. autoclass:: omni.isaac.orbit.actuators.model.ActuatorNetMLPCfg
   :members:
   :show-inheritance:

Long Short-term Memory (LSTM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: omni.isaac.orbit.actuators.model.ActuatorNetLSTM
   :members:
   :show-inheritance:

.. autoclass:: omni.isaac.orbit.actuators.model.ActuatorNetLSTMCfg
   :members:
   :show-inheritance:
