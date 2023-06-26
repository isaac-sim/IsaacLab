omni.isaac.orbit.actuators.group
================================

Actuator groups apply the same actuator model over a collection of actuated joints.
It deals with both explicit and implicit models, and processes the input joint
configuration and actions accordingly. The joint names, that are a part of a given
actuator group, are configured using regex expressions. These expressions are matched
with the joint names in the robot's kinematic tree. For instance, in the Franka Panda
Emika arm, the first four joints and last three joints can be captured using the regex
expressions ``"panda_joint[1-4]"`` and ``"panda_joint[5-7]"``,

For a given actuator group, it is possible to provide multiple joint-level command types
(e.g. position, velocity, torque, etc.). The command types are processed as a list of strings.
Each string has two sub-strings joined by underscore:

- **type of command mode:** "p" (position), "v" (velocity), "t" (torque)
- **type of command resolving:** "abs" (absolute), "rel" (relative)

For instance, the command type ``"p_abs"`` defines a position command in absolute mode, while ``"v_rel"``
defines a velocity command in relative mode.

To facilitate easier composability, the actuator group handles the offsets and scalings applied to
the input commands. These are set through the :class:`ActuatorControlCfg` and by default are set to zero
and one, respectively. Based on these, the input commands are processed as follows:

.. math::

   \text{command} = \text{offset} + \text{scaling} \times \text{input command}

where :math:`\text{command}` is the command that is sent to the actuator model, :math:`\text{offset}`
is the offset applied to the input command, :math:`\text{scaling}` is the scaling applied to the input
command, and :math:`\text{input command}` is the input command from the user.

1. **Relative command:** The offset is based on the current joint state. For instance, if the
   command type is "p_rel", the offset is the current joint position.
2. **Absolute command:** The offset is based on the values set in the :class:`ActuatorControlCfg`.
   For instance, if the command type is "p_abs", the offset is the value for :attr:`dof_pos_offset`
   in the :class:`ActuatorControlCfg` instance.

.. note::

   Currently, the following joint command types are supported: "p_abs", "p_rel", "v_abs", "v_rel", "t_abs".


On initialization, the actuator group performs the following:

1. **Sets the control mode into simulation:** The control mode is set into the simulator for each joint
   based on the command types and actuator models. For implicit actuator models, this is interpreted
   from the first entry in the input list of command types. For explicit actuator models, the control
   mode is always set to torque.
2. **Sets the joint stiffness and damping gains:** In case of implicit actuators, these are set into
   simulator directly, while for explicit actuators, the gains are set into the actuator model.
3. **Sets the joint torque limits:** In case of implicit actuators, these are set into simulator directly
   based on the parsed configuration. For explicit actuators, the torque limits are set high since the
   actuator model itself is responsible for enforcing the torque limits.

While computing the joint actions, the actuator group takes the following steps:

1. **Formats the input actions:** It formats the input actions to account for additional constraints over
   the joints. These include mimicking the input command over the joints, or considering non-holonomic steering
   constraint for a wheel base.
2. **Pre-process joint commands:** It splits the formatted commands based on number of command types. For
   instance, if the input command types are "p_abs" and "v_abs", the input command is split into two tensors.
   Over each tensor, the scaling and offset are applied.
3. **Computes the joint actions:** The joint actions are computed based on the actuator model. For implicit
   actuator models, the joint actions are returned directly. For explicit actuator models, the joint actions
   are computed by calling the :meth:`IdealActuator.compute` and :meth:`IdealActuator.clip_torques` method.


Consider a scene with multiple Franka Emika Panda robots present in the stage at the USD paths *"/World/Robot_{n}"*
where *n* is the number of the instance. The following is an example of using the default actuator group to control
the robot with position and velocity commands:

.. code-block:: python

   import torch
   from omni.isaac.core.articulations import ArticulationView
   from omni.isaac.orbit.actuator.model import IdealActuatorCfg
   from omni.isaac.orbit.actuator.group import ActuatorControlCfg, ActuatorGroupCfg, ActuatorGroup

   # Note: We assume the stage is created and simulation is playing.

   # create an articulation view for the arms
   # Reference: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_core.html
   articulation_view = ArticulationView(
      "/World/Robot_.*", "panda_arm", reset_xform_properties=False
   )
   articulation_view.initialize()

   # create a configuration instance
   # -- model
   model_cfg = IdealActuatorCfg(motor_torque_limit=40, gear_ratio=1)
   # -- control
   control_cfg = ActuatorControlCfg(
      command_types=["p_abs", "v_abs"],
      stiffness={".*": 1000},
      damping={".*": 10}
   )
   # -- group
   group_cfg = ActuatorGroupCfg(
       dof_names=["panda_joint[1-7]"],
       model_cfg=model_cfg,
       control_cfg=control_cfg,
   )
   # create the actuator group
   group = ActuatorGroup(group_cfg, articulation_view)
   # clear history in the actuator model (if any)
   group.reset()

   # create random commands
   shape = (articulation_view.count, 7)
   dof_pos_des, dof_vel_des = torch.rand(*shape), torch.rand(*shape)
   # concatenate the commands into a single tensor
   group_actions = torch.cat([dof_pos_des, dof_vel_des], dim=1)
   # check that commands are of the correct shape
   assert group_actions.shape == (group.num_articulation, group.control_dim)

   # read current joint state
   dof_pos = articulation_view.get_joint_positions(joint_indices=group.dof_indices)
   dof_vel = articulation_view.get_joint_velocities(joint_indices=group.dof_indices)

   # compute the joint actions
   joint_actions = group.compute(group_actions, dof_pos, dof_vel)
   # set actions into simulator
   articulation_view.apply_actions(joint_actions)


Actuator Control Configuration
------------------------------

.. autoclass:: omni.isaac.orbit.actuators.group.ActuatorControlCfg
   :members:
   :undoc-members:
   :show-inheritance:

Default Actuator Group
----------------------

.. autoclass:: omni.isaac.orbit.actuators.group.ActuatorGroup
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omni.isaac.orbit.actuators.group.ActuatorGroupCfg
   :members:
   :undoc-members:
   :show-inheritance:


Gripper Actuator Group
-----------------------

.. autoclass:: omni.isaac.orbit.actuators.group.GripperActuatorGroup
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omni.isaac.orbit.actuators.group.GripperActuatorGroupCfg
   :members:
   :undoc-members:
   :show-inheritance:

Non-Holonomic Kinematics Group
------------------------------

.. autoclass:: omni.isaac.orbit.actuators.group.NonHolonomicKinematicsGroup
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: omni.isaac.orbit.actuators.group.NonHolonomicKinematicsGroupCfg
   :members:
   :undoc-members:
   :show-inheritance:
