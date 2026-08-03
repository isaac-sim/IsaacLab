.. _overview_sensors_joint_wrench:

.. currentmodule:: isaaclab

Joint Wrench Sensor
===================

The joint wrench sensor reports incoming joint reaction wrenches for selected
articulation bodies. It exposes force [N] and torque [N·m] buffers separately,
with entries ordered by the sensor's :attr:`~isaaclab.sensors.JointWrenchSensor.body_names`.
The default convention is ``incoming_joint_frame``, which expresses each wrench
in the child-side joint frame at the child-side joint anchor.

The sensor is configured on an articulation prim and can then be used directly
or through manager terms such as :func:`~isaaclab.envs.mdp.body_incoming_wrench`.
For example, the Ant environment adds a joint wrench sensor to the scene:

.. literalinclude:: ../../../../../source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/ant_manager_env_cfg.py
  :language: python
  :lines: 91-95

The same environment uses :class:`~isaaclab.managers.SceneEntityCfg` to select
the reported foot bodies for an observation term:

.. literalinclude:: ../../../../../source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/ant_manager_env_cfg.py
  :language: python
  :lines: 133-142

Direct access to the sensor data follows the usual scene lookup pattern.

.. code-block:: python

  joint_wrench = scene["joint_wrench"]
  foot_ids, _ = joint_wrench.find_bodies([".*foot"])

  force = joint_wrench.data.force.torch[:, foot_ids]
  torque = joint_wrench.data.torque.torch[:, foot_ids]
  wrench = torch.cat((force, torque), dim=-1)

The resulting ``wrench`` tensor has shape ``(num_envs, num_selected_bodies, 6)``
and stores the force components followed by the torque components for each
selected body.
