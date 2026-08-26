.. _concepts_sensors_joint_wrench:
.. _overview_sensors_joint_wrench:

.. currentmodule:: isaaclab

Joint Wrench Sensor
===================

A :class:`~sensors.JointWrenchSensor` reports the incoming reaction wrench at each selected
articulation body's parent joint. It exposes force [N] and torque [N·m] separately, with entries
ordered by :attr:`~sensors.JointWrenchSensor.body_names`.

Wrench convention
-----------------

The ``incoming_joint_frame`` convention expresses the wrench in the child-side joint frame at the
child-side joint anchor. This matches the placement of a six-axis force/torque sensor mounted at the
joint. Backend implementations convert their native solver output to this common convention.

Configure the sensor
--------------------

Set :attr:`~sensors.JointWrenchSensorCfg.prim_path` to the articulation root. The sensor reports one
entry for each non-root articulation body:

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/ant_manager_env_cfg.py
   :language: python
   :lines: 91-95

Manager-based environments can select a body subset through
:class:`~isaaclab.managers.SceneEntityCfg` and use
:func:`~isaaclab.envs.mdp.body_incoming_wrench` as an observation term:

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/ant_manager_env_cfg.py
   :language: python
   :lines: 133-142

Read the data
-------------

For ``E`` environments and ``B`` reported bodies, ``force.torch`` and ``torque.torch`` each have
shape ``(E, B, 3)``. Both buffers are ``None`` before simulation initialization.

.. code-block:: python

   joint_wrench = scene["joint_wrench"]
   foot_ids, _ = joint_wrench.find_bodies([".*foot"])

   force = joint_wrench.data.force.torch[:, foot_ids]
   torque = joint_wrench.data.torque.torch[:, foot_ids]
   wrench = torch.cat((force, torque), dim=-1)

The composed ``wrench`` has shape ``(E, num_selected_bodies, 6)`` with force components followed by
torque components. Use ``body_names`` or :meth:`~sensors.JointWrenchSensor.find_bodies` instead of
assuming backend body order.
