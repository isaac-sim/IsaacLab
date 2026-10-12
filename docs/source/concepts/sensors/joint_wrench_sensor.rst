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

PhysX's ``get_link_incoming_joint_force()`` already returns the wrench in the child-side joint frame,
referenced at its anchor, so the PhysX sensor exposes those components directly. Applying the USD
``localPos1`` and ``localRot1`` again would shift and rotate the wrench twice.

Read through articulation data
------------------------------

For observations that must follow the articulation's ``body_ordering``, use
:attr:`~assets.ArticulationData.body_joint_wrench`. The data container applies the
articulation's existing body map, so indices from ``robot.find_bodies()`` select the same bodies
in the wrench and other articulation data. No separate sensor is required.

Set ``robot_cfg.enable_joint_wrench = True`` before constructing the articulation when using
Newton. This requests the extended solver state; it is opt-in because wrench computation adds
solver work and is incompatible with MJWarp's sensor-disabled deterministic mode. PhysX and
OVPhysX provide the data on demand without this flag.

.. code-block:: python

   robot = scene["robot"]
   body_ids, _ = robot.find_bodies([".*foot"])
   wrench = robot.data.body_joint_wrench.torch[:, body_ids]
   force = wrench[..., :3]
   torque = wrench[..., 3:]

The tensor has shape ``(num_envs, num_bodies, 6)`` and uses the same incoming joint frame
convention as the sensor. Newton fills free and world-fixed root entries with zero and excludes
loop-closing constraints; PhysX and OVPhysX expose their root reactions. Read after a physics
step following a reset or state write, since these are solver results from the last step.

The standalone sensor API below retains backend-native entry order and its own update/reset timing.

Configure the sensor
--------------------

Set :attr:`~sensors.JointWrenchSensorCfg.prim_path` to the articulation root. Reported body coverage
depends on the physics backend:

* PhysX and OVPhysX report every articulation link, including the root link.
* Newton reports the child link of each non-free joint in the articulation tree, including fixed
  connections between bodies such as a welded wrist sensor or tool flange. Free joints, fixed joints
  to the world, and loop-closing constraints are excluded. The reported wrench at a weld includes
  the loads transmitted by its child subtree, even though the joint has no degrees of freedom.

Use :attr:`~sensors.JointWrenchSensor.body_names` or
:meth:`~sensors.JointWrenchSensor.find_bodies` instead of assuming that different backends expose
the same number or order of entries:

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/ant_manager_env_cfg.py
   :language: python
   :start-at: joint_wrench = JointWrenchSensorCfg
   :end-at: joint_wrench = JointWrenchSensorCfg

Manager-based environments can select a body subset through
:class:`~isaaclab.managers.SceneEntityCfg` and use
:func:`~isaaclab.envs.mdp.body_incoming_wrench` as an observation term:

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/core/locomotion/ant/ant_manager_env_cfg.py
   :language: python
   :start-at: feet_body_forces = ObsTerm(
   :end-at: actions = ObsTerm(func=mdp.last_action)

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
torque components. ``B`` and the entry ordering can differ between backends.
