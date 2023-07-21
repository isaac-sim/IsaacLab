Changelog
---------


0.6.2 (2023-07-21)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :mod:`omni.isaac.orbit.command_generators` to generate different commands based on the desired task.
  It allows the user to generate commands for different tasks in the same environment without having to write
  custom code for each task.


0.6.1 (2023-07-16)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the :meth:`omni.isaac.orbit.utils.math.quat_apply_yaw` to compute the yaw quaternion correctly.


Added
^^^^^^^

* Added functions to convert string and callable objects in :mod:`omni.isaac.orbit.utils.string`.


0.6.0 (2023-07-16)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the argument :attr:`sort_keys` to the :meth:`omni.isaac.orbit.utils.io.yaml.dump_yaml` method to allow
  enabling/disabling of sorting of keys in the output yaml file.

Fixed
^^^^^

* Fixed the ordering of terms in :mod:`omni.isaac.orbit.core.utils.configclass` to be consistent in the order in which
  they are defined. Previously, the ordering was done alphabetically which made it inconsistent with the order in which
  the parameters were defined.

Changed
^^^^^^^

* Changed the default value of the argument :attr:`sort_keys` in the :meth:`omni.isaac.orbit.utils.io.yaml.dump_yaml`
  method to ``False``.
* Moved the old config classes in :mod:`omni.isaac.orbit.core.utils.configclass` to
  :mod:`omni.isaac.orbit.compat.utils.configclass` so that users can still run their old code where alphabetical
  ordering was used.


0.5.0 (2023-07-04)

Added
^^^^^

* Added a generalized :class:`omni.isaac.orbit.sensors.SensorBase` class that leverages the ideas of views to
  handle multiple sensors in a single class.
* Added the classes :class:`omni.isaac.orbit.sensors.RayCaster`, :class:`omni.isaac.orbit.sensors.ContactSensor`,
  and :class:`omni.isaac.orbit.sensors.Camera` that output a batched tensor of sensor data.

Changed
^^^^^^^

* Renamed the parameter ``sensor_tick`` to ``update_freq`` to make it more intuitive.
* Moved the old sensors in :mod:`omni.isaac.orbit.sensors` to :mod:`omni.isaac.orbit.compat.sensors`.
* Modified the standalone scripts to use the :mod:`omni.isaac.orbit.compat.sensors` module.


0.4.4 (2023-07-05)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the :meth:`omni.isaac.orbit.terrains.trimesh.utils.make_plane` method to handle the case when the
  plane origin does not need to be centered.
* Added the :attr:`omni.isaac.orbit.terrains.TerrainGeneratorCfg.seed` to make generation of terrains reproducible.
  The default value is ``None`` which means that the seed is not set.

Changed
^^^^^^^

* Changed the saving of ``origins`` in :class:`omni.isaac.orbit.terrains.TerrainGenerator` class to be in CSV format
  instead of NPY format.


0.4.3 (2023-06-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :class:`omni.isaac.orbit.markers.PointInstancerMarker` class that wraps around
  `UsdGeom.PointInstancer <https://graphics.pixar.com/usd/dev/api/class_usd_geom_point_instancer.html>`_
  to directly work with torch and numpy arrays.

Changed
^^^^^^^

* Moved the old markers in :mod:`omni.isaac.orbit.markers` to :mod:`omni.isaac.orbit.compat.markers`.
* Modified the standalone scripts to use the :mod:`omni.isaac.orbit.compat.markers` module.


0.4.2 (2023-06-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the sub-module :mod:`omni.isaac.orbit.terrains` to allow procedural generation of terrains and supporting
  importing of terrains from different sources (meshes, usd files or default ground plane).


0.4.1 (2023-06-27)
~~~~~~~~~~~~~~~~~~

* Added the :class:`omni.isaac.orbit.app.AppLauncher` class to allow controlled instantiation of
  the `SimulationApp <https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html>`_
  and extension loading for remote deployment and ROS bridges.

Changed
^^^^^^^

* Modified all standalone scripts to use the :class:`omni.isaac.orbit.app.AppLauncher` class.


0.4.0 (2023-05-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a helper class :class:`omni.isaac.orbit.asset_loader.UrdfLoader` that coverts a URDF file to instanceable USD
  file based on the input configuration object.


0.3.2 (2023-04-27)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added safe-printing of functions while using the :meth:`omni.isaac.orbit.utils.dict.print_dict` function.


0.3.1 (2023-04-23)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a modified version of ``lula_franka_gen.urdf`` which includes an end-effector frame.
* Added a standalone script ``play_rmpflow.py`` to show RMPFlow controller.

Fixed
^^^^^

* Fixed the splitting of commands in the :meth:`ActuatorGroup.compute` method. Earlier it was reshaping the
  commands to the shape ``(num_actuators, num_commands)`` which was causing the commands to be split incorrectly.
* Fixed the processing of actuator command in the :meth:`RobotBase._process_actuators_cfg` to deal with multiple
  command types when using "implicit" actuator group.

0.3.0 (2023-04-20)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added the destructor to the keyboard devices to unsubscribe from carb.

Added
^^^^^

* Added the :class:`Se2Gamepad` and :class:`Se3Gamepad` for gamepad teleoperation support.


0.2.8 (2023-04-10)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed bugs in :meth:`axis_angle_from_quat` in the ``omni.isaac.orbit.core.utils.math`` to handle quaternion with negative w component.
* Fixed bugs in :meth:`subtract_frame_transforms` in the ``omni.isaac.orbit.core.utils.math`` by adding the missing final rotation.


0.2.7 (2023-04-07)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed repetition in applying mimic multiplier for "p_abs" in the :class:`GripperActuatorGroup` class.
* Fixed bugs in :meth:`reset_buffers` in the :class:`RobotBase` and :class:`LeggedRobot` classes.

0.2.6 (2023-03-16)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :class:`CollisionPropertiesCfg` to rigid/articulated object and robot base classes.
* Added the :class:`PhysicsMaterialCfg` to the :class:`SingleArm` class for tool sites.

Changed
^^^^^^^

* Changed the default control mode of the :obj:`PANDA_HAND_MIMIC_GROUP_CFG` to be from ``"v_abs"`` to ``"p_abs"``.
  Using velocity control for the mimic group can cause the hand to move in a jerky manner.


0.2.5 (2023-03-08)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the indices used for the Jacobian and dynamics quantities in the :class:`MobileManipulator` class.


0.2.4 (2023-03-04)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :meth:`apply_nested_physics_material` to the ``omni.isaac.orbit.core.utils.kit``.
* Added the :meth:`sample_cylinder` to sample points from a cylinder's surface.
* Added documentation about the issue in using instanceable asset as markers.

Fixed
^^^^^

* Simplified the physics material application in the rigid object and legged robot classes.

Removed
^^^^^^^

* Removed the ``geom_prim_rel_path`` argument in the :class:`RigidObjectCfg.MetaInfoCfg` class.


0.2.3 (2023-02-24)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the end-effector body index used for getting the Jacobian in the :class:`SingleArm` and :class:`MobileManipulator` classes.


0.2.2 (2023-01-27)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the :meth:`set_world_pose_ros` and :meth:`set_world_pose_from_view` in the :class:`Camera` class.

Deprecated
^^^^^^^^^^

* Removed the :meth:`set_world_pose_from_ypr` method from the :class:`Camera` class.


0.2.1 (2023-01-26)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the :class:`Camera` class to support different fisheye projection types.


0.2.0 (2023-01-25)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for warp backend in camera utilities.
* Extended the ``play_camera.py`` with ``--gpu`` flag to use GPU replicator backend.

0.1.1 (2023-01-24)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed setting of physics material on the ground plane when using :meth:`omni.isaac.orbit.utils.kit.create_ground_plane` function.


0.1.0 (2023-01-17)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of the extension with experimental API.
* Available robot configurations:

  * **Quadrupeds:** Unitree A1, ANYmal B, ANYmal C
  * **Single-arm manipulators:** Franka Emika arm, UR5
  * **Mobile manipulators:** Clearpath Ridgeback with Franka Emika arm or UR5
