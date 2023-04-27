Changelog
---------

0.3.2 (2023-04-27)
~~~~~~~~~~~~~~~~~~

Added
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

* Fixed setting of physics material on the ground plane when using :meth:``omni.isaac.orbit.utils.kit.create_ground_plane`` function.


0.1.0 (2023-01-17)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of the extension with experimental API.
* Available robot configurations:

  * **Quadrupeds:** Unitree A1, ANYmal B, ANYmal C
  * **Single-arm manipulators:** Franka Emika arm, UR5
  * **Mobile manipulators:** Clearpath Ridgeback with Franka Emika arm or UR5
