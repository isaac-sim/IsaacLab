Changelog
---------

0.9.0 (2023-08-18)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Introduces a new set of asset interfaces. These interfaces simplify the spawning of assets into the scene
  and initializing the physics handle by putting that inside post-startup physics callbacks. With this, users
  no longer need to worry about the :meth:`spawn` and :meth:`initialize` calls.
* Added utility methods to :mod:`omni.isaac.orbit.utils.string` module that resolve regex expressions based
  on passed list of target keys.

Changed
^^^^^^^

* Renamed all references of joints in an articulation from "dof" to "joint". This makes it consistent with the
  terminology used in robotics.

Deprecated
^^^^^^^^^^

* Removed the previous modules for objects and robots. Instead the :class:`Articulation` and :class:`RigidObject`
  should be used.


0.8.12 (2023-08-18)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added other properties provided by ``PhysicsScene`` to the :class:`omni.isaac.orbit.sim.SimulationContext`
  class to allow setting CCD, solver iterations, etc.
* Added commonly used functions to the :class:`SimulationContext` class itself to avoid having additional
  imports from Isaac Sim when doing simple tasks such as setting camera view or retrieving the simulation settings.

Fixed
^^^^^

* Switched the notations of default buffer values in :class:`omni.isaac.orbit.sim.PhysxCfg` from multiplication
  to scientific notation to avoid confusion with the values.


0.8.11 (2023-08-18)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Adds utility functions and configuration objects in the :mod:`omni.isaac.orbit.sim.spawners`
  to create the following prims in the scene:

  * :mod:`omni.isaac.orbit.sim.spawners.from_file`: Create a prim from a USD/URDF file.
  * :mod:`omni.isaac.orbit.sim.spawners.shapes`: Create USDGeom prims for shapes (box, sphere, cylinder, capsule, etc.).
  * :mod:`omni.isaac.orbit.sim.spawners.materials`: Create a visual or physics material prim.
  * :mod:`omni.isaac.orbit.sim.spawners.lights`: Create a USDLux prim for different types of lights.
  * :mod:`omni.isaac.orbit.sim.spawners.sensors`: Create a USD prim for supported sensors.

Changed
^^^^^^^

* Modified the :class:`SimulationContext` class to take the default physics material using the material spawn
  configuration object.


0.8.10 (2023-08-17)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added methods for defining different physics-based schemas in the :mod:`omni.isaac.orbit.sim.schemas` module.
  These methods allow creating the schema if it doesn't exist at the specified prim path and modify
  its properties based on the configuration object.


0.8.9 (2023-08-09)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved the :class:`omni.isaac.orbit.asset_loader.UrdfLoader` class to the :mod:`omni.isaac.orbit.sim.loaders`
  module to make it more accessible to the user.


0.8.8 (2023-08-09)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configuration classes and functions for setting different physics-based schemas in the
  :mod:`omni.isaac.orbit.sim.schemas` module. These allow modifying properties of the physics solver
  on the asset using configuration objects.


0.8.7 (2023-08-03)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added support for `__post_init__ <https://docs.python.org/3/library/dataclasses.html#post-init-processing>`_ in
  the :class:`omni.isaac.orbit.utils.configclass` decorator.


0.8.6 (2023-08-03)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for callable classes in the :class:`omni.isaac.orbit.managers.ManagerBase`.


0.8.5 (2023-08-03)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the :class:`omni.isaac.orbit.markers.Visualizationmarkers` class so that the markers are not visible in camera rendering mode.

Changed
^^^^^^^

* Simplified the creation of the point instancer in the :class:`omni.isaac.orbit.markers.Visualizationmarkers` class. It now creates a new
  prim at the next available prim path if a prim already exists at the given path.


0.8.4 (2023-08-02)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :class:`omni.isaac.orbit.sim.SimulationContext` class to the :mod:`omni.isaac.orbit.sim` module.
  This class inherits from the :class:`omni.isaac.core.simulation_context.SimulationContext` class and adds
  the ability to create a simulation context from a configuration object.


0.8.3 (2023-08-02)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved the :class:`ActuatorBase` class to the :mod:`omni.isaac.orbit.actuators.actuator_base` module.
* Renamed the :mod:`omni.isaac.orbit.actuators.actuator` module to :mod:`omni.isaac.orbit.actuators.actuator_pd`
  to make it more explicit that it contains the PD actuator models.


0.8.2 (2023-08-02)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Cleaned up the :class:`omni.isaac.orbit.terrain.TerrainImporter` class to take all the parameters from the configuration
  object. This makes it consistent with the other classes in the package.
* Moved the configuration classes for terrain generator and terrain importer into separate files to resolve circular
  dependency issues.


0.8.1 (2023-08-02)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added a hack into :class:`omni.isaac.orbit.app.AppLauncher` class to remove orbit packages from the path before launching
  the simulation application. This prevents the warning messages that appears when the user launches the ``SimulationApp``.

Added
^^^^^

* Enabled necessary viewport extensions in the :class:`omni.isaac.orbit.app.AppLauncher` class itself if ``VIEWPORT_ENABLED``
  flag is true.


0.8.0 (2023-07-26)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :class:`ActionManager` class to the :mod:`omni.isaac.orbit.managers` module to handle actions in the
  environment through action terms.
* Added contact force history to the :class:`omni.isaac.orbit.sensors.ContactSensor` class. The history is stored
  in the ``net_forces_w_history`` attribute of the sensor data.

Changed
^^^^^^^

* Implemented lazy update of buffers in the :class:`omni.isaac.orbit.sensors.SensorBase` class. This allows the user
  to update the sensor data only when required, i.e. when the data is requested by the user. This helps avoid double
  computation of sensor data when a reset is called in the environment.

Deprecated
^^^^^^^^^^

* Removed the support for different backends in the sensor class. We only use Pytorch as the backend now.
* Removed the concept of actuator groups. They are now handled by the :class:`omni.isaac.orbit.managers.ActionManager`
  class. The actuator models are now directly handled by the robot class itself.


0.7.4 (2023-07-26)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the behavior of the :class:`omni.isaac.orbit.terrains.TerrainImporter` class. It now expects the terrain
  type to be specified in the configuration object. This allows the user to specify everything in the configuration
  object and not have to do an explicit call to import a terrain.

Fixed
^^^^^

* Fixed setting of quaternion orientations inside the :class:`omni.isaac.orbit.markers.Visualizationmarkers` class.
  Earlier, the orientation was being set into the point instancer in the wrong order (``wxyz`` instead of ``xyzw``).


0.7.3 (2023-07-25)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the issue with multiple inheritance in the :class:`omni.isaac.orbit.utils.configclass` decorator.
  Earlier, if the inheritance tree was more than one level deep and the lowest level configuration class was
  not updating its values from the middle level classes.


0.7.2 (2023-07-24)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the method :meth:`replace` to the :class:`omni.isaac.orbit.utils.configclass` decorator to allow
  creating a new configuration object with values replaced from keyword arguments. This function internally
  calls the `dataclasses.replace <https://docs.python.org/3/library/dataclasses.html#dataclasses.replace>`_.

Fixed
^^^^^

* Fixed the handling of class types as member values in the :meth:`omni.isaac.orbit.utils.configclass`. Earlier it was
  throwing an error since class types were skipped in the if-else block.


0.7.1 (2023-07-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :class:`TerminationManager`, :class:`CurriculumManager`, and :class:`RandomizationManager` classes
  to the :mod:`omni.isaac.orbit.managers` module to handle termination, curriculum, and randomization respectively.


0.7.0 (2023-07-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created a new :mod:`omni.isaac.orbit.managers` module for all the managers related to the environment / scene.
  This includes the :class:`omni.isaac.orbit.managers.ObservationManager` and :class:`omni.isaac.orbit.managers.RewardManager`
  classes that were previously in the :mod:`omni.isaac.orbit.utils.mdp` module.
* Added the :class:`omni.isaac.orbit.managers.ManagerBase` class to handle the creation of managers.
* Added configuration classes for :class:`ObservationTermCfg` and :class:`RewardTermCfg` to allow easy creation of
  observation and reward terms.

Changed
^^^^^^^

* Changed the behavior of :class:`ObservationManager` and :class:`RewardManager` classes to accept the key ``func``
  in each configuration term to be a callable. This removes the need to inherit from the base class
  and allows more reusability of the functions across different environments.
* Moved the old managers to the :mod:`omni.isaac.orbit.compat.utils.mdp` module.
* Modified the necessary scripts to use the :mod:`omni.isaac.orbit.compat.utils.mdp` module.


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
^^^^^

* Added functions to convert string and callable objects in :mod:`omni.isaac.orbit.utils.string`.


0.6.0 (2023-07-16)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the argument :attr:`sort_keys` to the :meth:`omni.isaac.orbit.utils.io.yaml.dump_yaml` method to allow
  enabling/disabling of sorting of keys in the output yaml file.

Fixed
^^^^^

* Fixed the ordering of terms in :mod:`omni.isaac.orbit.utils.configclass` to be consistent in the order in which
  they are defined. Previously, the ordering was done alphabetically which made it inconsistent with the order in which
  the parameters were defined.

Changed
^^^^^^^

* Changed the default value of the argument :attr:`sort_keys` in the :meth:`omni.isaac.orbit.utils.io.yaml.dump_yaml`
  method to ``False``.
* Moved the old config classes in :mod:`omni.isaac.orbit.utils.configclass` to
  :mod:`omni.isaac.orbit.compat.utils.configclass` so that users can still run their old code where alphabetical
  ordering was used.


0.5.0 (2023-07-04)
~~~~~~~~~~~~~~~~~~

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

* Fixed bugs in :meth:`axis_angle_from_quat` in the ``omni.isaac.orbit.utils.math`` to handle quaternion with negative w component.
* Fixed bugs in :meth:`subtract_frame_transforms` in the ``omni.isaac.orbit.utils.math`` by adding the missing final rotation.


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

* Added :meth:`apply_nested_physics_material` to the ``omni.isaac.orbit.utils.kit``.
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
