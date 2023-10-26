Changelog
---------

0.9.22 (2023-10-26)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a :class:`omni.isaac.orbit.command_generators.NullCommandGenerator` class for no command environments.
  This is easier to work with than having checks for :obj:`None` in the command generator.

Fixed
^^^^^

* Moved the randomization manager to the :class:`omni.isaac.orbit.envs.BaseEnv` class with the default
  settings to reset the scene to the defaults specified in the configurations of assets.
* Moved command generator to the :class:`omni.isaac.orbit.envs.RlEnv` class to have all task-specification
  related classes in the same place.


0.9.21 (2023-10-26)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Decreased the priority of callbacks in asset and sensor base classes. This may help in preventing
  crashes when warm starting the simulation.
* Fixed no rendering mode when running the environment from the GUI. Earlier the function
  :meth:`SimulationContext.set_render_mode` was erroring out.


0.9.20 (2023-10-25)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Changed naming in :class:`omni.isaac.orbit.sim.SimulationContext.RenderMode` to use ``NO_GUI_OR_RENDERING``
  and ``NO_RENDERING`` instead of ``HEADLESS`` for clarity.
* Changed :class:`omni.isaac.orbit.sim.SimulationContext` to be capable of handling livestreaming and
  offscreen rendering.
* Changed :class:`omni.isaac.orbit.app.AppLauncher` envvar ``VIEWPORT_RECORD`` to the more descriptive
  ``OFFSCREEN_RENDER``.


0.9.19 (2023-10-25)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added Gym observation and action spaces for the :class:`omni.isaac.orbit.envs.RLEnv` class.


0.9.18 (2023-10-23)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created :class:`omni.issac.orbit.sim.converters.asset_converter.AssetConverter` to serve as a base
  class for all asset converters.
* Added :class:`omni.issac.orbit.sim.converters.mesh_converter.MeshConverter` to handle loading and conversion
  of mesh files (OBJ, STL and FBX) into USD format.
* Added script `convert_mesh.py` to ``source/tools`` to allow users to convert a mesh to USD via command line arguments.

Changed
^^^^^^^

* Renamed the submodule :mod:`omni.isaac.orbit.sim.loaders` to :mod:`omni.isaac.orbit.sim.converters` to be more
  general with the functionality of the module.
* Updated `check_instanceable.py` script to convert relative paths to absolute paths.


0.9.17 (2023-10-22)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added setters and getters for term configurations in the :class:`RandomizationManager`, :class:`RewardManager`
  and :class:`TerminationManager` classes. This allows the user to modify the term configurations after the
  manager has been created.
* Added the method :meth:`compute_group` to the :class:`omni.isaac.orbit.managers.ObservationManager` class to
  compute the observations for only a given group.
* Added the curriculum term for modifying reward weights after certain environment steps.


0.9.16 (2023-10-22)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for keyword arguments for terms in the :class:`omni.isaac.orbit.managers.ManagerBase`.

Fixed
^^^^^

* Fixed resetting of buffers in the :class:`TerminationManager` class. Earlier, the values were being set
  to ``0.0`` instead of ``False``.


0.9.15 (2023-10-22)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added base yaw heading and body acceleration into :class:`omni.isaac.orbit.assets.RigidObjectData` class.
  These quantities are computed inside the :class:`RigidObject` class.

Fixed
^^^^^

* Fixed the :meth:`omni.isaac.orbit.assets.RigidObject.set_external_force_and_torque` method to correctly
  deal with the body indices.
* Fixed a bug in the :meth:`omni.isaac.orbit.utils.math.wrap_to_pi` method to prevent self-assignment of
  the input tensor.


0.9.14 (2023-10-21)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added 2-D drift (i.e. along x and y) to the :class:`omni.isaac.orbit.sensors.RayCaster` class.
* Added flags to the :class:`omni.isaac.orbit.sensors.ContactSensorCfg` to optionally obtain the
  sensor origin and air time information. Since these are not required by default, they are
  disabled by default.

Fixed
^^^^^

* Fixed the handling of contact sensor history buffer in the :class:`omni.isaac.orbit.sensors.ContactSensor` class.
  Earlier, the buffer was not being updated correctly.


0.9.13 (2023-10-20)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the issue with double :obj:`Ellipsis` when indexing tensors with multiple dimensions.
  The fix now uses :obj:`slice(None)` instead of :obj:`Ellipsis` to index the tensors.


0.9.12 (2023-10-18)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed bugs in actuator model implementation for actuator nets. Earlier the DC motor clipping was not working.
* Fixed bug in applying actuator model in the :class:`omni.isaac.orbit.asset.Articulation` class. The new
  implementation caches the outputs from explicit actuator model into the ``joint_pos_*_sim`` buffer to
  avoid feedback loops in the tensor operation.


0.9.11 (2023-10-17)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the support for semantic tags into the :class:`omni.isaac.orbit.sim.spawner.SpawnerCfg` class. This allows
  the user to specify the semantic tags for a prim when spawning it into the scene. It follows the same format as
  Omniverse Replicator.


0.9.10 (2023-10-16)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added `livestream` and `ros` CLI args to :class:`omni.isaac.orbit.app.AppLauncher` class.
* Added a static function :meth:`omni.isaac.orbit.app.AppLauncher.add_app_launcher_args`, which
  appends the arguments needed for :class:`omni.isaac.orbit.app.AppLauncher` to the argument parser.

Changed
^^^^^^^

* Within :class:`omni.isaac.orbit.app.AppLauncher`, removed `REMOTE_DEPLOYMENT` env-var processing
  in the favor of ``HEADLESS`` and ``LIVESTREAM`` env-vars. These have clearer uses and better parity
  with the CLI args.


0.9.9 (2023-10-12)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the property :attr:`omni.isaac.orbit.assets.Articulation.is_fixed_base` to the articulation class to
  check if the base of the articulation is fixed or floating.
* Added the task-space action term corresponding to the differential inverse-kinematics controller.

Fixed
^^^^^

* Simplified the :class:`omni.isaac.orbit.controllers.DifferentialIKController` to assume that user provides the
  correct end-effector poses and Jacobians. Earlier it was doing internal frame transformations which made the
  code more complicated and error-prone.


0.9.8 (2023-09-30)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the boundedness of class objects that register callbacks into the simulator.
  These include devices, :class:`AssetBase`, :class:`SensorBase` and :class:`CommandGenerator`.
  The fix ensures that object gets deleted when the user deletes the object.


0.9.7 (2023-09-26)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Modified the :class:`omni.isaac.orbit.markers.VisualizationMarkers` to use the
  :class:`omni.isaac.orbit.sim.spawner.SpawnerCfg` class instead of their
  own configuration objects. This makes it consistent with the other ways to spawn assets in the scene.

Added
^^^^^

* Added the method :meth:`copy` to configclass to allow copying of configuration objects.


0.9.6 (2023-09-26)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Changed class-level configuration classes to refer to class types using ``class_type`` attribute instead
  of ``cls`` or ``cls_name``.


0.9.5 (2023-09-25)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added future import of ``annotations`` to have a consistent behavior across Python versions.
* Removed the type-hinting from docstrings to simplify maintenance of the documentation. All type-hints are
  now in the code itself.


0.9.4 (2023-08-29)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`omni.isaac.orbit.scene.InteractiveScene`, as the central scene unit that contains all entities
  that are part of the simulation. These include the terrain, sensors, articulations, rigid objects etc.
  The scene groups the common operations of these entities and allows to access them via their unique names.
* Added :mod:`omni.isaac.orbit.envs` module that contains environment definitions that encapsulate the different
  general (scene, action manager, observation manager) and RL-specific (reward and termination manager) managers.
* Added :class:`omni.isaac.orbit.managers.SceneEntityCfg` to handle which scene elements are required by the
  manager's terms. This allows the manager to parse useful information from the scene elements, such as the
  joint and body indices, and pass them to the term.
* Added :class:`omni.isaac.orbit.sim.SimulationContext.RenderMode` to handle different rendering modes based on
  what the user wants to update (viewport, cameras, or UI elements).

Fixed
^^^^^

* Fixed the :class:`omni.isaac.orbit.command_generators.CommandGeneratorBase` to register a debug visualization
  callback similar to how sensors and robots handle visualization.


0.9.3 (2023-08-23)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Enabled the `faulthander <https://docs.python.org/3/library/faulthandler.html>`_ to catch segfaults and print
  the stack trace. This is enabled by default in the :class:`omni.isaac.orbit.app.AppLauncher` class.

Fixed
^^^^^

* Re-added the :mod:`omni.isaac.orbit.utils.kit` to the ``compat`` directory and fixed all the references to it.
* Fixed the deletion of Replicator nodes for the :class:`omni.isaac.orbit.sensors.Camera` class. Earlier, the
  Replicator nodes were not being deleted when the camera was deleted. However, this does not prevent the random
  crashes that happen when the camera is deleted.
* Fixed the :meth:`omni.isaac.orbit.utils.math.convert_quat` to support both numpy and torch tensors.

Changed
^^^^^^^

* Renamed all the scripts inside the ``test`` directory to follow the convention:

  * ``test_<module_name>.py``: Tests for the module ``<module_name>`` using unittest.
  * ``check_<module_name>``: Check for the module ``<module_name>`` using python main function.


0.9.2 (2023-08-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the ability to color meshes in the :class:`omni.isaac.orbit.terrain.TerrainGenerator` class. Currently,
  it only supports coloring the mesh randomly (``"random"``), based on the terrain height (``"height"``), and
  no coloring (``"none"``).

Fixed
^^^^^

* Modified the :class:`omni.isaac.orbit.terrain.TerrainImporter` class to configure visual and physics materials
  based on the configuration object.


0.9.1 (2023-08-18)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Introduced three different rotation conventions in the :class:`omni.isaac.orbit.sensors.Camera` class. These
  conventions are:

  * ``opengl``: the camera is looking down the -Z axis with the +Y axis pointing up
  * ``ros``: the camera is looking down the +Z axis with the +Y axis pointing down
  * ``world``: the camera is looking along the +X axis with the -Z axis pointing down

  These can be used to declare the camera offset in :class:`omni.isaac.orbit.sensors.CameraCfg.OffsetCfg` class
  and in :meth:`omni.isaac.orbit.sensors.Camera.set_world_pose` method. Additionally, all conventions are
  saved to :class:`omni.isaac.orbit.sensors.CameraData` class for easy access.

Changed
^^^^^^^

* Adapted all the sensor classes to follow a structure similar to the :class:`omni.issac.orbit.assets.AssetBase`.
  Hence, the spawning and initialization of sensors manually by the users is avoided.
* Removed the :meth:`debug_vis` function since that this functionality is handled by a render callback automatically
  (based on the passed configuration for the :class:`omni.isaac.orbit.sensors.SensorBaseCfg.debug_vis` flag).


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

* Added a helper class :class:`omni.isaac.orbit.asset_loader.UrdfLoader` that converts a URDF file to instanceable USD
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
