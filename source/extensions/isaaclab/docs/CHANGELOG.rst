Changelog
---------

0.18.5 (2024-06-26)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the gravity vector direction used inside the :class:`isaaclab.assets.RigidObjectData`class.
  Earlier, the gravity direction was hard-coded as (0, 0, -1) which may be different from the actual
  gravity direction in the simulation. Now, the gravity direction is obtained from the simulation context
  and used to compute the projection of the gravity vector on the object.


0.18.4 (2024-06-26)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed double reference count of the physics sim view inside the asset classes. This was causing issues
  when destroying the asset class instance since the physics sim view was not being properly released.

Added
^^^^^

* Added the attribute :attr:`~isaaclab.assets.AssetBase.is_initialized` to check if the asset and sensor
  has been initialized properly. This can be used to ensure that the asset or sensor is ready to use in the simulation.


0.18.3 (2024-06-25)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the docstrings at multiple places related to the different buffer implementations inside the
  :mod:`isaaclab.utils.buffers` module. The docstrings were not clear and did not provide enough
  information about the classes and their methods.

Added
^^^^^

* Added the field for fixed tendom names in the :class:`isaaclab.assets.ArticulationData` class.
  Earlier, this information was not exposed which was inconsistent with other name related information
  such as joint or body names.

Changed
^^^^^^^

* Renamed the fields ``min_num_time_lags`` and ``max_num_time_lags`` to ``min_delay`` and
  ``max_delay`` in the :class:`isaaclab.actuators.DelayedPDActuatorCfg` class. This is to make
  the naming simpler to understand.


0.18.2 (2024-06-25)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved the configuration for tile-rendered camera into its own file named ``tiled_camera_cfg.py``.
  This makes it easier to follow where the configuration is located and how it is related to the class.


0.18.1 (2024-06-25)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Ensured that a parity between class and its configuration class is explicitly visible in the
  :mod:`isaaclab.envs` module. This makes it easier to follow where definitions are located and how
  they are related. This should not be a breaking change as the classes are still accessible through the same module.


0.18.0 (2024-06-13)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the rendering logic to render at the specified interval. Earlier, the substep parameter had no effect and rendering
  would happen once every env.step() when active.

Changed
^^^^^^^

* Renamed :attr:`isaaclab.sim.SimulationCfg.substeps` to :attr:`isaaclab.sim.SimulationCfg.render_interval`.
  The render logic is now integrated in the decimation loop of the environment.


0.17.13 (2024-06-13)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the orientation reset logic in :func:`isaaclab.envs.mdp.events.reset_root_state_uniform` to make it relative to
  the default orientation. Earlier, the position was sampled relative to the default and the orientation not.


0.17.12 (2024-06-13)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the class :class:`isaaclab.utils.buffers.TimestampedBuffer` to store timestamped data.

Changed
^^^^^^^

* Added time-stamped buffers in the classes :class:`isaaclab.assets.RigidObjectData` and :class:`isaaclab.assets.ArticulationData`
  to update some values lazily and avoid unnecessary computations between physics updates. Before, all the data was always
  updated at every step, even if it was not used by the task.


0.17.11 (2024-05-30)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :class:`isaaclab.sensor.ContactSensor` not loading correctly in extension mode.
  Earlier, the :attr:`isaaclab.sensor.ContactSensor.body_physx_view` was not initialized when
  :meth:`isaaclab.sensor.ContactSensor._debug_vis_callback` is called which references it.


0.17.10 (2024-05-30)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed compound classes being directly assigned in ``default_factory`` generator method
  :meth:`isaaclab.utils.configclass._return_f`, which resulted in shared references such that modifications to
  compound objects were reflected across all instances generated from the same ``default_factory`` method.


0.17.9 (2024-05-30)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``variants`` attribute to the :class:`isaaclab.sim.from_files.UsdFileCfg` class to select USD
  variants when loading assets from USD files.


0.17.8 (2024-05-28)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Implemented the reset methods in the action terms to avoid returning outdated data.


0.17.7 (2024-05-28)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added debug visualization utilities in the :class:`isaaclab.managers.ActionManager` class.


0.17.6 (2024-05-27)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``wp.init()`` call in Warp utils.


0.17.5 (2024-05-22)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Websocket livestreaming is no longer supported. Valid livestream options are {0, 1, 2}.
* WebRTC livestream is now set with livestream=2.


0.17.4 (2024-05-17)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Modified the noise functions to also support add, scale, and abs operations on the data. Added aliases
  to ensure backward compatibility with the previous functions.

  * Added :attr:`isaaclab.utils.noise.NoiseCfg.operation` for the different operations.
  * Renamed ``constant_bias_noise`` to :func:`isaaclab.utils.noise.constant_noise`.
  * Renamed ``additive_uniform_noise`` to :func:`isaaclab.utils.noise.uniform_noise`.
  * Renamed ``additive_gaussian_noise`` to :func:`isaaclab.utils.noise.gaussian_noise`.


0.17.3 (2024-05-15)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Set ``hide_ui`` flag in the app launcher for livestream.
* Fix native client livestream extensions.


0.17.2 (2024-05-09)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Renamed ``_range`` to ``distribution_params`` in ``events.py`` for methods that defined a distribution.
* Apply additive/scaling randomization noise on default data instead of current data.
* Changed material bucketing logic to prevent exceeding 64k materials.

Fixed
^^^^^

* Fixed broadcasting issues with indexing when environment and joint IDs are provided.
* Fixed incorrect tensor dimensions when setting a subset of environments.

Added
^^^^^

* Added support for randomization of fixed tendon parameters.
* Added support for randomization of dof limits.
* Added support for randomization of gravity.
* Added support for Gaussian sampling.
* Added default buffers to Articulation/Rigid object data classes for randomization.


0.17.1 (2024-05-10)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added attribute :attr:`isaaclab.sim.converters.UrdfConverterCfg.override_joint_dynamics` to properly parse
  joint dynamics in :class:`isaaclab.sim.converters.UrdfConverter`.


0.17.0 (2024-05-07)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Renamed ``BaseEnv`` to :class:`isaaclab.envs.ManagerBasedEnv`.
* Renamed ``base_env.py`` to ``manager_based_env.py``.
* Renamed ``BaseEnvCfg`` to :class:`isaaclab.envs.ManagerBasedEnvCfg`.
* Renamed ``RLTaskEnv`` to :class:`isaaclab.envs.ManagerBasedRLEnv`.
* Renamed ``rl_task_env.py`` to ``manager_based_rl_env.py``.
* Renamed ``RLTaskEnvCfg`` to :class:`isaaclab.envs.ManagerBasedRLEnvCfg`.
* Renamed ``rl_task_env_cfg.py`` to ``rl_env_cfg.py``.
* Renamed ``OIGEEnv`` to :class:`isaaclab.envs.DirectRLEnv`.
* Renamed ``oige_env.py`` to ``direct_rl_env.py``.
* Renamed ``RLTaskEnvWindow`` to :class:`isaaclab.envs.ui.ManagerBasedRLEnvWindow`.
* Renamed ``rl_task_env_window.py`` to ``manager_based_rl_env_window.py``.
* Renamed all references of ``BaseEnv``, ``BaseEnvCfg``, ``RLTaskEnv``, ``RLTaskEnvCfg``,  ``OIGEEnv``, and ``RLTaskEnvWindow``.

Added
^^^^^

* Added direct workflow base class :class:`isaaclab.envs.DirectRLEnv`.


0.16.4 (2024-05-06)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added :class:`isaaclab.sensors.TiledCamera` to support tiled rendering with RGB and depth.


0.16.3 (2024-04-26)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed parsing of filter prim path expressions in the :class:`isaaclab.sensors.ContactSensor` class.
  Earlier, the filter prim paths given to the physics view was not being parsed since they were specified as
  regex expressions instead of glob expressions.


0.16.2 (2024-04-25)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Simplified the installation procedure, isaaclab -e is no longer needed
* Updated torch dependency to 2.2.2


0.16.1 (2024-04-20)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added attribute :attr:`isaaclab.sim.ArticulationRootPropertiesCfg.fix_root_link` to fix the root link
  of an articulation to the world frame.


0.16.0 (2024-04-16)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the function :meth:`isaaclab.utils.math.quat_unique` to standardize quaternion representations,
  i.e. always have a non-negative real part.
* Added events terms for randomizing mass by scale, simulation joint properties (stiffness, damping, armature,
  and friction)

Fixed
^^^^^

* Added clamping of joint positions and velocities in event terms for resetting joints. The simulation does not
  throw an error if the set values are out of their range. Hence, users are expected to clamp them before setting.
* Fixed :class:`isaaclab.envs.mdp.EMAJointPositionToLimitsActionCfg` to smoothen the actions
  at environment frequency instead of simulation frequency.

* Renamed the following functions in :meth:`isaaclab.envs.mdp` to avoid confusions:

  * Observation: :meth:`joint_pos_norm` -> :meth:`joint_pos_limit_normalized`
  * Action: :class:`ExponentialMovingAverageJointPositionAction` -> :class:`EMAJointPositionToLimitsAction`
  * Termination: :meth:`base_height` -> :meth:`root_height_below_minimum`
  * Termination: :meth:`joint_pos_limit` -> :meth:`joint_pos_out_of_limit`
  * Termination: :meth:`joint_pos_manual_limit` -> :meth:`joint_pos_out_of_manual_limit`
  * Termination: :meth:`joint_vel_limit` -> :meth:`joint_vel_out_of_limit`
  * Termination: :meth:`joint_vel_manual_limit` -> :meth:`joint_vel_out_of_manual_limit`
  * Termination: :meth:`joint_torque_limit` -> :meth:`joint_effort_out_of_limit`

Deprecated
^^^^^^^^^^

* Deprecated the function :meth:`isaaclab.envs.mdp.add_body_mass` in favor of
  :meth:`isaaclab.envs.mdp.randomize_rigid_body_mass`. This supports randomizing the mass based on different
  operations (add, scale, or set) and sampling distributions.


0.15.13 (2024-04-16)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Improved startup performance by enabling rendering-based extensions only when necessary and caching of nucleus directory.
* Renamed the flag ``OFFSCREEN_RENDER`` or ``--offscreen_render`` to ``ENABLE_CAMERAS`` or ``--enable_cameras`` respectively.


0.15.12 (2024-04-16)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Replaced calls to the ``check_file_path`` function in the :mod:`isaaclab.sim.spawners.from_files`
  with the USD stage resolve identifier function. This helps speed up the loading of assets from file paths
  by avoiding Nucleus server calls.


0.15.11 (2024-04-15)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :meth:`isaaclab.sim.SimulationContext.has_rtx_sensors` method to check if any
  RTX-related sensors such as cameras have been created in the simulation. This is useful to determine
  if simulation requires RTX rendering during step or not.

Fixed
^^^^^

* Fixed the rendering of RTX-related sensors such as cameras inside the :class:`isaaclab.envs.RLTaskEnv` class.
  Earlier the rendering did not happen inside the step function, which caused the sensor data to be empty.


0.15.10 (2024-04-11)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed sharing of the same memory address between returned tensors from observation terms
  in the :class:`isaaclab.managers.ObservationManager` class. Earlier, the returned
  tensors could map to the same memory address, causing issues when the tensors were modified
  during scaling, clipping or other operations.


0.15.9 (2024-04-04)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed assignment of individual termination terms inside the :class:`isaaclab.managers.TerminationManager`
  class. Earlier, the terms were being assigned their values through an OR operation which resulted in incorrect
  values. This regression was introduced in version 0.15.1.


0.15.8 (2024-04-02)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added option to define ordering of points for the mesh-grid generation in the
  :func:`isaaclab.sensors.ray_caster.patterns.grid_pattern`. This parameter defaults to 'xy'
  for backward compatibility.


0.15.7 (2024-03-28)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Adds option to return indices/data in the specified query keys order in
  :class:`isaaclab.managers.SceneEntityCfg` class, and the respective
  :func:`isaaclab.utils.string.resolve_matching_names_values` and
  :func:`isaaclab.utils.string.resolve_matching_names` functions.


0.15.6 (2024-03-28)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Extended the :class:`isaaclab.app.AppLauncher` class to support the loading of experience files
  from the command line. This allows users to load a specific experience file when running the application
  (such as for multi-camera rendering or headless mode).

Changed
^^^^^^^

* Changed default loading of experience files in the :class:`isaaclab.app.AppLauncher` class from the ones
  provided by Isaac Sim to the ones provided in Isaac Lab's ``source/apps`` directory.


0.15.5 (2024-03-23)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the env origins in :meth:`_compute_env_origins_grid` of :class:`isaaclab.terrain.TerrainImporter`
  to match that obtained from the Isaac Sim :class:`omni.isaac.cloner.GridCloner` class.

Added
^^^^^

* Added unit test to ensure consistency between environment origins generated by IsaacSim's Grid Cloner and those
  produced by the TerrainImporter.


0.15.4 (2024-03-22)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the :class:`isaaclab.envs.mdp.actions.NonHolonomicActionCfg` class to use
  the correct variable when applying actions.


0.15.3 (2024-03-21)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added unit test to check that :class:`isaaclab.scene.InteractiveScene` entity data is not shared between separate instances.

Fixed
^^^^^

* Moved class variables in :class:`isaaclab.scene.InteractiveScene` to correctly  be assigned as
  instance variables.
* Removed custom ``__del__`` magic method from :class:`isaaclab.scene.InteractiveScene`.


0.15.2 (2024-03-21)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added resolving of relative paths for the main asset USD file when using the
  :class:`isaaclab.sim.converters.UrdfConverter` class. This is to ensure that the material paths are
  resolved correctly when the main asset file is moved to a different location.


0.15.1 (2024-03-19)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the imitation learning workflow example script, updating Isaac Lab and Robomimic API calls.
* Removed the resetting of :attr:`_term_dones` in the :meth:`isaaclab.managers.TerminationManager.reset`.
  Previously, the environment cleared out all the terms. However, it impaired reading the specific term's values externally.


0.15.0 (2024-03-17)
~~~~~~~~~~~~~~~~~~~

Deprecated
^^^^^^^^^^

* Renamed :class:`isaaclab.managers.RandomizationManager` to :class:`isaaclab.managers.EventManager`
  class for clarification as the manager takes care of events such as reset in addition to pure randomizations.
* Renamed :class:`isaaclab.managers.RandomizationTermCfg` to :class:`isaaclab.managers.EventTermCfg`
  for consistency with the class name change.


0.14.1 (2024-03-16)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added simulation schemas for joint drive and fixed tendons. These can be configured for assets imported
  from file formats.
* Added logging of tendon properties to the articulation class (if they are present in the USD prim).


0.14.0 (2024-03-15)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the ordering of body names used in the :class:`isaaclab.assets.Articulation` class. Earlier,
  the body names were not following the same ordering as the bodies in the articulation. This led
  to issues when using the body names to access data related to the links from the articulation view
  (such as Jacobians, mass matrices, etc.).

Removed
^^^^^^^

* Removed the attribute :attr:`body_physx_view` from the :class:`isaaclab.assets.RigidObject`
  and :class:`isaaclab.assets.Articulation` classes. These were causing confusions when used
  with articulation view since the body names were not following the same ordering.


0.13.1 (2024-03-14)
~~~~~~~~~~~~~~~~~~~

Removed
^^^^^^^

* Removed the :mod:`isaaclab.compat` module. This module was used to provide compatibility
  with older versions of Isaac Sim. It is no longer needed since we have most of the functionality
  absorbed into the main classes.


0.13.0 (2024-03-12)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for the following data types inside the :class:`isaaclab.sensors.Camera` class:
  ``instance_segmentation_fast`` and ``instance_id_segmentation_fast``. These are GPU-supported annotations
  and are faster than the regular annotations.

Fixed
^^^^^

* Fixed handling of semantic filtering inside the :class:`isaaclab.sensors.Camera` class. Earlier,
  the annotator was given ``semanticTypes`` as an argument. However, with Isaac Sim 2023.1, the annotator
  does not accept this argument. Instead the mapping needs to be set to the synthetic data interface directly.
* Fixed the return shape of colored images for segmentation data types inside the
  :class:`isaaclab.sensors.Camera` class. Earlier, the images were always returned as ``int32``. Now,
  they are casted to ``uint8`` 4-channel array before returning if colorization is enabled for the annotation type.

Removed
^^^^^^^

* Dropped support for ``instance_segmentation`` and ``instance_id_segmentation`` annotations in the
  :class:`isaaclab.sensors.Camera` class. Their "fast" counterparts should be used instead.
* Renamed the argument :attr:`isaaclab.sensors.CameraCfg.semantic_types` to
  :attr:`isaaclab.sensors.CameraCfg.semantic_filter`. This is more aligned with Replicator's terminology
  for semantic filter predicates.
* Replaced the argument :attr:`isaaclab.sensors.CameraCfg.colorize` with separate colorized
  arguments for each annotation type (:attr:`~isaaclab.sensors.CameraCfg.colorize_instance_segmentation`,
  :attr:`~isaaclab.sensors.CameraCfg.colorize_instance_id_segmentation`, and
  :attr:`~isaaclab.sensors.CameraCfg.colorize_semantic_segmentation`).


0.12.4 (2024-03-11)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^


* Adapted randomization terms to deal with ``slice`` for the body indices. Earlier, the terms were not
  able to handle the slice object and were throwing an error.
* Added ``slice`` type-hinting to all body and joint related methods in the rigid body and articulation
  classes. This is to make it clear that the methods can handle both list of indices and slices.


0.12.3 (2024-03-11)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added signal handler to the :class:`isaaclab.app.AppLauncher` class to catch the ``SIGINT`` signal
  and close the application gracefully. This is to prevent the application from crashing when the user
  presses ``Ctrl+C`` to close the application.


0.12.2 (2024-03-10)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added observation terms for states of a rigid object in world frame.
* Added randomization terms to set root state with randomized orientation and joint state within user-specified limits.
* Added reward term for penalizing specific termination terms.

Fixed
^^^^^

* Improved sampling of states inside randomization terms. Earlier, the code did multiple torch calls
  for sampling different components of the vector. Now, it uses a single call to sample the entire vector.


0.12.1 (2024-03-09)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added an option to the last actions observation term to get a specific term by name from the action manager.
  If None, the behavior remains the same as before (the entire action is returned).


0.12.0 (2024-03-08)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added functionality to sample flat patches on a generated terrain. This can be configured using
  :attr:`isaaclab.terrains.SubTerrainBaseCfg.flat_patch_sampling` attribute.
* Added a randomization function for setting terrain-aware root state. Through this, an asset can be
  reset to a randomly sampled flat patches.

Fixed
^^^^^

* Separated normal and terrain-base position commands. The terrain based commands rely on the
  terrain to sample flat patches for setting the target position.
* Fixed command resample termination function.

Changed
^^^^^^^

* Added the attribute :attr:`isaaclab.envs.mdp.commands.UniformVelocityCommandCfg.heading_control_stiffness`
  to control the stiffness of the heading control term in the velocity command term. Earlier, this was
  hard-coded to 0.5 inside the term.

Removed
^^^^^^^

* Removed the function :meth:`sample_new_targets` in the terrain importer. Instead the attribute
  :attr:`isaaclab.terrains.TerrainImporter.flat_patches` should be used to sample new targets.


0.11.3 (2024-03-04)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Corrects the functions :func:`isaaclab.utils.math.axis_angle_from_quat` and :func:`isaaclab.utils.math.quat_error_magnitude`
  to accept tensors of the form (..., 4) instead of (N, 4). This brings us in line with our documentation and also upgrades one of our functions
  to handle higher dimensions.


0.11.2 (2024-03-04)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added checks for default joint position and joint velocity in the articulation class. This is to prevent
  users from configuring values for these quantities that might be outside the valid range from the simulation.


0.11.1 (2024-02-29)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Replaced the default values for ``joint_ids`` and ``body_ids`` from ``None`` to ``slice(None)``
  in the :class:`isaaclab.managers.SceneEntityCfg`.
* Adapted rewards and observations terms so that the users can query a subset of joints and bodies.


0.11.0 (2024-02-27)
~~~~~~~~~~~~~~~~~~~

Removed
^^^^^^^

* Dropped support for Isaac Sim<=2022.2. As part of this, removed the components of :class:`isaaclab.app.AppLauncher`
  which handled ROS extension loading. We no longer need them in Isaac Sim>=2023.1 to control the load order to avoid crashes.
* Upgraded Dockerfile to use ISAACSIM_VERSION=2023.1.1 by default.


0.10.28 (2024-02-29)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Implemented relative and moving average joint position action terms. These allow the user to specify
  the target joint positions as relative to the current joint positions or as a moving average of the
  joint positions over a window of time.


0.10.27 (2024-02-28)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added UI feature to start and stop animation recording in the stage when running an environment.
  To enable this feature, please pass the argument ``--disable_fabric`` to the environment script to allow
  USD read/write operations. Be aware that this will slow down the simulation.


0.10.26 (2024-02-26)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a viewport camera controller class to the :class:`isaaclab.envs.BaseEnv`. This is useful
  for applications where the user wants to render the viewport from different perspectives even when the
  simulation is running in headless mode.


0.10.25 (2024-02-26)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Ensures that all path arguments in :mod:`isaaclab.sim.utils` are cast to ``str``. Previously,
  we had handled path types as strings without casting.


0.10.24 (2024-02-26)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added tracking of contact time in the :class:`isaaclab.sensors.ContactSensor` class. Previously,
  only the air time was being tracked.
* Added contact force threshold, :attr:`isaaclab.sensors.ContactSensorCfg.force_threshold`, to detect
  when the contact sensor is in contact. Previously, this was set to hard-coded 1.0 in the sensor class.


0.10.23 (2024-02-21)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixes the order of size arguments in :meth:`isaaclab.terrains.height_field.random_uniform_terrain`. Previously, the function would crash if the size along x and y were not the same.


0.10.22 (2024-02-14)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed "divide by zero" bug in :class:`~isaaclab.sim.SimulationContext` when setting gravity vector.
  Now, it is correctly disabled when the gravity vector is set to zero.


0.10.21 (2024-02-12)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the printing of articulation joint information when the articulation has only one joint.
  Earlier, the function was performing a squeeze operation on the tensor, which caused an error when
  trying to index the tensor of shape (1,).


0.10.20 (2024-02-12)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Adds :attr:`isaaclab.sim.PhysxCfg.enable_enhanced_determinism` to enable improved
  determinism from PhysX. Please note this comes at the expense of performance.


0.10.19 (2024-02-08)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed environment closing so that articulations, objects, and sensors are cleared properly.


0.10.18 (2024-02-05)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Pinned :mod:`torch` version to 2.0.1 in the setup.py to keep parity version of :mod:`torch` supplied by
  Isaac 2023.1.1, and prevent version incompatibility between :mod:`torch` ==2.2 and
  :mod:`typing-extensions` ==3.7.4.3


0.10.17 (2024-02-02)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^^

* Fixed carb setting ``/app/livestream/enabled`` to be set as False unless live-streaming is specified
  by :class:`isaaclab.app.AppLauncher` settings. This fixes the logic of :meth:`SimulationContext.render`,
  which depended on the config in previous versions of Isaac defaulting to false for this setting.


0.10.16 (2024-01-29)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^^

* Added an offset parameter to the height scan observation term. This allows the user to specify the
  height offset of the scan from the tracked body. Previously it was hard-coded to be 0.5.


0.10.15 (2024-01-29)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed joint torque computation for implicit actuators. Earlier, the torque was always zero for implicit
  actuators. Now, it is computed approximately by applying the PD law.


0.10.14 (2024-01-22)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the tensor shape of :attr:`isaaclab.sensors.ContactSensorData.force_matrix_w`. Earlier, the reshaping
  led to a mismatch with the data obtained from PhysX.


0.10.13 (2024-01-15)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed running of environments with a single instance even if the :attr:`replicate_physics`` flag is set to True.


0.10.12 (2024-01-10)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed indexing of source and target frames in the :class:`isaaclab.sensors.FrameTransformer` class.
  Earlier, it always assumed that the source frame body is at index 0. Now, it uses the body index of the
  source frame to compute the transformation.

Deprecated
^^^^^^^^^^

* Renamed quantities in the :class:`isaaclab.sensors.FrameTransformerData` class to be more
  consistent with the terminology used in the asset classes. The following quantities are deprecated:

  * ``target_rot_w`` -> ``target_quat_w``
  * ``source_rot_w`` -> ``source_quat_w``
  * ``target_rot_source`` -> ``target_quat_source``


0.10.11 (2024-01-08)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed attribute error raised when calling the :class:`isaaclab.envs.mdp.TerrainBasedPositionCommand`
  command term.
* Added a dummy function in :class:`isaaclab.terrain.TerrainImporter` that returns environment
  origins as terrain-aware sampled targets. This function should be implemented by child classes based on
  the terrain type.


0.10.10 (2023-12-21)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed reliance on non-existent ``Viewport`` in :class:`isaaclab.sim.SimulationContext` when loading livestreaming
  by ensuring that the extension ``omni.kit.viewport.window`` is enabled in :class:`isaaclab.app.AppLauncher` when
  livestreaming is enabled


0.10.9 (2023-12-21)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed invalidation of physics views inside the asset and sensor classes. Earlier, they were left initialized
  even when the simulation was stopped. This caused issues when closing the application.


0.10.8 (2023-12-20)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the :class:`isaaclab.envs.mdp.actions.DifferentialInverseKinematicsAction` class
  to account for the offset pose of the end-effector.


0.10.7 (2023-12-19)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added a check to ray-cast and camera sensor classes to ensure that the sensor prim path does not
  have a regex expression at its leaf. For instance, ``/World/Robot/camera_.*`` is not supported
  for these sensor types. This behavior needs to be fixed in the future.


0.10.6 (2023-12-19)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for using articulations as visualization markers. This disables all physics APIs from
  the articulation and allows the user to use it as a visualization marker. It is useful for creating
  visualization markers for the end-effectors or base of the robot.

Fixed
^^^^^

* Fixed hiding of debug markers from secondary images when using the
  :class:`isaaclab.markers.VisualizationMarkers` class. Earlier, the properties were applied on
  the XForm prim instead of the Mesh prim.


0.10.5 (2023-12-18)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed test ``check_base_env_anymal_locomotion.py``, which
  previously called :func:`torch.jit.load` with the path to a policy (which would work
  for a local file), rather than calling
  :func:`isaaclab.utils.assets.read_file` on the path to get the file itself.


0.10.4 (2023-12-14)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed potentially breaking import of omni.kit.widget.toolbar by ensuring that
  if live-stream is enabled, then the :mod:`omni.kit.widget.toolbar`
  extension is loaded.

0.10.3 (2023-12-12)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the attribute :attr:`isaaclab.actuators.ActuatorNetMLPCfg.input_order`
  to specify the order of the input tensors to the MLP network.

Fixed
^^^^^

* Fixed computation of metrics for the velocity command term. Earlier, the norm was being computed
  over the entire batch instead of the last dimension.
* Fixed the clipping inside the :class:`isaaclab.actuators.DCMotor` class. Earlier, it was
  not able to handle the case when configured saturation limit was set to None.


0.10.2 (2023-12-12)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added a check in the simulation stop callback in the :class:`isaaclab.sim.SimulationContext` class
  to not render when an exception is raised. The while loop in the callback was preventing the application
  from closing when an exception was raised.


0.10.1 (2023-12-06)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added command manager class with terms defined by :class:`isaaclab.managers.CommandTerm`. This
  allow for multiple types of command generators to be used in the same environment.


0.10.0 (2023-12-04)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Modified the sensor and asset base classes to use the underlying PhysX views instead of Isaac Sim views.
  Using Isaac Sim classes led to a very high load time (of the order of minutes) when using a scene with
  many assets. This is because Isaac Sim supports USD paths which are slow and not required.

Added
^^^^^

* Added faster implementation of USD stage traversal methods inside the :class:`isaaclab.sim.utils` module.
* Added properties :attr:`isaaclab.assets.AssetBase.num_instances` and
  :attr:`isaaclab.sensor.SensorBase.num_instances` to obtain the number of instances of the asset
  or sensor in the simulation respectively.

Removed
^^^^^^^

* Removed dependencies on Isaac Sim view classes. It is no longer possible to use :attr:`root_view` and
  :attr:`body_view`. Instead use :attr:`root_physx_view` and :attr:`body_physx_view` to access the underlying
  PhysX views.


0.9.55 (2023-12-03)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the Nucleus directory path in the :attr:`isaaclab.utils.assets.NVIDIA_NUCLEUS_DIR`.
  Earlier, it was referring to the ``NVIDIA/Assets`` directory instead of ``NVIDIA``.


0.9.54 (2023-11-29)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed pose computation in the :class:`isaaclab.sensors.Camera` class to obtain them from XFormPrimView
  instead of using ``UsdGeomCamera.ComputeLocalToWorldTransform`` method. The latter is not updated correctly
  during GPU simulation.
* Fixed initialization of the annotator info in the class :class:`isaaclab.sensors.Camera`. Previously
  all dicts had the same memory address which caused all annotators to have the same info.
* Fixed the conversion of ``uint32`` warp arrays inside the :meth:`isaaclab.utils.array.convert_to_torch`
  method. PyTorch does not support this type, so it is converted to ``int32`` before converting to PyTorch tensor.
* Added render call inside :meth:`isaaclab.sim.SimulationContext.reset` to initialize Replicator
  buffers when the simulation is reset.


0.9.53 (2023-11-29)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the behavior of passing :obj:`None` to the :class:`isaaclab.actuators.ActuatorBaseCfg`
  class. Earlier, they were resolved to fixed default values. Now, they imply that the values are loaded
  from the USD joint drive configuration.

Added
^^^^^

* Added setting of joint armature and friction quantities to the articulation class.


0.9.52 (2023-11-29)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the warning print in :meth:`isaaclab.sim.utils.apply_nested` method
  to be more descriptive. Earlier, it was printing a warning for every instanced prim.
  Now, it only prints a warning if it could not apply the attribute to any of the prims.

Added
^^^^^

* Added the method :meth:`isaaclab.utils.assets.retrieve_file_path` to
  obtain the absolute path of a file on the Nucleus server or locally.

Fixed
^^^^^

* Fixed hiding of STOP button in the :class:`AppLauncher` class when running the
  simulation in headless mode.
* Fixed a bug with :meth:`isaaclab.sim.utils.clone` failing when the input prim path
  had no parent (example: "/Table").


0.9.51 (2023-11-29)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the :meth:`isaaclab.sensor.SensorBase.update` method to always recompute the buffers if
  the sensor is in visualization mode.

Added
^^^^^

* Added available entities to the error message when accessing a non-existent entity in the
  :class:`InteractiveScene` class.
* Added a warning message when the user tries to reference an invalid prim in the :class:`FrameTransformer` sensor.


0.9.50 (2023-11-28)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Hid the ``STOP`` button in the UI when running standalone Python scripts. This is to prevent
  users from accidentally clicking the button and stopping the simulation. They should only be able to
  play and pause the simulation from the UI.

Removed
^^^^^^^

* Removed :attr:`isaaclab.sim.SimulationCfg.shutdown_app_on_stop`. The simulation is always rendering
  if it is stopped from the UI. The user needs to close the window or press ``Ctrl+C`` to close the simulation.


0.9.49 (2023-11-27)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added an interface class, :class:`isaaclab.managers.ManagerTermBase`, to serve as the parent class
  for term implementations that are functional classes.
* Adapted all managers to support terms that are classes and not just functions clearer. This allows the user to
  create more complex terms that require additional state information.


0.9.48 (2023-11-24)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed initialization of drift in the :class:`isaaclab.sensors.RayCasterCamera` class.


0.9.47 (2023-11-24)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Automated identification of the root prim in the :class:`isaaclab.assets.RigidObject` and
  :class:`isaaclab.assets.Articulation` classes. Earlier, the root prim was hard-coded to
  the spawn prim path. Now, the class searches for the root prim under the spawn prim path.


0.9.46 (2023-11-24)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed a critical issue in the asset classes with writing states into physics handles.
  Earlier, the states were written over all the indices instead of the indices of the
  asset that were being updated. This caused the physics handles to refresh the states
  of all the assets in the scene, which is not desirable.


0.9.45 (2023-11-24)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`isaaclab.command_generators.UniformPoseCommandGenerator` to generate
  poses in the asset's root frame by uniformly sampling from a given range.


0.9.44 (2023-11-16)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added methods :meth:`reset` and :meth:`step` to the :class:`isaaclab.envs.BaseEnv`. This unifies
  the environment interface for simple standalone applications with the class.


0.9.43 (2023-11-16)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Replaced subscription of physics play and stop events in the :class:`isaaclab.assets.AssetBase` and
  :class:`isaaclab.sensors.SensorBase` classes with subscription to time-line play and stop events.
  This is to prevent issues in cases where physics first needs to perform mesh cooking and handles are not
  available immediately. For instance, with deformable meshes.


0.9.42 (2023-11-16)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed setting of damping values from the configuration for :class:`ActuatorBase` class. Earlier,
  the stiffness values were being set into damping when a dictionary configuration was passed to the
  actuator model.
* Added dealing with :class:`int` and :class:`float` values in the configurations of :class:`ActuatorBase`.
  Earlier, a type-error was thrown when integer values were passed to the actuator model.


0.9.41 (2023-11-16)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the naming and shaping issues in the binary joint action term.


0.9.40 (2023-11-09)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Simplified the manual initialization of Isaac Sim :class:`ArticulationView` class. Earlier, we basically
  copied the code from the Isaac Sim source code. Now, we just call their initialize method.

Changed
^^^^^^^

* Changed the name of attribute :attr:`default_root_state_w` to :attr:`default_root_state`. The latter is
  more correct since the data is actually in the local environment frame and not the simulation world frame.


0.9.39 (2023-11-08)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Changed the reference of private ``_body_view`` variable inside the :class:`RigidObject` class
  to the public ``body_view`` property. For a rigid object, the private variable is not defined.


0.9.38 (2023-11-07)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Upgraded the :class:`isaaclab.envs.RLTaskEnv` class to support Gym 0.29.0 environment definition.

Added
^^^^^

* Added computation of ``time_outs`` and ``terminated`` signals inside the termination manager. These follow the
  definition mentioned in `Gym 0.29.0 <https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/>`_.
* Added proper handling of observation and action spaces in the :class:`isaaclab.envs.RLTaskEnv` class.
  These now follow closely to how Gym VecEnv handles the spaces.


0.9.37 (2023-11-06)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed broken visualization in :mod:`isaaclab.sensors.FrameTramsformer` class by overwriting the
  correct ``_debug_vis_callback`` function.
* Moved the visualization marker configurations of sensors to their respective sensor configuration classes.
  This allows users to set these configurations from the configuration object itself.


0.9.36 (2023-11-03)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added explicit deleting of different managers in the :class:`isaaclab.envs.BaseEnv` and
  :class:`isaaclab.envs.RLTaskEnv` classes. This is required since deleting the managers
  is order-sensitive (many managers need to be deleted before the scene is deleted).


0.9.35 (2023-11-02)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the error: ``'str' object has no attribute '__module__'`` introduced by adding the future import inside the
  :mod:`isaaclab.utils.warp.kernels` module. Warp language does not support the ``__future__`` imports.


0.9.34 (2023-11-02)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added missing import of ``from __future__ import annotations`` in the :mod:`isaaclab.utils.warp`
  module. This is needed to have a consistent behavior across Python versions.


0.9.33 (2023-11-02)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the :class:`isaaclab.command_generators.NullCommandGenerator` class. Earlier,
  it was having a runtime error due to infinity in the resampling time range. Now, the class just
  overrides the parent methods to perform no operations.


0.9.32 (2023-11-02)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Renamed the :class:`isaaclab.envs.RLEnv` class to :class:`isaaclab.envs.RLTaskEnv` to
  avoid confusions in terminologies between environments and tasks.


0.9.31 (2023-11-02)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :class:`isaaclab.sensors.RayCasterCamera` class, as a ray-casting based camera for
  "distance_to_camera", "distance_to_image_plane" and "normals" annotations. It has the same interface and
  functionalities as the USD Camera while it is on average 30% faster.


0.9.30 (2023-11-01)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added skipping of None values in the :class:`InteractiveScene` class when creating the scene from configuration
  objects. Earlier, it was throwing an error when the user passed a None value for a scene element.
* Added ``kwargs`` to the :class:`RLEnv` class to allow passing additional arguments from gym registry function.
  This is now needed since the registry function passes args beyond the ones specified in the constructor.


0.9.29 (2023-11-01)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the material path resolution inside the :class:`isaaclab.sim.converters.UrdfConverter` class.
  With Isaac Sim 2023.1, the material paths from the importer are always saved as absolute paths. This caused
  issues when the generated USD file was moved to a different location. The fix now resolves the material paths
  relative to the USD file location.


0.9.28 (2023-11-01)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the way the :func:`isaaclab.sim.spawners.from_files.spawn_ground_plane` function sets the
  height of the ground. Earlier, it was reading the height from the configuration object. Now, it expects the
  desired transformation as inputs to the function. This makes it consistent with the other spawner functions.


0.9.27 (2023-10-31)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Removed the default value of the argument ``camel_case`` in setters of USD attributes. This is to avoid
  confusion with the naming of the attributes in the USD file.

Fixed
^^^^^

* Fixed the selection of material prim in the :class:`isaaclab.sim.spawners.materials.spawn_preview_surface`
  method. Earlier, the created prim was being selected in the viewport which interfered with the selection of
  prims by the user.
* Updated :class:`isaaclab.sim.converters.MeshConverter` to use a different stage than the default stage
  for the conversion. This is to avoid the issue of the stage being closed when the conversion is done.


0.9.26 (2023-10-31)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the sensor implementation for :class:`isaaclab.sensors.FrameTransformer` class. Currently,
  it handles obtaining the transformation between two frames in the same articulation.


0.9.25 (2023-10-27)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :mod:`isaaclab.envs.ui` module to put all the UI-related classes in one place. This currently
  implements the :class:`isaaclab.envs.ui.BaseEnvWindow` and :class:`isaaclab.envs.ui.RLEnvWindow`
  classes. Users can inherit from these classes to create their own UI windows.
* Added the attribute :attr:`isaaclab.envs.BaseEnvCfg.ui_window_class_type` to specify the UI window class
  to be used for the environment. This allows the user to specify their own UI window class to be used for the
  environment.


0.9.24 (2023-10-27)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the behavior of setting up debug visualization for assets, sensors and command generators.
  Earlier it was raising an error if debug visualization was not enabled in the configuration object.
  Now it checks whether debug visualization is implemented and only sets up the callback if it is
  implemented.


0.9.23 (2023-10-27)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed a typo in the :class:`AssetBase` and :class:`SensorBase` that effected the class destructor.
  Earlier, a tuple was being created in the constructor instead of the actual object.


0.9.22 (2023-10-26)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a :class:`isaaclab.command_generators.NullCommandGenerator` class for no command environments.
  This is easier to work with than having checks for :obj:`None` in the command generator.

Fixed
^^^^^

* Moved the randomization manager to the :class:`isaaclab.envs.BaseEnv` class with the default
  settings to reset the scene to the defaults specified in the configurations of assets.
* Moved command generator to the :class:`isaaclab.envs.RlEnv` class to have all task-specification
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

* Changed naming in :class:`isaaclab.sim.SimulationContext.RenderMode` to use ``NO_GUI_OR_RENDERING``
  and ``NO_RENDERING`` instead of ``HEADLESS`` for clarity.
* Changed :class:`isaaclab.sim.SimulationContext` to be capable of handling livestreaming and
  offscreen rendering.
* Changed :class:`isaaclab.app.AppLauncher` envvar ``VIEWPORT_RECORD`` to the more descriptive
  ``OFFSCREEN_RENDER``.


0.9.19 (2023-10-25)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added Gym observation and action spaces for the :class:`isaaclab.envs.RLEnv` class.


0.9.18 (2023-10-23)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created :class:`isaaclab.sim.converters.asset_converter.AssetConverter` to serve as a base
  class for all asset converters.
* Added :class:`isaaclab.sim.converters.mesh_converter.MeshConverter` to handle loading and conversion
  of mesh files (OBJ, STL and FBX) into USD format.
* Added script ``convert_mesh.py`` to ``source/tools`` to allow users to convert a mesh to USD via command line arguments.

Changed
^^^^^^^

* Renamed the submodule :mod:`isaaclab.sim.loaders` to :mod:`isaaclab.sim.converters` to be more
  general with the functionality of the module.
* Updated ``check_instanceable.py`` script to convert relative paths to absolute paths.


0.9.17 (2023-10-22)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added setters and getters for term configurations in the :class:`RandomizationManager`, :class:`RewardManager`
  and :class:`TerminationManager` classes. This allows the user to modify the term configurations after the
  manager has been created.
* Added the method :meth:`compute_group` to the :class:`isaaclab.managers.ObservationManager` class to
  compute the observations for only a given group.
* Added the curriculum term for modifying reward weights after certain environment steps.


0.9.16 (2023-10-22)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for keyword arguments for terms in the :class:`isaaclab.managers.ManagerBase`.

Fixed
^^^^^

* Fixed resetting of buffers in the :class:`TerminationManager` class. Earlier, the values were being set
  to ``0.0`` instead of ``False``.


0.9.15 (2023-10-22)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added base yaw heading and body acceleration into :class:`isaaclab.assets.RigidObjectData` class.
  These quantities are computed inside the :class:`RigidObject` class.

Fixed
^^^^^

* Fixed the :meth:`isaaclab.assets.RigidObject.set_external_force_and_torque` method to correctly
  deal with the body indices.
* Fixed a bug in the :meth:`isaaclab.utils.math.wrap_to_pi` method to prevent self-assignment of
  the input tensor.


0.9.14 (2023-10-21)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added 2-D drift (i.e. along x and y) to the :class:`isaaclab.sensors.RayCaster` class.
* Added flags to the :class:`isaaclab.sensors.ContactSensorCfg` to optionally obtain the
  sensor origin and air time information. Since these are not required by default, they are
  disabled by default.

Fixed
^^^^^

* Fixed the handling of contact sensor history buffer in the :class:`isaaclab.sensors.ContactSensor` class.
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
* Fixed bug in applying actuator model in the :class:`isaaclab.asset.Articulation` class. The new
  implementation caches the outputs from explicit actuator model into the ``joint_pos_*_sim`` buffer to
  avoid feedback loops in the tensor operation.


0.9.11 (2023-10-17)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the support for semantic tags into the :class:`isaaclab.sim.spawner.SpawnerCfg` class. This allows
  the user to specify the semantic tags for a prim when spawning it into the scene. It follows the same format as
  Omniverse Replicator.


0.9.10 (2023-10-16)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``--livestream`` and ``--ros`` CLI args to :class:`isaaclab.app.AppLauncher` class.
* Added a static function :meth:`isaaclab.app.AppLauncher.add_app_launcher_args`, which
  appends the arguments needed for :class:`isaaclab.app.AppLauncher` to the argument parser.

Changed
^^^^^^^

* Within :class:`isaaclab.app.AppLauncher`, removed ``REMOTE_DEPLOYMENT`` env-var processing
  in the favor of ``HEADLESS`` and ``LIVESTREAM`` env-vars. These have clearer uses and better parity
  with the CLI args.


0.9.9 (2023-10-12)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the property :attr:`isaaclab.assets.Articulation.is_fixed_base` to the articulation class to
  check if the base of the articulation is fixed or floating.
* Added the task-space action term corresponding to the differential inverse-kinematics controller.

Fixed
^^^^^

* Simplified the :class:`isaaclab.controllers.DifferentialIKController` to assume that user provides the
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

* Modified the :class:`isaaclab.markers.VisualizationMarkers` to use the
  :class:`isaaclab.sim.spawner.SpawnerCfg` class instead of their
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

* Added :class:`isaaclab.scene.InteractiveScene`, as the central scene unit that contains all entities
  that are part of the simulation. These include the terrain, sensors, articulations, rigid objects etc.
  The scene groups the common operations of these entities and allows to access them via their unique names.
* Added :mod:`isaaclab.envs` module that contains environment definitions that encapsulate the different
  general (scene, action manager, observation manager) and RL-specific (reward and termination manager) managers.
* Added :class:`isaaclab.managers.SceneEntityCfg` to handle which scene elements are required by the
  manager's terms. This allows the manager to parse useful information from the scene elements, such as the
  joint and body indices, and pass them to the term.
* Added :class:`isaaclab.sim.SimulationContext.RenderMode` to handle different rendering modes based on
  what the user wants to update (viewport, cameras, or UI elements).

Fixed
^^^^^

* Fixed the :class:`isaaclab.command_generators.CommandGeneratorBase` to register a debug visualization
  callback similar to how sensors and robots handle visualization.


0.9.3 (2023-08-23)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Enabled the `faulthander <https://docs.python.org/3/library/faulthandler.html>`_ to catch segfaults and print
  the stack trace. This is enabled by default in the :class:`isaaclab.app.AppLauncher` class.

Fixed
^^^^^

* Re-added the :mod:`isaaclab.utils.kit` to the ``compat`` directory and fixed all the references to it.
* Fixed the deletion of Replicator nodes for the :class:`isaaclab.sensors.Camera` class. Earlier, the
  Replicator nodes were not being deleted when the camera was deleted. However, this does not prevent the random
  crashes that happen when the camera is deleted.
* Fixed the :meth:`isaaclab.utils.math.convert_quat` to support both numpy and torch tensors.

Changed
^^^^^^^

* Renamed all the scripts inside the ``test`` directory to follow the convention:

  * ``test_<module_name>.py``: Tests for the module ``<module_name>`` using unittest.
  * ``check_<module_name>``: Check for the module ``<module_name>`` using python main function.


0.9.2 (2023-08-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the ability to color meshes in the :class:`isaaclab.terrain.TerrainGenerator` class. Currently,
  it only supports coloring the mesh randomly (``"random"``), based on the terrain height (``"height"``), and
  no coloring (``"none"``).

Fixed
^^^^^

* Modified the :class:`isaaclab.terrain.TerrainImporter` class to configure visual and physics materials
  based on the configuration object.


0.9.1 (2023-08-18)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Introduced three different rotation conventions in the :class:`isaaclab.sensors.Camera` class. These
  conventions are:

  * ``opengl``: the camera is looking down the -Z axis with the +Y axis pointing up
  * ``ros``: the camera is looking down the +Z axis with the +Y axis pointing down
  * ``world``: the camera is looking along the +X axis with the -Z axis pointing down

  These can be used to declare the camera offset in :class:`isaaclab.sensors.CameraCfg.OffsetCfg` class
  and in :meth:`isaaclab.sensors.Camera.set_world_pose` method. Additionally, all conventions are
  saved to :class:`isaaclab.sensors.CameraData` class for easy access.

Changed
^^^^^^^

* Adapted all the sensor classes to follow a structure similar to the :class:`isaaclab.assets.AssetBase`.
  Hence, the spawning and initialization of sensors manually by the users is avoided.
* Removed the :meth:`debug_vis` function since that this functionality is handled by a render callback automatically
  (based on the passed configuration for the :class:`isaaclab.sensors.SensorBaseCfg.debug_vis` flag).


0.9.0 (2023-08-18)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Introduces a new set of asset interfaces. These interfaces simplify the spawning of assets into the scene
  and initializing the physics handle by putting that inside post-startup physics callbacks. With this, users
  no longer need to worry about the :meth:`spawn` and :meth:`initialize` calls.
* Added utility methods to :mod:`isaaclab.utils.string` module that resolve regex expressions based
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

* Added other properties provided by ``PhysicsScene`` to the :class:`isaaclab.sim.SimulationContext`
  class to allow setting CCD, solver iterations, etc.
* Added commonly used functions to the :class:`SimulationContext` class itself to avoid having additional
  imports from Isaac Sim when doing simple tasks such as setting camera view or retrieving the simulation settings.

Fixed
^^^^^

* Switched the notations of default buffer values in :class:`isaaclab.sim.PhysxCfg` from multiplication
  to scientific notation to avoid confusion with the values.


0.8.11 (2023-08-18)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Adds utility functions and configuration objects in the :mod:`isaaclab.sim.spawners`
  to create the following prims in the scene:

  * :mod:`isaaclab.sim.spawners.from_file`: Create a prim from a USD/URDF file.
  * :mod:`isaaclab.sim.spawners.shapes`: Create USDGeom prims for shapes (box, sphere, cylinder, capsule, etc.).
  * :mod:`isaaclab.sim.spawners.materials`: Create a visual or physics material prim.
  * :mod:`isaaclab.sim.spawners.lights`: Create a USDLux prim for different types of lights.
  * :mod:`isaaclab.sim.spawners.sensors`: Create a USD prim for supported sensors.

Changed
^^^^^^^

* Modified the :class:`SimulationContext` class to take the default physics material using the material spawn
  configuration object.


0.8.10 (2023-08-17)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added methods for defining different physics-based schemas in the :mod:`isaaclab.sim.schemas` module.
  These methods allow creating the schema if it doesn't exist at the specified prim path and modify
  its properties based on the configuration object.


0.8.9 (2023-08-09)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved the :class:`isaaclab.asset_loader.UrdfLoader` class to the :mod:`isaaclab.sim.loaders`
  module to make it more accessible to the user.


0.8.8 (2023-08-09)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configuration classes and functions for setting different physics-based schemas in the
  :mod:`isaaclab.sim.schemas` module. These allow modifying properties of the physics solver
  on the asset using configuration objects.


0.8.7 (2023-08-03)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added support for `__post_init__ <https://docs.python.org/3/library/dataclasses.html#post-init-processing>`_ in
  the :class:`isaaclab.utils.configclass` decorator.


0.8.6 (2023-08-03)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for callable classes in the :class:`isaaclab.managers.ManagerBase`.


0.8.5 (2023-08-03)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the :class:`isaaclab.markers.Visualizationmarkers` class so that the markers are not visible in camera rendering mode.

Changed
^^^^^^^

* Simplified the creation of the point instancer in the :class:`isaaclab.markers.Visualizationmarkers` class. It now creates a new
  prim at the next available prim path if a prim already exists at the given path.


0.8.4 (2023-08-02)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :class:`isaaclab.sim.SimulationContext` class to the :mod:`isaaclab.sim` module.
  This class inherits from the :class:`omni.isaac.core.simulation_context.SimulationContext` class and adds
  the ability to create a simulation context from a configuration object.


0.8.3 (2023-08-02)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved the :class:`ActuatorBase` class to the :mod:`isaaclab.actuators.actuator_base` module.
* Renamed the :mod:`isaaclab.actuators.actuator` module to :mod:`isaaclab.actuators.actuator_pd`
  to make it more explicit that it contains the PD actuator models.


0.8.2 (2023-08-02)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Cleaned up the :class:`isaaclab.terrain.TerrainImporter` class to take all the parameters from the configuration
  object. This makes it consistent with the other classes in the package.
* Moved the configuration classes for terrain generator and terrain importer into separate files to resolve circular
  dependency issues.


0.8.1 (2023-08-02)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added a hack into :class:`isaaclab.app.AppLauncher` class to remove Isaac Lab packages from the path before launching
  the simulation application. This prevents the warning messages that appears when the user launches the ``SimulationApp``.

Added
^^^^^

* Enabled necessary viewport extensions in the :class:`isaaclab.app.AppLauncher` class itself if ``VIEWPORT_ENABLED``
  flag is true.


0.8.0 (2023-07-26)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :class:`ActionManager` class to the :mod:`isaaclab.managers` module to handle actions in the
  environment through action terms.
* Added contact force history to the :class:`isaaclab.sensors.ContactSensor` class. The history is stored
  in the ``net_forces_w_history`` attribute of the sensor data.

Changed
^^^^^^^

* Implemented lazy update of buffers in the :class:`isaaclab.sensors.SensorBase` class. This allows the user
  to update the sensor data only when required, i.e. when the data is requested by the user. This helps avoid double
  computation of sensor data when a reset is called in the environment.

Deprecated
^^^^^^^^^^

* Removed the support for different backends in the sensor class. We only use Pytorch as the backend now.
* Removed the concept of actuator groups. They are now handled by the :class:`isaaclab.managers.ActionManager`
  class. The actuator models are now directly handled by the robot class itself.


0.7.4 (2023-07-26)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the behavior of the :class:`isaaclab.terrains.TerrainImporter` class. It now expects the terrain
  type to be specified in the configuration object. This allows the user to specify everything in the configuration
  object and not have to do an explicit call to import a terrain.

Fixed
^^^^^

* Fixed setting of quaternion orientations inside the :class:`isaaclab.markers.Visualizationmarkers` class.
  Earlier, the orientation was being set into the point instancer in the wrong order (``wxyz`` instead of ``xyzw``).


0.7.3 (2023-07-25)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the issue with multiple inheritance in the :class:`isaaclab.utils.configclass` decorator.
  Earlier, if the inheritance tree was more than one level deep and the lowest level configuration class was
  not updating its values from the middle level classes.


0.7.2 (2023-07-24)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the method :meth:`replace` to the :class:`isaaclab.utils.configclass` decorator to allow
  creating a new configuration object with values replaced from keyword arguments. This function internally
  calls the `dataclasses.replace <https://docs.python.org/3/library/dataclasses.html#dataclasses.replace>`_.

Fixed
^^^^^

* Fixed the handling of class types as member values in the :meth:`isaaclab.utils.configclass`. Earlier it was
  throwing an error since class types were skipped in the if-else block.


0.7.1 (2023-07-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :class:`TerminationManager`, :class:`CurriculumManager`, and :class:`RandomizationManager` classes
  to the :mod:`isaaclab.managers` module to handle termination, curriculum, and randomization respectively.


0.7.0 (2023-07-22)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created a new :mod:`isaaclab.managers` module for all the managers related to the environment / scene.
  This includes the :class:`isaaclab.managers.ObservationManager` and :class:`isaaclab.managers.RewardManager`
  classes that were previously in the :mod:`isaaclab.utils.mdp` module.
* Added the :class:`isaaclab.managers.ManagerBase` class to handle the creation of managers.
* Added configuration classes for :class:`ObservationTermCfg` and :class:`RewardTermCfg` to allow easy creation of
  observation and reward terms.

Changed
^^^^^^^

* Changed the behavior of :class:`ObservationManager` and :class:`RewardManager` classes to accept the key ``func``
  in each configuration term to be a callable. This removes the need to inherit from the base class
  and allows more reusability of the functions across different environments.
* Moved the old managers to the :mod:`isaaclab.compat.utils.mdp` module.
* Modified the necessary scripts to use the :mod:`isaaclab.compat.utils.mdp` module.


0.6.2 (2023-07-21)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :mod:`isaaclab.command_generators` to generate different commands based on the desired task.
  It allows the user to generate commands for different tasks in the same environment without having to write
  custom code for each task.


0.6.1 (2023-07-16)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the :meth:`isaaclab.utils.math.quat_apply_yaw` to compute the yaw quaternion correctly.

Added
^^^^^

* Added functions to convert string and callable objects in :mod:`isaaclab.utils.string`.


0.6.0 (2023-07-16)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the argument :attr:`sort_keys` to the :meth:`isaaclab.utils.io.yaml.dump_yaml` method to allow
  enabling/disabling of sorting of keys in the output yaml file.

Fixed
^^^^^

* Fixed the ordering of terms in :mod:`isaaclab.utils.configclass` to be consistent in the order in which
  they are defined. Previously, the ordering was done alphabetically which made it inconsistent with the order in which
  the parameters were defined.

Changed
^^^^^^^

* Changed the default value of the argument :attr:`sort_keys` in the :meth:`isaaclab.utils.io.yaml.dump_yaml`
  method to ``False``.
* Moved the old config classes in :mod:`isaaclab.utils.configclass` to
  :mod:`isaaclab.compat.utils.configclass` so that users can still run their old code where alphabetical
  ordering was used.


0.5.0 (2023-07-04)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a generalized :class:`isaaclab.sensors.SensorBase` class that leverages the ideas of views to
  handle multiple sensors in a single class.
* Added the classes :class:`isaaclab.sensors.RayCaster`, :class:`isaaclab.sensors.ContactSensor`,
  and :class:`isaaclab.sensors.Camera` that output a batched tensor of sensor data.

Changed
^^^^^^^

* Renamed the parameter ``sensor_tick`` to ``update_freq`` to make it more intuitive.
* Moved the old sensors in :mod:`isaaclab.sensors` to :mod:`isaaclab.compat.sensors`.
* Modified the standalone scripts to use the :mod:`isaaclab.compat.sensors` module.


0.4.4 (2023-07-05)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the :meth:`isaaclab.terrains.trimesh.utils.make_plane` method to handle the case when the
  plane origin does not need to be centered.
* Added the :attr:`isaaclab.terrains.TerrainGeneratorCfg.seed` to make generation of terrains reproducible.
  The default value is ``None`` which means that the seed is not set.

Changed
^^^^^^^

* Changed the saving of ``origins`` in :class:`isaaclab.terrains.TerrainGenerator` class to be in CSV format
  instead of NPY format.


0.4.3 (2023-06-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :class:`isaaclab.markers.PointInstancerMarker` class that wraps around
  `UsdGeom.PointInstancer <https://graphics.pixar.com/usd/dev/api/class_usd_geom_point_instancer.html>`_
  to directly work with torch and numpy arrays.

Changed
^^^^^^^

* Moved the old markers in :mod:`isaaclab.markers` to :mod:`isaaclab.compat.markers`.
* Modified the standalone scripts to use the :mod:`isaaclab.compat.markers` module.


0.4.2 (2023-06-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the sub-module :mod:`isaaclab.terrains` to allow procedural generation of terrains and supporting
  importing of terrains from different sources (meshes, usd files or default ground plane).


0.4.1 (2023-06-27)
~~~~~~~~~~~~~~~~~~

* Added the :class:`isaaclab.app.AppLauncher` class to allow controlled instantiation of
  the `SimulationApp <https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html>`_
  and extension loading for remote deployment and ROS bridges.

Changed
^^^^^^^

* Modified all standalone scripts to use the :class:`isaaclab.app.AppLauncher` class.


0.4.0 (2023-05-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a helper class :class:`isaaclab.asset_loader.UrdfLoader` that converts a URDF file to instanceable USD
  file based on the input configuration object.


0.3.2 (2023-04-27)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added safe-printing of functions while using the :meth:`isaaclab.utils.dict.print_dict` function.


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

* Fixed bugs in :meth:`axis_angle_from_quat` in the ``isaaclab.utils.math`` to handle quaternion with negative w component.
* Fixed bugs in :meth:`subtract_frame_transforms` in the ``isaaclab.utils.math`` by adding the missing final rotation.


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

* Added :meth:`apply_nested_physics_material` to the ``isaaclab.utils.kit``.
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

* Fixed setting of physics material on the ground plane when using :meth:`isaaclab.utils.kit.create_ground_plane` function.


0.1.0 (2023-01-17)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of the extension with experimental API.
* Available robot configurations:

  * **Quadrupeds:** Unitree A1, ANYmal B, ANYmal C
  * **Single-arm manipulators:** Franka Emika arm, UR5
  * **Mobile manipulators:** Clearpath Ridgeback with Franka Emika arm or UR5
