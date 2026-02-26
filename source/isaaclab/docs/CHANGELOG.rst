Changelog
---------

4.2.1 (2026-02-25)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Migrated all MDP action terms to use new ``_index`` write/set APIs with keyword-only arguments.

* Migrated all MDP event terms to use new ``_index`` write/set APIs (mass, inertia, COM,
  joint properties, root state resets, fixed tendon parameters).

* Updated ``InteractiveScene.set_state`` to use new ``_index`` APIs for root pose/velocity
  and joint state writes.

* Updated ``SceneEntityCfg`` body resolution to use ``find_sensors``/``num_sensors`` for
  ContactSensor entities.


4.2.0 (2026-02-24)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Refined base asset class abstractions (:class:`~isaaclab.assets.BaseArticulation`,
  :class:`~isaaclab.assets.BaseRigidObject`) to better support multiple backends.
  Removed abstract method requirements that forced unnecessary boilerplate in backend
  implementations, making it easier to add new physics backends.

* Unified docstrings across all base asset classes with precise shape and dtype annotations
  for warp array properties and write methods, ensuring consistent documentation between
  PhysX and Newton backend implementations.


4.1.0 (2026-02-18)
~~~~~~~~~~~~~~~~~~

Removed
^^^^^^^

* Removed hard dependency on the Isaac Sim Cloner for scene replication. Replication now uses internal utilities
  :func:`~isaaclab.scene.cloner.usd_replicate` and :func:`~isaaclab.scene.cloner.physx_replicate`, reducing coupling
  to Isaac Sim. Public APIs in :class:`~isaaclab.scene.interactive_scene.InteractiveScene` remain unchanged; code
  directly importing the external Cloner should migrate to these utilities.

Added
^^^^^

* Added optional random prototype selection during environment cloning in
  :class:`~isaaclab.scene.interactive_scene.InteractiveScene` via
  :attr:`~isaaclab.scene.interactive_scene_cfg.InteractiveSceneCfg.random_heterogeneous_cloning`.
  Defaults to ``True``; round-robin (modulo) mapping remains available by setting it to ``False``.

* Added flexible per-object cloning path in
  :class:`~isaaclab.scene.interactive_scene.InteractiveScene`: when environments are heterogeneous
  (different prototypes across envs), replication switches to per-object instead of whole-env cloning.
  This reduces PhysX cloning time in heterogeneous scenes.


4.0.0 (2026-02-22)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated :class:`~isaaclab.sim.converters.MjcfConverter` and
  :class:`~isaaclab.sim.converters.MjcfConverterCfg` for the rewritten MJCF importer in Isaac Sim 5.0.
  The converter now uses the ``MJCFImporter`` / ``MJCFImporterConfig`` API backed by the
  ``mujoco-usd-converter`` library. The old settings ``fix_base``, ``link_density``,
  ``import_inertia_tensor``, ``import_sites``, and ``make_instanceable`` have been removed
  (handled automatically by the new converter). New settings ``merge_mesh``,
  ``collision_from_visuals``, and ``collision_type`` have been added. The ``convert_mjcf.py``
  CLI tool has been updated accordingly. Note that the new importer produces assets with nested
  rigid bodies (``RigidBodyAPI`` applied per link) instead of a flat hierarchy.



3.5.3 (2026-02-22)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Refactored ``SimulationContext.clear_instance`` to delegate stage teardown to
  :func:`~isaaclab.sim.utils.close_stage` instead of manually clearing the stage cache,
  thread-local context, and Kit USD context inline.
* Updated :func:`~isaaclab.sim.utils.close_stage` to also close the Kit USD context stage
  (``omni.usd.get_context().close_stage()``) when Kit is running, making it a complete
  stage teardown function.


3.5.2 (2026-02-23)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* ``NUCLEUS_ASSET_ROOT_DIR`` and derived Nucleus path constants are now parsed from ``apps/isaaclab.python.kit``


3.5.1 (2026-02-21)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed ``omni.usd`` calls with pure USD (``pxr``) equivalents in sim utils and sensors.

Deprecated
^^^^^^^^^^

* ``create_new_stage_in_memory`` — use ``create_new_stage`` instead.
* ``is_stage_loading`` — Kit-only, no production callers.


3.5.0 (2026-02-21)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* The in-memory stage created with ``SimulationCfg(create_stage_in_memory=True)`` is now automatically
  attached to the USD context at :class:`~isaaclab.sim.SimulationContext` creation. This ensures proper
  stage lifecycle events for viewport and physics systems, preventing test isolation issues.


3.4.3 (2026-02-22)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Migrated settings access from ``carb.settings`` to :class:`~isaaclab.app.settings_manager.SettingsManager`.
  Application code and tests now use :func:`~isaaclab.app.settings_manager.get_settings_manager` or
  :meth:`~isaaclab.sim.SimulationContext.get_setting` / :meth:`~isaaclab.sim.SimulationContext.set_setting`
  instead of ``carb.settings.get_settings()``.


3.4.2 (2026-02-20)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Replaced PhysX schema interactions via ``pxr.PhysxSchema`` API helpers with direct prim schema apply/get calls.
* Replaced ``omni.kit.commands.execute("ChangePropertyCommand")`` uses with direct ``CreateAttribute`` + ``Set`` calls.


Removed
^^^^^^^

* Removed :func:`~isaaclab.sim.utils.attach_stage_to_usd_context`. This function is no longer needed
  since the in-memory stage is now automatically attached to the USD context at ``SimulationContext``
  creation. Remove any calls to this function from your code.

Fixed
^^^^^

* Fixed :func:`~isaaclab.sim.utils.add_labels` to use :class:`UsdSemantics.LabelsAPI` directly
  instead of the Replicator API for Isaac Sim 5.0+. This resolves ``'NoneType' object has no
  attribute 'GetEditTarget'`` errors when using stage-in-memory mode.


3.4.0 (2026-02-18)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Integrated TeleopCore as the teleoperation backend via the new :mod:`isaaclab_teleop` extension.
  The :class:`~isaaclab_teleop.IsaacTeleopDevice` provides a unified teleoperation interface that
  replaces the previous XR-specific device and retargeter classes.

Deprecated
^^^^^^^^^^

* Deprecated the existing XR teleoperation solution. :class:`~isaaclab.devices.openxr.OpenXRDevice`,
  :class:`~isaaclab.devices.openxr.OpenXRDeviceCfg`, :class:`~isaaclab.devices.openxr.ManusVive`,
  :class:`~isaaclab.devices.RetargeterBase`, :class:`~isaaclab.devices.RetargeterCfg`, and all
  retargeters under :mod:`isaaclab.devices.openxr.retargeters` are deprecated in favor of
  :class:`~isaaclab_teleop.IsaacTeleopDevice`. Existing imports will continue to work but emit
  :class:`DeprecationWarning` when ``isaaclab_teleop`` is installed.


3.3.0 (2026-02-13)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Migrated all asset and sensor data properties from ``torch.Tensor`` to ``wp.array`` (NVIDIA Warp).
  All ``.data.*`` properties on :class:`~isaaclab.assets.Articulation`,
  :class:`~isaaclab.assets.RigidObject`, :class:`~isaaclab.assets.RigidObjectCollection`, and
  sensor data classes now return ``wp.array``. Use ``wp.to_torch()`` to convert back to
  ``torch.Tensor`` when needed.

* Split all asset write methods into ``_index`` and ``_mask`` variants. The old ``env_ids``
  parameter has been replaced by explicit ``_index`` methods (sparse indexed data) and ``_mask``
  methods (full data with boolean mask). For example,
  ``write_root_link_pose_to_sim(data, env_ids)`` is now
  ``write_root_link_pose_to_sim_index(data, env_ids)`` or
  ``write_root_link_pose_to_sim_mask(data, env_mask)``.

* Refactored :mod:`isaaclab.utils.wrench_composer` to use warp kernels internally.

* Updated all MDP action, observation, reward, termination, command, and event functions
  to wrap ``wp.array`` data accesses with ``wp.to_torch()`` for torch compatibility.

* Updated mock interfaces for all assets and sensors to produce ``wp.array``-backed data.

Added
^^^^^

* Added :class:`~isaaclab.utils.buffers.TimestampedBufferWarp` for warp-native timestamped
  data buffers, replacing ``TimestampedBuffer`` in warp-backed asset and sensor classes.

* Added shared warp math kernels in :mod:`isaaclab.utils.warp.kernels` for quaternion
  operations, coordinate transforms, and velocity computations.


3.2.0 (2026-02-06)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Refactored :class:`~isaaclab.sim.SimulationContext` to use :class:`~isaaclab.physics.PhysicsManager`
  abstraction layer for cleaner separation between simulation orchestration and physics backend.


3.1.0 (2026-02-05)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab.test.benchmark.BaseIsaacLabBenchmark` class replacing dependency on
  ``isaacsim.benchmark.services`` for benchmarking workflows. This provides a standalone framework
  for measuring performance of Isaac Lab components.

* Added measurement types in :mod:`isaaclab.test.benchmark.measurements`:

  * :class:`~isaaclab.test.benchmark.SingleMeasurement`: Single floating-point measurement with unit.
  * :class:`~isaaclab.test.benchmark.StatisticalMeasurement`: Mean, std, and sample count.
  * :class:`~isaaclab.test.benchmark.DictMeasurement`: Dictionary-valued measurement.
  * :class:`~isaaclab.test.benchmark.ListMeasurement`: List-valued measurement.
  * :class:`~isaaclab.test.benchmark.BooleanMeasurement`: Boolean measurement.

* Added metadata types in :mod:`isaaclab.test.benchmark.measurements`:

  * :class:`~isaaclab.test.benchmark.StringMetadata`: String metadata.
  * :class:`~isaaclab.test.benchmark.IntMetadata`: Integer metadata.
  * :class:`~isaaclab.test.benchmark.FloatMetadata`: Float metadata.
  * :class:`~isaaclab.test.benchmark.DictMetadata`: Dictionary metadata.

* Added :class:`~isaaclab.test.benchmark.TestPhase` for organizing measurements and metadata
  into logical phases within a benchmark.

* Added :class:`~isaaclab.test.benchmark.BenchmarkMonitor` context manager for async system
  resource monitoring during blocking operations like RL training loops.

* Added pluggable backend architecture in :mod:`isaaclab.test.benchmark.backends`:

  * ``json``: Full JSON output with all phases, measurements, and metadata.
  * ``osmo``: Osmo KPI format for CI/CD integration.
  * ``omniperf``: OmniPerf format for database upload.

* Added system recorders in :mod:`isaaclab.test.benchmark.recorders`:

  * :class:`~isaaclab.test.benchmark.recorders.CPUInfoRecorder`: CPU information capture.
  * :class:`~isaaclab.test.benchmark.recorders.GPUInfoRecorder`: GPU information capture.
  * :class:`~isaaclab.test.benchmark.recorders.MemoryInfoRecorder`: Memory usage tracking.
  * :class:`~isaaclab.test.benchmark.recorders.VersionInfoRecorder`: Software version capture.

* Added CLI arguments for benchmark scripts: ``--benchmark_backend``, ``--output_path``.

* Added shell scripts for running benchmark suites:

  * ``scripts/benchmarks/run_non_rl_benchmarks.sh``: Non-RL environment stepping benchmarks.
  * ``scripts/benchmarks/run_physx_benchmarks.sh``: PhysX micro-benchmarks.
  * ``scripts/benchmarks/run_training_benchmarks.sh``: RL training benchmarks.

Changed
^^^^^^^

* Refactored benchmark scripts to use new :class:`~isaaclab.test.benchmark.BaseIsaacLabBenchmark`
  class instead of ``isaacsim.benchmark.services``.

Removed
^^^^^^^

* Removed hard dependency on ``isaacsim.benchmark.services`` extension for benchmarking.
  The extension is now optional and only used for frametime recorders when available.


3.0.3 (2026-02-05)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Modified all the base classes so that they implement the shorthands and the deprecation cycle to IsaacLab 4.0


3.0.2 (2026-02-04)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Removed exact version pinning for URDF asset importer extension that is incompatible with Isaac Sim 6.0.


3.0.1 (2026-02-04)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :file:`isaaclab.sh` to use libstdc++ CXXABI_1.3.15 from conda for systems that lack that version (e.g., Ubuntu 22.04).


3.0.0 (2026-02-02)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added albedo annotator for faster diffuse albedo rendering. This path will be the most performant when GUI is not required and only albedo and/or depth annotations are requested.

Changed
^^^^^^^

* Updated Isaac Lab to be compatible with Isaac Sim 6.0.0.
* Updated the required Python version to 3.12 for Isaac Lab installation.
* Updated the required PyTorch version to 2.9.0+cu128 and torchvision to 0.24.0 for Isaac Lab installation.
* Updated numpy to 2.3.1 following version in Kit 109.0.
* Updated dex-retargeting to 0.5.0 with numpy 2.0+ dependency.
* Removed explicit URDF importer extension version dependency in :class:`~isaaclab.sim.converters.urdf_converter.UrdfConverter` and related code.


2.1.2 (2026-01-30)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab.test.benchmark` module providing a comprehensive benchmarking framework
  for measuring performance of Isaac Lab components. Includes:

  * :class:`BenchmarkConfig`: Configuration dataclass for benchmark execution parameters
    (iterations, warmup steps, instances, device).
  * :class:`BenchmarkResult`: Dataclass capturing timing statistics (mean, std in microseconds),
    skip status, and dependency information.
  * :class:`MethodBenchmark`: Definition class for methods to benchmark with multi-mode
    input generators.
  * Input generator helpers for creating standardized tensors and Warp masks:
    ``make_tensor_env_ids``, ``make_tensor_joint_ids``, ``make_tensor_body_ids``,
    ``make_warp_env_mask``, ``make_warp_joint_mask``, ``make_warp_body_mask``.
  * :func:`benchmark_method`: Core function for benchmarking with warmup phases,
    GPU synchronization, and graceful error handling.
  * I/O utilities: :func:`get_hardware_info`, :func:`get_git_info`, :func:`print_hardware_info`,
    :func:`print_results`, :func:`export_results_json`, :func:`export_results_csv`.


2.1.1 (2026-02-03)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab.test.mock_interfaces` module providing mock implementations for unit testing
  without requiring Isaac Sim. Includes:

  * Mock assets: :class:`MockArticulation`, :class:`MockRigidObject`, :class:`MockRigidObjectCollection`
    with full state tracking and property management.
  * Mock sensors: :class:`MockContactSensor`, :class:`MockImu`, :class:`MockFrameTransformer`
    with configurable data outputs.
  * Utility classes: :class:`MockArticulationBuilder`, :class:`MockSensorBuilder`,
    :class:`MockWrenchComposer` for flexible mock construction.
  * Factory functions for common robot morphologies (quadruped, humanoid).
  * Patching utilities and decorators for easy test injection.


2.1.0 (2026-02-02)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab.sensors.contact_sensor.BaseContactSensor` and
  :class:`~isaaclab.sensors.contact_sensor.BaseContactSensorData` abstract base classes that define
  the interface for contact sensors. These classes provide a backend-agnostic API for contact sensing.
* Added :class:`~isaaclab.sensors.imu.BaseImu` and :class:`~isaaclab.sensors.imu.BaseImuData` abstract
  base classes that define the interface for IMU sensors. These classes provide a backend-agnostic
  API for inertial measurement.
* Added :class:`~isaaclab.sensors.frame_transformer.BaseFrameTransformer` and
  :class:`~isaaclab.sensors.frame_transformer.BaseFrameTransformerData` abstract base classes that
  define the interface for frame transformer sensors. These classes provide a backend-agnostic API
  for coordinate frame transformations.

Changed
^^^^^^^

* Refactored the sensor classes (:class:`~isaaclab.sensors.ContactSensor`,
  :class:`~isaaclab.sensors.Imu`, :class:`~isaaclab.sensors.FrameTransformer`) to follow the
  multi-backend architecture. The classes now act as factory wrappers that instantiate the
  appropriate backend-specific implementation (PhysX by default).
* Refactored the sensor data classes (:class:`~isaaclab.sensors.ContactSensorData`,
  :class:`~isaaclab.sensors.ImuData`, :class:`~isaaclab.sensors.FrameTransformerData`) to use the
  factory pattern for backend-specific instantiation.
* Moved PhysX-specific sensor tests to the ``isaaclab_physx`` package:

  * ``test_contact_sensor.py`` → ``isaaclab_physx/test/sensors/``
  * ``test_imu.py`` → ``isaaclab_physx/test/sensors/``
  * ``test_frame_transformer.py`` → ``isaaclab_physx/test/sensors/``


2.0.0 (2026-01-30)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab.assets.BaseArticulation` and :class:`~isaaclab.assets.BaseArticulationData`
  abstract base classes that define the interface for articulation assets. These classes provide
  a backend-agnostic API for articulation operations.
* Added :class:`~isaaclab.assets.BaseRigidObject` and :class:`~isaaclab.assets.BaseRigidObjectData`
  abstract base classes that define the interface for rigid object assets. These classes provide
  a backend-agnostic API for rigid object operations.
* Added :class:`~isaaclab.assets.BaseRigidObjectCollection` and :class:`~isaaclab.assets.BaseRigidObjectCollectionData`
  abstract base classes that define the interface for rigid object collection assets. These classes
  provide a backend-agnostic API for managing collections of rigid objects.
* Added :mod:`~isaaclab.utils.backend_utils` module with utilities for managing simulation backends.

Changed
^^^^^^^

* Refactored the asset classes to follow a multi-backend architecture. The core :mod:`isaaclab.assets`
  module now provides abstract base classes that define the interface, while backend-specific
  implementations are provided in separate packages (e.g., ``isaaclab_physx``).
* The concrete :class:`~isaaclab.assets.Articulation`, :class:`~isaaclab.assets.RigidObject`,
  and :class:`~isaaclab.assets.RigidObjectCollection` classes in the ``isaaclab`` package
  now inherit from their respective base classes, and using the backend-specific implementations provided
  in the ``isaaclab_physx`` package, provide the default PhysX-based implementation.
* Moved :class:`DeformableObject`, :class:`DeformableObjectCfg`, and :class:`DeformableObjectData`
  to the ``isaaclab_physx`` package since deformable bodies are specific to PhysX simulation.
* Moved :class:`SurfaceGripper` and :class:`SurfaceGripperCfg` to the ``isaaclab_physx`` package
  since surface grippers rely on PhysX-specific contact APIs.

Deprecated
^^^^^^^^^^

* Deprecated the ``root_physx_view`` property on :class:`~isaaclab.assets.Articulation`,
  :class:`~isaaclab.assets.RigidObject`, and :class:`~isaaclab.assets.RigidObjectCollection`
  in favor of the backend-agnostic ``root_view`` property.

* Deprecated the ``object_*`` naming convention in :class:`~isaaclab.assets.RigidObjectCollection`
  and :class:`~isaaclab.assets.RigidObjectCollectionData` in favor of ``body_*``:

  **RigidObjectCollection methods:**

  * ``write_object_state_to_sim()`` → use ``write_body_state_to_sim()``
  * ``write_object_link_state_to_sim()`` → use ``write_body_link_state_to_sim()``
  * ``write_object_pose_to_sim()`` → use ``write_body_pose_to_sim()``
  * ``write_object_link_pose_to_sim()`` → use ``write_body_link_pose_to_sim()``
  * ``write_object_com_pose_to_sim()`` → use ``write_body_com_pose_to_sim()``
  * ``write_object_velocity_to_sim()`` → use ``write_body_com_velocity_to_sim()``
  * ``write_object_com_velocity_to_sim()`` → use ``write_body_com_velocity_to_sim()``
  * ``write_object_link_velocity_to_sim()`` → use ``write_body_link_velocity_to_sim()``
  * ``find_objects()`` → use ``find_bodies()``

  **RigidObjectCollectionData properties:**

  * ``default_object_state`` → use ``default_body_state``
  * ``object_names`` → use ``body_names``
  * ``object_pose_w``, ``object_pos_w``, ``object_quat_w`` → use ``body_pose_w``, ``body_pos_w``, ``body_quat_w``
  * ``object_vel_w``, ``object_lin_vel_w``, ``object_ang_vel_w`` → use ``body_vel_w``, ``body_lin_vel_w``, ``body_ang_vel_w``
  * ``object_acc_w``, ``object_lin_acc_w``, ``object_ang_acc_w`` → use ``body_acc_w``, ``body_lin_acc_w``, ``body_ang_acc_w``
  * And all other ``object_*`` properties (see :ref:`migrating-to-isaaclab-3-0` for complete list).

Migration
^^^^^^^^^

* See :ref:`migrating-to-isaaclab-3-0` for detailed migration instructions.


1.0.0 (2026-01-30)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a tool to find hard-coded quaternions in the codebase and help user convert them to the new XYZW ordering.

Changed
^^^^^^^

* Changed the quaternion ordering to match warp, PhysX, and Newton native XYZW quaternion ordering.


0.54.5 (2026-01-30)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :meth:`~isaaclab.utils.timer.Timer.get_timer_statistics` to get the statistics of the elapsed time of a timer.

Changed
^^^^^^^

* Changed :class:`~isaaclab.utils.timer.Timer` class to use the online Welford's algorithm to compute the mean and standard deviation of the elapsed time.


0.54.4 (2026-02-04)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :class:`~isaaclab.envs.mdp.actions.JointPositionToLimitsAction` and
  :class:`~isaaclab.envs.mdp.actions.EMAJointPositionToLimitsAction` ignoring
  ``preserve_order=True`` when the number of specified joints matches the total
  number of joints in the asset. The optimization that replaced joint indices with
  ``slice(None)`` now correctly checks for the ``preserve_order`` flag.


0.54.3 (2026-01-28)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved :mod:`isaaclab.sensors.tacsl_sensor` to :mod:`isaaclab_contrib.sensors.tacsl_sensor` module,
  since it is not completely ready for release yet.


0.54.2 (2026-01-25)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added test suite for ray caster patterns with comprehensive parameterized tests.

Fixed
^^^^^

* Fixed incorrect horizontal angle calculation in :func:`~isaaclab.sensors.ray_caster.patterns.patterns.lidar_pattern`
  that caused the actual angular resolution to differ from the requested resolution.


0.54.1 (2026-01-28)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Teleop: update carb settings to be compatible with Isaac Sim 6.0/Kit XR 110.0


0.54.0 (2026-01-13)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added Fabric backend support to :class:`~isaaclab.sim.views.XformPrimView` for GPU-accelerated
  batch transform operations on all Boundable prims using Warp kernels.
* Added :mod:`~isaaclab.sim.utils.fabric_utils` module with Warp kernels for efficient Fabric matrix operations.

Changed
^^^^^^^

* Changed :class:`~isaaclab.sensors.camera.Camera` to use Fabric backend for faster pose queries.


0.53.2 (2026-01-14)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab.assets.utils.wrench_composer.WrenchComposer` to compose forces and torques at the body's center of mass frame.
* Added :meth:`~isaaclab.assets.Articulation.instantaneous_wrench_composer` to add or set instantaneous external wrenches to the articulation.
* Added :meth:`~isaaclab.assets.Articulation.permanent_wrench_composer` to add or set permanent external wrenches to the articulation.
* Added :meth:`~isaaclab.assets.RigidObject.instantaneous_wrench_composer` to add or set instantaneous external wrenches to the rigid object.
* Added :meth:`~isaaclab.assets.RigidObject.permanent_wrench_composer` to add or set permanent external wrenches to the rigid object.
* Added :meth:`~isaaclab.assets.RigidObjectCollection.instantaneous_wrench_composer` to add or set instantaneous external wrenches to the rigid object collection.
* Added :meth:`~isaaclab.assets.RigidObjectCollection.permanent_wrench_composer` to add or set permanent external wrenches to the rigid object collection.
* Added unit tests for the wrench composer.
* Added kernels for the wrench composer in the :mod:`isaaclab.utils.warp.kernels` module.

Changed
^^^^^^^

* Deprecated :meth:`~isaaclab.assets.Articulation.set_external_force_and_torque`  in favor of :meth:`~isaaclab.assets.Articulation.permanent_wrench_composer.set_forces_and_torques`.
* Deprecated :meth:`~isaaclab.assets.RigidObject.set_external_force_and_torque`  in favor of :meth:`~isaaclab.assets.RigidObject.permanent_wrench_composer.set_forces_and_torques`.
* Deprecated :meth:`~isaaclab.assets.RigidObjectCollection.set_external_force_and_torque`  in favor of :meth:`~isaaclab.assets.RigidObjectCollection.permanent_wrench_composer.set_forces_and_torques`.
* Modified the tests of the articulation, rigid object, and rigid object collection to use the new permanent and instantaneous external wrench functions and test them.

0.53.1 (2026-01-08)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added function :func:`~isaaclab.sim.utils.prims.change_prim_property` to change attributes on a USD prim.
  This replaces the previously used USD command ``ChangeProperty`` that depends on Omniverse Kit API.

Changed
^^^^^^^

* Replaced occurrences of ``ChangeProperty`` USD command to :func:`~isaaclab.sim.utils.prims.change_prim_property`.


0.53.0 (2026-01-07)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab.sim.views.XformPrimView` class to provide a
  view of the USD Xform operations. Compared to Isaac Sim implementation,
  this class optimizes several operations using USD SDF API.

Changed
^^^^^^^

* Switched the sensor classes to use the :class:`~isaaclab.sim.views.XformPrimView`
  class for the internal view wherever applicable.

Removed
^^^^^^^

* Removed the usage of :class:`isaacsim.core.utils.prims.XformPrim`
  class from the sensor classes.


0.52.2 (2026-01-06)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Improved logic for the URDF importer extension version pinning: the older extension version
  is now pinned only on Isaac Sim 5.1 and later, while older Isaac Sim versions no longer
  attempt to pin to a version that does not exist.


0.52.1 (2026-01-02)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed FrameTransformer body name collision when tracking bodies with the same name but different hierarchical paths
  (e.g., Robot/left_hand vs Robot_1/left_hand). The sensor now uses the full prim path (with env_* patterns normalized)
  as the unique body identifier instead of just the leaf body name. This ensures bodies at different hierarchy levels
  are tracked separately. The change is backwards compatible: user-facing frame names still default to leaf names when
  not explicitly provided, while internal body tracking uses full paths to avoid collisions. Works for both
  environment-scoped paths (e.g., /World/envs/env_0/Robot) and non-environment paths (e.g., /World/Robot).


0.52.0 (2026-01-02)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`~isaaclab.sim.utils.transforms` module to handle USD Xform operations.
* Added passing of ``stage`` to the :func:`~isaaclab.sim.utils.prims.create_prim` function
  inside spawning functions to allow for the creation of prims in a specific stage.

Changed
^^^^^^^

* Changed :func:`~isaaclab.sim.utils.prims.create_prim` function to use the :mod:`~isaaclab.sim.utils.transforms`
  module for USD Xform operations. It removes the usage of Isaac Sim's XformPrim class.


0.51.2 (2025-12-30)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :attr:`~isaaclab.managers.ObservationManager.get_active_iterable_terms`
  to handle observation data when not concatenated along the last dimension.


0.51.1 (2025-12-29)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :func:`~isaaclab.utils.version.get_isaac_sim_version` to get the version of Isaac Sim.
  This function caches the version of Isaac Sim and returns it immediately if it has already been computed.
  This helps avoid parsing the VERSION file from disk multiple times.

Changed
^^^^^^^

* Changed the function :meth:`~isaaclab.utils.version.compare_versions` to use :mod:`packaging.version.Version` module.
* Changed occurrences of :func:`isaacsim.core.version.get_version` to :func:`~isaaclab.utils.version.get_isaac_sim_version`.

Removed
^^^^^^^

* Removed storing of Isaac Sim version inside the environment base classes defined inside
  :mod:`isaaclab.envs` module.


0.51.0 (2025-12-29)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added tests for the :mod:`isaaclab.sim.utils.prims` module.
* Added tests for the :mod:`isaaclab.sim.utils.stage` module.
* Created :mod:`isaaclab.sim.utils.legacy` sub-module to keep deprecated functions.

Removed
^^^^^^^

* Removed many unused USD prim and stage related operations from the :mod:`isaaclab.sim.utils` module.
* Moved :mod:`isaaclab.sim.utils.nucleus` sub-module to the ``tests/deps/isaacsim`` directory as it
  is only being used for Isaac Sim check scripts.

Changed
^^^^^^^

* Changed the organization of the :mod:`isaaclab.sim.utils` module to make it clearer which functions
  are related to the stage and which are related to the prims.
* Modified imports of :mod:`~isaaclab.sim.utils.stage` and :mod:`~isaaclab.sim.utils.prims` modules
  to only use the :mod:`isaaclab.sim.utils` module.
* Moved ``logger.py`` to the :mod:`isaaclab.utils` module.


0.50.7 (2025-12-29)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved ``pretrained_checkpoint.py`` to the :mod:`isaaclab_rl.utils` module.


0.50.6 (2025-12-18)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed issue where :meth:~isaaclab.envs.mdp.observations.body_pose_w` was modifying the original body pose data
  when using slice or int for body_ids in the observation config. A clone of the data is now created to avoid modifying
  the original data.


0.50.5 (2025-12-15)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab.sensors.MultiMeshRayCaster` sensor to support tracking of dynamic meshes for ray-casting.
  We keep the previous implementation of :class:`~isaaclab.sensors.RayCaster` for backwards compatibility.
* Added :mod:`isaaclab.utils.mesh` sub-module to perform various Trimesh and USD Mesh related operations.


0.50.4 (2025-12-15)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :attr:`~isaaclab.sim.PhysxCfg.enable_external_forces_every_iteration` to enable external forces every position
  iteration. This can help improve the accuracy of velocity updates. Consider enabling this flag if the velocities
  generated by the simulation are noisy.
* Added warning when :attr:`~isaaclab.sim.PhysxCfg.enable_external_forces_every_iteration` is set to False and
  the solver type is TGS.


0.50.3 (2025-12-11)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed missing mesh collision approximation attribute when running :class:`~isaaclab.sim.converters.MeshConverter`.
  The collision approximation attribute is now properly set on the USD prim when converting meshes with mesh collision
  properties.


0.50.2 (2025-11-21)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Prevent randomizing mass to zero in :meth:`~isaaclab.envs.mdp.events.randomize_mass_by_scale` to avoid physics errors.


0.50.1 (2025-11-25)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed advanced indexing issue in resetting prev action
  in :class:`~isaaclab.envs.mdp.actions.JointPositionToLimitsAction` .


0.50.0 (2025-12-8)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Implemented ability to attach an imu sensor to xform primitives in a usd file. This PR is based on work by '@GiulioRomualdi'
  here: #3094 Addressing issue #3088.


0.49.3 (2025-12-03)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`G1TriHandUpperBodyMotionControllerGripperRetargeter` and :class:`G1TriHandUpperBodyMotionControllerGripperRetargeterCfg` for retargeting the gripper state from motion controllers.
* Added unit tests for the retargeters.


0.49.2 (2025-11-17)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :attr:`~isaaclab.sensors.contact_sensor.ContactSensorCfg.track_friction_forces` to toggle tracking of friction forces between sensor bodies and filtered bodies.
* Added :attr:`~isaaclab.sensors.contact_sensor.ContactSensorData.friction_forces_w` data field for tracking friction forces.


0.49.1 (2025-11-26)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed import from ``isaacsim.core.utils.prims`` to ``isaaclab.sim.utils.prims`` across repo to reduce IsaacLab dependencies.

0.49.0 (2025-11-10)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Updated the URDF Importer version to 2.4.31 to avoid issues with merging joints on the latest URDF importer in Isaac Sim 5.1


0.48.9 (2025-11-21)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Add navigation state API to IsaacLabManagerBasedRLMimicEnv
* Add optional custom recorder config to MimicEnvCfg


0.48.8 (2025-10-15)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :attr:`preserve_order` flag to :class:`~isaaclab.envs.mdp.actions.actions_cfg.JointPositionToLimitsActionCfg`


0.48.7 (2025-11-25)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed import from ``isaaclab.sim.utils`` to ``isaaclab.sim.utils.stage`` in ``isaaclab.devices.openxr.xr_anchor_utils.py``
  to properly propagate the Isaac Sim stage context.


0.48.6 (2025-11-18)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added OpenXR motion controller support for the G1 robot locomanipulation environment
  ``Isaac-PickPlace-Locomanipulation-G1-Abs-v0``. This enables teleoperation using XR motion controllers
  in addition to hand tracking.
* Added :class:`OpenXRDeviceMotionController` for motion controller-based teleoperation with headset anchoring control.
* Added motion controller-specific retargeters:
  * :class:`G1TriHandControllerUpperBodyRetargeterCfg` for upper body and hand control using motion controllers.
  * :class:`G1LowerBodyStandingControllerRetargeterCfg` for lower body control using motion controllers.


0.48.5 (2025-11-14)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed import from ``isaacsim.core.utils.stage`` to ``isaaclab.sim.utils.stage`` to reduce IsaacLab dependencies.


0.48.4 (2025-11-14)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Refactored modules related to the actuator configs in order to remediate a circular import necessary to support future
  actuator drive model improvements.


0.48.3 (2025-11-13)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved retargeter and device declaration out of factory and into the devices/retargeters themselves.


0.48.2 (2025-11-13)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed from using :meth:`isaacsim.core.utils.torch.set_seed` to :meth:`~isaaclab.utils.seed.configure_seed`


0.48.1 (2025-11-10)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab.devices.haply.HaplyDevice` class for SE(3) teleoperation with dual Haply Inverse3 and Versegrip devices,
  supporting robot manipulation with haptic feedback.
* Added demo script ``scripts/demos/haply_teleoperation.py`` and documentation guide in
  ``docs/source/how-to/haply_teleoperation.rst`` for Haply-based robot teleoperation.


0.48.0 (2025-11-03)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Detected contacts are reported with the threshold of 0.0 (instead of 1.0). This increases the sensitivity of contact
  detection.

Fixed
^^^^^

* Removed passing the boolean flag to :meth:`isaaclab.sim.schemas.activate_contact_sensors` when activating contact
  sensors. This was incorrectly modifying the threshold attribute to 1.0 when contact sensors were activated.


0.47.11 (2025-11-03)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the bug where effort limits were being overridden in :class:`~isaaclab.actuators.ActuatorBase` when the ``effort_limit`` parameter is set to None.
* Corrected the unit tests for three effort limit scenarios with proper assertions


0.47.10 (2025-11-06)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``num_rerenders_on_reset`` parameter to ManagerBasedEnvCfg and DirectRLEnvCfg to configure the number
  of render steps to perform after reset. This enables more control over DLSS rendering behavior after reset.

Changed
^^^^^^^

* Added deprecation warning for ``rerender_on_reset`` parameter in ManagerBasedEnv and DirectRLEnv.


0.47.9 (2025-11-05)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Fixed termination term bookkeeping in :class:`~isaaclab.managers.TerminationManager`:
  per-step termination and last-episode termination bookkeeping are now separated.
  last-episode dones are now updated once per step from all term outputs, avoiding per-term overwrites
  and ensuring Episode_Termination metrics reflect the actual triggering terms.


0.47.8 (2025-11-06)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added parameter :attr:`~isaaclab.terrains.TerrainImporterCfg.use_terrain_origins` to allow generated sub terrains with grid origins.


0.47.7 (2025-10-31)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed Pink IK controller qpsolver from osqp to daqp.
* Changed Null Space matrix computation in Pink IK's Null Space Posture Task to a faster matrix pseudo inverse computation.


0.47.6 (2025-11-01)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed an issue in recurrent policy evaluation in RSL-RL framework where the recurrent state was not reset after an episode termination.


0.47.5 (2025-10-30)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added docstrings notes to clarify the friction coefficient modeling in Isaac Sim 4.5 and 5.0.


0.47.4 (2025-10-30)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Enhanced :meth:`~isaaclab.managers.RecorderManager.export_episodes` method to support customizable sequence of demo IDs:

  - Added argument ``demo_ids`` to :meth:`~isaaclab.managers.RecorderManager.export_episodes` to accept a sequence of integers
    for custom episode identifiers.

* Enhanced :meth:`~isaaclab.utils.datasets.HDF5DatasetFileHandler.write_episode` method to support customizable episode identifiers:

  - Added argument ``demo_id`` to :meth:`~isaaclab.utils.datasets.HDF5DatasetFileHandler.write_episode` to accept a custom integer
    for episode identifier.


0.47.3 (2025-10-22)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the data type conversion in :class:`~isaaclab.sensors.tiled_camera.TiledCamera` to
  support the correct data type when converting from numpy arrays to warp arrays on the CPU.


0.47.2 (2025-10-17)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :meth:`~isaaclab.sim.utils.resolve_prim_pose` to resolve the pose of a prim with respect to another prim.
* Added :meth:`~isaaclab.sim.utils.resolve_prim_scale` to resolve the scale of a prim in the world frame.


0.47.1 (2025-10-17)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Suppressed yourdfpy warnings when trying to load meshes from hand urdfs in dex_retargeting. These mesh files are not
  used by dex_retargeting, but the parser is incorrectly configured by dex_retargeting to load them anyway which results
  in warning spam.


0.47.0 (2025-10-14)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Removed pickle utilities for saving and loading configurations as pickle contains security vulnerabilities in its APIs.
  Configurations can continue to be saved and loaded through yaml.


0.46.11 (2025-10-15)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for modifying the :attr:`/rtx/domeLight/upperLowerStrategy` Sim rendering setting.


0.46.10 (2025-10-13)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ARM64 architecture for pink ik and dex-retargetting setup installations.


0.46.9 (2025-10-09)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :meth:`~isaaclab.devices.keyboard.se3_keyboard.Se3Keyboard.__del__` to use the correct method name
  for unsubscribing from keyboard events "unsubscribe_to_keyboard_events" instead of "unsubscribe_from_keyboard_events".


0.46.8 (2025-10-02)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed scaling factor for retargeting of GR1T2 hand.


0.46.7 (2025-09-30)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed finger joint indices with manus extension.


0.46.6 (2025-09-30)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added argument :attr:`traverse_instance_prims` to :meth:`~isaaclab.sim.utils.get_all_matching_child_prims` and
  :meth:`~isaaclab.sim.utils.get_first_matching_child_prim` to control whether to traverse instance prims
  during the traversal. Earlier, instanced prims were skipped since :meth:`Usd.Prim.GetChildren` did not return
  instanced prims, which is now fixed.

Changed
^^^^^^^

* Made parsing of instanced prims in :meth:`~isaaclab.sim.utils.get_all_matching_child_prims` and
  :meth:`~isaaclab.sim.utils.get_first_matching_child_prim` as the default behavior.
* Added parsing of instanced prims in :meth:`~isaaclab.sim.utils.make_uninstanceable` to make all prims uninstanceable.


0.46.5 (2025-10-14)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Exposed parameter :attr:`~isaaclab.sim.spawners.PhysxCfg.solve_articulation_contact_last`
  to configure USD attribute ``physxscene:solveArticulationContactLast``. This parameter may
  help improve solver stability with grippers, which previously required reducing simulation time-steps.
  :class:`~isaaclab.sim.spawners.PhysxCfg`


0.46.4 (2025-10-06)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Fixed :attr:`~isaaclab.sim.simulation_context.SimulationContext.device` to return the device from the configuration.
  Previously, it was returning the device from the simulation manager, which was causing a performance overhead.


0.46.3 (2025-09-17)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Modified setter to support for viscous and dynamic joint friction coefficients in articulation based on IsaacSim 5.0.
* Added randomization of viscous and dynamic joint friction coefficients in event term.


0.46.2 (2025-09-13)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Fixed missing actuator indices in :meth:`~isaaclab.envs.mdp.events.randomize_actuator_gains`


0.46.1 (2025-09-10)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved IO descriptors output directory to a subfolder under the task log directory.


0.46.0 (2025-09-06)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added argument :attr:`traverse_instance_prims` to :meth:`~isaaclab.sim.utils.get_all_matching_child_prims` and
  :meth:`~isaaclab.sim.utils.get_first_matching_child_prim` to control whether to traverse instance prims
  during the traversal. Earlier, instanced prims were skipped since :meth:`Usd.Prim.GetChildren` did not return
  instanced prims, which is now fixed.

Changed
^^^^^^^

* Made parsing of instanced prims in :meth:`~isaaclab.sim.utils.get_all_matching_child_prims` and
  :meth:`~isaaclab.sim.utils.get_first_matching_child_prim` as the default behavior.
* Added parsing of instanced prims in :meth:`~isaaclab.sim.utils.make_uninstanceable` to make all prims uninstanceable.


0.45.16 (2025-09-06)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added teleoperation environments for Unitree G1. This includes an environment with lower body fixed and upper body
  controlled by IK, and an environment with the lower body controlled by a policy and the upper body controlled by IK.


0.45.15 (2025-09-05)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added action terms for using RMPFlow in Manager-Based environments.


0.45.14 (2025-09-08)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab.ui.xr_widgets.TeleopVisualizationManager` and :class:`~isaaclab.ui.xr_widgets.XRVisualization`
  classes to provide real-time visualization of teleoperation and inverse kinematics status in XR environments.


0.45.13 (2025-09-08)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab.devices.openxr.manus_vive.ManusVive` to support teleoperation with Manus gloves and Vive trackers.


0.45.12 (2025-09-05)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab.envs.mdp.actions.SurfaceGripperBinaryAction` for supporting surface grippers in Manager-Based workflows.

Changed
^^^^^^^

* Added AssetBase inheritance for :class:`~isaaclab.assets.surface_gripper.SurfaceGripper`.


0.45.11 (2025-09-04)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixes a high memory usage and perf slowdown issue in episode data by removing the use of torch.cat when appending to the episode data
  at each timestep. The use of torch.cat was causing the episode data to be copied at each timestep, which causes high memory usage and
  significant performance slowdown when recording longer episode data.
* Patches the configclass to allow validate dict with key is not a string.

Added
^^^^^

* Added optional episode metadata (ep_meta) to be stored in the HDF5 data attributes.
* Added option to record data pre-physics step.
* Added joint_target data to episode data. Joint target data can be optionally recorded by the user and replayed to improve
  determinism of replay.


0.45.10 (2025-09-02)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed regression in reach task configuration where the gripper command was being returned.
* Added :attr:`~isaaclab.devices.Se3GamepadCfg.gripper_term` to :class:`~isaaclab.devices.Se3GamepadCfg`
  to control whether the gamepad device should return a gripper command.
* Added :attr:`~isaaclab.devices.Se3SpaceMouseCfg.gripper_term` to :class:`~isaaclab.devices.Se3SpaceMouseCfg`
  to control whether the spacemouse device should return a gripper command.
* Added :attr:`~isaaclab.devices.Se3KeyboardCfg.gripper_term` to :class:`~isaaclab.devices.Se3KeyboardCfg`
  to control whether the keyboard device should return a gripper command.


0.45.9 (2025-08-27)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed removing import of pink_ik controller from isaaclab.controllers which is causing pinocchio import error.


0.45.8 (2025-07-25)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created :attr:`~isaaclab.controllers.pink_ik.PinkIKControllerCfg.target_eef_link_names` to :class:`~isaaclab.controllers.pink_ik.PinkIKControllerCfg`
  to specify the target end-effector link names for the pink inverse kinematics controller.

Changed
^^^^^^^

* Updated pink inverse kinematics controller configuration for the following tasks (Isaac-PickPlace-GR1T2, Isaac-NutPour-GR1T2, Isaac-ExhaustPipe-GR1T2)
  to increase end-effector tracking accuracy and speed. Also added a null-space regularizer that enables turning on of waist degrees-of-freedom.
* Improved the test_pink_ik script to more comprehensive test on controller accuracy. Also, migrated to use pytest. With the current IK controller
  improvements, our unit tests pass position and orientation accuracy test within **(1 mm, 1 degree)**. Previously, the position accuracy tolerances
  were set to **(30 mm, 10 degrees)**.
* Included a new config parameter :attr:`fail_on_ik_error` to :class:`~isaaclab.controllers.pink_ik.PinkIKControllerCfg`
  to control whether the IK controller raise an exception if robot joint limits are exceeded. In the case of an exception, the controller will hold the
  last joint position. This adds to stability of the controller and avoids operator experiencing what is perceived as sudden large delays in robot control.


0.45.7 (2025-08-21)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added periodic logging when checking if a USD path exists on a Nucleus server
  to improve user experience when the checks takes a while.


0.45.6 (2025-08-22)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :meth:`~isaaclab.envs.mdp.events.randomize_rigid_body_com` to broadcasts the environment ids.


0.45.5 (2025-08-21)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :meth:`~isaaclab.assets.Articulation.write_joint_friction_coefficient_to_sim` to set the friction coefficients in the simulation.
* Fixed :meth:`~isaaclab.assets.Articulation.write_joint_dynamic_friction_coefficient_to_sim` to set the friction coefficients in the simulation.* Added :meth:`~isaaclab.envs.ManagerBasedEnvCfg.export_io_descriptors` to toggle the export of the IO descriptors.
* Fixed :meth:`~isaaclab.assets.Articulation.write_joint_viscous_friction_coefficient_to_sim` to set the friction coefficients in the simulation.



0.45.4 (2025-08-21)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added unit tests for :class:`~isaaclab.sensor.sensor_base`


0.45.3 (2025-08-20)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :meth:`isaaclab.envs.mdp.terminations.joint_effort_out_of_limit` so that it correctly reports whether a joint
  effort limit has been violated. Previously, the implementation marked a violation when the applied and computed
  torques were equal; in fact, equality should indicate no violation, and vice versa.


0.45.2 (2025-08-18)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :meth:`~isaaclab.managers.ObservationManager.get_IO_descriptors` to export the IO descriptors for the observation manager.
* Added :meth:`~isaaclab.envs.ManagerBasedEnvCfg.io_descriptors_output_dir` to configure the directory to export the IO descriptors to.
* Added :meth:`~isaaclab.envs.ManagerBasedEnvCfg.export_io_descriptors` to toggle the export of the IO descriptors.
* Added the option to export the Observation and Action of the managed environments into a YAML file. This can be used to more easily
  deploy policies trained in Isaac Lab.


0.45.1 (2025-08-16)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added validations for scale-based randomization ranges across mass, actuator, joint, and tendon parameters.

Changed
^^^^^^^

* Refactored randomization functions into classes with initialization-time checks to avoid runtime overhead.


0.45.0 (2025-08-07)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :attr:`~isaaclab.sensors.contact_sensor.ContactSensorCfg.track_contact_points` to toggle tracking of contact
  point locations between sensor bodies and filtered bodies.
* Added :attr:`~isaaclab.sensors.contact_sensor.ContactSensorCfg.max_contact_data_per_prim` to configure the maximum
  amount of contacts per sensor body.
* Added :attr:`~isaaclab.sensors.contact_sensor.ContactSensorData.contact_pos_w` data field for tracking contact point
  locations.


0.44.12 (2025-08-12)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed IndexError in :meth:`isaaclab.envs.mdp.events.reset_joints_by_scale`,
  :meth:`isaaclab.envs.mdp.events.reset_joints_by_offsets` by adding dimension to env_ids when indexing.


0.44.11 (2025-08-11)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed rendering preset mode when an experience CLI arg is provided.


0.44.10 (2025-08-06)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the old termination manager in :class:`~isaaclab.managers.TerminationManager` term_done logging that
  logs the instantaneous term done count at reset. This let to inaccurate aggregation of termination count,
  obscuring the what really happening during the training. Instead we log the episodic term done.


0.44.9 (2025-07-30)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``from __future__ import annotations`` to manager_based_rl_mimic_env.py to fix Sphinx
  doc warnings for IsaacLab Mimic docs.


0.44.8 (2025-07-30)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Improved handling of deprecated flag :attr:`~isaaclab.sensors.RayCasterCfg.attach_yaw_only`.
  Previously, the flag was only handled if it was set to True. This led to a bug where the yaw was not accounted for
  when the flag was set to False.
* Fixed the handling of interval-based events inside :class:`~isaaclab.managers.EventManager` to properly handle
  their resets. Previously, only class-based events were properly handled.


0.44.7 (2025-07-30)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a new argument ``is_global`` to :meth:`~isaaclab.assets.Articulation.set_external_force_and_torque`,
  :meth:`~isaaclab.assets.RigidObject.set_external_force_and_torque`, and
  :meth:`~isaaclab.assets.RigidObjectCollection.set_external_force_and_torque` allowing to set external wrenches
  in the global frame directly from the method call rather than having to set the frame in the configuration.

Removed
^^^^^^^^

* Removed :attr:`xxx_external_wrench_frame` flag from asset configuration classes in favor of direct argument
  passed to the :meth:`set_external_force_and_torque` function.


0.44.6 (2025-07-28)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Tweak default behavior for rendering preset modes.


0.44.5 (2025-07-28)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :meth:`isaaclab.scene.reset_to` to properly accept None as valid argument.
* Added tests to verify that argument types.


0.44.4 (2025-07-22)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added safe callbacks for stage in memory attaching.
* Remove on prim deletion callback workaround


0.44.3 (2025-07-21)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed rendering preset mode regression.


0.44.2 (2025-07-22)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated teleop scripts to print to console vs omni log.


0.44.1 (2025-07-17)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated test_pink_ik.py test case to pytest format.


0.44.0 (2025-07-21)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the way clipping is handled for :class:`~isaaclab.actuator.DCMotor` for torque-speed points in when in
  negative power regions.

Added
^^^^^

* Added unit tests for :class:`~isaaclab.actuator.ImplicitActuator`, :class:`~isaaclab.actuator.IdealPDActuator`,
  and :class:`~isaaclab.actuator.DCMotor` independent of :class:`~isaaclab.assets.Articulation`


0.43.0 (2025-07-21)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updates torch version to 2.7.0 and torchvision to 0.22.0.
  Some dependencies now require torch>=2.6, and given the vulnerabilities in Torch 2.5.1,
  we are updating the torch version to 2.7.0 to also include Blackwell support. Since Isaac Sim 4.5 has not updated the
  torch version, we are now overwriting the torch installation step in isaaclab.sh when running ``./isaaclab.sh -i``.


0.42.26 (2025-06-29)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added MangerBasedRLEnv support for composite gym observation spaces.
* A test for the composite gym observation spaces in ManagerBasedRLEnv is added to ensure that the observation spaces
  are correctly configured base on the clip.


0.42.25 (2025-07-11)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :attr:`~isaaclab.sensors.ContactSensorData.force_matrix_w_history` that tracks the history of the filtered
  contact forces in the world frame.


0.42.24 (2025-06-25)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added new curriculum mdp :func:`~isaaclab.envs.mdp.curriculums.modify_env_param` and
  :func:`~isaaclab.envs.mdp.curriculums.modify_env_param` that enables flexible changes to any configurations in the
  env instance


0.42.23 (2025-07-11)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :meth:`isaaclab.envs.mdp.events.reset_joints_by_scale`, :meth:`isaaclab.envs.mdp.events.reset_joints_by_offsets`
  restricting the resetting joint indices be that user defined joint indices.


0.42.22 (2025-07-11)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed missing attribute in :class:`~isaaclab.sensors.ray_caster.RayCasterCamera` class and its reset method when no
  env_ids are passed.


0.42.21 (2025-07-09)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added input param ``update_history`` to :meth:`~isaaclab.managers.ObservationManager.compute`
  to control whether the history buffer should be updated.
* Added unit test for :class:`~isaaclab.envs.ManagerBasedEnv`.

Fixed
^^^^^

* Fixed :class:`~isaaclab.envs.ManagerBasedEnv` and :class:`~isaaclab.envs.ManagerBasedRLEnv` to not update the history
  buffer on recording.


0.42.20 (2025-07-10)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added unit tests for multiple math functions:
  :func:`~isaaclab.utils.math.scale_transform`.
  :func:`~isaaclab.utils.math.unscale_transform`.
  :func:`~isaaclab.utils.math.saturate`.
  :func:`~isaaclab.utils.math.normalize`.
  :func:`~isaaclab.utils.math.copysign`.
  :func:`~isaaclab.utils.math.convert_quat`.
  :func:`~isaaclab.utils.math.quat_conjugate`.
  :func:`~isaaclab.utils.math.quat_from_euler_xyz`.
  :func:`~isaaclab.utils.math.quat_from_matrix`.
  :func:`~isaaclab.utils.math.euler_xyz_from_quat`.
  :func:`~isaaclab.utils.math.matrix_from_euler`.
  :func:`~isaaclab.utils.math.quat_from_angle_axis`.
  :func:`~isaaclab.utils.math.axis_angle_from_quat`.
  :func:`~isaaclab.utils.math.skew_symmetric_matrix`.
  :func:`~isaaclab.utils.math.combine_transform`.
  :func:`~isaaclab.utils.math.subtract_transform`.
  :func:`~isaaclab.utils.math.compute_pose_error`.

Changed
^^^^^^^

* Changed the implementation of :func:`~isaaclab.utils.math.copysign` to better reflect the documented functionality.


0.42.19 (2025-07-09)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added clone_in_fabric config flag to :class:`~isaaclab.scene.interactive_scene_cfg.InteractiveSceneCfg`
* Enable clone_in_fabric for envs which work with limited benchmark_non_rl.py script


0.42.18 (2025-07-07)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed texture and color randomization to use new replicator functional APIs.


0.42.17 (2025-07-08)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed hanging quat_rotate calls to point to quat_apply in :class:`~isaaclab.assets.articulation.ArticulationData` and
  :class:`~isaaclab.assets.articulation.RigidObjectCollectionData`


0.42.16 (2025-07-08)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ability to set platform height independent of object height for trimesh terrains.


0.42.15 (2025-07-01)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :attr:`abs_height_noise` and :attr:`rel_height_noise` to give minimum and maximum absolute and relative noise to
  :class:`isaaclab.terrrains.trimesh.MeshRepeatedObjectsTerrainCfg`
* Added deprecation warnings to the existing :attr:`max_height_noise` but still functions.


0.42.14 (2025-07-03)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed unittest tests that are floating inside pytests for articulation and rendering


0.42.13 (2025-07-07)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated gymnasium to v1.2.0. This update includes fixes for a memory leak that appears when recording
  videos with the ``--video`` flag.


0.42.12 (2025-06-27)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added unit test for :func:`~isaaclab.utils.math.quat_inv`.

Fixed
^^^^^

* Fixed the implementation mistake in :func:`~isaaclab.utils.math.quat_inv`.


0.42.11 (2025-06-25)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :func:`~isaaclab.utils.dict.update_class_from_dict` preventing setting flat Iterables with different lengths.


0.42.10 (2025-06-25)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``sample_bias_per_component`` flag to :class:`~isaaclab.utils.noise.noise_model.NoiseModelWithAdditiveBias`
  to enable independent per-component bias sampling, which is now the default behavior. If set to False, the previous
  behavior of sharing the same bias value across all components is retained.


0.42.9 (2025-06-18)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed data inconsistency between read_body, read_link, read_com when write_body, write_com, write_joint performed, in
  :class:`~isaaclab.assets.Articulation`, :class:`~isaaclab.assets.RigidObject`, and
  :class:`~isaaclab.assets.RigidObjectCollection`
* added pytest that check against these data consistencies


0.42.8 (2025-06-24)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* :class:`~isaaclab.utils.noise.NoiseModel` support for manager-based workflows.

Changed
^^^^^^^

* Renamed :func:`~isaaclab.utils.noise.NoiseModel.apply` method to :func:`~isaaclab.utils.noise.NoiseModel.__call__`.


0.42.7 (2025-06-12)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed potential issues in :func:`~isaaclab.envs.mdp.events.randomize_visual_texture_material` related to handling
  visual prims during texture randomization.


0.42.6 (2025-06-11)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Remove deprecated usage of quat_rotate from articulation data class and replace with quat_apply.


0.42.5 (2025-05-22)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed collision filtering logic for CPU simulation. The automatic collision filtering feature
  currently has limitations for CPU simulation. Collision filtering needs to be manually enabled when using
  CPU simulation.


0.42.4 (2025-06-03)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Removes the hardcoding to :class:`~isaaclab.terrains.terrain_generator.TerrainGenerator` in
  :class:`~isaaclab.terrains.terrain_generator.TerrainImporter` and instead the ``class_type`` is used which is
  passed in the ``TerrainGeneratorCfg``.


0.42.3 (2025-03-20)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Made separate data buffers for poses and velocities for the :class:`~isaaclab.assets.Articulation`,
  :class:`~isaaclab.assets.RigidObject`, and :class:`~isaaclab.assets.RigidObjectCollection` classes.
  Previously, the two data buffers were stored together in a single buffer requiring an additional
  concatenation operation when accessing the data.
* Cleaned up ordering of members inside the data classes for the assets to make them easier
  to comprehend. This reduced the code duplication within the class and made the class
  more readable.


0.42.2 (2025-05-31)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Updated gymnasium to >= 1.0
* Added support for specifying module:task_name as task name to avoid module import for ``gym.make``


0.42.1 (2025-06-02)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added time observation functions to ~isaaclab.envs.mdp.observations module,
  :func:`~isaaclab.envs.mdp.observations.current_time_s` and :func:`~isaaclab.envs.mdp.observations.remaining_time_s`.

Changed
^^^^^^^

* Moved initialization of ``episode_length_buf`` outside of :meth:`load_managers()` of
  :class:`~isaaclab.envs.ManagerBasedRLEnv` to make it available for mdp functions.


0.42.0 (2025-06-02)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for stage in memory and cloning in fabric. This will help improve performance for scene setup and lower
  overall startup time.


0.41.0 (2025-05-19)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added simulation schemas for spatial tendons. These can be configured for assets imported
  from file formats.
* Added support for spatial tendons.


0.40.14 (2025-05-29)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added deprecation warning for :meth:`~isaaclab.utils.math.quat_rotate` and
  :meth:`~isaaclab.utils.math.quat_rotate_inverse`

Changed
^^^^^^^

* Changed all calls to :meth:`~isaaclab.utils.math.quat_rotate` and :meth:`~isaaclab.utils.math.quat_rotate_inverse` to
  :meth:`~isaaclab.utils.math.quat_apply` and :meth:`~isaaclab.utils.math.quat_apply_inverse` for speed.


0.40.13 (2025-05-19)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Raising exceptions in step, render and reset if they occurred inside the initialization callbacks
  of assets and sensors.used from the experience files and the double definition is removed.


0.40.12 (2025-01-30)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added method :meth:`omni.isaac.lab.assets.AssetBase.set_visibility` to set the visibility of the asset
  in the simulation.


0.40.11 (2025-05-16)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for concatenation of observations along different dimensions in
  :class:`~isaaclab.managers.observation_manager.ObservationManager`.

Changed
^^^^^^^

* Updated the :class:`~isaaclab.managers.command_manager.CommandManager` to update the command counter after the
  resampling call.


0.40.10 (2025-05-16)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed penetration issue for negative border height in :class:`~isaaclab.terrains.terrain_generator.TerrainGeneratorCfg`.


0.40.9 (2025-05-20)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed the implementation of :meth:`~isaaclab.utils.math.quat_box_minus`

Added
^^^^^

* Added :meth:`~isaaclab.utils.math.quat_box_plus`
* Added :meth:`~isaaclab.utils.math.rigid_body_twist_transform`


0.40.8 (2025-05-15)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :meth:`omni.isaac.lab.sensors.camera.camera.Camera.set_intrinsic_matrices` preventing setting of unused USD
  camera parameters.
* Fixed :meth:`omni.isaac.lab.sensors.camera.camera.Camera._update_intrinsic_matrices` preventing unused USD camera
  parameters from being used to calculate :attr:`omni.isaac.lab.sensors.camera.CameraData.intrinsic_matrices`
* Fixed :meth:`omni.isaac.lab.spawners.sensors.sensors_cfg.PinholeCameraCfg.from_intrinsic_matrix` preventing setting of
  unused USD camera parameters.


0.40.7 (2025-05-14)
~~~~~~~~~~~~~~~~~~~

* Added a new attribute :attr:`articulation_root_prim_path` to the :class:`~isaaclab.assets.ArticulationCfg` class
  to allow explicitly specifying the prim path of the articulation root.


0.40.6 (2025-05-14)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Enabled external cameras in XR.


0.40.5 (2025-05-23)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added feature for animation recording through baking physics operations into OVD files.


0.40.4 (2025-05-17)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed livestreaming options to use ``LIVESTREAM=1`` for WebRTC over public networks and ``LIVESTREAM=2`` for WebRTC over private networks.


0.40.3 (2025-05-20)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Made modifications to :func:`isaaclab.envs.mdp.image` to handle image normalization for normal maps.


0.40.2 (2025-05-14)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Refactored remove_camera_configs to be a function that can be used in the record_demos and teleop scripts.


0.40.1 (2025-05-14)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed spacemouse device add callback function to work with record_demos/teleop_se3_agent scripts.


0.40.0 (2025-05-03)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added check in RecorderManager to ensure that the success indicator is only set if the termination manager is present.
* Added semantic tags in :func:`isaaclab.sim.spawners.from_files.spawn_ground_plane`.
  This allows for :attr:`semantic_segmentation_mapping` to be used when using the ground plane spawner.


0.39.0 (2025-04-01)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :meth:`~isaaclab.env.mdp.observations.joint_effort`


0.38.0 (2025-04-01)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :meth:`~isaaclab.envs.mdp.observations.body_pose_w`
* Added :meth:`~isaaclab.envs.mdp.observations.body_projected_gravity_b`


0.37.5 (2025-05-12)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a new teleop configuration class :class:`~isaaclab.devices.DevicesCfg` to support multiple teleoperation
  devices declared in the environment configuration file.
* Implemented a factory function to create teleoperation devices based on the device configuration.


0.37.4 (2025-05-12)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Remove isaacsim.xr.openxr from openxr experience file.
* Use Performance AR profile for XR rendering.


0.37.3 (2025-05-08)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Updated PINK task space action to record processed actions.
* Added new recorder term for recording post step processed actions.


0.37.2 (2025-05-06)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Migrated OpenXR device to use the new OpenXR handtracking API from omni.kit.xr.core.


0.37.1 (2025-05-05)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Removed xr rendering mode.


0.37.0 (2025-04-24)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated pytorch to latest 2.7.0 with cuda 12.8 for Blackwell support.
  Torch is now installed as part of the isaaclab.sh/bat scripts to ensure the correct version is installed.
* Removed :attr:`~isaaclab.sim.spawners.PhysicsMaterialCfg.improve_patch_friction` as it has been deprecated and removed from the simulation.
  The simulation will always behave as if this attribute is set to true.


0.36.23 (2025-04-24)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed ``return_latest_camera_pose`` option in :class:`~isaaclab.sensors.TiledCameraCfg` from not being used to the
  argument ``update_latest_camera_pose`` in :class:`~isaaclab.sensors.CameraCfg` with application in both
  :class:`~isaaclab.sensors.Camera` and :class:`~isaaclab.sensors.TiledCamera`.


0.36.22 (2025-04-23)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^^^

* Adds correct type check for ManagerTermBase class in event_manager.py.


0.36.21 (2025-04-15)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Removed direct call of qpsovlers library from pink_ik controller and changed solver from quadprog to osqp.


0.36.20 (2025-04-09)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added call to set cuda device after each ``app.update()`` call in :class:`~isaaclab.sim.SimulationContext`.
  This is now required for multi-GPU workflows because some underlying logic in ``app.update()`` is modifying
  the cuda device, which results in NCCL errors on distributed setups.


0.36.19 (2025-04-01)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added check in RecorderManager to ensure that the success indicator is only set if the termination manager is present.


0.36.18 (2025-03-26)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a dynamic text instruction widget that provides real-time feedback
  on the number of successful recordings during demonstration sessions.


0.36.17 (2025-03-26)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added override in AppLauncher to apply patch for ``pxr.Gf.Matrix4d`` to work with Pinocchio 2.7.0.


0.36.16 (2025-03-25)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Modified rendering mode default behavior when the launcher arg :attr:`enable_cameras` is not set.


0.36.15 (2025-03-25)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added near plane distance configuration for XR device.


0.36.14 (2025-03-24)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed default render settings in :class:`~isaaclab.sim.SimulationCfg` to None, which means that
  the default settings will be used from the experience files and the double definition is removed.


0.36.13 (2025-03-24)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added headpose support to OpenXRDevice.


0.36.12 (2025-03-19)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added parameter to show warning if Pink IK solver fails to find a solution.


0.36.11 (2025-03-19)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed default behavior of :class:`~isaaclab.actuators.ImplicitActuator` if no :attr:`effort_limits_sim` or
  :attr:`effort_limit` is set.


0.36.10 (2025-03-17)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* App launcher to update the cli arguments if conditional defaults are used.


0.36.9 (2025-03-18)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^^^

* Xr rendering mode, which is default when xr is used.


0.36.8 (2025-03-17)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Removed ``scalar_first`` from scipy function usage to support older versions of scipy.


0.36.7 (2025-03-14)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Changed the import structure to only import ``pinocchio`` when ``pink-ik`` or ``dex-retargeting`` is being used.
  This also solves for the problem that ``pink-ik`` and ``dex-retargeting`` are not supported in windows.
* Removed ``isaacsim.robot_motion.lula`` and ``isaacsim.robot_motion.motion_generation`` from the default loaded Isaac Sim extensions.
* Moved pink ik action config to a separate file.


0.36.6 (2025-03-13)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Worked around an issue where the render mode is set to ``"RayTracedLighting"`` instead of ``"RaytracedLighting"`` by
  some dependencies.


0.36.5 (2025-03-11)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^^^

* Added 3 rendering mode presets: performance, balanced, and quality.
* Preset settings are stored in ``apps/rendering_modes``.
* Presets can be set with cli arg ``--rendering_mode`` or with :class:`RenderCfg`.
* Preset rendering settings can be overwritten with :class:`RenderCfg`.
* :class:`RenderCfg` supports all native RTX carb settings.

Changed
^^^^^^^
* :class:`RenderCfg` default settings are unset.


0.36.4 (2025-03-11)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the OpenXR kit file ``isaaclab.python.xr.openxr.kit`` to inherit from ``isaaclab.python.kit`` instead of
  ``isaaclab.python.rendering.kit`` which is not appropriate.


0.36.3 (2025-03-10)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added the PinkIKController controller class that interfaces Isaac Lab with the Pink differential inverse kinematics solver
  to allow control of multiple links in a robot using a single solver.


0.36.2 (2025-03-07)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Allowed users to exit on 1 Ctrl+C instead of consecutive 2 key strokes.
* Allowed physics reset during simulation through :meth:`reset` in :class:`~isaaclab.sim.SimulationContext`.


0.36.1 (2025-03-10)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :attr:`semantic_segmentation_mapping` for camera configs to allow specifying colors for semantics.


0.36.0 (2025-03-07)
~~~~~~~~~~~~~~~~~~~

Removed
^^^^^^^

* Removed the storage of tri-meshes and warp meshes inside the :class:`~isaaclab.terrains.TerrainImporter` class.
  Initially these meshes were added for ray-casting purposes. However, since the ray-caster reads the terrains
  directly from the USD files, these meshes are no longer needed.
* Deprecated the :attr:`warp_meshes` and :attr:`meshes` attributes from the
  :class:`~isaaclab.terrains.TerrainImporter` class. These attributes now return an empty dictionary
  with a deprecation warning.

Changed
^^^^^^^

* Changed the prim path of the "plane" terrain inside the :class:`~isaaclab.terrains.TerrainImporter` class.
  Earlier, the terrain was imported directly as the importer's prim path. Now, the terrain is imported as
  ``{importer_prim_path}/{name}``, where ``name`` is the name of the terrain.


0.35.0 (2025-03-07)
~~~~~~~~~~~~~~~~~~~

* Improved documentation of various attributes in the :class:`~isaaclab.assets.ArticulationData` class to make
  it clearer which values represent the simulation and internal class values. In the new convention,
  the ``default_xxx`` attributes are whatever the user configured from their configuration of the articulation
  class, while the ``xxx`` attributes are the values from the simulation.
* Updated the soft joint position limits inside the :meth:`~isaaclab.assets.Articulation.write_joint_pos_limits_to_sim`
  method to use the new limits passed to the function.
* Added setting of :attr:`~isaaclab.assets.ArticulationData.default_joint_armature` and
  :attr:`~isaaclab.assets.ArticulationData.default_joint_friction` attributes in the
  :class:`~isaaclab.assets.Articulation` class based on user configuration.

Changed
^^^^^^^

* Removed unnecessary buffer creation operations inside the :class:`~isaaclab.assets.Articulation` class.
  Earlier, the class initialized a variety of buffer data with zeros and in the next function assigned
  them the value from PhysX. This made the code bulkier and more complex for no reason.
* Renamed parameters for a consistent nomenclature. These changes are backwards compatible with previous releases
  with a deprecation warning for the old names.

  * ``joint_velocity_limits`` → ``joint_vel_limits`` (to match attribute ``joint_vel`` and ``joint_vel_limits``)
  * ``joint_limits`` → ``joint_pos_limits`` (to match attribute ``joint_pos`` and ``soft_joint_pos_limits``)
  * ``default_joint_limits`` → ``default_joint_pos_limits``
  * ``write_joint_limits_to_sim`` → ``write_joint_position_limit_to_sim``
  * ``joint_friction`` → ``joint_friction_coeff``
  * ``default_joint_friction`` → ``default_joint_friction_coeff``
  * ``write_joint_friction_to_sim`` → ``write_joint_friction_coefficient_to_sim``
  * ``fixed_tendon_limit`` → ``fixed_tendon_pos_limits``
  * ``default_fixed_tendon_limit`` → ``default_fixed_tendon_pos_limits``
  * ``set_fixed_tendon_limit`` → ``set_fixed_tendon_position_limit``


0.34.13 (2025-03-06)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a new event mode called "prestartup", which gets called right after the scene design is complete
  and before the simulation is played.
* Added a callback to resolve the scene entity configurations separately once the simulation plays,
  since the scene entities cannot be resolved before the simulation starts playing
  (as we currently rely on PhysX to provide us with the joint/body ordering)


0.34.12 (2025-03-06)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Updated the mimic API :meth:`target_eef_pose_to_action` in :class:`isaaclab.envs.ManagerBasedRLMimicEnv` to take a dictionary of
  eef noise values instead of a single noise value.
* Added support for optional subtask constraints based on DexMimicGen to the mimic configuration class :class:`isaaclab.envs.MimicEnvCfg`.
* Enabled data compression in HDF5 dataset file handler :class:`isaaclab.utils.datasets.hdf5_dataset_file_handler.HDF5DatasetFileHandler`.


0.34.11 (2025-03-04)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed issue in :class:`~isaaclab.sensors.TiledCamera` and :class:`~isaaclab.sensors.Camera` where segmentation outputs only display the first tile
  when scene instancing is enabled. A workaround is added for now to disable instancing when segmentation
  outputs are requested.


0.34.10 (2025-03-04)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the issue of misalignment in the motion vectors from the :class:`TiledCamera`
  with other modalities such as RGBA and depth.


0.34.9 (2025-03-04)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added methods inside the :class:`omni.isaac.lab.assets.Articulation` class to set the joint
  position and velocity for the articulation. Previously, the joint position and velocity could
  only be set using the :meth:`omni.isaac.lab.assets.Articulation.write_joint_state_to_sim` method,
  which didn't allow setting the joint position and velocity separately.


0.34.8 (2025-03-02)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the propagation of the :attr:`activate_contact_sensors` attribute to the
  :class:`~isaaclab.sim.spawners.wrappers.wrappers_cfg.MultiAssetSpawnerCfg` class. Previously, this value
  was always set to False, which led to incorrect contact sensor settings for the spawned assets.


0.34.7 (2025-03-02)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Enabled the physics flag for disabling contact processing in the :class:`~isaaclab.sim.SimulationContact`
  class. This means that by default, no contact reporting is done by the physics engine, which should provide
  a performance boost in simulations with no contact processing requirements.
* Disabled the physics flag for disabling contact processing in the :class:`~isaaclab.sensors.ContactSensor`
  class when the sensor is created to allow contact reporting for the sensor.

Removed
^^^^^^^

* Removed the attribute ``disable_contact_processing`` from :class:`~isaaclab.sim.SimulationContact`.


0.34.6 (2025-03-01)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a new attribute :attr:`is_implicit_model` to the :class:`isaaclab.actuators.ActuatorBase` class to
  indicate if the actuator model is implicit or explicit. This helps checking that the correct model type
  is being used when initializing the actuator models.

Fixed
^^^^^

* Added copy of configurations to :class:`~isaaclab.assets.AssetBase` and :class:`~isaaclab.sensors.SensorBase`
  to prevent modifications of the configurations from leaking outside of the classes.
* Fixed the case where setting velocity/effort limits for the simulation in the
  :class:`~isaaclab.actuators.ActuatorBaseCfg` class was not being used to update the actuator-specific
  velocity/effort limits.

Changed
^^^^^^^

* Moved warnings and checks for implicit actuator models to the :class:`~isaaclab.actuators.ImplicitActuator` class.
* Reverted to IsaacLab v1.3 behavior where :attr:`isaaclab.actuators.ImplicitActuatorCfg.velocity_limit`
  attribute was not used for setting the velocity limits in the simulation. This makes it possible to deploy
  policies from previous release without any changes. If users want to set the velocity limits for the simulation,
  they should use the :attr:`isaaclab.actuators.ImplicitActuatorCfg.velocity_limit_sim` attribute instead.


0.34.5 (2025-02-28)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added IP address support for WebRTC livestream to allow specifying IP address to stream across networks.
  This feature requires an updated livestream extension, which is current only available in the pre-built Isaac Lab 2.0.1 docker image.
  Support for other Isaac Sim builds will become available in Isaac Sim 5.0.


0.34.4 (2025-02-27)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Refactored retargeting code from Se3Handtracking class into separate modules for better modularity
* Added scaffolding for developing additional retargeters (e.g. dex)


0.34.3 (2025-02-26)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Enablec specifying the placement of the simulation when viewed in an XR device. This is achieved by
  adding an ``XrCfg`` environment configuration with ``anchor_pos`` and ``anchor_rot`` parameters.


0.34.2 (2025-02-21)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed setting of root velocities inside the event term :meth:`reset_root_state_from_terrain`. Earlier, the indexing
  based on the environment IDs was missing.


0.34.1 (2025-02-17)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Ensured that the loaded torch JIT models inside actuator networks are correctly set to eval mode
  to prevent any unexpected behavior during inference.


0.34.0 (2025-02-14)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added attributes :attr:`velocity_limits_sim` and :attr:`effort_limits_sim` to the
  :class:`isaaclab.actuators.ActuatorBaseCfg` class to separate solver limits from actuator limits.


0.33.17 (2025-02-13)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed Imu sensor based observations at first step by updating scene during initialization for
  :class:`~isaaclab.envs.ManagerBasedEnv`, :class:`~isaaclab.envs.DirectRLEnv`, and :class:`~isaaclab.envs.DirectMARLEnv`


0.33.16 (2025-02-09)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Removes old deprecation warning from :attr:`isaaclab.assets.RigidObectData.body_state_w`


0.33.15 (2025-02-09)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed not updating the ``drift`` when calling :func:`~isaaclab.sensors.RayCaster.reset`


0.33.14 (2025-02-01)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed not updating the timestamp of ``body_link_state_w`` and ``body_com_state_w`` when ``write_root_pose_to_sim`` and ``write_joint_state_to_sim`` in the ``Articulation`` class are called.


0.33.13 (2025-01-30)
~~~~~~~~~~~~~~~~~~~~

* Fixed resampling of interval time left for the next event in the :class:`~isaaclab.managers.EventManager`
  class. Earlier, the time left for interval-based events was not being resampled on episodic resets. This led
  to the event being triggered at the wrong time after the reset.


0.33.12 (2025-01-28)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed missing import in ``line_plot.py``


0.33.11 (2025-01-25)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :attr:`isaaclab.scene.InteractiveSceneCfg.filter_collisions` to allow specifying whether collision masking across environments is desired.

Changed
^^^^^^^

* Automatic collision filtering now happens as part of the replicate_physics call. When replicate_physics is not enabled, we call the previous
  ``filter_collisions`` API to mask collisions between environments.


0.33.10 (2025-01-22)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* In :meth:`isaaclab.assets.Articulation.write_joint_limits_to_sim`, we previously added a check for if default joint positions exceed the
  new limits being set. When this is True, we log a warning message to indicate that the default joint positions will be clipped to be within
  the range of the new limits. However, the warning message can become overly verbose in a randomization setting where this API is called on
  every environment reset. We now default to only writing the message to info level logging if called within randomization, and expose a
  parameter that can be used to choose the logging level desired.


0.33.9 (2025-01-22)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed typo in /physics/autoPopupSimulationOutputWindow setting in :class:`~isaaclab.sim.SimulationContext`


0.33.8 (2025-01-17)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Removed deprecation of :attr:`isaaclab.assets.ArticulationData.root_state_w` and
  :attr:`isaaclab.assets.ArticulationData.body_state_w` derived properties.
* Removed deprecation of :meth:`isaaclab.assets.Articulation.write_root_state_to_sim`.
* Replaced calls to :attr:`isaaclab.assets.ArticulationData.root_com_state_w` and
  :attr:`isaaclab.assets.ArticulationData.root_link_state_w` with corresponding calls to
  :attr:`isaaclab.assets.ArticulationData.root_state_w`.
* Replaced calls to :attr:`isaaclab.assets.ArticulationData.body_com_state_w` and
  :attr:`isaaclab.assets.ArticulationData.body_link_state_w` properties with corresponding calls to
  :attr:`isaaclab.assets.ArticulationData.body_state_w` properties.
* Removed deprecation of :attr:`isaaclab.assets.RigidObjectData.root_state_w` derived properties.
* Removed deprecation of :meth:`isaaclab.assets.RigidObject.write_root_state_to_sim`.
* Replaced calls to :attr:`isaaclab.assets.RigidObjectData.root_com_state_w` and
  :attr:`isaaclab.assets.RigidObjectData.root_link_state_w` properties with corresponding calls to
  :attr:`isaaclab.assets.RigidObjectData.root_state_w` properties.
* Removed deprecation of :attr:`isaaclab.assets.RigidObjectCollectionData.root_state_w` derived properties.
* Removed deprecation of :meth:`isaaclab.assets.RigidObjectCollection.write_root_state_to_sim`.
* Replaced calls to :attr:`isaaclab.assets.RigidObjectCollectionData.root_com_state_w` and
  :attr:`isaaclab.assets.RigidObjectData.root_link_state_w` properties with corresponding calls to
  :attr:`isaaclab.assets.RigidObjectData.root_state_w` properties.
* Fixed indexing issue in ``write_root_link_velocity_to_sim`` in :class:`isaaclab.assets.RigidObject`
* Fixed index broadcasting in ``write_object_link_velocity_to_sim`` and ``write_object_com_pose_to_sim`` in
  the :class:`isaaclab.assets.RigidObjectCollection` class.


0.33.7 (2025-01-14)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the respawn of only wrong object samples in :func:`repeated_objects_terrain` of :mod:`isaaclab.terrains.trimesh` module.
  Previously, the function was respawning all objects in the scene instead of only the wrong object samples, which in worst case
  could lead to infinite respawn loop.


0.33.6 (2025-01-16)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added initial unit tests for multiple tiled cameras, including tests for initialization, groundtruth annotators, different poses, and different resolutions.


0.33.5 (2025-01-13)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved the definition of ``/persistent/isaac/asset_root/*`` settings from :class:`AppLauncher` to the app files.
  This is needed to prevent errors where ``isaaclab_assets`` was loaded prior to the carbonite setting being set.


0.33.4 (2025-01-10)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added an optional parameter in the :meth:`record_pre_reset` method in
  :class:`~isaaclab.managers.RecorderManager` to override the export config upon invoking.


0.33.3 (2025-01-08)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed docstring in articulation data :class:`isaaclab.assets.ArticulationData`.
  In body properties sections, the second dimension should be num_bodies but was documented as 1.


0.33.2 (2025-01-02)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added body tracking as an origin type to :class:`isaaclab.envs.ViewerCfg` and :class:`isaaclab.envs.ui.ViewportCameraController`.


0.33.1 (2024-12-26)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added kinematics initialization call for populating kinematic prim transforms to fabric for rendering.
* Added ``enable_env_ids`` flag for cloning and replication to replace collision filtering.


0.33.0 (2024-12-22)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed populating default_joint_stiffness and default_joint_damping values for ImplicitActuator instances in :class:`isaaclab.assets.Articulation`


0.32.2 (2024-12-17)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added null-space (position) control option to :class:`isaaclab.controllers.OperationalSpaceController`.
* Added test cases that uses null-space control for :class:`isaaclab.controllers.OperationalSpaceController`.
* Added information regarding null-space control to the tutorial script and documentation of
  :class:`isaaclab.controllers.OperationalSpaceController`.
* Added arguments to set specific null-space joint position targets within
  :class:`isaaclab.envs.mdp.actions.OperationalSpaceControllerAction` class.


0.32.1 (2024-12-17)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added a default and generic implementation of the :meth:`get_object_poses` function
  in the :class:`ManagerBasedRLMimicEnv` class.
* Added a ``EXPORT_NONE`` mode in the :class:`DatasetExportMode` class and updated
  :class:`~isaaclab.managers.RecorderManager` to enable recording without exporting
  the data to a file.


0.32.0 (2024-12-16)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Previously, physx returns the rigid bodies and articulations velocities in the com of bodies rather than the
  link frame, while poses are in link frames. We now explicitly provide :attr:`body_link_state` and
  :attr:`body_com_state` APIs replacing the previous :attr:`body_state` API. Previous APIs are now marked as
  deprecated. Please update any code using the previous pose and velocity APIs to use the new
  ``*_link_*`` or ``*_com_*`` APIs in :attr:`isaaclab.assets.RigidBody`,
  :attr:`isaaclab.assets.RigidBodyCollection`, and :attr:`isaaclab.assets.Articulation`.


0.31.0 (2024-12-16)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`ManagerBasedRLMimicEnv` and config classes for mimic data generation workflow for imitation learning.


0.30.3 (2024-12-16)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed ordering of logging and resamping in the command manager, where we were logging the metrics
  after resampling the commands. This leads to incorrect logging of metrics when inside the resample call,
  the metrics tensors get reset.


0.30.2 (2024-12-16)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed errors within the calculations of :class:`isaaclab.controllers.OperationalSpaceController`.

Added
^^^^^

* Added :class:`isaaclab.controllers.OperationalSpaceController` to API documentation.
* Added test cases for :class:`isaaclab.controllers.OperationalSpaceController`.
* Added a tutorial for :class:`isaaclab.controllers.OperationalSpaceController`.
* Added the implementation of :class:`isaaclab.envs.mdp.actions.OperationalSpaceControllerAction` class.


0.30.1 (2024-12-15)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added call to update articulation kinematics after reset to ensure states are updated for non-rendering sensors.
  Previously, some changes in reset such as modifying joint states would not be reflected in the rigid body
  states immediately after reset.


0.30.0 (2024-12-15)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added UI interface to the Managers in the ManagerBasedEnv and MangerBasedRLEnv classes.
* Added UI widgets for :class:`LiveLinePlot` and :class:`ImagePlot`.
* Added ``ManagerLiveVisualizer/Cfg``: Given a ManagerBase (i.e. action_manager, observation_manager, etc) and a
  config file this class creates the the interface between managers and the UI.
* Added :class:`EnvLiveVisualizer`: A 'manager' of ManagerLiveVisualizer. This is added to the ManagerBasedEnv
  but is only called during the initialization of the managers in load_managers
* Added ``get_active_iterable_terms`` implementation methods to ActionManager, ObservationManager, CommandsManager,
  CurriculumManager, RewardManager, and TerminationManager. This method exports the active term data and labels
  for each manager and is called by ManagerLiveVisualizer.
* Additions to :class:`BaseEnvWindow` and :class:`RLEnvWindow` to register ManagerLiveVisualizer UI interfaces
  for the chosen managers.


0.29.0 (2024-12-15)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added observation history computation to :class:`isaaclab.manager.observation_manager.ObservationManager`.
* Added ``history_length`` and ``flatten_history_dim`` configuration parameters to :class:`isaaclab.manager.manager_term_cfg.ObservationTermCfg`
* Added ``history_length`` and ``flatten_history_dim`` configuration parameters to :class:`isaaclab.manager.manager_term_cfg.ObservationGroupCfg`
* Added full buffer property to :class:`isaaclab.utils.buffers.circular_buffer.CircularBuffer`


0.28.4 (2024-12-15)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added action clip to all :class:`isaaclab.envs.mdp.actions`.


0.28.3 (2024-12-14)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added check for error below threshold in state machines to ensure the state has been reached.


0.28.2 (2024-12-13)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the shape of ``quat_w`` in the ``apply_actions`` method of :attr:`~isaaclab.env.mdp.NonHolonomicAction`
  (previously (N,B,4), now (N,4) since the number of root bodies B is required to be 1). Previously ``apply_actions``
  errored because ``euler_xyz_from_quat`` requires inputs of shape (N,4).


0.28.1 (2024-12-13)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the internal buffers for ``set_external_force_and_torque`` where the buffer values would be stale if zero
  values are sent to the APIs.


0.28.0 (2024-12-12)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Adapted the :class:`~isaaclab.sim.converters.UrdfConverter` to use the latest URDF converter API from Isaac Sim 4.5.
  The physics articulation root can now be set separately, and the joint drive gains can be set on a per joint basis.


0.27.33 (2024-12-11)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Introduced an optional ``sensor_cfg`` parameter to the :meth:`~isaaclab.envs.mdp.rewards.base_height_l2` function,
  enabling the use of :class:`~isaaclab.sensors.RayCaster` for height adjustments. For flat terrains, the function
  retains its previous behavior.
* Improved documentation to clarify the usage of the :meth:`~isaaclab.envs.mdp.rewards.base_height_l2` function in
  both flat and rough terrain settings.


0.27.32 (2024-12-11)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Modified :class:`isaaclab.envs.mdp.actions.DifferentialInverseKinematicsAction` class to use the geometric
  Jacobian computed w.r.t. to the root frame of the robot. This helps ensure that root pose does not affect the tracking.


0.27.31 (2024-12-09)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Introduced configuration options in :class:`Se3HandTracking` to:

  - Zero out rotation around the x/y axes
  - Apply smoothing and thresholding to position and rotation deltas for reduced jitter
  - Use wrist-based rotation reference as an alternative to fingertip-based rotation

* Switched the default position reference in :class:`Se3HandTracking` to the wrist joint pose, providing more stable
  relative-based positioning.


0.27.30 (2024-12-09)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the initial state recorder term in :class:`isaaclab.envs.mdp.recorders.InitialStateRecorder` to
  return only the states of the specified environment IDs.


0.27.29 (2024-12-06)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the enforcement of :attr:`~isaaclab.actuators.ActuatorBaseCfg.velocity_limits` at the
  :attr:`~isaaclab.assets.Articulation.root_physx_view` level.


0.27.28 (2024-12-06)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* If a USD that contains an articulation root is loaded using a
  :attr:`isaaclab.assets.RigidBody` we now fail unless the articulation root is explicitly
  disabled. Using an articulation root for rigid bodies is not needed and decreases overall performance.


0.27.27 (2024-12-06)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Corrected the projection types of fisheye camera in :class:`isaaclab.sim.spawners.sensors.sensors_cfg.FisheyeCameraCfg`.
  Earlier, the projection names used snakecase instead of camelcase.


0.27.26 (2024-12-06)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added option to define the clipping behavior for depth images generated by
  :class:`~isaaclab.sensors.RayCasterCamera`, :class:`~isaaclab.sensors.Camera`, and :class:`~isaaclab.sensors.TiledCamera`

Changed
^^^^^^^

* Unified the clipping behavior for the depth images of all camera implementations. Per default, all values exceeding
  the range are clipped to zero for both ``distance_to_image_plane`` and ``distance_to_camera`` depth images. Prev.
  :class:`~isaaclab.sensors.RayCasterCamera` clipped the values to the maximum value of the depth image,
  :class:`~isaaclab.sensors.Camera` did not clip them and had a different behavior for both types.


0.27.25 (2024-12-05)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the condition in ``isaaclab.sh`` that checks whether ``pre-commit`` is installed before attempting installation.


0.27.24 (2024-12-05)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Removed workaround in :class:`isaaclab.sensors.TiledCamera` and :class:`isaaclab.sensors.Camera`
  that was previously required to prevent frame offsets in renders. The denoiser setting is no longer
  automatically modified based on the resolution of the cameras.


0.27.23 (2024-12-04)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added the attributes :attr:`~isaaclab.envs.DirectRLEnvCfg.wait_for_textures` and
  :attr:`~isaaclab.envs.ManagerBasedEnvCfg.wait_for_textures` to enable assets loading check
  during :class:`~isaaclab.DirectRLEnv` and :class:`~isaaclab.ManagerBasedEnv` reset method when
  rtx sensors are added to the scene.


0.27.22 (2024-12-04)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the order of the incoming parameters in :class:`isaaclab.envs.DirectMARLEnv` to correctly use
  ``NoiseModel`` in marl-envs.


0.27.21 (2024-12-04)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab.managers.RecorderManager` and its utility classes to record data from the simulation.
* Added :class:`~isaaclab.utils.datasets.EpisodeData` to store data for an episode.
* Added :class:`~isaaclab.utils.datasets.DatasetFileHandlerBase` as a base class for handling dataset files.
* Added :class:`~isaaclab.utils.datasets.HDF5DatasetFileHandler` as a dataset file handler implementation to
  export and load episodes from HDF5 files.
* Added ``record_demos.py`` script to record human-teleoperated demos for a specified task and export to an HDF5 file.
* Added ``replay_demos.py`` script to replay demos loaded from an HDF5 file.


0.27.20 (2024-12-02)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed :class:`isaaclab.envs.DirectMARLEnv` to inherit from ``Gymnasium.Env`` due to requirement from Gymnasium
  v1.0.0 requiring all environments to be a subclass of ``Gymnasium.Env`` when using the ``make`` interface.


0.27.19 (2024-12-02)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``isaaclab.utils.pretrained_checkpoints`` containing constants and utility functions used to manipulate
  paths and load checkpoints from Nucleus.


0.27.18 (2024-11-28)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Renamed Isaac Sim imports to follow Isaac Sim 4.5 naming conventions.


0.27.17 (2024-11-20)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``create_new_stage`` setting in :class:`~isaaclab.app.AppLauncher` to avoid creating a default new
  stage on startup in Isaac Sim. This helps reduce the startup time when launching Isaac Lab.


0.27.16 (2024-11-15)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the class :class:`~isaaclab.devices.Se3HandTracking` which enables XR teleop for manipulators.


0.27.15 (2024-11-09)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed indexing in :meth:`isaaclab.assets.Articulation.write_joint_limits_to_sim` to correctly process
  non-None ``env_ids`` and ``joint_ids``.


0.27.14 (2024-10-23)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the class :class:`~isaaclab.assets.RigidObjectCollection` which allows to spawn
  multiple objects in each environment and access/modify the quantities with a unified (env_ids, object_ids) API.


0.27.13 (2024-10-30)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the attributes :attr:`~isaaclab.sim.converters.MeshConverterCfg.translation`, :attr:`~isaaclab.sim.converters.MeshConverterCfg.rotation`,
  :attr:`~isaaclab.sim.converters.MeshConverterCfg.scale` to translate, rotate, and scale meshes
  when importing them with :class:`~isaaclab.sim.converters.MeshConverter`.


0.27.12 (2024-11-04)
~~~~~~~~~~~~~~~~~~~~

Removed
^^^^^^^

* Removed TensorDict usage in favor of Python dictionary in sensors


0.27.11 (2024-10-31)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support to define tuple of floats to scale observation terms by expanding the
  :attr:`isaaclab.managers.manager_term_cfg.ObservationManagerCfg.scale` attribute.


0.27.10 (2024-11-01)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Cached the PhysX view's joint paths before looping over them when processing fixed joint tendons
  inside the :class:`Articulation` class. This helps improve the processing time for the tendons.


0.27.9 (2024-11-01)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the :class:`isaaclab.utils.types.ArticulationActions` class to store the joint actions
  for an articulation. Earlier, the class from Isaac Sim was being used. However, it used a different
  type for the joint actions which was not compatible with the Isaac Lab framework.


0.27.8 (2024-11-01)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added sanity check if the term is a valid type inside the command manager.
* Corrected the iteration over ``group_cfg_items`` inside the observation manager.


0.27.7 (2024-10-28)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added frozen encoder feature extraction observation space with ResNet and Theia


0.27.6 (2024-10-25)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed usage of ``meshes`` property in :class:`isaaclab.sensors.RayCasterCamera` to use ``self.meshes``
  instead of the undefined ``RayCaster.meshes``.
* Fixed issue in :class:`isaaclab.envs.ui.BaseEnvWindow` where undefined configs were being accessed when
  creating debug visualization elements in UI.


0.27.5 (2024-10-25)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added utilities for serializing/deserializing Gymnasium spaces.


0.27.4 (2024-10-18)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Updated installation path instructions for Windows in the Isaac Lab documentation to remove redundancy in the
  use of %USERPROFILE% for path definitions.


0.27.3 (2024-10-22)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the issue with using list or tuples of ``configclass`` within a ``configclass``. Earlier, the list of
  configclass objects were not converted to dictionary properly when ``to_dict`` function was called.


0.27.2 (2024-10-21)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``--kit_args`` to :class:`~isaaclab.app.AppLauncher` to allow passing command line arguments directly to
  Omniverse Kit SDK.


0.27.1 (2024-10-20)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab.sim.RenderCfg` and the attribute :attr:`~isaaclab.sim.SimulationCfg.render` for
  specifying render related settings.


0.27.0 (2024-10-14)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a method to :class:`~isaaclab.utils.configclass` to check for attributes with values of
  type ``MISSING``. This is useful when the user wants to check if a certain attribute has been set or not.
* Added the configuration validation check inside the constructor of all the core classes
  (such as sensor base, asset base, scene and environment base classes).
* Added support for environments without commands by leaving the attribute
  :attr:`isaaclab.envs.ManagerBasedRLEnvCfg.commands` as None. Before, this had to be done using
  the class :class:`isaaclab.command_generators.NullCommandGenerator`.
* Moved the ``meshes`` attribute in the :class:`isaaclab.sensors.RayCaster` class from class variable to instance variable.
  This prevents the meshes to overwrite each other.


0.26.0 (2024-10-16)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added Imu sensor implementation that directly accesses the physx view :class:`isaaclab.sensors.Imu`. The
  sensor comes with a configuration class :class:`isaaclab.sensors.ImuCfg` and data class
  :class:`isaaclab.sensors.ImuData`.
* Moved and renamed :meth:`isaaclab.sensors.camera.utils.convert_orientation_convention` to
  :meth:`isaaclab.utils.math.convert_camera_frame_orientation_convention`
* Moved :meth:`isaaclab.sensors.camera.utils.create_rotation_matrix_from_view` to
  :meth:`isaaclab.utils.math.create_rotation_matrix_from_view`


0.25.2 (2024-10-16)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for different Gymnasium spaces (``Box``, ``Discrete``, ``MultiDiscrete``, ``Tuple`` and ``Dict``)
  to define observation, action and state spaces in the direct workflow.
* Added :meth:`sample_space` to environment utils to sample supported spaces where data containers are torch tensors.

Changed
^^^^^^^

* Mark the :attr:`num_observations`, :attr:`num_actions` and :attr:`num_states` in :class:`DirectRLEnvCfg` as deprecated
  in favor of :attr:`observation_space`, :attr:`action_space` and :attr:`state_space` respectively.
* Mark the :attr:`num_observations`, :attr:`num_actions` and :attr:`num_states` in :class:`DirectMARLEnvCfg` as deprecated
  in favor of :attr:`observation_spaces`, :attr:`action_spaces` and :attr:`state_space` respectively.


0.25.1 (2024-10-10)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed potential issue where default joint positions can fall outside of the limits being set with Articulation's
  ``write_joint_limits_to_sim`` API.


0.25.0 (2024-10-06)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added configuration classes for spawning assets from a list of individual asset configurations randomly
  at the specified prim paths.


0.24.20 (2024-10-07)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the :meth:`isaaclab.envs.mdp.events.randomize_rigid_body_material` function to
  correctly sample friction and restitution from the given ranges.


0.24.19 (2024-10-05)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added new functionalities to the FrameTransformer to make it more general. It is now possible to track:

  * Target frames that aren't children of the source frame prim_path
  * Target frames that are based upon the source frame prim_path


0.24.18 (2024-10-04)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixes parsing and application of ``size`` parameter for :class:`~isaaclab.sim.spawn.GroundPlaneCfg` to correctly
  scale the grid-based ground plane.


0.24.17 (2024-10-04)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the deprecation notice for using ``pxr.Semantics``. The corresponding modules use ``Semantics`` module
  directly.


0.24.16 (2024-10-03)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Renamed the observation function :meth:`grab_images` to :meth:`image` to follow convention of noun-based naming.
* Renamed the function :meth:`convert_perspective_depth_to_orthogonal_depth` to a shorter name
  :meth:`isaaclab.utils.math.orthogonalize_perspective_depth`.


0.24.15 (2024-09-20)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :meth:`grab_images` to be able to use images for an observation term in manager-based environments.


0.24.14 (2024-09-20)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the method :meth:`convert_perspective_depth_to_orthogonal_depth` to convert perspective depth
  images to orthogonal depth images. This is useful for the :meth:`~isaaclab.utils.math.unproject_depth`,
  since it expects orthogonal depth images as inputs.


0.24.13 (2024-09-08)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved the configuration of visualization markers for the command terms to their respective configuration classes.
  This allows users to modify the markers for the command terms without having to modify the command term classes.


0.24.12 (2024-09-18)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed outdated fetching of articulation data by using the method ``update_articulations_kinematic`` in
  :class:`isaaclab.assets.ArticulationData`. Before if an articulation was moved during a reset, the pose of the
  links were outdated if fetched before the next physics step. Adding this method ensures that the pose of the links
  is always up-to-date. Similarly ``update_articulations_kinematic`` was added before any render step to ensure that the
  articulation displays correctly after a reset.


0.24.11 (2024-09-11)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added skrl's JAX environment variables to :class:`~isaaclab.app.AppLauncher`
  to support distributed multi-GPU and multi-node training using JAX


0.24.10 (2024-09-10)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added config class, support, and tests for MJCF conversion via standalone python scripts.


0.24.9 (2024-09-09)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a seed parameter to the :attr:`isaaclab.envs.ManagerBasedEnvCfg` and :attr:`isaaclab.envs.DirectRLEnvCfg`
  classes to set the seed for the environment. This seed is used to initialize the random number generator for the environment.
* Adapted the workflow scripts to set the seed for the environment using the seed specified in the learning agent's configuration
  file or the command line argument. This ensures that the simulation results are reproducible across different runs.


0.24.8 (2024-09-08)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Modified:meth:`quat_rotate` and :meth:`quat_rotate_inverse` operations to use :meth:`torch.einsum`
  for faster processing of high dimensional input tensors.


0.24.7 (2024-09-06)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added support for property attributes in the :meth:``isaaclab.utils.configclass`` method.
  Earlier, the configclass decorator failed to parse the property attributes correctly and made them
  instance variables instead.


0.24.6 (2024-09-05)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Adapted the ``A`` and ``D`` button bindings inside :meth:`isaaclab.device.Se3Keyboard` to make them now
  more-intuitive to control the y-axis motion based on the right-hand rule.


0.24.5 (2024-08-29)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added alternative data type "distance_to_camera" in :class:`isaaclab.sensors.TiledCamera` class to be
  consistent with all other cameras (equal to type "depth").


0.24.4 (2024-09-02)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added missing SI units to the documentation of :class:`isaaclab.sensors.Camera` and
  :class:`isaaclab.sensors.RayCasterCamera`.
* Added test to check :attr:`isaaclab.sensors.RayCasterCamera.set_intrinsic_matrices`


0.24.3 (2024-08-29)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the support for class-bounded methods when creating a configclass
  out of them. Earlier, these methods were being made as instance methods
  which required initialization of the class to call the class-methods.


0.24.2 (2024-08-28)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a class method to initialize camera configurations with an intrinsic matrix in the
  :class:`isaaclab.sim.spawner.sensors.PinholeCameraCfg`
  :class:`isaaclab.sensors.ray_caster.patterns_cfg.PinholeCameraPatternCfg` classes.

Fixed
^^^^^

* Fixed the ray direction in :func:`isaaclab.sensors.ray_caster.patterns.patterns.pinhole_camera_pattern` to
  point to the center of the pixel instead of the top-left corner.
* Fixed the clipping of the "distance_to_image_plane" depth image obtained using the
  :class:`isaaclab.sensors.ray_caster.RayCasterCamera` class. Earlier, the depth image was being clipped
  before the depth image was generated. Now, the clipping is applied after the depth image is generated. This makes
  the behavior equal to the USD Camera.


0.24.1 (2024-08-21)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Disabled default viewport in certain headless scenarios for better performance.


0.24.0 (2024-08-17)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added additional annotators for :class:`isaaclab.sensors.camera.TiledCamera` class.

Changed
^^^^^^^

* Updated :class:`isaaclab.sensors.TiledCamera` to latest RTX tiled rendering API.
* Single channel outputs for :class:`isaaclab.sensors.TiledCamera`, :class:`isaaclab.sensors.Camera` and :class:`isaaclab.sensors.RayCasterCamera` now has shape (H, W, 1).
* Data type for RGB output for :class:`isaaclab.sensors.TiledCamera` changed from ``torch.float`` to ``torch.uint8``.
* Dimension of RGB output for :class:`isaaclab.sensors.Camera` changed from (H, W, 4) to (H, W, 3). Use type ``rgba`` to retrieve the previous dimension.


0.23.1 (2024-08-17)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated torch to version 2.4.0.


0.23.0 (2024-08-16)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added direct workflow base class :class:`isaaclab.envs.DirectMARLEnv` for multi-agent environments.


0.22.1 (2024-08-17)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added APIs to interact with the physics simulation of deformable objects. This includes setting the
  material properties, setting kinematic targets, and getting the state of the deformable object.
  For more information, please refer to the :mod:`isaaclab.assets.DeformableObject` class.


0.22.0 (2024-08-14)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`~isaaclab.utils.modifiers` module to provide framework for configurable and custom
  observation data modifiers.
* Adapted the :class:`~isaaclab.managers.ObservationManager` class to support custom modifiers.
  These are applied to the observation data before applying any noise or scaling operations.


0.21.2 (2024-08-13)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Moved event mode-based checks in the :meth:`isaaclab.managers.EventManager.apply` method outside
  the loop that iterates over the event terms. This prevents unnecessary checks and improves readability.
* Fixed the logic for global and per environment interval times when using the "interval" mode inside the
  event manager. Earlier, the internal lists for these times were of unequal lengths which led to wrong indexing
  inside the loop that iterates over the event terms.


0.21.1 (2024-08-06)
~~~~~~~~~~~~~~~~~~~

* Added a flag to preserve joint ordering inside the :class:`isaaclab.envs.mdp.JointAction` action term.


0.21.0 (2024-08-05)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added the command line argument ``--device`` in :class:`~isaaclab.app.AppLauncher`. Valid options are:

  * ``cpu``: Use CPU.
  * ``cuda``: Use GPU with device ID ``0``.
  * ``cuda:N``: Use GPU, where N is the device ID. For example, ``cuda:0``. The default value is ``cuda:0``.

Changed
^^^^^^^

* Simplified setting the device throughout the code by relying on :attr:`isaaclab.sim.SimulationCfg.device`
  to activate gpu/cpu pipelines.

Removed
^^^^^^^

* Removed the parameter :attr:`isaaclab.sim.SimulationCfg.use_gpu_pipeline`. This is now directly inferred from
  :attr:`isaaclab.sim.SimulationCfg.device`.
* Removed the command line input argument ``--device_id`` in :class:`~isaaclab.app.AppLauncher`. The device id can
  now be set using the ``--device`` argument, for example with ``--device cuda:0``.


0.20.8 (2024-08-02)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the handling of observation terms with different shapes in the
  :class:`~isaaclab.managers.ObservationManager` class. Earlier, the constructor would throw an error if the
  shapes of the observation terms were different. Now, this operation only happens when the terms in an observation
  group are being concatenated. Otherwise, the terms are stored as a dictionary of tensors.
* Improved the error message when the observation terms are not of the same shape in the
  :class:`~isaaclab.managers.ObservationManager` class and the terms are being concatenated.


0.20.7 (2024-08-02)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Performance improvements for material randomization in events.

Added
^^^^^

* Added minimum randomization frequency for reset mode randomizations.


0.20.6 (2024-08-02)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Removed the hierarchy from :class:`~isaaclab.assets.RigidObject` class to
  :class:`~isaaclab.assets.Articulation` class. Previously, the articulation class overrode  almost
  all the functions of the rigid object class making the hierarchy redundant. Now, the articulation class
  is a standalone class that does not inherit from the rigid object class. This does add some code
  duplication but the simplicity and clarity of the code is improved.


0.20.5 (2024-08-02)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :attr:`isaaclab.terrain.TerrainGeneratorCfg.border_height` to set the height of the border
  around the terrain.


0.20.4 (2024-08-02)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the caching of terrains when using the :class:`isaaclab.terrains.TerrainGenerator` class.
  Earlier, the random sampling of the difficulty levels led to different hash values for the same terrain
  configuration. This caused the terrains to be re-generated even when the same configuration was used.
  Now, the numpy random generator is seeded with the same seed to ensure that the difficulty levels are
  sampled in the same order between different runs.


0.20.3 (2024-08-02)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the setting of translation and orientation when spawning a mesh prim. Earlier, the translation
  and orientation was being applied both on the parent Xform and the mesh prim. This was causing the
  mesh prim to be offset by the translation and orientation of the parent Xform, which is not the intended
  behavior.


0.20.2 (2024-08-02)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Modified the computation of body acceleration for rigid body data to use PhysX APIs instead of
  numerical finite-differencing. This removes the need for computation of body acceleration at
  every update call of the data buffer.


0.20.1 (2024-07-30)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the :meth:`isaaclab.utils.math.wrap_to_pi` method to handle the wrapping of angles correctly.
  Earlier, the method was not wrapping the angles to the range [-pi, pi] correctly when the angles were outside
  the range [-2*pi, 2*pi].


0.20.0 (2024-07-26)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Support for the Isaac Sim 4.1.0 release.

Removed
^^^^^^^

* The ``mdp.add_body_mass`` method in the events. Please use the
  :meth:`isaaclab.envs.mdp.randomize_rigid_body_mass` method instead.
* The classes ``managers.RandomizationManager`` and ``managers.RandomizationTermCfg`` are replaced with
  :class:`isaaclab.managers.EventManager` and :class:`isaaclab.managers.EventTermCfg` classes.
* The following properties in :class:`isaaclab.sensors.FrameTransformerData`:

  * ``target_rot_source`` --> :attr:`~isaaclab.sensors.FrameTransformerData.target_quat_w`
  * ``target_rot_w`` --> :attr:`~isaaclab.sensors.FrameTransformerData.target_quat_source`
  * ``source_rot_w`` --> :attr:`~isaaclab.sensors.FrameTransformerData.source_quat_w`

* The kit experience file ``isaaclab.backwards.compatible.kit``. This is followed by dropping the support for
  Isaac Sim 2023.1.1 completely.


0.19.4 (2024-07-13)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added the call to "startup" events when using the :class:`~isaaclab.envs.ManagerBasedEnv` class.
  Earlier, the "startup" events were not being called when the environment was initialized. This issue
  did not occur when using the :class:`~isaaclab.envs.ManagerBasedRLEnv` class since the "startup"
  events were called in the constructor.


0.19.3 (2024-07-13)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added schemas for setting and modifying deformable body properties on a USD prim.
* Added API to spawn a deformable body material in the simulation.
* Added APIs to spawn rigid and deformable meshes of primitive shapes (cone, cylinder, sphere, box, capsule)
  in the simulation. This is possible through the :mod:`isaaclab.sim.spawners.meshes` module.


0.19.2 (2024-07-05)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Modified cloning scheme based on the attribute :attr:`~isaaclab.scene.InteractiveSceneCfg.replicate_physics`
  to determine whether environment is homogeneous or heterogeneous.


0.19.1 (2024-07-05)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a lidar pattern function :func:`~isaaclab.sensors.ray_caster.patterns.patterns.lidar_pattern` with
  corresponding config :class:`~isaaclab.sensors.ray_caster.patterns_cfg.LidarPatternCfg`.


0.19.0 (2024-07-04)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed parsing of articulations with nested rigid links while using the :class:`isaaclab.assets.Articulation`
  class. Earlier, the class initialization failed when the articulation had nested rigid links since the rigid
  links were not being parsed correctly by the PhysX view.

Removed
^^^^^^^

* Removed the attribute :attr:`body_physx_view` from the :class:`isaaclab.assets.Articulation` and
  :class:`isaaclab.assets.RigidObject` classes. These were causing confusions when used with articulation
  view since the body names were not following the same ordering.
* Dropped support for Isaac Sim 2023.1.1. The minimum supported version is now Isaac Sim 4.0.0.


0.18.6 (2024-07-01)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the environment stepping logic. Earlier, the environments' rendering logic was updating the kit app which
  would in turn step the physics :attr:`isaaclab.sim.SimulationCfg.render_interval` times. Now, a render
  call only does rendering and does not step the physics.


0.18.5 (2024-06-26)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the gravity vector direction used inside the :class:`isaaclab.assets.RigidObjectData` class.
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
  provided by Isaac Sim to the ones provided in Isaac Lab's ``apps`` directory.


0.15.5 (2024-03-23)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the env origins in :meth:`_compute_env_origins_grid` of :class:`isaaclab.terrain.TerrainImporter`
  to match that obtained from the Isaac Sim :class:`isaacsim.core.cloner.GridCloner` class.

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

* Fixes the order of size arguments in :meth:`isaaclab.terrains.height_field.random_uniform_terrain`. Previously, the function
  would crash if the size along x and y were not the same.


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
  This class inherits from the :class:`isaacsim.core.api.simulation_context.SimulationContext` class and adds
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
  the SimulationApp and extension loading for remote deployment and ROS bridges.

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
