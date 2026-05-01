Changelog
---------

0.5.29 (2026-04-30)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Added fused :meth:`~isaaclab_physx.assets.Articulation.write_joint_state_to_sim_index`
  that writes joint position and velocity in a single kernel launch instead of two.
* Cached ``.view(wp.float32)`` results in root pose/velocity writers and wrench
  composer views in ``write_data_to_sim`` to avoid per-call wrapper allocations.
* Pre-allocated pinned CPU buffers for all joint property and body property writers,
  replacing per-call ``wp.clone(device="cpu")`` allocations with ``wp.copy`` into
  reusable pinned memory.


0.5.28 (2026-04-29)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed camera observation hang when a visualizer (e.g. KitVisualizer for XR
  teleop) is active and ``--enable_cameras`` is set.
  :func:`~isaaclab_physx.renderers.isaac_rtx_renderer_utils.ensure_isaac_rtx_render_update`
  now performs the initial ``app.update()`` on the very first call for a new
  :class:`~isaaclab.sim.SimulationContext`, even when a visualizer reports that
  it pumps the Kit app loop, because the visualizer has not had a chance to pump
  yet at that point.


0.5.27 (2026-04-27)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed ``Simulation view object is invalidated and cannot be used again to call
  getDofVelocities`` raised on the first ``scene.update()`` after ``sim.reset()``
  with recent Isaac Sim ``develop`` builds. Isaac Sim's
  ``isaacsim.core.simulation_manager.SimulationManager`` recently became reactive
  to timeline ``STOP`` events (after its ``_on_stop`` was decorated with
  ``@staticmethod`` upstream), and its ``invalidate_physics()`` was clobbering
  the shared ``omni.physics.tensors`` simulation view that
  :class:`~isaaclab_physx.physics.PhysxManager` and PhysX articulation views
  rely on. The ``isaaclab_physx`` package init now disables the original Isaac
  Sim ``SimulationManager``'s default timeline/stage callbacks via
  ``enable_all_default_callbacks(False)`` before swapping the module attribute,
  so :class:`PhysxManager` is the single owner of the simulation lifecycle.


0.5.26 (2026-04-27)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed ``import isaaclab_physx`` eagerly importing ``isaacsim``, ``omni``,
  and ``carb`` backend modules when used for pure-data config loading before
  ``SimulationApp`` has launched. The ``SimulationManager`` patch now checks
  ``sys.modules`` lazily instead of force-importing the target module, allowing
  env-cfg classes that reference :class:`~isaaclab_physx.physics.PhysxCfg` to
  be constructed without a running Kit instance (regression caught by
  ``test_env_cfg_no_forbidden_imports``).

Changed
^^^^^^^

* Migrated :func:`~isaaclab_physx.renderers.kit_viewport_utils.set_kit_renderer_camera_view`
  off the deprecated ``isaacsim.core.utils.viewports.set_camera_view`` to
  ``isaacsim.core.rendering_manager.ViewportManager.set_camera_view``, matching the
  pattern used by the Kit perspective video helper.


0.5.25 (2026-04-27)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated imports of the PhysX tensors API from ``omni.physics.tensors.impl.api`` to
  ``omni.physics.tensors.api`` to track the upstream Isaac Sim module relocation
  (the ``impl`` submodule was removed).
* Migrated the PhysX scene data provider, PhysX asset micro-benchmarks, and cross-backend asset
  interface tests off ``isaacsim.core.simulation_manager.SimulationManager`` to
  :class:`~isaaclab_physx.physics.PhysxManager` (imported as ``SimulationManager`` to mirror the
  Newton backend's ``NewtonManager as SimulationManager`` convention).
* Updated optional-extension enablement and Kit perspective capture helpers to use non-deprecated
  Isaac Sim module paths (``isaacsim.core.experimental.utils.app`` and ``isaacsim.core.rendering_manager``).


0.5.24 (2026-04-24)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated :class:`~isaaclab_physx.sim.views.FabricFrameView` to match the new
  :class:`~isaaclab.sim.views.BaseFrameView` ProxyArray return contract. See
  the ``isaaclab`` 4.6.15 changelog for migration guidance.


0.5.23 (2026-04-24)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Properties on the following data classes now return
  :class:`~isaaclab.utils.warp.ProxyArray` instead of raw ``wp.array``:
  :class:`~isaaclab_physx.assets.articulation.ArticulationData`,
  :class:`~isaaclab_physx.assets.rigid_object.RigidObjectData`,
  :class:`~isaaclab_physx.assets.rigid_object_collection.RigidObjectCollectionData`,
  :class:`~isaaclab_physx.assets.deformable_object.DeformableObjectData`,
  :class:`~isaaclab_physx.sensors.contact_sensor.ContactSensorData`,
  :class:`~isaaclab_physx.sensors.frame_transformer.FrameTransformerData`,
  :class:`~isaaclab_physx.sensors.imu.ImuData`, and
  :class:`~isaaclab_physx.sensors.pva.PvaData`.
  Use ``.torch`` for a cached zero-copy ``torch.Tensor`` view, or ``.warp`` for
  the underlying ``wp.array``. Implicit torch operations (arithmetic,
  ``torch.*`` functions) work during the deprecation period but emit a warning.


0.5.22 (2026-04-23)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed ``RuntimeError: NewtonWarpRenderer requires a Newton model but the scene data provider
  returned None`` when a Direct env (e.g. ``ShadowHandVisionEnv``, ``CartpoleCameraEnv``)
  uses ``physx`` physics with the ``newton_warp`` renderer. The
  :class:`~isaaclab_physx.scene_data_providers.PhysxSceneDataProvider` now falls back to a
  USD-traversal Newton build when the cloner-time prebuilt artifact is absent, and stashes
  the freshly built artifact on the simulation context so subsequent providers reuse it.


0.5.21 (2026-04-22)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_physx.sim.views.XformPrimView` providing the PhysX/Fabric
  backend implementation for xform prim views.

Changed
^^^^^^^

* Renamed :class:`~isaaclab_physx.sim.views.FabricXformPrimView` to
  :class:`~isaaclab_physx.sim.views.FabricFrameView`. Old name is kept as a deprecated alias.


0.5.20 (2026-04-21)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated ``write_data_to_sim`` in :class:`~isaaclab_physx.assets.Articulation`,
  :class:`~isaaclab_physx.assets.RigidObject`, and :class:`~isaaclab_physx.assets.RigidObjectCollection`
  to use the dual-buffer :class:`~isaaclab.utils.wrench_composer.WrenchComposer`. Composed wrenches are
  applied to PhysX with ``is_global=False`` after body-frame composition.


0.5.19 (2026-04-20)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed Newton ``shape_color`` not reflecting the post-clone USD stage when the
  PhysX scene data provider builds or reloads the Newton model by calling
  :func:`~isaaclab.sim.utils.newton_model_utils.replace_newton_shape_colors` on
  the artifact, per-environment, and filtered Newton models in
  :class:`~isaaclab_physx.scene_data_providers.PhysxSceneDataProvider`.

0.5.18 (2026-04-16)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed flaky first-frame textured rendering by replacing the event-based RTX
  streaming subscription with a synchronous
  ``UsdContext.get_stage_streaming_status()`` query in
  :func:`~isaaclab_physx.renderers.isaac_rtx_renderer_utils.ensure_isaac_rtx_render_update`.


0.5.17 (2026-04-14)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_physx.sim.schemas.DeformableBodyPropertiesCfg` with
  namespace-aware property routing. Properties are organized into
  ``omniphysics:``, ``physxDeformableBody:``, and ``physxCollision:`` prefixed
  parent classes, allowing correct USD attribute mapping for the updated PhysX
  deformable body schema.
* Added :func:`~isaaclab_physx.sim.schemas.define_deformable_body_properties` and
  :func:`~isaaclab_physx.sim.schemas.modify_deformable_body_properties` to
  ``isaaclab_physx.sim.schemas``, supporting both surface and volume deformable
  types via the ``deformable_type`` parameter.
* Added :class:`~isaaclab_physx.sim.spawners.materials.DeformableBodyMaterialCfg`
  and :class:`~isaaclab_physx.sim.spawners.materials.SurfaceDeformableBodyMaterialCfg`
  with namespace-aware property routing for ``omniphysics:`` and
  ``physxDeformableBody:`` material attributes.
* Added :class:`~isaaclab_physx.sim.spawners.spawner_cfg.DeformableObjectSpawnerCfg`
  for configuring deformable body properties and materials when spawning.
* Added surface deformable body support to
  :class:`~isaaclab_physx.assets.DeformableObject`. The asset now detects whether
  the deformable is a surface or volume type based on the applied material API
  and creates the appropriate PhysX tensor view
  (``create_surface_deformable_body_view`` vs ``create_volume_deformable_body_view``).

Changed
^^^^^^^

* Changed :attr:`~isaaclab_physx.assets.DeformableObject.root_view` return type
  from ``physx.SoftBodyView`` to ``physx.DeformableBodyView`` to align with the
  updated PhysX API.
* Changed :attr:`~isaaclab_physx.assets.DeformableObject.material_physx_view`
  return type from ``physx.SoftBodyMaterialView`` to
  ``physx.DeformableMaterialView``.
* Changed deformable body root prim discovery to check for
  ``OmniPhysicsDeformableBodyAPI`` instead of ``PhysxDeformableBodyAPI``.
* Changed material prim discovery to check for ``OmniPhysicsDeformableMaterialAPI``
  instead of ``PhysxDeformableBodyMaterialAPI``.
* Changed PhysX view API calls to use updated method names:
  ``get_simulation_nodal_positions``, ``set_simulation_nodal_positions``,
  ``set_simulation_nodal_velocities``, ``get_simulation_nodal_kinematic_targets``,
  ``set_simulation_nodal_kinematic_targets``.
* Changed property accessors to use updated PhysX view attributes:
  ``max_simulation_elements_per_body``, ``max_collision_elements_per_body``,
  ``max_simulation_nodes_per_body``, ``max_collision_nodes_per_body``.
* Changed kinematic target operations to raise ``ValueError`` when called on
  surface deformable bodies, which do not support kinematic targets.


0.5.16 (2026-04-13)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Renamed the PhysX IMU sensor implementation to
  :class:`~isaaclab_physx.sensors.Pva`. The ``isaaclab_physx.sensors.imu``
  module now contains a new lightweight IMU sensor that only provides angular
  velocity and linear acceleration.
* Changed :class:`~isaaclab_physx.sensors.Pva` to no longer accept a
  ``gravity_bias`` parameter. Linear acceleration is now pure finite
  differencing of velocity without any gravity contribution.
* Changed :class:`~isaaclab_physx.sensors.Imu` to unconditionally include
  gravity in accelerometer readings. The gravity vector is queried from the
  PhysX simulation at initialization instead of being user-configured.

Added
^^^^^

* Added :class:`~isaaclab_physx.sensors.Imu` PhysX backend for the new
  lightweight IMU sensor with simplified Warp kernels that only compute
  angular velocity and linear acceleration.

Fixed
^^^^^

* Fixed unused ``body_pos`` variable in the IMU Warp kernel.
* Fixed ``phsyx`` typo in :class:`~isaaclab.sensors.pva.BasePva` docstring.


0.5.15 (2026-04-13)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :meth:`~isaaclab_physx.assets.RigidObject.set_material_properties_index`,
  :meth:`~isaaclab_physx.assets.RigidObject.set_material_properties_mask`,
  :meth:`~isaaclab_physx.assets.Articulation.set_material_properties_index`,
  :meth:`~isaaclab_physx.assets.Articulation.set_material_properties_mask`,
  :meth:`~isaaclab_physx.assets.RigidObjectCollection.set_material_properties_index`, and
  :meth:`~isaaclab_physx.assets.RigidObjectCollection.set_material_properties_mask`
  methods for setting collision shape material properties (friction, restitution).
  These methods follow the standard ``_index``/``_mask`` pattern, providing a unified
  API across PhysX and Newton backends.


0.5.14 (2026-04-06)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the simulation training loop not pausing when the Kit GUI timeline is
  paused. :meth:`~isaaclab_physx.physics.PhysxManager.wait_for_playing` now
  blocks and keeps the GUI responsive until the timeline is resumed or stopped.
* Fixed articulation visualization freezing after pausing and unpausing the
  simulation through the headed GUI in Isaac Sim 5.1+. Articulation meshes now
  remain visually updated after resuming.


0.5.13 (2026-03-25)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed device mismatch in :meth:`~isaaclab_physx.assets.RigidObjectCollection.reshape_view_to_data_2d`
  and :meth:`~isaaclab_physx.assets.RigidObjectCollection.reshape_view_to_data_3d` that caused
  ``wp.clone`` to fail with CUDA errors when PhysX returns data on CPU (e.g., masses, COMs, inertias)
  while the simulation runs on GPU. The strided view now correctly uses ``data.device`` instead of
  ``self.device``, matching the fix already present in :class:`~isaaclab_physx.assets.RigidObjectCollectionData`.


0.5.12 (2026-03-16)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed ``test_body_incoming_joint_wrench_b_single_joint`` computing the expected
  wrench in the parent body's frame instead of the child body's frame. The expected
  wrench is now expressed in
  :attr:`~isaaclab_physx.assets.ArticulationData.body_incoming_joint_wrench_b`'s
  actual convention (child body frame) and body indices are resolved by name to be
  robust across backends. Also corrected the docstring for
  :attr:`~isaaclab_physx.assets.ArticulationData.body_incoming_joint_wrench_b` to
  accurately describe the frame convention.


0.5.11 (2026-03-13)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed articulation root prim discovery failing when the
  ``physxArticulation:articulationEnabled`` attribute is not authored on the
  USD prim. The predicate now treats an unset attribute as enabled (the PhysX
  default) instead of rejecting the prim.


0.5.10 (2026-03-13)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Removed verbose ``logger.info`` calls from
  :class:`~isaaclab_physx.assets.RigidObject` and
  :class:`~isaaclab_physx.assets.Articulation` initialization that logged body
  names, joint names, and instance counts. Articulation joint parameter tables and
  actuator group summaries are retained.


0.5.9 (2026-03-11)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed device mismatch in
  :class:`~isaaclab_physx.assets.RigidObjectCollectionData` where
  ``_reshape_view_to_data_2d`` and ``_reshape_view_to_data_3d`` created
  strided pointer views with the target GPU device instead of the source
  array's device. PhysX returns masses, COMs, and inertias on CPU, so the
  strided view incorrectly claimed a CPU pointer lived on GPU. This caused
  ``CUDA error 1: invalid argument`` during ``wp.clone`` on GPUs without
  HMM (Heterogeneous Memory Management).


0.5.8 (2026-03-10)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Removed redundant ``ArticulationView`` from
  :class:`~isaaclab_physx.scene_data_providers.PhysxSceneDataProvider`.
  Creating a single ``ArticulationView`` for heterogeneous articulation types
  (e.g. Robot + Cabinet) triggered PhysX "Incorrect DofIdx" errors. The
  ``RigidBodyView`` already covers all body transforms including articulation
  links, so the articulation view was unnecessary. Articulation paths from
  prebuilt artifacts are now merged into rigid body paths for the
  ``RigidBodyView``.
* Fixed pre-existing test fixture in
  ``test_physx_scene_data_provider_visualizer_contract.py`` where
  ``_make_provider()`` was missing the
  ``_force_usd_fallback_for_newton_model_build`` attribute and the force
  fallback test used an incorrect attribute name.


0.5.7 (2026-03-06)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Made several PhysX articulation tests more backend-agnostic by relaxing
  PhysX-specific assumptions in ``test_articulation.py``.


0.5.6 (2026-03-03)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fix asset writer methods in :class:`~isaaclab_physx.assets.Articulation`,
  :class:`~isaaclab_physx.assets.RigidObject`, and
  :class:`~isaaclab_physx.assets.RigidObjectCollection` to use public data
  properties instead of internal timestamped buffer ``.data`` fields, removing
  redundant manual timestamp updates.


0.5.5 (2026-03-02)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Replaced all ``wp.nonzero()`` calls in
  :class:`~isaaclab_physx.assets.Articulation`,
  :class:`~isaaclab_physx.assets.RigidObject`, and
  :class:`~isaaclab_physx.assets.RigidObjectCollection` mask methods with
  ``torch.nonzero()`` via new ``_resolve_env_mask``, ``_resolve_body_mask``,
  ``_resolve_joint_mask``, ``_resolve_fixed_tendon_mask``, and
  ``_resolve_spatial_tendon_mask`` helpers, fixing mask-based writers that previously
  raised errors at runtime.

* Fixed device mismatch in ``RigidObjectCollection._env_body_ids_to_view_ids`` where GPU
  index arrays were passed to a CPU kernel launch. Inputs are now cloned to the target
  device before use.

* Added ``_get_cpu_env_ids`` helper to :class:`~isaaclab_physx.assets.Articulation`,
  :class:`~isaaclab_physx.assets.RigidObject`, and
  :class:`~isaaclab_physx.assets.RigidObjectCollection` to safely clone environment
  indices to CPU for PhysX model-property setters.

* Fixed ``MockArticulationViewWarp`` to support the mock test infrastructure.


0.5.4 (2026-03-01)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* fixed :func:`~isaaclab_physx.cloner.physx_replicate` to not exclude self replication by default.


0.5.3 (2026-02-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added runtime shape and dtype validation to all write methods in
  :class:`~isaaclab_physx.assets.Articulation`,
  :class:`~isaaclab_physx.assets.RigidObject`,
  :class:`~isaaclab_physx.assets.RigidObjectCollection`,
  :class:`~isaaclab_physx.assets.DeformableObject`, and
  :class:`~isaaclab_physx.assets.SurfaceGripper` using
  :meth:`~isaaclab.assets.AssetBase.assert_shape_and_dtype`. Validates input dimensions
  and types before kernel launch to catch mismatches early.


0.5.2 (2026-02-25)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added runtime shape and dtype validation to all write methods in
  :class:`~isaaclab_physx.assets.Articulation`,
  :class:`~isaaclab_physx.assets.RigidObject`,
  :class:`~isaaclab_physx.assets.RigidObjectCollection`,
  :class:`~isaaclab_physx.assets.DeformableObject`, and
  :class:`~isaaclab_physx.assets.SurfaceGripper` using
  :meth:`~isaaclab.assets.AssetBase.assert_shape_and_dtype`. Validates input dimensions
  and types before kernel launch to catch mismatches early.


0.5.1 (2026-02-25)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated ContactSensor ``body_names`` property to use ``num_sensors`` instead of
  deprecated ``num_bodies``.


0.5.0 (2026-02-24)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Aligned asset API with the multi-backend architecture. Base class abstract methods
  in :class:`~isaaclab.assets.BaseArticulation` and :class:`~isaaclab.assets.BaseRigidObject`
  have been refined so that PhysX and Newton backends share a consistent interface.

* Improved docstrings across all asset classes with precise shape and dtype annotations
  for warp array properties and write methods.

* Migrated tests to use the new ``_index`` / ``_mask`` write method APIs, removing
  usage of deprecated write methods.


0.4.1 (2026-02-18)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed a bug in :meth:~isaaclab_physx.assets.Articulation.process_actuators_cfg where explicit actuator joints could receive non-zero PhysX stiffness/damping, causing double PD control.


0.4.0 (2026-02-13)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Migrated all PhysX asset classes to warp backend:
  :class:`~isaaclab_physx.assets.Articulation`,
  :class:`~isaaclab_physx.assets.RigidObject`,
  :class:`~isaaclab_physx.assets.RigidObjectCollection`,
  :class:`~isaaclab_physx.assets.DeformableObject`, and
  :class:`~isaaclab_physx.assets.SurfaceGripper`.
  Internal state buffers now use ``wp.array`` with structured warp types
  (``wp.vec3f``, ``wp.quatf``, ``wp.transformf``, ``wp.spatial_vectorf``).

* Migrated all PhysX sensor classes to warp backend:
  :class:`~isaaclab_physx.sensors.ContactSensor`,
  :class:`~isaaclab_physx.sensors.Imu`, and
  :class:`~isaaclab_physx.sensors.FrameTransformer`.

* Split all write methods into ``_index`` and ``_mask`` variants for explicit
  sparse-index vs. boolean-mask semantics.

Added
^^^^^

* Added warp kernel modules for fused GPU computations:

  * :mod:`isaaclab_physx.assets.kernels` — shared kernels for root state extraction,
    velocity transforms, and data write-back.
  * :mod:`isaaclab_physx.assets.articulation.kernels` — articulation-specific kernels
    for joint state, body properties, and COM computations.
  * :mod:`isaaclab_physx.assets.deformable_object.kernels` — nodal state and mean
    vertex computations.
  * :mod:`isaaclab_physx.assets.rigid_object_collection.kernels` — 2D indexed kernels
    for multi-body collections.
  * :mod:`isaaclab_physx.sensors.contact_sensor.kernels` — contact force aggregation
    and history buffer management.
  * :mod:`isaaclab_physx.sensors.imu.kernels` — fused IMU update combining acceleration,
    gyroscope, and gravity projection.
  * :mod:`isaaclab_physx.sensors.frame_transformer.kernels` — frame transform computations.

* Added warp-based mock PhysX views for unit testing:
  ``MockArticulationViewWarp``, ``MockRigidBodyViewWarp``, ``MockRigidContactViewWarp``.


0.3.0 (2026-02-11)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Refactored :class:`~isaaclab_physx.physics.PhysxManager` to properly handle physics initialization
  order. ``attach_stage()`` is now called before ``start_simulation()`` to ensure GPU buffers are
  correctly allocated.
* Removed ``device`` field from :class:`~isaaclab_physx.physics.PhysxManagerCfg`. Device is now
  inherited from :attr:`SimulationCfg.device`.

Added
^^^^^

* Added :class:`~isaaclab_physx.physics.PhysxManager` as the concrete PhysX backend implementation
  of :class:`~isaaclab.physics.PhysicsManager`.
* Added :class:`~isaaclab_physx.physics.IsaacEvents` enum for PhysX-specific simulation events.
* Added monkey-patching of ``isaacsim.core.simulation_manager.SimulationManager`` in package init
  to ensure Isaac Sim uses :class:`~isaaclab_physx.physics.PhysxManager` for callback handling.


0.2.0 (2026-02-05)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated all PhysX benchmarks in :mod:`isaaclab_physx.benchmark` to use the new
  :class:`~isaaclab.test.benchmark.BaseIsaacLabBenchmark` framework from ``isaaclab.test.benchmark``.

* Added support for configurable output backends via ``--benchmark_backend`` argument.
  Supported backends: ``json``, ``osmo``, ``omniperf``.


0.1.4 (2026-02-05)
~~~~~~~~~~~~~~~~~~

Removed
^^^^^^^

* Removed all the deprecated properties and shorthands in the assets. They now live in the base classes.


0.1.3 (2026-02-03)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_physx.benchmark` module containing performance micro-benchmarks for
  PhysX asset classes. Includes:

  * ``benchmark_articulation.py``: Benchmarks for setter/writer methods on
    :class:`~isaaclab_physx.assets.Articulation` including root state, joint state,
    joint parameters, and body property operations.
  * ``benchmark_articulation_data.py``: Benchmarks for property accessors on
    :class:`~isaaclab_physx.assets.ArticulationData` covering root link/COM properties,
    joint properties, and body link/COM properties.
  * ``benchmark_rigid_object.py``: Benchmarks for setter/writer methods on
    :class:`~isaaclab_physx.assets.RigidObject` including root state and body property operations.
  * ``benchmark_rigid_object_data.py``: Benchmarks for property accessors on
    :class:`~isaaclab_physx.assets.RigidObjectData`.
  * ``benchmark_rigid_object_collection.py``: Benchmarks for setter/writer methods on
    :class:`~isaaclab_physx.assets.RigidObjectCollection` including body state, pose,
    velocity, and property operations.
  * ``benchmark_rigid_object_collection_data.py``: Benchmarks for property accessors on
    :class:`~isaaclab_physx.assets.RigidObjectCollectionData`.

  All benchmarks support configurable iterations, warmup steps, instance counts, multiple
  input modes (torch list, torch tensor), and output to JSON/CSV formats with hardware
  information capture.


0.1.2 (2026-02-03)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_physx.test.mock_interfaces` module providing mock PhysX view implementations
  for unit testing without requiring Isaac Sim. Includes:

  * :class:`MockRigidBodyView`: Mock for ``physx.RigidBodyView`` with transforms, velocities,
    accelerations, and mass properties.
  * :class:`MockArticulationView`: Mock for ``physx.ArticulationView`` with root/link states,
    DOF properties, and joint control.
  * :class:`MockRigidContactView`: Mock for ``physx.RigidContactView`` with contact forces,
    positions, normals, and friction data.
  * Factory functions including pre-configured quadruped and humanoid views.
  * Patching utilities and decorators for easy test injection.


0.1.0 (2026-01-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_physx.sensors` module containing PhysX-specific sensor implementations:

  * :class:`~isaaclab_physx.sensors.ContactSensor` and :class:`~isaaclab_physx.sensors.ContactSensorData`:
    PhysX-specific implementation for contact force sensing. Extends
    :class:`~isaaclab.sensors.contact_sensor.BaseContactSensor` with PhysX tensor API for contact
    force queries, contact filtering, and contact point tracking.

  * :class:`~isaaclab_physx.sensors.Imu` and :class:`~isaaclab_physx.sensors.ImuData`:
    PhysX-specific implementation for inertial measurement unit simulation. Extends
    :class:`~isaaclab.sensors.imu.BaseImu` with PhysX rigid body views for velocity and
    acceleration computation.

  * :class:`~isaaclab_physx.sensors.FrameTransformer` and :class:`~isaaclab_physx.sensors.FrameTransformerData`:
    PhysX-specific implementation for coordinate frame transformations. Extends
    :class:`~isaaclab.sensors.frame_transformer.BaseFrameTransformer` with PhysX rigid body views
    for efficient frame pose queries.

* Added PhysX-specific sensor tests moved from ``isaaclab/test/sensors/``:

  * ``test_contact_sensor.py``
  * ``test_imu.py``
  * ``test_frame_transformer.py``
  * ``check_contact_sensor.py``
  * ``check_imu_sensor.py``

Deprecated
^^^^^^^^^^

* Deprecated the ``pose_w``, ``pos_w``, and ``quat_w`` properties on
  :class:`~isaaclab_physx.sensors.ContactSensorData` and :class:`~isaaclab_physx.sensors.ImuData`.
  These properties will be removed in a future release. Please use a dedicated sensor (e.g.,
  :class:`~isaaclab.sensors.FrameTransformer`) to measure sensor poses in world frame.


0.1.0 (2026-01-28)
~~~~~~~~~~~~~~~~~~~

This is the initial release of the ``isaaclab_physx`` extension, which provides PhysX-specific
implementations of Isaac Lab asset classes. This extension enables a multi-backend architecture
where simulation backend-specific code is separated from the core Isaac Lab API.

Added
^^^^^

* Added :mod:`isaaclab_physx.assets` module containing PhysX-specific asset implementations:

  * :class:`~isaaclab_physx.assets.Articulation` and :class:`~isaaclab_physx.assets.ArticulationData`:
    PhysX-specific implementation for articulated rigid body systems (e.g., robots). Extends
    :class:`~isaaclab.assets.BaseArticulation` with PhysX tensor API integration for efficient
    GPU-accelerated simulation of multi-joint systems.

  * :class:`~isaaclab_physx.assets.RigidObject` and :class:`~isaaclab_physx.assets.RigidObjectData`:
    PhysX-specific implementation for single rigid body assets. Extends
    :class:`~isaaclab.assets.BaseRigidObject` with PhysX tensor API for efficient rigid body
    state queries and writes.

  * :class:`~isaaclab_physx.assets.RigidObjectCollection` and :class:`~isaaclab_physx.assets.RigidObjectCollectionData`:
    PhysX-specific implementation for collections of rigid objects. Extends
    :class:`~isaaclab.assets.BaseRigidObjectCollection` with batched ``(env_ids, object_ids)``
    API for efficient multi-object state management.

  * :class:`~isaaclab_physx.assets.DeformableObject`, :class:`~isaaclab_physx.assets.DeformableObjectCfg`,
    and :class:`~isaaclab_physx.assets.DeformableObjectData`: PhysX-specific implementation for
    soft body simulation using finite element methods (FEM). Moved from ``isaaclab.assets``.

  * :class:`~isaaclab_physx.assets.SurfaceGripper` and :class:`~isaaclab_physx.assets.SurfaceGripperCfg`:
    PhysX-specific implementation for surface gripper simulation using contact APIs. Moved from
    ``isaaclab.assets``.

* Added backward-compatible wrapper methods in :class:`~isaaclab_physx.assets.RigidObjectCollection`
  and :class:`~isaaclab_physx.assets.RigidObjectCollectionData` that delegate to the new
  ``body_*`` naming convention.

Deprecated
^^^^^^^^^^

* Deprecated the ``root_physx_view`` property on :class:`~isaaclab_physx.assets.Articulation`,
  :class:`~isaaclab_physx.assets.RigidObject`, :class:`~isaaclab_physx.assets.RigidObjectCollection`,
  and :class:`~isaaclab_physx.assets.DeformableObject` in favor of the ``root_view`` property.
  The ``root_physx_view`` property will be removed in a future release.

* Deprecated the ``object_*`` naming convention in :class:`~isaaclab_physx.assets.RigidObjectCollection`
  and :class:`~isaaclab_physx.assets.RigidObjectCollectionData` in favor of ``body_*``. The following
  methods and properties are deprecated and will be removed in a future release:

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
  * ``object_link_pose_w`` → use ``body_link_pose_w``
  * ``object_link_vel_w`` → use ``body_link_vel_w``
  * ``object_com_pose_w`` → use ``body_com_pose_w``
  * ``object_com_vel_w`` → use ``body_com_vel_w``
  * ``object_state_w`` → use ``body_state_w``
  * ``object_link_state_w`` → use ``body_link_state_w``
  * ``object_com_state_w`` → use ``body_com_state_w``
  * ``object_com_acc_w`` → use ``body_com_acc_w``
  * ``object_pose_w`` → use ``body_pose_w``
  * ``object_pos_w`` → use ``body_pos_w``
  * ``object_quat_w`` → use ``body_quat_w``
  * ``object_vel_w`` → use ``body_vel_w``
  * ``object_lin_vel_w`` → use ``body_lin_vel_w``
  * ``object_ang_vel_w`` → use ``body_ang_vel_w``
  * ``object_acc_w`` → use ``body_acc_w``
  * And all other ``object_*`` properties (see :ref:`migrating-to-isaaclab-3-0` for complete list).

Migration
^^^^^^^^^

* See :ref:`migrating-to-isaaclab-3-0` for detailed migration instructions.
