Changelog
---------

0.11.0 (2026-05-17)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added Newton backend for :class:`~isaaclab.sensors.ray_caster.RayCaster` /
  :class:`~isaaclab.sensors.ray_caster.RayCasterCamera` /
  :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCaster` /
  :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCasterCamera`. Site-based,
  matching :class:`~isaaclab_newton.sensors.pva.Pva` and
  :class:`~isaaclab_newton.sensors.frame_transformer.FrameTransformer`:
  registers body-attached sites via
  :meth:`~isaaclab_newton.physics.NewtonManager.cl_register_site` for both
  the sensor frame and any tracked target meshes, and reads per-step
  transforms off :class:`~newton.sensors.SensorFrameTransform` against a
  world-origin reference. Static parents/targets bypass the site
  machinery and serve cached per-env ``wp.transformf`` arrays.

Changed
^^^^^^^

* Changed Newton tracked target mesh updates to copy site poses directly into
  Warp mesh pose tables instead of staging through torch views.


0.10.0 (2026-05-16)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added runtime verification of the ``omni::cubric::IAdapter`` interface
  version in :mod:`~isaaclab_newton.physics._cubric` as defense-in-depth
  against future ABI shifts. The shim falls back to the CPU path on
  major-version mismatch or older-minor.

Changed
^^^^^^^

* Bumped the ``newton[sim]`` pin from ``v1.2.0rc2`` to ``v1.2.0``
  (stable). Upstream release notes: `newton-physics/newton v1.2.0
  <https://github.com/newton-physics/newton/releases/tag/v1.2.0>`_.
* Updated :class:`~isaaclab_newton.renderers.NewtonWarpRenderer` to accept
  :class:`~isaaclab.utils.warp.ProxyArray` in :meth:`set_outputs` and :meth:`update_camera`,
  matching the updated :class:`~isaaclab.renderers.BaseRenderer` interface. Output buffers are
  reinterpreted directly from the ProxyArray's underlying warp array, removing the previous
  :func:`warp.from_torch` conversion path.

Fixed
^^^^^

* Fixed per-environment string identifiers (e.g. ``mujoco:tendon_label``)
  keeping the source proto path after replication.
  :func:`~isaaclab_newton.cloner.newton_replicate._rename_builder_labels`
  now also walks string-typed custom-attribute columns whose frequency
  declares a ``references="world"`` companion, rewriting their per-row
  source-path prefix to the destination world root in the same pass that
  handles built-in label arrays. Adds ``constraint_mimic`` and
  ``equality_constraint`` to that built-in pass for completeness. The
  prefix match uses a path-separator boundary so a source path that is a
  string prefix of another (e.g. ``/Sources/protoA`` vs
  ``/Sources/protoAB``) does not cross-contaminate during the rename.


0.9.1 (2026-05-15)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the acceleration-arrow debug visualizer in
  :class:`~isaaclab_newton.sensors.pva.Pva` drawing arrows in undefined directions for
  bodies with effectively zero acceleration. Such bodies are now skipped from the
  visualization.


0.9.0 (2026-05-14)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added Newton implementations of
  :attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w`,
  :attr:`~isaaclab.assets.BaseArticulationData.body_com_jacobian_w`, and
  :attr:`~isaaclab.assets.BaseArticulationData.mass_matrix` on
  :class:`~isaaclab_newton.assets.ArticulationData`. The properties wrap
  ``ArticulationView.eval_jacobian`` and ``ArticulationView.eval_mass_matrix``
  with view-sized output buffers cached via the standard timestamped-buffer
  pattern. Per-step behavior is allocation-free and safe under CUDA-graph
  capture: source / scratch / output buffers are pre-allocated in
  ``_create_buffers``, and
  :func:`~isaaclab_newton.assets.articulation.kernels.gather_jacobian_rows`
  and :func:`~isaaclab_newton.assets.articulation.kernels.gather_mass_matrix_rows`
  Warp kernels gather just this view's rows from the model-sized buffers
  Newton populates. The DoF axis preserves the leading 6 floating-base
  columns Newton fills for floating-base articulations (matching the
  cross-library industry convention and PhysX's layout).
* Added the
  :func:`~isaaclab_newton.assets.articulation.kernels.shift_jacobian_com_to_origin`
  Warp kernel applying the
  ``v_origin = v_com - omega x (R · body_com_pos_b)`` shift to the
  linear-velocity rows of the gathered, view-sized Jacobian, so the link-
  origin form matches the cross-backend
  :attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w`
  contract.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.get_state` and
  :meth:`~isaaclab_newton.physics.NewtonManager.update_visualization_state` so
  Newton-based renderers, visualizers, and video recorders can fetch a Newton
  ``Model``/``State`` regardless of the active sim backend. When the sim
  backend is PhysX the manager builds a shadow Newton model directly from the
  USD stage (via
  :meth:`~isaaclab_newton.physics.NewtonManager.instantiate_builder_from_stage`)
  and refreshes ``state_0.body_q`` from rigid-body transforms supplied by the
  :class:`~isaaclab.scene_data.SceneDataProvider` each render
  frame.

Changed
^^^^^^^

* :attr:`~isaaclab_newton.assets.ArticulationData.gravity_compensation_forces`
  raises :class:`NotImplementedError` with a message pointing at the
  upstream gap. Newton's ``ArticulationView`` does not expose an
  inverse-dynamics primitive yet (upstream Newton issues
  `#2497 <https://github.com/newton-physics/newton/issues/2497>`_,
  `#2529 <https://github.com/newton-physics/newton/issues/2529>`_,
  `#2625 <https://github.com/newton-physics/newton/issues/2625>`_).
  OSC users on Newton must set ``gravity_compensation=False`` until
  upstream lands the primitive.
* **Breaking:** :class:`~isaaclab_newton.renderers.NewtonWarpRenderer`,
  :class:`~isaaclab_newton.video_recording.NewtonGlPerspectiveVideo`, and the
  Newton/Rerun/Viser visualizers now read Newton ``Model``/``State`` from
  :class:`~isaaclab_newton.physics.NewtonManager` instead of the removed
  ``BaseSceneDataProvider.get_newton_model()`` / ``get_newton_state()``.
* Switched the Newton install to ``newton[sim]`` so that ``mujoco`` and
  ``mujoco-warp`` are pulled in transitively via Newton's ``[sim]`` extra.
  The explicit ``mujoco==3.8.0`` and ``mujoco-warp==3.8.0.1`` pins were
  removed from :mod:`isaaclab_newton` — Newton is now the single source of
  truth for those versions.

Removed
^^^^^^^

* **Breaking:** Removed the ``isaaclab_newton.scene_data_providers`` package
  (``NewtonSceneDataProvider``). Replace direct uses with
  :meth:`~isaaclab_newton.physics.NewtonManager.get_model` /
  :meth:`~isaaclab_newton.physics.NewtonManager.get_state` and the
  Warp-native :class:`~isaaclab.scene_data.SceneDataProvider`.

Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.sensors.ContactSensor` metadata extraction
  after the migration to Newton 1.1, where ``sensing_obj_type`` and
  ``counterpart_type`` became scalar strings and ``counterpart_indices``
  became per-row.
* Fixed per-environment string identifiers (e.g. ``mujoco:tendon_label``)
  keeping the source proto path after replication.
  :func:`~isaaclab_newton.cloner.newton_replicate._rename_builder_labels`
  now also walks string-typed custom-attribute columns whose frequency
  declares a ``references="world"`` companion, rewriting their per-row
  source-path prefix to the destination world root in the same pass that
  handles built-in label arrays. Adds ``constraint_mimic`` and
  ``equality_constraint`` to that built-in pass for completeness. The
  prefix match uses a path-separator boundary so a source path that is a
  string prefix of another (e.g. ``/Sources/protoA`` vs
  ``/Sources/protoAB``) does not cross-contaminate during the rename.


0.8.1 (2026-05-13)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed Newton integration to use the packaged Newton 1.2.0 release candidate
  and updated transform conversion calls for Warp 1.13 compatibility.

Fixed
^^^^^

* Fixed a spurious ``[Error][carb] Client passed into the framework is nullptr.``
  log emitted from :meth:`~isaaclab_newton.physics._cubric.CubricBindings.initialize`
  when the first ``tryAcquireInterfaceWithClient`` attempt returned null. The
  helper used to retry with ``clientName=None``, which Carbonite has rejected as
  invalid since 2018 — the retry only emitted a misleading error log. Removed
  the null-client retry; the existing ``acquireInterfaceWithClient`` fallback
  with the ``isaaclab.cubric`` client name still handles configurations where
  the plugin needs to be loaded on demand.


0.8.0 (2026-05-12)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_newton.sim.schemas.NewtonRigidBodyPropertiesCfg` and
  :class:`~isaaclab_newton.sim.schemas.NewtonJointDrivePropertiesCfg` as Newton-targeted
  bases for solver-specific subclasses. Currently empty (no Newton-native ``newton:*``
  rigid-body or joint-drive attributes today); reserved as the family root for any
  future Newton-native fields.
* Added :class:`~isaaclab_newton.sim.schemas.MujocoRigidBodyPropertiesCfg` (subclasses
  :class:`NewtonRigidBodyPropertiesCfg`) with :attr:`gravcomp` for body-level gravity
  compensation (``mjc:gravcomp``).
* Added :class:`~isaaclab_newton.sim.schemas.MujocoJointDrivePropertiesCfg` (subclasses
  :class:`NewtonJointDrivePropertiesCfg`) with :attr:`actuatorgravcomp` for joint-level
  gravity compensation routing (``mjc:actuatorgravcomp`` via ``MjcJointAPI``).
* Added :class:`~isaaclab_newton.sim.schemas.NewtonCollisionPropertiesCfg` with
  :attr:`contact_margin` and :attr:`contact_gap` (``newton:*`` via ``NewtonCollisionAPI``).
* Added :class:`~isaaclab_newton.sim.schemas.NewtonMeshCollisionPropertiesCfg` with
  :attr:`max_hull_vertices` (``newton:maxHullVertices`` via ``NewtonMeshCollisionAPI``).
* Added :class:`~isaaclab_newton.sim.schemas.NewtonMaterialPropertiesCfg` with
  :attr:`torsional_friction` and :attr:`rolling_friction` (``newton:*`` via ``NewtonMaterialAPI``).
* Added :class:`~isaaclab_newton.sim.schemas.NewtonArticulationRootPropertiesCfg` with
  :attr:`self_collision_enabled` (``newton:selfCollisionEnabled`` via ``NewtonArticulationRootAPI``).

Changed
^^^^^^^

* Split :class:`~isaaclab_newton.renderers.NewtonWarpRenderer` construction
  into a pre-physics ``__init__`` (stores cfg and registers the Newton-Warp
  scene-data requirement on
  :class:`~isaaclab.sim.SimulationContext`) and a post-physics
  :meth:`~isaaclab_newton.renderers.NewtonWarpRenderer.initialize` (reads
  the built Newton model.


0.7.2 (2026-05-11)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Changed rigid object collection spawning to honor planned ``spawn_path``
  values while falling back to ``prim_path`` for direct construction.


0.7.1 (2026-05-09)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.assets.Articulation` joint friction docs to identify Newton friction as a force or
  torque value instead of a unitless coefficient.
* Fixed :class:`~isaaclab_newton.sensors.contact_sensor.ContactSensor` to use
  current Newton contact sensor API names, removing deprecation warnings from
  Newton contact sensor test runs.
* Fixed stale Newton forward-kinematics state after explicit pose writes so
  downstream collision queries and :attr:`~isaaclab_newton.assets.RigidObjectData.body_link_pose_w`
  reads use updated transforms.


0.7.0 (2026-05-08)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Bumped Newton pin to ``v1.2.0rc2``. Pulls in IsaacLab-relevant fixes from
  `newton-physics/newton#2678 <https://github.com/newton-physics/newton/pull/2678>`_
  and `newton-physics/newton#2720
  <https://github.com/newton-physics/newton/pull/2720>`_ (``SolverKamino``
  reset under ``world_mask``), the upstream tendon-scoping fix from
  `newton-physics/newton#2659
  <https://github.com/newton-physics/newton/pull/2659>`_ ("Scope USD
  custom-frequency parsing"), and a VRAM-leak fix on example reset
  (`newton-physics/newton#2710
  <https://github.com/newton-physics/newton/pull/2710>`_).
* Newton ``v1.2.0rc2`` requires ``warp-lang==1.13.0``, ``mujoco==3.8.0``,
  and ``mujoco-warp==3.8.0.1``. ``warp-lang``/``mujoco``/``mujoco-warp``
  pins live in :mod:`isaaclab` and ``tools/wheel_builder/res/python_packages.toml``;
  the Newton pin is mirrored across :mod:`isaaclab_newton`,
  :mod:`isaaclab_visualizers` (3×), :mod:`isaaclab_physx` (``[newton]``
  extra), and the wheel-builder TOML.
* Updated ``wp.math.transform_to_matrix`` to ``wp.transform_to_matrix`` in
  :mod:`~isaaclab_newton.physics.newton_manager` and
  :mod:`~isaaclab_ov.renderers.ovrtx_renderer_kernels` to match the
  ``warp-lang`` 1.13 API (the ``wp.math`` namespace was removed).
* Adapted :class:`~isaaclab_newton.renderers.NewtonWarpRenderer` to
  Newton ``v1.2.0rc2``'s explicit shape-BVH lifecycle.
  :meth:`~newton.sensors.SensorTiledCamera.update` no longer auto-builds
  the BVH when a non-``None`` state is passed and the underlying
  ``RenderContext.render`` now raises ``RuntimeError("build_bvh_shape()
  must be called before rendering shapes.")`` if it was never built. The
  renderer now calls ``newton.geometry.build_bvh_shape`` once after
  sensor construction and ``newton.geometry.refit_bvh_shape`` each frame
  before :meth:`~newton.sensors.SensorTiledCamera.update`, since env
  body poses move every step.


0.6.0 (2026-05-08)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Modified the newton renderer to use the new patterns from renderer/camera decoupling.
* Changed :class:`~isaaclab_newton.physics.NewtonManager` to dispatch through
  solver-specific manager subclasses while preserving the existing
  ``NewtonCfg(solver_cfg=...)`` configuration pattern.

Deprecated
^^^^^^^^^^

* Deprecated :attr:`~isaaclab_newton.physics.NewtonSolverCfg.solver_type` for
  manager dispatch in favor of
  :attr:`~isaaclab_newton.physics.NewtonSolverCfg.class_type`. Existing configs
  remain valid, but new code should rely on ``class_type``.

Removed
^^^^^^^

* **Breaking:** Removed
  ``isaaclab_newton.cloner.newton_replicate.create_newton_visualizer_prebuild_clone_fn``.
  Callers that need a Newton model for visualization should call
  :func:`~isaaclab_newton.cloner.newton_replicate.newton_visualizer_prebuild`
  directly with the ``(sources, destinations, env_ids, mask, positions)`` bundle
  derived from :meth:`~isaaclab.sim.SimulationContext.get_clone_plans`.
* Removed the unimplemented ``ArticulationData.body_incoming_joint_wrench_b``
  accessor. Add :class:`~isaaclab.sensors.JointWrenchSensorCfg` to the scene
  and read :attr:`~isaaclab.sensors.JointWrenchSensorData.force` and
  :attr:`~isaaclab.sensors.JointWrenchSensorData.torque` instead.

Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.sensors.JointWrenchSensor` initialization for
  USD assets whose articulation root is nested below the configured asset prim.


0.5.26 (2026-04-30)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_newton.sensors.JointWrenchSensor`.


0.5.25 (2026-04-28)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_newton.physics.KaminoSolverCfg` to support Newton's Kamino
  solver backend, a Proximal-ADMM based solver for constrained rigid multi-body dynamics.
* Added fused :meth:`~isaaclab_newton.assets.Articulation.write_joint_state_to_sim_index`
  and :meth:`~isaaclab_newton.assets.Articulation.write_joint_state_to_sim_mask` that
  write joint position and velocity in a single kernel launch instead of two.

Changed
^^^^^^^

* Removed dead state-buffer output parameters from 8 root pose/velocity warp kernels
  in :mod:`~isaaclab_newton.assets.kernels`, reducing kernel argument marshalling
  overhead.

Fixed
^^^^^

* Replaced boolean ``_fk_dirty`` and ``_kamino_needs_fk`` flags with per-world
  reset masks (``_world_reset_mask`` and ``_fk_reset_mask``). Asset write methods
  now call :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_fk` with
  ``env_mask``/``env_ids`` and ``articulation_ids``, so ``eval_fk`` and
  ``SolverKamino.reset()`` only operate on dirtied environments. Rigid object
  and rigid object collection write methods now also trigger FK invalidation.
* Fixed CUDA error 700 (illegal memory access) when calling ``SolverKamino.reset()``
  after CUDA graph capture. ``StateKamino.from_newton()`` lazily allocates
  ``body_f_total``, ``joint_q_prev``, and ``joint_lambdas`` via ``wp.clone``/``wp.zeros``
  during the first ``step()`` inside graph capture. These memory-pool addresses become
  stale without a warm-up ``wp.capture_launch`` replay to pin them before any eager
  ``solver.reset()`` call.


0.5.24 (2026-04-27)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_newton.physics.NewtonShapeCfg` exposing
  per-shape collision defaults (``margin``, ``gap``) via
  :attr:`~isaaclab_newton.physics.NewtonCfg.default_shape_cfg`.
  :meth:`~isaaclab_newton.physics.NewtonManager.create_builder` now
  forwards the wrapper onto Newton's upstream
  ``ModelBuilder.default_shape_cfg`` via
  :func:`~isaaclab.utils.checked_apply`. The previous code only set
  ``gap`` and left ``margin`` at Newton's upstream default of ``0.0``,
  causing all non-Anymal-D robots to fail to learn rough-terrain
  locomotion on triangle-mesh terrain. ``RoughPhysicsCfg`` opts in to
  ``margin=0.01``.


0.5.23 (2026-04-24)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView` to match the
  new :class:`~isaaclab.sim.views.BaseFrameView` ProxyArray return contract.
  See the ``isaaclab`` 4.6.15 changelog for migration guidance.


0.5.22 (2026-04-23)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Properties on the following data classes now return
  :class:`~isaaclab.utils.warp.ProxyArray` instead of raw ``wp.array``:
  :class:`~isaaclab_newton.assets.articulation.ArticulationData`,
  :class:`~isaaclab_newton.assets.rigid_object.RigidObjectData`,
  :class:`~isaaclab_newton.assets.rigid_object_collection.RigidObjectCollectionData`,
  :class:`~isaaclab_newton.sensors.contact_sensor.ContactSensorData`,
  :class:`~isaaclab_newton.sensors.frame_transformer.FrameTransformerData`,
  :class:`~isaaclab_newton.sensors.imu.ImuData`, and
  :class:`~isaaclab_newton.sensors.pva.PvaData`.
  Use ``.torch`` for a cached zero-copy ``torch.Tensor`` view, or ``.warp`` for
  the underlying ``wp.array``. Implicit torch operations (arithmetic,
  ``torch.*`` functions) work during the deprecation period but emit a warning.


0.5.21 (2026-04-23)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed flakiness in ``test_body_root_state_properties`` by bounding the random spin velocity so
  numerical drift stays within the position tolerance over the simulated trajectory.


0.5.20 (2026-04-22)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_newton.sim.views.XformPrimView` providing the Newton
  backend implementation for xform prim views.

Changed
^^^^^^^

* Renamed :class:`~isaaclab_newton.sim.views.NewtonSiteXformPrimView` to
  :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView`. Old name is kept as a deprecated alias.


0.5.19 (2026-04-22)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated ``write_data_to_sim`` in :class:`~isaaclab_newton.assets.Articulation`,
  :class:`~isaaclab_newton.assets.RigidObject`, and :class:`~isaaclab_newton.assets.RigidObjectCollection`
  to use the dual-buffer :class:`~isaaclab.utils.wrench_composer.WrenchComposer`. Composed wrenches are
  applied after body-frame composition.
* Updated the PhysX Tensor API docstring link in :class:`~isaaclab_newton.assets.ArticulationData`
  from ``omni.physics.tensors.impl.api`` to ``omni.physics.tensors.api`` to track the upstream
  Isaac Sim module relocation (the ``impl`` submodule was removed).


0.5.18 (2026-04-21)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Upgraded Newton from ``2684d75`` to ``a27277e``. Includes collision improvements, contact quality fixes,
  hydroelastic contact optimization, and memory usage fixes in CollisionPipeline. For details see
  ``Newton changelog <https://github.com/newton-physics/newton/blob/main/CHANGELOG.md>``.
* Pinned ``mujoco`` and ``mujoco-warp`` to ``3.6.0`` to align with the Newton library.


0.5.17 (2026-04-20)
~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed Newton visualization colors drifting from the USD stage by calling
  :func:`~isaaclab.sim.utils.newton_model_utils.replace_newton_shape_colors`
  after the model is finalized in :class:`~isaaclab_newton.physics.NewtonManager`.

Changed
^^^^^^^

* Changed Newton Warp tiled camera outputs to clear with a light linear gray
  (0xFFEEEEEE, 93% gray, fully opaque) background via ``SensorTiledCamera.ClearData``
  in :class:`~isaaclab_newton.renderers.NewtonWarpRenderer`.

0.5.16 (2026-04-17)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed incorrect attribute name ``contact_margin`` on Newton
  ``ShapeConfig`` in
  :meth:`~isaaclab_newton.physics.NewtonManager.create_builder`. The
  field was renamed to ``gap`` in Newton PR #1732. The typo created a
  dead attribute so the intended 1 cm default shape gap was never applied.


0.5.15 (2026-04-16)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_newton.sensors.pva.Pva` sensor wrapping Newton's
  body state (``body_q``, ``body_qd``, ``body_qdd``) to provide world-frame
  pose and body-frame velocities/accelerations.


0.5.14 (2026-04-14)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_newton.sensors.Imu` sensor wrapping Newton's
  ``SensorIMU``, providing angular velocity and linear acceleration in the
  sensor's body frame.


0.5.13 (2026-04-13)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_newton.physics.NewtonCollisionPipelineCfg` to expose Newton
  collision pipeline parameters via :attr:`~isaaclab_newton.physics.NewtonCfg.collision_cfg`.
* Added :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.tolerance` for solver convergence control.

Fixed
^^^^^

* Fixed truthiness check on hydroelastic config dict in collision pipeline
  initialization. An explicit ``is not None`` check is now used so that
  :class:`~isaaclab_newton.physics.newton_collision_cfg.HydroelasticSDFCfg`
  with all-default values is no longer silently skipped.


0.5.12 (2026-04-13)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``set_friction_index/mask`` and ``set_restitution_index/mask`` methods to
  Newton assets for native material property randomization.


0.5.11 (2026-04-13)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_newton.sensors.frame_transformer.FrameTransformer` sensor
  wrapping Newton's ``SensorFrameTransform``. Supports per-env source/target site
  registration, wildcard body matching, and zero-copy transform views.


0.5.10 (2026-04-05)
~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed NaN after env reset caused by stale ``body_q`` in the collision
  pipeline. Added :meth:`~isaaclab_newton.physics.NewtonManager.invalidate_fk`
  so articulation write methods trigger ``eval_fk`` before the next
  ``collide()``.

Fixed
^^^^^

* Fixed ``test_body_incoming_joint_wrench_b_single_joint`` computing the expected
  wrench in the parent body's frame instead of the child body's frame. The expected
  wrench is now expressed in the child body's own frame and body indices are resolved
  by name to be robust across backends.


0.5.9 (2026-03-13)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed overly tight numerical tolerances in
  ``test_object_state_properties`` for
  :class:`~isaaclab_newton.assets.RigidObjectCollection` that caused
  spurious failures on CPU. Aligned tolerances with the equivalent
  rigid object test (``test_rigid_object.py``, ``atol=2e-3, rtol=2e-3``).


0.5.8 (2026-03-13)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fix ``test_filter_enables_force_matrix`` failing with ``TypeError`` due to
  ``pytest.mark.flaky(reruns=3)`` being incompatible with the installed
  ``flaky`` plugin. Replace with ``@flaky(max_runs=4, min_passes=1)`` decorator.


0.5.7 (2026-03-13)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Removed verbose ``logger.info`` calls from
  :class:`~isaaclab_newton.assets.RigidObject`,
  :class:`~isaaclab_newton.assets.RigidObjectCollection`, and
  :class:`~isaaclab_newton.assets.Articulation` initialization that logged body
  names, joint names, and instance counts. Articulation joint parameter tables and
  actuator group summaries are retained.


0.5.6 (2026-03-10)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed dtype mismatch in :class:`~isaaclab_newton.assets.RigidObjectCollection`
  where ``write_body_com_pose_to_sim_index`` and ``write_body_link_velocity_to_sim_index``
  passed ``body_com_pose_b`` (``wp.transformf``) instead of ``body_com_pos_b``
  (``wp.vec3f``) to the underlying warp kernels.

* Fixed :attr:`~isaaclab_newton.assets.ArticulationData.body_inertia`,
  :attr:`~isaaclab_newton.assets.RigidObjectData.body_inertia`, and
  :attr:`~isaaclab_newton.assets.RigidObjectCollectionData.body_inertia`
  returning raw ``mat33f`` arrays instead of ``(N, B, 9)`` float32. The
  previous ptr-based reshape assumed ``float32`` with ``ndim == 4``, but
  Newton returns ``mat33f`` dtype with ``ndim == 2``. Fixed the pointer
  aliasing to correctly reinterpret each 36-byte ``mat33f`` element as 9
  contiguous ``float32`` values.


0.5.5 (2026-03-10)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.renderers.NewtonWarpRenderer` to raise a clear
  ``RuntimeError`` when the Newton model is unavailable instead of deferring to
  a confusing ``AttributeError`` on ``render_context.world_count``.


0.5.4 (2026-02-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added contact sensor support via :class:`newton.sensors.SensorContact` with
  Isaac Lab pattern conversion (``.*`` to fnmatch, USD path normalization)
  inlined in :meth:`~isaaclab_newton.physics.NewtonManager.add_contact_sensor`.

Changed
^^^^^^^

* Changed :class:`~isaaclab_newton.sensors.contact_sensor.ContactSensor` to
  flatten Newton's per-world nested ``sensing_objs`` and ``counterparts``
  attributes.

Fixed
^^^^^

* Fixed ``RigidObjectData.body_inertia`` shape from ``(N, B, 3, 3)`` to ``(N, B, 9)``.


0.5.3 (2026-03-09)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :attr:`~isaaclab_newton.assets.RigidObjectData.body_inertia` to return a
  ``(num_instances, num_bodies, 9)`` float32 strided view, matching the articulation fix in 0.5.2.

* Fixed non-contiguous array handling in ``RigidObjectData`` position, quaternion, and
  spatial-vector extraction helpers. The ``source`` buffer shape and kernel dispatch ``dim``
  now use the input array's shape instead of the (possibly uninitialized) output shape.


0.5.2 (2026-03-06)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed :attr:`~isaaclab_newton.assets.ArticulationData.body_inertia` in
  :class:`~isaaclab_newton.assets.ArticulationData` to return a ``(num_instances, num_bodies, 9)``
  float32 array as documented, instead of a ``(num_instances, num_bodies, 3, 3)`` array. The
  ``(N, B, 3, 3)`` shape caused a broadcasting error in
  :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_mass` and a dimension mismatch when the
  ``write_body_inertia_to_buffer_*`` kernels were called via
  :meth:`~isaaclab_newton.assets.Articulation.set_inertias_index` and
  :meth:`~isaaclab_newton.assets.Articulation.set_inertias_mask`. The fix creates a ``(N, B, 9)``
  view over the same memory using explicit strides, collapsing the two contiguous trailing
  dimensions without copying data.

* Fixed ``AttributeError: 'NoneType' object has no attribute 'device'`` in
  :meth:`~isaaclab_newton.physics.NewtonManager.step` when ``use_cuda_graph=True`` but the CUDA
  graph was not captured (e.g., when RTX/Fabric USD sync is active). The step condition now
  checks ``cls._graph is not None`` directly instead of repeating the capture-time heuristic.


0.5.1 (2026-03-06)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_newton.assets.RigidObjectCollection` and
  :class:`~isaaclab_newton.assets.RigidObjectCollectionData` for managing
  collections of independent rigid bodies. Uses a single
  ``ArticulationView`` with a combined fnmatch pattern to get direct
  ``(num_envs, num_bodies)`` bindings into Newton's state, avoiding the
  scatter/gather overhead needed by PhysX.

* Added :class:`~isaaclab_newton.test.mock_interfaces.views.MockNewtonCollectionView`
  for unit testing the collection data class without simulation.

* Added Newton backend to the rigid object collection interface conformance
  tests (``test_rigid_object_collection_iface.py``).


0.5.0 (2026-03-06)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added full Newton articulation test suite (``test_articulation.py``) — 194 passed,
  8 skipped, 4 xfailed — adapted from PhysX tests with Newton-specific imports, sim
  config, and solver tolerance adjustments.

* Added full Newton rigid body test suite (``test_rigid_object.py``) — 74 passed,
  25 skipped — adapted from PhysX tests with Newton-specific mass/COM APIs and
  ``_newton_sim_context()`` helper for device/gravity/dt configuration.

Fixed
^^^^^

* Fixed ``ArticulationData`` and ``RigidObjectData`` to rebind simulation pointers
  on full sim reset via ``PHYSICS_READY`` callback, preventing stale warp array
  references after ``sim.reset()`` recreates the Newton model.

* Fixed ``ArticulationData`` to force ``eval_fk`` after joint state writes so that
  link poses are consistent with joint positions before the next ``sim.step()``.

* Fixed lazy initialization of ``TimestampedBuffer`` properties in
  ``RigidObjectData`` (velocity-in-body-frame and deprecated state properties)
  that were left as ``None`` and caused ``AttributeError`` on first access.

* Fixed ``None`` guards for timestamp invalidation in ``RigidObject`` write methods
  (``write_root_pose_to_sim``, ``write_root_velocity_to_sim``) to avoid
  ``AttributeError`` when optional buffers have not been initialized.

* Fixed ``is_contiguous`` usage in ``RigidObjectData`` — warp 1.12.0rc2 exposes it
  as a property, not a method.

* Fixed ``body_com_pose_b`` → ``body_com_pos_b`` kernel input naming in
  ``RigidObjectData`` for ``root_com_pose_w`` and ``root_link_vel_w`` properties.

* Fixed ``wp.from_torch()`` called on warp arrays in ``RigidObjectData`` body
  inertia binding — replaced with direct ``.view()``/``.reshape()`` on warp arrays.

* Improved CPU support in ``NewtonManager``: added device guards for CUDA graph
  operations that are not available on CPU.

* Fixed explicit mask resolution in asset write methods to correctly handle both
  index-based and mask-based sparse writes.


0.4.1 (2026-03-03)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fix asset writer methods in :class:`~isaaclab_newton.assets.Articulation` and
  :class:`~isaaclab_newton.assets.RigidObject` to use public data properties
  instead of internal timestamped buffer ``.data`` fields, removing redundant
  manual timestamp updates.


0.4.0 (2026-03-01)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_newton.physics.NewtonManager` physics backend with
  MuJoCo-Warp, XPBD, and Featherstone solvers, CUDA-graph support, and
  backend-agnostic callback dispatch via :class:`~isaaclab.physics.PhysicsEvent`.

Changed
^^^^^^^

* Implemented ``newton_replicate`` to build per-environment worlds from USD
  prototypes using Newton's ``ModelBuilder``.

* Renamed ``NewtonContactSensorCfg`` to ``ContactSensorCfg`` and made it
  backend-agnostic with lazy ``class_type`` resolution.

* Pinned ``mujoco-warp==3.5.0`` and ``warp-lang==1.12.0rc2`` in ``setup.py``.


0.3.0 (2026-02-25)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_newton.test.mock_interfaces` test infrastructure module with
  structured mock views, factory functions, and unit tests — mirroring the
  ``isaaclab_physx`` mock interface pattern:

  * :class:`~isaaclab_newton.test.mock_interfaces.views.MockNewtonArticulationView`:
    extracted from monolithic ``mock_newton.py`` into its own module with lazy
    initialization, individual ``set_mock_*`` methods, ``_noop_setters`` flag,
    and numpy-based ``set_random_mock_data()``.

  * Factory functions: ``create_mock_articulation_view()``,
    ``create_mock_quadruped_view()``, ``create_mock_humanoid_view()`` for
    convenient test setup.

* Added unit tests for mock interfaces:
  ``test_mock_articulation_view.py`` and ``test_factories.py``.

Changed
^^^^^^^

* Restructured ``mock_newton.py``: moved ``MockNewtonArticulationView`` to
  ``views/mock_articulation_view.py`` and removed ``torch`` dependency from
  the mock module (replaced with ``numpy`` for random data generation).


0.2.3 (2026-02-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added runtime shape and dtype validation to all write methods in
  :class:`~isaaclab_newton.assets.Articulation` and
  :class:`~isaaclab_newton.assets.RigidObject` using
  :meth:`~isaaclab.assets.AssetBase.assert_shape_and_dtype` and
  :meth:`~isaaclab.assets.AssetBase.assert_shape_and_dtype_mask`.


0.2.2 (2026-02-26)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added runtime shape and dtype validation to all write methods in
  :class:`~isaaclab_newton.assets.Articulation` and
  :class:`~isaaclab_newton.assets.RigidObject` using
  :meth:`~isaaclab.assets.AssetBase.assert_shape_and_dtype` and
  :meth:`~isaaclab.assets.AssetBase.assert_shape_and_dtype_mask`.


0.2.1 (2026-02-25)

Removed
^^^^^^^

imgui-bundle dependency.

0.2.0 (2026-02-24)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_newton.assets` module containing Newton-specific asset implementations:

  * :class:`~isaaclab_newton.assets.Articulation` and :class:`~isaaclab_newton.assets.ArticulationData`:
    Newton-specific implementation for articulated rigid body systems (e.g., robots). Extends
    :class:`~isaaclab.assets.BaseArticulation` with Newton's ``ArticulationView`` API for
    GPU-accelerated simulation of multi-joint systems.

  * :class:`~isaaclab_newton.assets.RigidObject` and :class:`~isaaclab_newton.assets.RigidObjectData`:
    Newton-specific implementation for single rigid body assets. Extends
    :class:`~isaaclab.assets.BaseRigidObject` with Newton's simulation API for efficient
    rigid body state queries and writes.

* Added warp kernel modules for fused GPU computations:

  * :mod:`isaaclab_newton.assets.kernels` — shared kernels for root state extraction,
    velocity transforms, COM/link frame conversions, and data write-back.
  * :mod:`isaaclab_newton.assets.articulation.kernels` — articulation-specific kernels
    for joint state, soft limits, actuator state updates, and friction properties.

* All ``.data.*`` properties use ``wp.array`` with structured warp types
  (``wp.vec3f``, ``wp.quatf``, ``wp.transformf``, ``wp.spatial_vectorf``),
  matching the same convention used by ``isaaclab_physx``.

* All write methods follow the ``_index`` / ``_mask`` split for explicit
  sparse-index vs. boolean-mask semantics.


0.1.0 (2026-02-16)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added empty package
