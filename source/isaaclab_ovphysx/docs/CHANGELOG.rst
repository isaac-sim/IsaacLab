Changelog
---------

3.2.0 (2026-06-23)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_ovphysx.sensors.JointWrenchSensor` and
  :class:`~isaaclab_ovphysx.sensors.JointWrenchSensorData` so the factory
  :class:`~isaaclab.sensors.JointWrenchSensor` dispatches under the OVPhysX
  backend.
* Added :class:`~isaaclab_ovphysx.sensors.FrameTransformer` and
  :class:`~isaaclab_ovphysx.sensors.FrameTransformerData` for OVPhysX
  frame transform sensing.

Removed
^^^^^^^

* Removed :attr:`~isaaclab_ovphysx.assets.ArticulationData.body_incoming_joint_wrench_b`
  to match the PhysX and Newton backends. Add
  :class:`~isaaclab.sensors.JointWrenchSensorCfg` to the scene and read
  :attr:`~isaaclab.sensors.JointWrenchSensorData.force` and
  :attr:`~isaaclab.sensors.JointWrenchSensorData.torque` instead.


3.1.0 (2026-06-18)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_ovphysx.sensors.Imu` and
  :class:`~isaaclab_ovphysx.sensors.ImuData` implementing the
  :class:`~isaaclab.sensors.imu.BaseImu` /
  :class:`~isaaclab.sensors.imu.BaseImuData` contracts on the OVPhysX
  backend. Reports angular velocity and proper linear acceleration in
  the sensor body frame using ovphysx tensor bindings on the rigid-body
  ancestor of the sensor prim path.
* Added :mod:`isaaclab_ovphysx.sensors.ray_caster` with
  :class:`~isaaclab_ovphysx.sensors.ray_caster.RayCaster`,
  :class:`~isaaclab_ovphysx.sensors.ray_caster.RayCasterCamera`,
  :class:`~isaaclab_ovphysx.sensors.ray_caster.MultiMeshRayCaster`, and
  :class:`~isaaclab_ovphysx.sensors.ray_caster.MultiMeshRayCasterCamera`.
  Mirrors :mod:`isaaclab_physx.sensors.ray_caster` structure: a single
  ``_OvPhysxRayCasterMixin`` carries the backend-specific pose-tracking
  surface, reading body poses via the ovphysx
  ``create_tensor_binding(pattern=..., tensor_type=RIGID_BODY_POSE)``
  API. Static (non-physics) sensor frames fall back to a one-time USD
  pose snapshot. Unblocks ``Isaac-Velocity-Rough-Anymal-D-v0`` (the
  height_scanner now dispatches under OVPhysX).
* Added :class:`~isaaclab_ovphysx.sensors.Pva` and
  :class:`~isaaclab_ovphysx.sensors.PvaData` implementing the
  :class:`~isaaclab.sensors.pva.BasePva` /
  :class:`~isaaclab.sensors.pva.BasePvaData` contracts on the OVPhysX
  backend. Reports world-frame pose, body-frame linear and angular
  velocities, body-frame coordinate linear and angular accelerations,
  and projected gravity using ovphysx tensor bindings on the rigid-body
  ancestor of the sensor prim path. Linear and angular accelerations
  are coordinate accelerations (zero at rest, ``-g`` in freefall) and
  do not include the IMU's gravity bias — projected gravity is reported
  separately as the unit gravity direction vector.
* Added :meth:`~isaaclab_ovphysx.physics.OvPhysxManager.get_gravity`
  classmethod mirroring PhysX's ``SimulationView.get_gravity()`` so
  backend-agnostic sensor code can read scene gravity through a single
  entry point.


3.0.6 (2026-06-17)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added a ``skip_forward`` argument to the root, body, and joint state writers (e.g.
  ``write_root_link_pose_to_sim_index``) to defer cached-buffer invalidation when several
  writes are batched before a single forward pass.

Fixed
^^^^^

* Fixed stale cached asset pose and velocity state after simulation state writes.


3.0.5 (2026-06-16)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Reused shared path-expression helpers when deriving OVPhysX schema-root view expressions.


3.0.4 (2026-06-10)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Re-enabled both CPU and GPU coverage in CI for OVPhysX tests by tagging
  :file:`test/assets/test_articulation.py`,
  :file:`test/assets/test_rigid_object.py`,
  :file:`test/assets/test_rigid_object_collection.py`,
  :file:`test/sensors/test_contact_sensor.py`, and
  :file:`test/sim/test_views_xform_prim_ovphysx.py` with the new
  ``device_split`` pytest marker, which causes the CI driver to invoke each
  file once per device in separate subprocesses. Works around the
  ``ovphysx<=0.3.7`` process-global device lock (gap G5).
* Stopped OVPhysX assets from enqueueing redundant USD replication work
  during scene cloning.


3.0.3 (2026-06-09)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Added an actionable install error when the optional ``ovphysx`` runtime wheel
  is missing from the :mod:`isaaclab_ovphysx` backend.


3.0.2 (2026-06-05)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the OVPhysX optional runtime dependency to install ``ovphysx==0.4.13``
  instead of accepting newer breaking releases.


3.0.1 (2026-06-03)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added torch tensor input support to
  :meth:`~isaaclab_ovphysx.assets.RigidObjectCollection.reshape_data_to_view_3d`.

Changed
^^^^^^^

* **Breaking:** :meth:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView.get_scales`
  now returns a :class:`~isaaclab.utils.warp.ProxyArray`, matching the updated
  :class:`~isaaclab.sim.views.BaseFrameView` contract. Callers that fed the
  return value into Warp kernels or ``set_scales`` need to extract the
  underlying array via ``.warp``.


3.0.0 (2026-05-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_ovphysx.assets.RigidObjectCollection` and
  :class:`~isaaclab_ovphysx.assets.RigidObjectCollectionData` for the
  OVPhysX backend, completing the rigid-body asset surface alongside
  :class:`~isaaclab_ovphysx.assets.RigidObject` and
  :class:`~isaaclab_ovphysx.assets.Articulation`. Supports
  ``(env, body)`` dual indexing and per-body property setters. Uses the
  ovphysx 0.4.3+ native fused multi-prim binding API
  (``create_tensor_binding(prim_paths=[...])``) so one binding spans all
  ``num_instances * num_bodies`` prims per tensor type, mirroring the
  strided-view reshape pattern used by the PhysX collection.
* Added :class:`~isaaclab_ovphysx.sensors.ContactSensor`,
  :class:`~isaaclab_ovphysx.sensors.ContactSensorCfg`, and
  :class:`~isaaclab_ovphysx.sensors.ContactSensorData` for the OVPhysX
  backend, satisfying the
  :class:`~isaaclab.sensors.contact_sensor.BaseContactSensor` and
  :class:`~isaaclab.sensors.contact_sensor.BaseContactSensorData`
  contracts. Wires net contact forces and the per-partner force matrix
  through the OVPhysX :class:`ovphysx.api.ContactBinding` API
  (``read_net_forces`` / ``read_force_matrix``); optional pose tracking
  reads through a ``RIGID_BODY_POSE`` :class:`ovphysx.api.TensorBinding`.
  Air/contact time tracking,
  :meth:`~isaaclab_ovphysx.sensors.ContactSensor.compute_first_contact`,
  :meth:`~isaaclab_ovphysx.sensors.ContactSensor.compute_first_air`,
  history buffers, and reset semantics mirror the PhysX backend.
* Added the shared
  :mod:`isaaclab_ovphysx.sensors.kernels` module with
  :func:`~isaaclab_ovphysx.sensors.kernels.concat_pos_and_quat_to_pose_kernel`
  and the 1D variant for reuse across future OVPhysX sensors.
* Added :class:`~isaaclab_ovphysx.sim.views.OvPhysxFrameView`, a
  Warp-native batched-prim view that reads world poses from the OVPhysX
  scene data provider's ``body_q`` array. Mirrors
  :class:`~isaaclab_newton.sim.views.NewtonSiteFrameView` in semantics
  and API: ``set_world_poses`` / ``set_local_poses`` update the view's
  internal ``site_local`` buffer and never mutate the physics state.
  Scales and visibility delegate to a lazy internal
  :class:`~isaaclab.sim.views.UsdFrameView`.

Changed
^^^^^^^

* Changed the existing
  ``source/isaaclab_ovphysx/test/sensors/check_contact_sensor.py``
  stubs to real tests adapted from the PhysX
  :mod:`isaaclab_physx.test.sensors.test_contact_sensor` suite. The
  three tests that exercise ``track_contact_points`` or
  ``track_friction_forces`` are decorated with
  :func:`pytest.mark.skip` until the OVPhysX wheel ships
  tensor-friendly per-sensor reads (see
  ``docs/superpowers/specs/2026-04-27-ovphysx-contact-api-gaps.md``);
  the test bodies are preserved so the decorator can be removed in a
  follow-up.
* Updated imports of :class:`~isaaclab.scene_data.SceneDataBackend` and
  :class:`~isaaclab.scene_data.SceneDataFormat` in
  :mod:`~isaaclab_ovphysx.physics.ovphysx_manager` to their new location in
  :mod:`isaaclab.scene_data` (previously :mod:`isaaclab.physics`).

Removed
^^^^^^^

* **Breaking:** Removed the five
  ``source/isaaclab_ovphysx/test/sensors/check_contact_sensor.py``
  ``pytest.skip("Contact sensor not yet supported by ovphysx
  backend.")`` placeholders in favour of the real test suite above.
  No public migration is required; the placeholder names did not
  appear in any external API.


2.1.0 (2026-05-19)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_ovphysx.physics.OvPhysxSceneDataBackend` and
  :meth:`~isaaclab_ovphysx.physics.OvPhysxManager.get_scene_data_backend`
  so the central
  :class:`~isaaclab.scene.scene_data_provider.SceneDataProvider`
  (introduced in #5128) can expose OVPhysX rigid-body transforms to
  Rerun, Viser, and the native Newton viewport. The backend creates one
  ovphysx ``TT.RIGID_BODY_POSE`` binding per distinct env-wildcard
  rigid-body prim path (cartpole produces 2 bindings, Allegro hand ~17,
  each covering all envs), reads each binding into a pre-allocated
  ``wp.float32`` staging buffer via ``TensorBinding.read(dst)``, and
  concatenates the per-binding reads into a single ``wp.transformf``
  merged buffer that the central provider consumes as
  :class:`~isaaclab.physics.scene_data_backend.SceneDataFormat.Transform`.


2.0.0 (2026-05-17)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_ovphysx.assets.Articulation` and
  :class:`~isaaclab_ovphysx.assets.ArticulationData` for multi-DOF articulated
  robots against the OVPhysX backend, satisfying the
  :class:`~isaaclab.assets.BaseArticulation` and
  :class:`~isaaclab.assets.BaseArticulationData` contracts. Public surface
  matches the PhysX/Newton conventions: kwarg-only ``write_*_to_sim_index`` /
  ``write_*_to_sim_mask`` writers and ``set_*_index`` / ``set_*_mask`` setters
  for root state, joint state, joint properties, body properties, joint
  command targets, fixed/spatial tendon properties, and external wrenches via
  :attr:`~isaaclab_ovphysx.assets.Articulation.instantaneous_wrench_composer`
  / :attr:`~isaaclab_ovphysx.assets.Articulation.permanent_wrench_composer`.
  The full IsaacLab actuator pipeline (``compute`` /
  ``_apply_actuator_model`` / ``_process_actuators_cfg``) is implemented on
  top of the wheel's ``DOF_ACTUATION_FORCE`` /
  ``DOF_POSITION_TARGET`` / ``DOF_VELOCITY_TARGET`` bindings.
* Added articulation-specific Warp kernels in
  :mod:`isaaclab_ovphysx.assets.articulation.kernels`: soft-limit refresh,
  default-joint-pos clamp, friction-component scatter (index + mask
  variants).  Six articulation kernels were also folded into the shared
  :mod:`isaaclab_ovphysx.assets.kernels` module for reuse with
  :class:`~isaaclab_ovphysx.assets.RigidObject` and
  :class:`~isaaclab_ovphysx.assets.RigidObjectCollection`.
* Added init-time validation in
  :meth:`~isaaclab_ovphysx.assets.Articulation._initialize_impl` that raises
  ``RuntimeError`` when ``cfg.prim_path`` resolves to no
  ``UsdPhysics.ArticulationRootAPI`` prim or to multiple roots, and
  ``ValueError`` (via :meth:`_validate_cfg`) when any default joint
  position is outside ``[lower, upper]`` or any default joint velocity
  exceeds the per-joint maximum.  Mirrors the PhysX backend.
* Added support for ``cfg.articulation_root_prim_path`` in
  :meth:`~isaaclab_ovphysx.assets.Articulation._initialize_impl`: when the
  user supplies an explicit subpath the binding pattern is extended
  directly instead of running the auto-discovery walk, and a
  ``RuntimeError`` is raised when the resulting expression resolves to no
  prim in the USD stage.

Changed
^^^^^^^

* **Breaking:** Renamed ``Articulation`` write/set methods to the dual
  ``*_index`` / ``*_mask`` form and dropped the legacy ``full_data``
  flag.  Index methods accept partial data shaped
  ``(len(env_ids), len(joint_or_body_ids), ...)``; mask methods accept
  full-shape data and a ``wp.bool`` mask.  All keyword-only arguments live
  after ``*,``; no positional fall-through.  Migration: replace
  ``write_X_to_sim(..., from_mask=True)`` with ``write_X_to_sim_mask(..., mask=...)``.
* **Breaking:** Removed the ``_write_body_state`` plumbing layer.
  Deprecated state-writer shims (``write_root_state_to_sim``,
  ``write_root_com_state_to_sim``, ``write_root_link_state_to_sim``,
  joint-state equivalents) now call the public ``write_*_to_sim_index``
  methods directly.  Behaviour is preserved.
* Changed ``Articulation.root_view`` to return the per-tensor-type bindings
  dict (``self._bindings``).  The OVPhysX wheel does not expose a single
  ``ArticulationView`` object; callers that previously walked
  ``root_view.shared_metatype`` / ``root_view.max_dofs`` should read from
  :attr:`~isaaclab_ovphysx.assets.Articulation.num_joints` /
  :attr:`~isaaclab_ovphysx.assets.Articulation.num_bodies` /
  :attr:`~isaaclab_ovphysx.assets.Articulation.body_names` /
  :attr:`~isaaclab_ovphysx.assets.Articulation.joint_names` instead.
* Changed every ``ArticulationData`` public property to return a
  :class:`~isaaclab.utils.ProxyArray` (warp + torch dual view); raw
  ``wp.array`` is reserved for one-shot config buffers.  Eager
  ``TimestampedBufferWarp`` allocation in :meth:`_create_buffers` makes
  every buffer a single source of truth — no
  ``_invalidate_caches`` / ``_ensure_*_buffers`` machinery.
* Changed ``Articulation`` body and DOF property writers to honor the
  wheel's actual binding device.  Tensor-type membership in
  :data:`isaaclab_ovphysx.tensor_types._CPU_ONLY_TYPES` now reflects what
  the wheel exposes: ``BODY_MASS``, ``BODY_COM_POSE``, ``BODY_INERTIA``,
  ``DOF_STIFFNESS``, ``DOF_DAMPING``, ``DOF_LIMIT``, ``DOF_MAX_VELOCITY``,
  ``DOF_MAX_FORCE``, ``DOF_ARMATURE``, ``DOF_FRICTION_PROPERTIES`` are
  CPU-only (write goes through pinned-host staging); fixed and spatial
  tendon bindings write directly from sim-device buffers.
* Changed :meth:`~isaaclab_ovphysx.assets.Articulation.write_joint_friction_coefficient_to_sim_index`
  / ``_mask`` to accept ``joint_dynamic_friction_coeff`` and
  ``joint_viscous_friction_coeff`` keyword arguments (each
  ``float | torch.Tensor | wp.array | None``).  ``None`` preserves the
  existing component on the wheel; matches the PhysX backend.
* Changed :meth:`~isaaclab_ovphysx.assets.Articulation.write_joint_position_limit_to_sim_index`
  / ``_mask`` to clamp ``default_joint_pos`` and refresh
  ``soft_joint_pos_limits`` when the new hard limits invalidate the
  defaults, matching the PhysX backend (with a
  ``warn_limit_violation`` log).
* Changed every fixed/spatial tendon ``set_*_index`` / ``set_*_mask`` setter
  to accept a scalar :class:`float` for the value argument; broadcast is
  materialized via :meth:`_broadcast_scalar_to_2d`.  Mirrors PhysX.
* Implemented the previously stubbed
  :meth:`~isaaclab_ovphysx.assets.Articulation.write_fixed_tendon_properties_to_sim_index`
  / ``_mask`` and
  :meth:`~isaaclab_ovphysx.assets.Articulation.write_spatial_tendon_properties_to_sim_index`
  / ``_mask``: each iterates the per-tensor bindings since the OVPhysX
  wheel has no batch ``set_*_tendon_properties`` setter.

Removed
^^^^^^^

* **Breaking:** Removed the ``full_data`` keyword-argument from every
  ``Articulation`` ``*_index`` writer/setter.  Index methods now strictly
  accept partial data; full-data callers should use the matching
  ``*_mask`` overload.
* Removed the stop-gap :mod:`isaaclab_ovphysx.assets.kernels_old` module;
  the six articulation kernels it housed
  (``_compose_root_com_pose``, ``_compute_heading``, ``_copy_first_body``,
  ``_projected_gravity``, ``_world_vel_to_body_ang``,
  ``_world_vel_to_body_lin``) are now in
  :mod:`isaaclab_ovphysx.assets.kernels`.


1.0.0 (2026-05-14)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_ovphysx.assets.RigidObject` and
  :class:`~isaaclab_ovphysx.assets.RigidObjectData` for single-actor rigid-body
  simulation against the OVPhysX backend, satisfying the
  :class:`~isaaclab.assets.BaseRigidObject` and
  :class:`~isaaclab.assets.BaseRigidObjectData` contracts. Public surface
  matches the PhysX/Newton conventions: ``write_root_*_to_sim_index`` /
  ``write_root_*_to_sim_mask`` writers (link- and com-frame variants),
  ``set_masses_*``, ``set_coms_*``, ``set_inertias_*`` setters, and the
  external-wrench composers exposed via
  :meth:`~isaaclab_ovphysx.assets.RigidObject.set_external_force_and_torque`.
* Added the ``RIGID_BODY_*`` :class:`TensorType` aliases in
  :mod:`isaaclab_ovphysx.tensor_types` (``POSE``, ``VELOCITY``, ``WRENCH``,
  ``MASS``, ``COM_POSE``, ``INERTIA``; plus ``ACCELERATION``, ``INV_MASS``,
  ``INV_INERTIA`` declared for forward compatibility once the wheel ships
  them).
* Added :class:`~isaaclab_ovphysx.assets.kernels` as a shared Warp-kernel
  module (frame conversions, state concatenation, finite-difference
  acceleration, index- and mask-style scatter writers) consumed by both the
  rigid-object and articulation assets.
* Added USD prim-scan validation in
  :meth:`~isaaclab_ovphysx.assets.RigidObject._initialize_impl`: a clear
  ``RuntimeError`` is raised when ``cfg.prim_path`` resolves to no
  ``UsdPhysics.RigidBodyAPI`` prim, multiple rigid-body prims, or a prim with
  an enabled ``UsdPhysics.ArticulationRootAPI``.

Changed
^^^^^^^

* Changed :meth:`~isaaclab_ovphysx.physics.OvPhysxManager._release_physx` to
  perform a soft reset (``physx.reset()``) and keep the cached
  :class:`ovphysx.PhysX` reference alive across
  :class:`~isaaclab.sim.SimulationContext` lifetimes, instead of dropping the
  reference and triggering the wheel's dual-Carbonite static-destructor race.
  :meth:`~isaaclab_ovphysx.physics.OvPhysxManager._warmup_and_load` now reuses
  the cached instance on subsequent calls.
* Changed :meth:`~isaaclab_ovphysx.physics.OvPhysxManager._warmup_and_load` to
  raise a clear ``RuntimeError`` when a later
  :class:`~isaaclab.sim.SimulationContext` requests a different device than
  the one the process is locked to, surfacing the wheel's process-global
  device-mode lock as a Python error before
  :exc:`ovphysx.types.PhysXDeviceError` would fire.
* Changed :meth:`~isaaclab_ovphysx.physics.OvPhysxManager._configure_physx_scene_prim`
  to apply the ``UsdPhysics.PhysxSceneAPI`` schema and
  ``enableSceneQuerySupport`` on both CPU and GPU; GPU-only attributes
  (``enableGPUDynamics``, ``broadphaseType``, the ``gpu*`` capacity attributes
  from :class:`~isaaclab_ovphysx.physics.OvPhysxCfg`) remain gated on
  ``device == "gpu"``.
* Inherits the base
  :attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w`,
  :attr:`~isaaclab.assets.BaseArticulationData.body_com_jacobian_w`,
  :attr:`~isaaclab.assets.BaseArticulationData.mass_matrix`, and
  :attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces`
  :class:`NotImplementedError` defaults — ovphysx's OmniGraph-based view
  does not expose articulation Jacobians, mass matrices, or gravity
  compensation. Use the PhysX or Newton backends for task-space
  controllers.


0.1.4 (2026-05-09)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed OvPhysX articulation tensor reads and writes for ``ovphysx`` 0.4
  compatibility.
* Restored DirectGPU startup settings for OvPhysX GPU simulations.


0.1.3 (2026-05-08)
~~~~~~~~~~~~~~~~~~

Removed
^^^^^^^

* Removed ``ArticulationData.body_incoming_joint_wrench_b`` to match the
  shared articulation data API. Code that needs incoming joint reaction
  wrenches should use a backend joint-wrench sensor instead of the articulation
  data object.


0.1.2 (2026-04-23)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Wrapped :attr:`~isaaclab_ovphysx.assets.ArticulationData.GRAVITY_VEC_W` and
  :attr:`~isaaclab_ovphysx.assets.ArticulationData.FORWARD_VEC_B` in
  :class:`~isaaclab.utils.warp.ProxyArray` to match the PhysX and Newton
  backends. Public observations such as
  :func:`~isaaclab.envs.mdp.observations.projected_gravity` access
  ``asset.data.GRAVITY_VEC_W.torch``; the previous raw ``wp.array`` lacked
  ``.torch`` and raised ``AttributeError`` on the ovphysx backend.


0.1.1 (2026-04-21)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Replaced private ``_find_names`` (fnmatch + regex) with the standard
  :func:`~isaaclab.utils.string.resolve_matching_names` for all finder
  methods, unifying name-resolution behavior across backends. Fnmatch-style
  glob patterns (e.g. ``joint_*``) are no longer supported; use regex
  equivalents (e.g. ``joint_.*``). ``find_fixed_tendons`` and
  ``find_spatial_tendons`` now raise ``ValueError`` on empty tendon lists,
  matching the PhysX backend.
* Changed ``find_joints`` ``joint_subset`` parameter from ``list[int]``
  (indices) to ``list[str]`` (names) to match the ``BaseArticulation``
  interface. Callers passing indices should convert to names first.


0.1.0 (2026-04-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of the ``isaaclab_ovphysx`` extension.
