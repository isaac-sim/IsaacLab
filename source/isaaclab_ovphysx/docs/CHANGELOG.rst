Changelog
---------

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
