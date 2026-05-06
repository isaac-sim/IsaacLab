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
