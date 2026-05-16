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
