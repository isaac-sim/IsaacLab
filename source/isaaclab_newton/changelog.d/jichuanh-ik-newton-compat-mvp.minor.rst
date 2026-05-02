Added
^^^^^

* Added :attr:`~isaaclab_newton.assets.Articulation.joint_to_jacobi_offset`
  override returning ``0``. Newton's
  ``ArticulationView.joint_dof_count`` already counts the 6
  floating-base DoFs on floating-base assets, so a state-space
  joint index is also the matching Jacobian column index without
  any shift.
* Added :meth:`~isaaclab_newton.assets.Articulation.get_jacobians`
  and :meth:`~isaaclab_newton.assets.Articulation.get_mass_matrix`
  wrapping ``ArticulationView.eval_jacobian`` and
  ``ArticulationView.eval_mass_matrix`` and returning view-sized
  arrays matching the PhysX shape contract. Per-step behavior is
  allocation-free and safe under CUDA graph capture: source / scratch
  / output buffers are pre-allocated in ``_create_buffers``, and new
  :func:`~isaaclab_newton.assets.articulation.kernels.gather_jacobian_rows`
  and :func:`~isaaclab_newton.assets.articulation.kernels.gather_mass_matrix_rows`
  Warp kernels gather just this view's rows from the model-sized
  buffers Newton populates.
* Added a new
  :func:`~isaaclab_newton.assets.articulation.kernels.shift_jacobian_com_to_origin`
  Warp kernel that applies the
  ``v_origin = v_com - omega x (R · body_com_pos_b)`` shift to the
  linear-velocity rows of the gathered, view-sized Jacobian, so the
  returned Jacobian's linear rows reference the link origin in world
  frame -- matching the cross-backend
  :meth:`~isaaclab.assets.BaseArticulation.get_jacobians` contract.

Fixed
^^^^^

* Fixed :meth:`~isaaclab_newton.assets.Articulation.get_jacobians` and
  :meth:`~isaaclab_newton.assets.Articulation.get_mass_matrix` returning
  the wrong DoF columns for floating-base articulations. The IsaacLab
  Newton view is constructed with ``exclude_joint_types=[FREE, FIXED]``
  so its joint count excludes the free-root joint, but Newton's
  :func:`newton.eval_jacobian` and :func:`newton.eval_mass_matrix`
  write the full articulation buffer with the free-root's 6 DoF columns
  at the start. The view-sized gather kernels now apply a matching
  ``dof_offset`` (0 fixed-base, 6 floating-base) so the returned
  buffers contain only the actuated joints' columns. Fixed-base assets
  (e.g. the Franka tracking-accuracy tests) are unaffected; floating-
  base assets (e.g. quadrupeds) previously returned root columns where
  the action terms expected actuated columns.

Changed
^^^^^^^

* :meth:`~isaaclab_newton.assets.Articulation.get_gravity_compensation_forces`
  raises :class:`NotImplementedError` with a message pointing at the
  upstream gap. Newton's ``ArticulationView`` does not expose an
  inverse-dynamics primitive yet (upstream Newton issues
  `#2497 <https://github.com/newton-physics/newton/issues/2497>`_,
  `#2529 <https://github.com/newton-physics/newton/issues/2529>`_,
  `#2625 <https://github.com/newton-physics/newton/issues/2625>`_).
  OSC users on Newton must set ``gravity_compensation=False`` until
  upstream lands the primitive.
