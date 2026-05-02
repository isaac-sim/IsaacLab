Added
^^^^^

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
