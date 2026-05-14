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
