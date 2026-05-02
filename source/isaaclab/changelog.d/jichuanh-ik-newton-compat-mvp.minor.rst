Added
^^^^^

* Added :meth:`~isaaclab.assets.BaseArticulation.get_jacobians`,
  :meth:`~isaaclab.assets.BaseArticulation.get_mass_matrix`, and
  :meth:`~isaaclab.assets.BaseArticulation.get_gravity_compensation_forces`
  abstract methods, so task-space controllers no longer call PhysX-only
  ``root_view`` accessors directly. Backends without a native
  primitive raise :class:`NotImplementedError`.
* Added :attr:`~isaaclab.assets.BaseArticulation.num_jacobi_joints`
  property reporting the size of the Jacobian's joint axis. Concrete
  with :class:`NotImplementedError` on the base class -- backends
  declare their own convention. Used by the task-space action terms
  to translate logical joint indices to Jacobian column indices via
  ``num_jacobi_joints - num_joints`` (0 or 6).

Changed
^^^^^^^

* Migrated :class:`~isaaclab.envs.mdp.actions.task_space_actions.DifferentialInverseKinematicsAction`,
  :class:`~isaaclab.envs.mdp.actions.task_space_actions.OperationalSpaceControllerAction`,
  and :class:`~isaaclab.envs.mdp.actions.rmpflow_task_space_actions.RMPFlowAction`
  to fetch dynamic quantities through the new
  :class:`~isaaclab.assets.BaseArticulation` accessors instead of the
  PhysX-only ``root_view``. The OSC action term now also gates the
  per-step mass-matrix and gravity-compensation fetches behind the
  controller cfg's :attr:`inertial_dynamics_decoupling`,
  :attr:`nullspace_control`, and :attr:`gravity_compensation` flags
  so backends without a native primitive are not invoked when the
  controller does not consume the result.
* Replaced the hard-coded ``+6`` floating-base Jacobian column offset
  in the three task-space action terms with
  ``num_jacobi_joints - num_joints`` so backends with different
  floating-base joint-axis conventions work without changes to the
  action terms.
