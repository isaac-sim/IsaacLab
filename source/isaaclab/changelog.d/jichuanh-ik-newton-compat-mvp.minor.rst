Added
^^^^^

* Added :meth:`~isaaclab.assets.BaseArticulation.get_jacobians`,
  :meth:`~isaaclab.assets.BaseArticulation.get_mass_matrix`, and
  :meth:`~isaaclab.assets.BaseArticulation.get_gravity_compensation_forces`
  abstract methods, so task-space controllers no longer call PhysX-only
  ``root_view`` accessors directly. Backends without a native
  primitive raise :class:`NotImplementedError`.
* Added :attr:`~isaaclab.assets.BaseArticulation.joint_to_jacobi_offset`
  property: the offset added to a state-space joint index to get the
  matching Jacobian column index. Concrete with
  :class:`NotImplementedError` on the base class so backends declare
  their own convention explicitly. Returns 0 on backends whose
  Jacobian counts the same DoFs as joint-state buffers, and 6 on
  PhysX floating-base where the Jacobian prepends 6 floating-base
  DoFs.

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
  in the three task-space action terms with the new
  :attr:`~isaaclab.assets.BaseArticulation.joint_to_jacobi_offset`
  property, so backends with different floating-base joint-axis
  conventions work without changes to the action terms.
