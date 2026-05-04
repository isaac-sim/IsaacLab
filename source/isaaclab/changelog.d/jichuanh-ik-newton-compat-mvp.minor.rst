Added
^^^^^

* Added :attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w` and
  :attr:`~isaaclab.assets.BaseArticulationData.body_com_jacobian_w` properties,
  exposing the per-body geometric Jacobian referenced at the link origin and
  body center of mass respectively. The pair mirrors the existing
  :attr:`~isaaclab.assets.BaseArticulationData.body_link_pose_w` /
  :attr:`~isaaclab.assets.BaseArticulationData.body_com_pose_w` and
  :attr:`~isaaclab.assets.BaseArticulationData.body_link_vel_w` /
  :attr:`~isaaclab.assets.BaseArticulationData.body_com_vel_w` exposure pattern.
  Backends without a native primitive raise :class:`NotImplementedError`.
* Added :attr:`~isaaclab.assets.BaseArticulationData.mass_matrix` property,
  exposing the joint-space generalized mass matrix ``M(q)``.
* Added :attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces`
  property, exposing the joint-space gravity-loading torque vector ``g(q)``.

The Jacobian / mass-matrix / gravity-comp joint axis is **actuated-only on
every backend** — shape ``(N, num_jacobi_bodies, 6, num_joints)`` for the
Jacobian and ``(N, num_joints, num_joints)`` for the mass matrix. PhysX
floating-base previously prepended 6 base-DoF columns; the data-layer
accessors strip them so consumers don't need to bridge the asymmetry.

Changed
^^^^^^^

* Migrated :class:`~isaaclab.envs.mdp.actions.task_space_actions.DifferentialInverseKinematicsAction`,
  :class:`~isaaclab.envs.mdp.actions.task_space_actions.OperationalSpaceControllerAction`,
  and :class:`~isaaclab.envs.mdp.actions.rmpflow_task_space_actions.RMPFlowAction`
  to fetch dynamic quantities through the new
  :class:`~isaaclab.assets.BaseArticulationData` properties instead of the
  PhysX-only ``root_view``. The OSC action term now also gates the
  per-step mass-matrix and gravity-compensation fetches behind the
  controller cfg's :attr:`inertial_dynamics_decoupling`,
  :attr:`nullspace_control`, and :attr:`gravity_compensation` flags
  so backends without a native primitive are not invoked when the
  controller does not consume the result.
* Removed the hard-coded ``+6`` floating-base Jacobian column offset
  in the three task-space action terms. Action-term joint-id arithmetic
  (``[i + offset for i in joint_ids]``) is gone; consumers index by the
  state-space joint id directly.
* PhysX backend's :attr:`body_link_jacobian_w` applies the COM→origin shift to
  PhysX's natively COM-referenced Jacobian. The previously-exposed
  ``Articulation.get_jacobians()`` was a passthrough that returned the raw
  COM-referenced Jacobian, while IK / OSC consumers also read
  :attr:`body_link_pose_w` as the EE pose setpoint — a frame mismatch that
  produced a ``ω × r_com_w`` per-body bias in tracking. The new property
  reads the same engine buffer and applies the shift so ``J · q_dot`` matches
  ``body_link_lin_vel_w``. Consumers that intentionally want the raw
  COM-referenced form can read :attr:`body_com_jacobian_w`.
