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
* Added :attr:`~isaaclab.assets.BaseArticulation.num_base_dofs` — number of
  free DoFs of the floating base (``0`` for fixed-base, ``6`` for floating-
  base). Use it to map an actuated-joint index ``j`` to its column in the
  Jacobian / mass matrix / gravity vector via ``j + num_base_dofs``.

The Jacobian / mass-matrix / gravity-comp DoF axis includes the floating-
base DoFs at the front: shape ``(N, num_jacobi_bodies, 6, num_joints +
num_base_dofs)`` for the Jacobian and ``(N, num_joints + num_base_dofs,
num_joints + num_base_dofs)`` for the mass matrix. This matches the
cross-library industry convention (Pinocchio's ``nv = 6 + n_actuated``,
Drake's ephemeral floating joint, MuJoCo's ``<freejoint/>``, RBDL's
``JointTypeFloatingBase``, OCS2's ``generalizedCoordinatesNum =
6 + actuatedJointsNum``, iDynTree's ``getFreeFloatingMassMatrix``
returning ``(6 + dofs, 6 + dofs)``).

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
* Action terms (DiffIK / OSC / RMPFlow / Pink) compute their Jacobian
  joint-axis indices via
  ``[j + asset.num_base_dofs for j in joint_ids]``, which is ``0`` for
  fixed-base and ``+6`` for floating-base. Pink IK previously hardcoded
  a private ``_physx_floating_joint_indices_offset = 6``; that was
  removed in favor of the cross-backend property.
* PhysX backend's :attr:`body_link_jacobian_w` applies the COM→origin shift to
  PhysX's natively COM-referenced Jacobian. The previously-exposed
  ``Articulation.get_jacobians()`` was a passthrough that returned the raw
  COM-referenced Jacobian, while IK / OSC consumers also read
  :attr:`body_link_pose_w` as the EE pose setpoint — a frame mismatch that
  produced a ``ω × r_com_w`` per-body bias in tracking. The new property
  reads the same engine buffer and applies the shift so ``J · q_dot`` matches
  ``body_link_lin_vel_w``. Consumers that intentionally want the raw
  COM-referenced form can read :attr:`body_com_jacobian_w`.
