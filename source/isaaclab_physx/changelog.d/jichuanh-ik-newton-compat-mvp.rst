Added
^^^^^

* Added PhysX implementations of
  :attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w`,
  :attr:`~isaaclab.assets.BaseArticulationData.body_com_jacobian_w`,
  :attr:`~isaaclab.assets.BaseArticulationData.mass_matrix`, and
  :attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces`
  on :class:`~isaaclab_physx.assets.ArticulationData`. The COM
  variant is a passthrough to ``physx.ArticulationView.get_jacobians``;
  the link-origin variant applies a new
  :func:`~isaaclab_physx.assets.articulation.kernels.shift_jacobian_com_to_origin`
  Warp kernel to convert the COM-referenced linear-velocity rows to
  link-origin references using each body's pose and COM offset.
* Added :attr:`~isaaclab_physx.assets.Articulation.joint_to_jacobi_offset`
  override returning ``6`` for floating-base articulations (PhysX prepends
  6 floating-base DoFs to the Jacobian's joint axis) and ``0`` for fixed-base.

Fixed
^^^^^

* Fixed a latent correctness bug in IK / OSC controllers on the PhysX
  backend, where the previously-exposed Jacobian was COM-referenced but
  the controllers used :attr:`~isaaclab_physx.assets.ArticulationData.body_link_pose_w`
  as the EE pose setpoint. The frame mismatch caused tracking error on
  bodies whose COM offset is non-trivial. The new
  :attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w`
  applies the COM→origin shift so the Jacobian and pose share a
  reference point.
