* Fixed :func:`~isaaclab.utils.math.euler_xyz_from_quat` returning roll and yaw angles that describe a different
  rotation at gimbal lock (pitch of ±90°). Roll and yaw are now computed from their half-angle sum and difference,
  so the returned angles reproduce the input rotation.
